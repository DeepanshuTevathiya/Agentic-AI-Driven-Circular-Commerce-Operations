"""
Lead 2 Supplier Gateway Abstraction & Deterministic Simulation Adapter
======================================================================
Defines the supplier gateway abstraction, typed transmission contracts,
and a deterministic simulated supplier adapter for Phase 5.4.1.

ARCHITECTURAL PRINCIPLES:
1. DECOUPLED FROM STATE MACHINE:
   Supplier network communication and transmission contracts live outside
   the domain state machine and persistence repository.
2. STRICT PHASE 4 APPROVAL GOVERNANCE:
   Only purchase orders with valid, verified human approval evidence
   can be transmitted. Unapproved, rejected, or tampered orders are rejected.
3. COMMERCIAL PURCHASE ORDER IMMUTABILITY:
   PurchaseOrder objects are treated as strictly immutable contracts of record.
4. DETERMINISTIC SIMULATION:
   No random number generators, wall-clock jitter, external APIs, or network
   calls. The same input and configured scenario always produce identical results.
5. STRICT IDEMPOTENCY:
   Repeating a transmission with the same idempotency key is an exact no-op
   and creates no duplicate supplier submission. Reusing a key with a differing
   PO or payload raises a clear conflict error.
"""

from abc import ABC, abstractmethod
from datetime import datetime as dt_datetime, timedelta, timezone
from enum import Enum
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..common.config import CATEGORY_SUPPLIER_PARAMS
from ..procurement.approval import can_be_dispatched, verify_approval_evidence
from ..procurement.schemas import PurchaseOrder, PurchaseOrderStatus
from .schemas import (
    SupplierAcknowledgment,
    SupplierAcknowledgmentLine,
    SupplierAcknowledgmentStatus,
    compute_payload_fingerprint,
    validate_and_normalize_iso_timestamp,
)


class SupplierGatewayError(Exception):
    """Base exception for supplier gateway and transmission errors."""
    pass


class SupplierGatewayIdempotencyConflictError(SupplierGatewayError, ValueError):
    """Raised when a transmission idempotency key is reused with differing payload or PO."""
    pass


class InvalidPurchaseOrderError(SupplierGatewayError, ValueError):
    """Raised when an unapproved, unverified, or invalid Purchase Order is submitted for transmission."""
    pass


class SupplierTransmissionStatus(str, Enum):
    """
    Validated outcome classification for a purchase order transmission.
    Distinguishes confirmed responses, pre-acceptance failure, and ambiguous outcome.
    """
    CONFIRMED_ACCEPTANCE = "CONFIRMED_ACCEPTANCE"
    CONFIRMED_REJECTION = "CONFIRMED_REJECTION"
    TRANSMISSION_FAILURE = "TRANSMISSION_FAILURE"
    AMBIGUOUS_OUTCOME = "AMBIGUOUS_OUTCOME"


class SupplierSimulationScenario(str, Enum):
    """
    Explicitly configured simulation scenarios for the deterministic adapter.
    """
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    TRANSMISSION_FAILURE = "TRANSMISSION_FAILURE"
    AMBIGUOUS_OUTCOME = "AMBIGUOUS_OUTCOME"


def _validate_non_empty_str(v: Any, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


class SupplierTransmissionResult(BaseModel):
    """
    Validated immutable outcome contract of a Purchase Order transmission to a supplier.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    transmission_id: str = Field(..., description="Unique transmission event identifier")
    idempotency_key: str = Field(..., description="Caller-provided transmission idempotency key")
    po_id: str = Field(..., description="Purchase Order identifier")
    supplier_id: str = Field(..., description="Target supplier identifier")
    status: SupplierTransmissionStatus = Field(..., description="Transmission outcome classification")
    transmitted_at: str = Field(..., description="ISO 8601 UTC timestamp of transmission")

    correlation_id: str = Field(..., description="Correlation ID tracking request across systems")
    fo_id: Optional[str] = Field(default=None, description="Associated Fulfilment Order identifier if known")
    supplier_reference: Optional[str] = Field(default=None, description="External vendor order reference")

    acknowledgment: Optional[SupplierAcknowledgment] = Field(
        default=None,
        description="Validated supplier acknowledgment when status is CONFIRMED_ACCEPTANCE or CONFIRMED_REJECTION"
    )

    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error classification if transmission failed or timed out"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Human-readable diagnostic description of failure or ambiguous outcome"
    )

    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured operational and debugging metadata"
    )

    @field_validator("transmission_id", "idempotency_key", "po_id", "supplier_id", "correlation_id")
    @classmethod
    def validate_required_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("fo_id", "supplier_reference", "error_code", "error_message")
    @classmethod
    def validate_optional_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("transmitted_at", mode="before")
    @classmethod
    def validate_transmission_timestamp(cls, v: Any, info) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

    @model_validator(mode="after")
    def validate_status_and_contracts(self) -> "SupplierTransmissionResult":
        # 1. CONFIRMED_ACCEPTANCE
        if self.status == SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE:
            if self.acknowledgment is None:
                raise ValueError("acknowledgment is required when transmission status is CONFIRMED_ACCEPTANCE.")
            if self.acknowledgment.status not in [
                SupplierAcknowledgmentStatus.ACCEPTED,
                SupplierAcknowledgmentStatus.PARTIALLY_ACCEPTED,
            ]:
                raise ValueError(
                    f"Acknowledgment status '{self.acknowledgment.status.value}' conflicts with transmission status "
                    f"'{self.status.value}'."
                )
            if self.error_message or self.error_code:
                raise ValueError("error_code and error_message must be None for CONFIRMED_ACCEPTANCE.")

        # 2. CONFIRMED_REJECTION
        elif self.status == SupplierTransmissionStatus.CONFIRMED_REJECTION:
            if self.acknowledgment is None:
                raise ValueError("acknowledgment is required when transmission status is CONFIRMED_REJECTION.")
            if self.acknowledgment.status != SupplierAcknowledgmentStatus.REJECTED:
                raise ValueError(
                    f"Acknowledgment status '{self.acknowledgment.status.value}' conflicts with transmission status "
                    f"'{self.status.value}'."
                )
            if self.error_message or self.error_code:
                raise ValueError("error_code and error_message must be None for CONFIRMED_REJECTION.")

        # 3. TRANSMISSION_FAILURE
        elif self.status == SupplierTransmissionStatus.TRANSMISSION_FAILURE:
            if self.acknowledgment is not None:
                raise ValueError("acknowledgment must be None when transmission status is TRANSMISSION_FAILURE.")
            if not self.error_code or not self.error_message:
                raise ValueError("error_code and error_message are mandatory for TRANSMISSION_FAILURE.")

        # 4. AMBIGUOUS_OUTCOME
        elif self.status == SupplierTransmissionStatus.AMBIGUOUS_OUTCOME:
            if self.acknowledgment is not None:
                raise ValueError("acknowledgment must be None when transmission outcome is AMBIGUOUS_OUTCOME.")
            if not self.error_code or not self.error_message:
                raise ValueError("error_code and error_message are mandatory for AMBIGUOUS_OUTCOME.")

        return self

    @property
    def is_success(self) -> bool:
        """True if the supplier confirmed acceptance."""
        return self.status == SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE

    @property
    def is_rejected(self) -> bool:
        """True if the supplier confirmed rejection."""
        return self.status == SupplierTransmissionStatus.CONFIRMED_REJECTION

    @property
    def is_failure(self) -> bool:
        """True if transmission definitely failed prior to supplier acceptance."""
        return self.status == SupplierTransmissionStatus.TRANSMISSION_FAILURE

    @property
    def is_ambiguous(self) -> bool:
        """True if supplier response was unavailable / unknown outcome."""
        return self.status == SupplierTransmissionStatus.AMBIGUOUS_OUTCOME


class SupplierGateway(ABC):
    """
    Abstract gateway interface for transmitting approved Purchase Orders to suppliers.
    Decoupled from state machine and database persistence.
    """

    @abstractmethod
    def transmit_purchase_order(
        self,
        po: PurchaseOrder,
        idempotency_key: str,
        *,
        fo_id: Optional[str] = None,
        transmission_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplierTransmissionResult:
        """
        Transmits an approved Purchase Order to the designated supplier.

        Args:
            po: Approved PurchaseOrder with verified approval evidence.
            idempotency_key: Unique caller-provided idempotency key.
            fo_id: Optional associated Fulfilment Order identifier.
            transmission_timestamp: Optional normalized ISO 8601 UTC timestamp.
            **kwargs: Gateway-specific operational parameters.

        Returns:
            Validated SupplierTransmissionResult.

        Raises:
            InvalidPurchaseOrderError: If PO is unapproved or lacks verified approval evidence.
            SupplierGatewayIdempotencyConflictError: If idempotency key is reused with differing payload.
        """
        pass

    @abstractmethod
    def reconcile_transmission(
        self,
        idempotency_key: str,
        po: PurchaseOrder,
        *,
        fo_id: Optional[str] = None,
        reconciliation_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplierTransmissionResult:
        """
        Reconciles an ambiguous, interrupted, or unconfirmed transmission with the supplier.
        Queries the supplier's state for the order associated with idempotency_key/po.
        """
        pass


@runtime_checkable
class SupplierGatewayProtocol(Protocol):
    """
    Runtime-checkable Protocol for structural subtyping of supplier gateways.
    """

    def transmit_purchase_order(
        self,
        po: PurchaseOrder,
        idempotency_key: str,
        *,
        fo_id: Optional[str] = None,
        transmission_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplierTransmissionResult:
        ...

    def reconcile_transmission(
        self,
        idempotency_key: str,
        po: PurchaseOrder,
        *,
        fo_id: Optional[str] = None,
        reconciliation_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplierTransmissionResult:
        ...


class SimulatedSupplierAdapter(SupplierGateway):
    """
    Deterministic simulated supplier adapter.

    Guarantees:
    - Pure, deterministic outputs: identical inputs and scenarios always yield identical outputs.
    - Zero random numbers, zero real HTTP/network calls, zero external APIs.
    - Explicit scenario configuration:
      1. ACCEPT (supplier accepts order)
      2. REJECT (supplier rejects order)
      3. TRANSMISSION_FAILURE (fails before supplier acceptance)
      4. AMBIGUOUS_OUTCOME (supplier accepts order, response dropped/unavailable)
    - Strong idempotency: duplicate keys return cached result without duplicate submission.
      Altered payloads or POs under the same key raise SupplierGatewayIdempotencyConflictError.
    - Phase 4 governance enforcement: rejects unapproved or unverified Purchase Orders.
    """

    def __init__(
        self,
        default_scenario: SupplierSimulationScenario = SupplierSimulationScenario.ACCEPT,
        default_rejection_reason: str = "Supplier capacity unavailable",
        default_lead_time_hours: Optional[float] = None,
        db_path: Optional[Union[str, Path]] = None,
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        self._default_scenario = default_scenario
        self._default_rejection_reason = default_rejection_reason
        self._default_lead_time_hours = default_lead_time_hours
        self.db_path = Path(db_path) if db_path is not None and str(db_path) != ":memory:" else db_path
        self._conn = connection

        # Scenario overrides
        self._po_scenarios: Dict[str, SupplierSimulationScenario] = {}
        self._supplier_scenarios: Dict[str, SupplierSimulationScenario] = {}
        self._po_rejection_reasons: Dict[str, str] = {}

        # Idempotency registry: idempotency_key -> (payload_fingerprint, SupplierTransmissionResult)
        self._transmissions: Dict[str, Tuple[str, SupplierTransmissionResult]] = {}

        # Simulated supplier internal submission audit (counts and records orders actually received by supplier)
        self._supplier_received_orders: Dict[str, List[Dict[str, Any]]] = {}

        if self.db_path is not None or self._conn is not None:
            self._init_storage()

    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        if self.db_path is not None:
            if str(self.db_path) != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA busy_timeout = 10000;")
            self._conn.row_factory = sqlite3.Row
            return self._conn
        return None

    def close(self) -> None:
        """Closes underlying SQLite connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _init_storage(self) -> None:
        conn = self._get_db_connection()
        if conn is None:
            return
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulated_supplier_orders (
                    po_id TEXT PRIMARY KEY,
                    supplier_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    rejection_reason TEXT,
                    details_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulated_supplier_transmissions (
                    idempotency_key TEXT PRIMARY KEY,
                    po_id TEXT NOT NULL,
                    payload_fingerprint TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    status TEXT NOT NULL,
                    transmitted_at TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );
                """
            )
            cur = conn.execute("SELECT * FROM simulated_supplier_orders")
            for row in cur.fetchall():
                po_id = row["po_id"]
                if po_id not in self._supplier_received_orders:
                    self._supplier_received_orders[po_id] = []
                self._supplier_received_orders[po_id].append({
                    "timestamp": row["received_at"],
                    "status": row["status"],
                    "scenario": row["scenario"],
                    "rejection_reason": row["rejection_reason"],
                })

            cur_tx = conn.execute("SELECT * FROM simulated_supplier_transmissions")
            for row in cur_tx.fetchall():
                try:
                    res = SupplierTransmissionResult.model_validate_json(row["result_json"])
                    self._transmissions[row["idempotency_key"]] = (row["payload_fingerprint"], res)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Scenario Configuration
    # -------------------------------------------------------------------------

    def set_default_scenario(self, scenario: SupplierSimulationScenario) -> None:
        """Sets the default scenario used when no specific override matches."""
        self._default_scenario = scenario

    def set_po_scenario(
        self,
        po_id: str,
        scenario: SupplierSimulationScenario,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Configures a scenario override for a specific Purchase Order ID."""
        clean_po_id = _validate_non_empty_str(po_id, "po_id")
        self._po_scenarios[clean_po_id] = scenario
        if rejection_reason and rejection_reason.strip():
            self._po_rejection_reasons[clean_po_id] = rejection_reason.strip()

    def set_supplier_scenario(
        self,
        supplier_id: str,
        scenario: SupplierSimulationScenario,
    ) -> None:
        """Configures a scenario override for all orders to a specific supplier."""
        clean_sup_id = _validate_non_empty_str(supplier_id, "supplier_id")
        self._supplier_scenarios[clean_sup_id] = scenario

    # -------------------------------------------------------------------------
    # Inspection & Diagnostics
    # -------------------------------------------------------------------------

    def get_submission_count(self, po_id: str) -> int:
        """Returns the number of times the simulated supplier physically received a submission for po_id."""
        clean_po_id = _validate_non_empty_str(po_id, "po_id")
        return len(self._supplier_received_orders.get(clean_po_id, []))

    def has_supplier_received(self, po_id: str) -> bool:
        """True if the simulated supplier received and processed the order (even if response timed out)."""
        return self.get_submission_count(po_id) > 0

    def get_transmission_record(self, idempotency_key: str) -> Optional[SupplierTransmissionResult]:
        """Looks up a past transmission result by its idempotency key."""
        clean_key = idempotency_key.strip() if idempotency_key else ""
        entry = self._transmissions.get(clean_key)
        return entry[1] if entry else None

    def reset(self) -> None:
        """Clears all transmissions, caches, scenario overrides, and persisted tables."""
        self._transmissions.clear()
        self._supplier_received_orders.clear()
        self._po_scenarios.clear()
        self._supplier_scenarios.clear()
        self._po_rejection_reasons.clear()
        conn = self._get_db_connection()
        if conn is not None:
            with conn:
                conn.execute("DELETE FROM simulated_supplier_orders")
                conn.execute("DELETE FROM simulated_supplier_transmissions")

    # -------------------------------------------------------------------------
    # Core Transmission Execution
    # -------------------------------------------------------------------------

    def transmit_purchase_order(
        self,
        po: PurchaseOrder,
        idempotency_key: str,
        *,
        fo_id: Optional[str] = None,
        transmission_timestamp: Optional[str] = None,
        override_scenario: Optional[SupplierSimulationScenario] = None,
        rejection_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplierTransmissionResult:
        """
        Executes a deterministic simulated transmission of an approved Purchase Order.
        """
        # 1. Validate idempotency key
        clean_key = _validate_non_empty_str(idempotency_key, "idempotency_key")

        # 2. Strict Phase 4 Approval Governance Validation
        if not isinstance(po, PurchaseOrder):
            raise InvalidPurchaseOrderError(f"Expected PurchaseOrder instance, got {type(po)}.")

        if po.status != PurchaseOrderStatus.APPROVED:
            raise InvalidPurchaseOrderError(
                f"Governance Violation: Purchase Order '{po.po_id}' has status '{po.status.value}'. "
                "Only orders in APPROVED status can be transmitted to a supplier."
            )

        if not can_be_dispatched(po) or not verify_approval_evidence(po):
            raise InvalidPurchaseOrderError(
                f"Governance Violation: Purchase Order '{po.po_id}' lacks verified human approval evidence. "
                "Cannot transmit to supplier."
            )

        # 3. Resolve effective scenario
        effective_scenario = (
            override_scenario
            or self._po_scenarios.get(po.po_id)
            or self._supplier_scenarios.get(po.supplier_id)
            or self._default_scenario
        )

        # 4. Canonical payload fingerprinting for strict idempotency
        clean_fo_id = fo_id.strip() if fo_id and fo_id.strip() else f"FO-{po.po_id.replace('PO-', '')}"
        
        fingerprint_payload: Dict[str, Any] = {
            "action": "TRANSMIT_PURCHASE_ORDER",
            "fo_id": clean_fo_id,
            "po_id": po.po_id,
            "supplier_id": po.supplier_id,
            "supplier_type": po.supplier_type.value,
            "store_id": po.store_id,
            "created_date": po.created_date,
            "urgency": po.urgency.value,
            "total_order_qty": po.total_order_qty,
            "total_order_cost": po.total_order_cost,
            "scenario": effective_scenario.value,
            "items": [
                {
                    "sku_id": item.sku_id,
                    "category": item.category,
                    "final_order_qty": item.final_order_qty,
                    "unit_procurement_cost": item.unit_procurement_cost,
                }
                for item in po.items
            ],
        }
        if rejection_reason and rejection_reason.strip():
            fingerprint_payload["rejection_reason"] = rejection_reason.strip()

        incoming_fingerprint = compute_payload_fingerprint(fingerprint_payload)

        # 5. Idempotency Check: return cached result or detect conflicting reuse
        if clean_key in self._transmissions:
            recorded_fp, recorded_result = self._transmissions[clean_key]
            if recorded_fp == incoming_fingerprint:
                # Exact idempotent retry: return existing result WITHOUT second simulated submission
                return recorded_result
            raise SupplierGatewayIdempotencyConflictError(
                f"Idempotency Conflict: key '{clean_key}' was previously used with fingerprint "
                f"'{recorded_fp}' for PO '{recorded_result.po_id}', but incoming fingerprint is "
                f"'{incoming_fingerprint}' for PO '{po.po_id}'."
            )

        # Also check persistent storage if configured to avoid duplicate submission across adapter instances
        conn = self._get_db_connection()
        if conn is not None:
            cur = conn.execute(
                "SELECT payload_fingerprint, result_json FROM simulated_supplier_transmissions WHERE idempotency_key = ?",
                (clean_key,),
            )
            row = cur.fetchone()
            if row:
                rec_fp = row["payload_fingerprint"]
                if rec_fp == incoming_fingerprint:
                    rec_result = SupplierTransmissionResult.model_validate_json(row["result_json"])
                    self._transmissions[clean_key] = (rec_fp, rec_result)
                    return rec_result
                raise SupplierGatewayIdempotencyConflictError(
                    f"Idempotency Conflict: key '{clean_key}' was previously recorded with fingerprint "
                    f"'{rec_fp}', but incoming fingerprint is '{incoming_fingerprint}'."
                )

        # 6. Resolve timestamps deterministically
        raw_ts = transmission_timestamp or po.approval_timestamp or "2024-11-15T12:00:00Z"
        clean_ts = validate_and_normalize_iso_timestamp(raw_ts, "transmission_timestamp")

        # 7. Generate deterministic identifiers
        key_hash = compute_payload_fingerprint({"k": clean_key})[:12]
        correlation_id = f"CORR-{po.po_id}-{key_hash}"
        transmission_id = f"TXN-{po.po_id}-{key_hash}"
        supplier_reference = f"SUP-REF-{po.po_id}"

        # 8. Execute scenario-specific deterministic simulation
        result: SupplierTransmissionResult

        if effective_scenario == SupplierSimulationScenario.ACCEPT:
            # Record simulated supplier submission
            self._record_supplier_submission(
                po.po_id, clean_ts, "ACCEPTED", supplier_id=po.supplier_id, scenario="ACCEPT"
            )

            # Deterministic lead time calculation
            expected_delivery_at = self._compute_expected_delivery(po, clean_ts)

            ack_lines = tuple(
                SupplierAcknowledgmentLine(
                    sku_id=item.sku_id,
                    accepted_quantity=item.final_order_qty,
                    declined_quantity=0.0,
                    notes=None,
                )
                for item in po.items
            )

            ack = SupplierAcknowledgment(
                acknowledgment_id=f"ACK-{po.po_id}",
                fo_id=clean_fo_id,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierAcknowledgmentStatus.ACCEPTED,
                acknowledged_at=clean_ts,
                expected_delivery_at=expected_delivery_at,
                lines=ack_lines,
                supplier_reference=supplier_reference,
                notes="Simulated supplier order acceptance.",
            )

            result = SupplierTransmissionResult(
                transmission_id=transmission_id,
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=clean_fo_id,
                supplier_reference=supplier_reference,
                acknowledgment=ack,
                details={
                    "simulation_scenario": effective_scenario.value,
                    "simulated_lead_time_hours": self._resolve_lead_time_hours(po),
                },
            )

        elif effective_scenario == SupplierSimulationScenario.REJECT:
            effective_rejection_reason = (
                rejection_reason
                or self._po_rejection_reasons.get(po.po_id)
                or self._default_rejection_reason
            )

            # Record simulated supplier submission
            self._record_supplier_submission(
                po.po_id, clean_ts, "REJECTED", supplier_id=po.supplier_id,
                scenario="REJECT", rejection_reason=effective_rejection_reason
            )

            ack_lines = tuple(
                SupplierAcknowledgmentLine(
                    sku_id=item.sku_id,
                    accepted_quantity=0.0,
                    declined_quantity=item.final_order_qty,
                    notes="Item rejected by vendor",
                )
                for item in po.items
            )

            ack = SupplierAcknowledgment(
                acknowledgment_id=f"ACK-REJ-{po.po_id}",
                fo_id=clean_fo_id,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierAcknowledgmentStatus.REJECTED,
                acknowledged_at=clean_ts,
                lines=ack_lines,
                rejection_reason=effective_rejection_reason,
                supplier_reference=supplier_reference,
                notes="Simulated supplier order rejection.",
            )

            result = SupplierTransmissionResult(
                transmission_id=transmission_id,
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.CONFIRMED_REJECTION,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=clean_fo_id,
                supplier_reference=supplier_reference,
                acknowledgment=ack,
                details={
                    "simulation_scenario": effective_scenario.value,
                    "rejection_reason": effective_rejection_reason,
                },
            )

        elif effective_scenario == SupplierSimulationScenario.TRANSMISSION_FAILURE:
            # Definite transmission failure: supplier was never reached, NO submission recorded
            result = SupplierTransmissionResult(
                transmission_id=transmission_id,
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.TRANSMISSION_FAILURE,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=clean_fo_id,
                supplier_reference=None,
                acknowledgment=None,
                error_code="TRANSMISSION_FAILED_PRE_ACCEPTANCE",
                error_message="Transmission to supplier failed before acceptance: connection refused or host unreachable.",
                details={
                    "simulation_scenario": effective_scenario.value,
                    "supplier_received": False,
                },
            )

        elif effective_scenario == SupplierSimulationScenario.AMBIGUOUS_OUTCOME:
            # Ambiguous outcome: supplier DID receive and accept the order, but transmission response
            # was dropped / unavailable to caller.
            self._record_supplier_submission(
                po.po_id, clean_ts, "ACCEPTED", supplier_id=po.supplier_id, scenario="AMBIGUOUS_OUTCOME"
            )

            result = SupplierTransmissionResult(
                transmission_id=transmission_id,
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.AMBIGUOUS_OUTCOME,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=clean_fo_id,
                supplier_reference=None,
                acknowledgment=None,
                error_code="RESPONSE_TIMEOUT_AMBIGUOUS",
                error_message="Supplier accepted the order, but transmission response was unavailable (ambiguous outcome).",
                details={
                    "simulation_scenario": effective_scenario.value,
                    "supplier_received": True,
                    "internal_supplier_status": "ACCEPTED",
                },
            )

        else:
            raise ValueError(f"Unsupported SupplierSimulationScenario: '{effective_scenario}'.")

        # 9. Commit to idempotency registry and storage
        self._transmissions[clean_key] = (incoming_fingerprint, result)
        conn = self._get_db_connection()
        if conn is not None:
            with conn:
                conn.execute(
                    """
                    INSERT INTO simulated_supplier_transmissions (
                        idempotency_key, po_id, payload_fingerprint, scenario, status, transmitted_at, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO UPDATE SET
                        payload_fingerprint = excluded.payload_fingerprint,
                        scenario = excluded.scenario,
                        status = excluded.status,
                        transmitted_at = excluded.transmitted_at,
                        result_json = excluded.result_json
                    """,
                    (
                        clean_key,
                        po.po_id,
                        incoming_fingerprint,
                        effective_scenario.value,
                        result.status.value,
                        clean_ts,
                        result.model_dump_json(),
                    ),
                )
        return result

    def reconcile_transmission(
        self,
        idempotency_key: str,
        po: PurchaseOrder,
        *,
        fo_id: Optional[str] = None,
        reconciliation_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplierTransmissionResult:
        """
        Reconciles an ambiguous, interrupted, or unconfirmed transmission with the simulated supplier.
        Queries the supplier's state for the order associated with po_id.
        """
        clean_key = _validate_non_empty_str(idempotency_key, "idempotency_key")
        if not isinstance(po, PurchaseOrder):
            raise InvalidPurchaseOrderError(f"Expected PurchaseOrder instance, got {type(po)}.")
        if not can_be_dispatched(po) or not verify_approval_evidence(po):
            raise InvalidPurchaseOrderError(
                f"Governance Violation: Purchase Order '{po.po_id}' lacks verified human approval evidence."
            )

        clean_ts = validate_and_normalize_iso_timestamp(
            reconciliation_timestamp or po.approval_timestamp or "2024-11-15T12:00:00Z",
            "reconciliation_timestamp",
        )
        clean_fo_id = fo_id.strip() if fo_id and fo_id.strip() else f"FO-{po.po_id.replace('PO-', '')}"

        # 1. Return existing confirmed result if previously resolved
        if clean_key in self._transmissions:
            recorded_fp, recorded_result = self._transmissions[clean_key]
            if recorded_result.status in (
                SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE,
                SupplierTransmissionStatus.CONFIRMED_REJECTION,
            ):
                return recorded_result

        conn = self._get_db_connection()
        if conn is not None:
            cur = conn.execute(
                "SELECT payload_fingerprint, status, result_json FROM simulated_supplier_transmissions WHERE idempotency_key = ?",
                (clean_key,),
            )
            row = cur.fetchone()
            if row:
                rec_status = row["status"]
                if rec_status in (
                    SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE.value,
                    SupplierTransmissionStatus.CONFIRMED_REJECTION.value,
                ):
                    rec_result = SupplierTransmissionResult.model_validate_json(row["result_json"])
                    self._transmissions[clean_key] = (row["payload_fingerprint"], rec_result)
                    return rec_result

        key_hash = compute_payload_fingerprint({"k": clean_key})[:12]
        correlation_id = f"CORR-{po.po_id}-{key_hash}"
        transmission_id = f"TXN-REC-{po.po_id}-{key_hash}"
        supplier_reference = f"SUP-REF-{po.po_id}"

        # Check if supplier received the order
        received = self.has_supplier_received(po.po_id)
        supplier_status = "ACCEPTED"
        rej_reason = None
        if received and po.po_id in self._supplier_received_orders and self._supplier_received_orders[po.po_id]:
            last_sub = self._supplier_received_orders[po.po_id][-1]
            supplier_status = last_sub.get("status", "ACCEPTED")
            rej_reason = last_sub.get("rejection_reason")

        if received and supplier_status == "ACCEPTED":
            # Confirmed acceptance upon reconciliation
            expected_delivery_at = self._compute_expected_delivery(po, clean_ts)
            ack_lines = tuple(
                SupplierAcknowledgmentLine(
                    sku_id=item.sku_id,
                    accepted_quantity=item.final_order_qty,
                    declined_quantity=0.0,
                    notes=None,
                )
                for item in po.items
            )
            ack = SupplierAcknowledgment(
                acknowledgment_id=f"ACK-REC-{po.po_id}",
                fo_id=clean_fo_id,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierAcknowledgmentStatus.ACCEPTED,
                acknowledged_at=clean_ts,
                expected_delivery_at=expected_delivery_at,
                lines=ack_lines,
                supplier_reference=supplier_reference,
                notes="Reconciled order acceptance after ambiguous transmission.",
            )
            result = SupplierTransmissionResult(
                transmission_id=transmission_id,
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=clean_fo_id,
                supplier_reference=supplier_reference,
                acknowledgment=ack,
                details={
                    "reconciled": True,
                    "reconciled_at": clean_ts,
                    "reconciled_status": "ACCEPTED",
                },
            )
        elif received and supplier_status == "REJECTED":
            effective_rej_reason = (
                rej_reason or self._po_rejection_reasons.get(po.po_id) or self._default_rejection_reason
            )
            ack_lines = tuple(
                SupplierAcknowledgmentLine(
                    sku_id=item.sku_id,
                    accepted_quantity=0.0,
                    declined_quantity=item.final_order_qty,
                    notes="Item rejected by vendor",
                )
                for item in po.items
            )
            ack = SupplierAcknowledgment(
                acknowledgment_id=f"ACK-REJ-REC-{po.po_id}",
                fo_id=clean_fo_id,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierAcknowledgmentStatus.REJECTED,
                acknowledged_at=clean_ts,
                lines=ack_lines,
                rejection_reason=effective_rej_reason,
                supplier_reference=supplier_reference,
                notes="Reconciled order rejection after ambiguous transmission.",
            )
            result = SupplierTransmissionResult(
                transmission_id=transmission_id,
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.CONFIRMED_REJECTION,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=clean_fo_id,
                supplier_reference=supplier_reference,
                acknowledgment=ack,
                details={
                    "reconciled": True,
                    "reconciled_at": clean_ts,
                    "reconciled_status": "REJECTED",
                },
            )
        else:
            # Supplier has no record of receiving this order
            result = SupplierTransmissionResult(
                transmission_id=transmission_id,
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.TRANSMISSION_FAILURE,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=clean_fo_id,
                supplier_reference=None,
                acknowledgment=None,
                error_code="ORDER_NOT_FOUND_AT_SUPPLIER",
                error_message="Reconciliation confirmed supplier has no record of receiving order.",
                details={
                    "reconciled": True,
                    "reconciled_at": clean_ts,
                    "supplier_received": False,
                },
            )

        # Update cache & persistence
        rec_fp = (
            self._transmissions[clean_key][0]
            if clean_key in self._transmissions
            else compute_payload_fingerprint({"k": clean_key, "po": po.po_id})
        )
        self._transmissions[clean_key] = (rec_fp, result)
        conn = self._get_db_connection()
        if conn is not None:
            with conn:
                conn.execute(
                    """
                    INSERT INTO simulated_supplier_transmissions (
                        idempotency_key, po_id, payload_fingerprint, scenario, status, transmitted_at, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO UPDATE SET
                        status = excluded.status,
                        transmitted_at = excluded.transmitted_at,
                        result_json = excluded.result_json
                    """,
                    (
                        clean_key,
                        po.po_id,
                        rec_fp,
                        "RECONCILED",
                        result.status.value,
                        clean_ts,
                        result.model_dump_json(),
                    ),
                )
        return result

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _record_supplier_submission(
        self,
        po_id: str,
        timestamp: str,
        status: str,
        supplier_id: str = "SIM_SUPP_DEFAULT",
        scenario: str = "ACCEPT",
        rejection_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records a simulated supplier receipt in the adapter's audit history and storage."""
        if po_id not in self._supplier_received_orders:
            self._supplier_received_orders[po_id] = []
        self._supplier_received_orders[po_id].append({
            "timestamp": timestamp,
            "status": status,
            "scenario": scenario,
            "rejection_reason": rejection_reason,
        })
        conn = self._get_db_connection()
        if conn is not None:
            with conn:
                conn.execute(
                    """
                    INSERT INTO simulated_supplier_orders (
                        po_id, supplier_id, status, received_at, scenario, rejection_reason, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(po_id) DO UPDATE SET
                        status = excluded.status,
                        received_at = excluded.received_at,
                        scenario = excluded.scenario,
                        rejection_reason = excluded.rejection_reason,
                        details_json = excluded.details_json
                    """,
                    (
                        po_id,
                        supplier_id,
                        status,
                        timestamp,
                        scenario,
                        rejection_reason,
                        json.dumps(details or {}),
                    ),
                )

    def _resolve_lead_time_hours(self, po: PurchaseOrder) -> float:
        """Determines delivery lead time in hours deterministically from config or default."""
        if self._default_lead_time_hours is not None:
            return float(self._default_lead_time_hours)

        # Derive max lead time across PO categories from CATEGORY_SUPPLIER_PARAMS
        max_lead_days = 1.5
        for item in po.items:
            params = CATEGORY_SUPPLIER_PARAMS.get(item.category)
            if params and params.lead_time_mean_days > max_lead_days:
                max_lead_days = params.lead_time_mean_days

        return round(max_lead_days * 24.0, 1)

    def _compute_expected_delivery(self, po: PurchaseOrder, transmitted_at: str) -> str:
        """Computes deterministic expected delivery arrival timestamp."""
        dt_base = dt_datetime.fromisoformat(transmitted_at.replace("Z", "+00:00"))
        lead_hours = self._resolve_lead_time_hours(po)
        dt_delivery = dt_base + timedelta(hours=lead_hours)
        return dt_delivery.strftime("%Y-%m-%dT%H:%M:%SZ")
