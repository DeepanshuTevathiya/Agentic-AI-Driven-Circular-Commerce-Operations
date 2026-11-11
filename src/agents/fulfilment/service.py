"""
Lead 2 Fulfilment Application Service Layer
==========================================
Orchestrates domain workflows and SQLite persistence for Purchase Order fulfilment.

RESPONSIBILITIES:
1. Loads the latest fulfilment order aggregate and database version.
2. Calls the pure state-machine operations defined in state_machine.py.
3. Persists resulting aggregates using optimistic concurrency versioning.
4. Handles idempotent retries without duplicate events or unnecessary version increments.
5. Guarantees multi-aggregate atomic supersession within a single SQLite transaction.

ARCHITECTURAL INVARIANTS:
- The state machine remains the sole authority for business rules.
- The repository remains the sole authority for database transactions and persistence.
- Approved PurchaseOrders remain strictly immutable.
"""
from __future__ import annotations

from datetime import datetime as dt_datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from src.agents.common.schemas import (
    DailyAgentInput,
    InventoryBatch,
    InventoryCalculationResult,
    InventoryPosition,
    InventoryRiskAssessment,
)

if TYPE_CHECKING:
    from src.agents.inventory.engine import InventoryEngine
    from src.agents.inventory.repository import OperationalInventoryRepository
    from src.agents.inventory.schemas import (
        BatchBalance,
        GrnPostingReconciliation,
        InventoryGovernanceError,
        InventoryLedgerEntry,
        InventoryPostingReason,
        InventoryTransactionType,
        StoreSkuBalance,
    )

from src.agents.fulfilment.gateway import (
    InvalidPurchaseOrderError,
    SimulatedSupplierAdapter,
    SupplierGateway,
    SupplierGatewayError,
    SupplierGatewayIdempotencyConflictError,
    SupplierSimulationScenario,
    SupplierTransmissionResult,
    SupplierTransmissionStatus,
)
from src.agents.fulfilment.repository import (
    FulfilmentOrderNotFoundError,
    FulfilmentRepository,
    IdempotencyConflictError,
    StaleUpdateError,
)
from src.agents.fulfilment.schemas import (
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DiscrepancyReason,
    DiscrepancyResolution,
    DiscrepancyResolutionType,
    FulfilmentBatch,
    FulfilmentEvent,
    FulfilmentEventType,
    FulfilmentOrder,
    FulfilmentOrderStatus,
    GoodsReceiptNote,
    LineReceiptReconciliation,
    OrderReceiptReconciliation,
    QualityHoldResolution,
    ShipmentReceiptReconciliation,
    SupplierAcknowledgment,
    calculate_order_receipt_reconciliation,
    compute_payload_fingerprint,
    generate_grn_fulfilment_batches,
)
from src.agents.fulfilment.state_machine import (
    FulfilmentGovernanceError,
    InvalidFulfilmentTransitionError,
    apply_goods_receipt,
    apply_shipment_dispatch,
    apply_supplier_acknowledgment,
    close_short_fulfilment_order,
    confirm_cancellation,
    create_fulfilment_order,
    reaffirm_original_terms,
    record_delivery_exception,
    reject_cancellation_request,
    request_cancellation,
    resolve_delivery_exception,
    resolve_goods_receipt_discrepancy,
    resolve_quality_hold,
    resolve_shipment_custody,
    supersede_fulfilment_order,
    transmit_fulfilment_order,
)
from src.agents.procurement.approval import can_be_dispatched, verify_approval_evidence
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderStatus,
    validate_and_normalize_iso_timestamp,
)
from src.agents.fulfilment.tracking import (
    InvalidTrackingEventError,
    InvalidTrackingTransitionError,
    ShipmentNotFoundError,
    ShipmentTrackingEvent,
    ShipmentTrackingRecord,
    ShipmentTrackingResult,
    ShipmentTrackingStatus,
    SimulatedTrackingProvider,
    TrackingError,
    TrackingIdempotencyConflictError,
    TrackingProvider,
    TrackingScenario,
)


def _validate_non_empty_str(v: Any, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


def _validate_po_matches_fo(po: PurchaseOrder, fo: FulfilmentOrder) -> None:
    """Verifies that an approved PurchaseOrder exactly matches the FulfilmentOrder commercial details."""
    if fo.po_id != po.po_id:
        raise FulfilmentGovernanceError(
            f"Commercial mismatch: PurchaseOrder po_id '{po.po_id}' does not match FulfilmentOrder po_id '{fo.po_id}'."
        )
    if fo.store_id != po.store_id:
        raise FulfilmentGovernanceError(
            f"Commercial mismatch: PurchaseOrder store_id '{po.store_id}' does not match FulfilmentOrder store_id '{fo.store_id}'."
        )
    if fo.supplier_id != po.supplier_id:
        raise FulfilmentGovernanceError(
            f"Commercial mismatch: PurchaseOrder supplier_id '{po.supplier_id}' does not match FulfilmentOrder supplier_id '{fo.supplier_id}'."
        )
    if fo.supplier_type.value != po.supplier_type.value:
        raise FulfilmentGovernanceError(
            f"Commercial mismatch: PurchaseOrder supplier_type '{po.supplier_type.value}' does not match FulfilmentOrder '{fo.supplier_type.value}'."
        )

    fo_lines_by_sku = {line.sku_id: line for line in fo.lines}
    if len(fo.lines) != len(po.items):
        raise FulfilmentGovernanceError(
            f"Commercial mismatch: PO item count ({len(po.items)}) does not match FO line count ({len(fo.lines)})."
        )

    for item in po.items:
        if item.sku_id not in fo_lines_by_sku:
            raise FulfilmentGovernanceError(f"Commercial mismatch: PO item SKU '{item.sku_id}' not found in FO lines.")
        line = fo_lines_by_sku[item.sku_id]
        if abs(line.ordered_quantity - item.final_order_qty) > 1e-4:
            raise FulfilmentGovernanceError(
                f"Commercial mismatch for SKU '{item.sku_id}': PO quantity {item.final_order_qty} != FO {line.ordered_quantity}."
            )
        if abs(line.unit_cost - item.unit_procurement_cost) > 1e-4:
            raise FulfilmentGovernanceError(
                f"Commercial mismatch for SKU '{item.sku_id}': PO unit cost ₹{item.unit_procurement_cost} != FO ₹{line.unit_cost}."
            )


def _resolve_idempotency_key(
    current_fo: FulfilmentOrder,
    idempotency_key: Optional[str],
    default_prefix: str,
    event_type: FulfilmentEventType,
    check_active_status: Optional[FulfilmentOrderStatus] = None,
    filter_fn: Optional[Callable[[FulfilmentEvent], bool]] = None,
) -> str:
    """
    Resolves the idempotency key for a domain transition.
    1. If an explicit non-empty idempotency_key is provided, uses it verbatim.
    2. If omitted:
       - If order is currently in check_active_status (an exact retry scenario),
         resolves to the key of the existing active event of this type so state machine can detect replay.
       - Otherwise (a new lifecycle occurrence), derives a sequence-aware key:
         '{prefix}:{fo_id}:{count + 1}'.
    """
    clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else None
    if clean_key:
        return clean_key

    matching_events = [
        e for e in current_fo.event_history
        if e.event_type == event_type and (filter_fn(e) if filter_fn else True)
    ]
    count = len(matching_events)

    if check_active_status is not None and current_fo.status == check_active_status and count > 0:
        return matching_events[-1].idempotency_key

    return f"{default_prefix}:{current_fo.fo_id}:{count + 1}"


class ShipmentTrackingService:
    """
    Application service managing the lifecycle, ingestion, and durable persistence
    of shipment tracking events and projections.
    Coordinates between TrackingProvider, domain tracking state machine, and FulfilmentRepository.
    """

    def __init__(
        self,
        repository: FulfilmentRepository,
        tracking_provider: Optional[TrackingProvider] = None,
    ) -> None:
        self.repository = repository
        self.provider = tracking_provider or SimulatedTrackingProvider()

    def register_shipment(
        self,
        fo_id: str,
        shipment_id: str,
        carrier_id: str,
        *,
        scenario: Optional[TrackingScenario] = None,
        base_timestamp: Optional[str] = None,
        estimated_delivery_at: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingRecord:
        """
        Registers a dispatched shipment for tracking.
        Enforces:
        1. FulfilmentOrder must exist in repository.
        2. Shipment ID must be associated with the FulfilmentOrder.
        3. Carrier ID must match the ASN dispatch carrier if one was specified.
        4. Idempotently returns existing record if already registered with matching carrier.
        5. Rejects conflicting re-registration with differing carrier.
        6. Persists initial tracking projection and events into SQLite.
        """
        clean_fo_id = _validate_non_empty_str(fo_id, "fo_id")
        clean_shipment_id = _validate_non_empty_str(shipment_id, "shipment_id")
        clean_carrier = _validate_non_empty_str(carrier_id, "carrier_id")

        # 1. Verify FO exists and is active (not cancelled or superseded)
        fo = self.repository.get_or_raise(clean_fo_id)
        if fo.status == FulfilmentOrderStatus.CANCELLED:
            raise FulfilmentGovernanceError(
                f"Cannot register tracking for cancelled or superseded FulfilmentOrder '{clean_fo_id}'."
            )

        # 2. Verify shipment belongs to FO
        if clean_shipment_id not in fo.shipment_ids:
            raise FulfilmentGovernanceError(
                f"Shipment '{clean_shipment_id}' is not associated with FulfilmentOrder '{clean_fo_id}'. "
                f"Active shipment IDs: {fo.shipment_ids}."
            )

        # 3. Verify carrier matches ASN if specified in dispatch events
        for evt in fo.event_history:
            if evt.event_type == FulfilmentEventType.SHIPMENT_DISPATCH and evt.related_reference_id == clean_shipment_id:
                asn_carrier = evt.details.get("carrier")
                if asn_carrier and asn_carrier.strip().upper() != clean_carrier.upper():
                    raise FulfilmentGovernanceError(
                        f"Carrier mismatch for shipment '{clean_shipment_id}': dispatch ASN specified "
                        f"carrier '{asn_carrier}', but registration specified '{clean_carrier}'."
                    )

        # 4. Check if already registered
        existing = self.repository.get_tracking_record(clean_shipment_id)
        if existing:
            if existing.carrier_id == clean_carrier:
                return existing
            raise TrackingIdempotencyConflictError(
                f"Tracking record for shipment '{clean_shipment_id}' already registered with carrier "
                f"'{existing.carrier_id}', cannot be re-registered with '{clean_carrier}'."
            )

        # 5. Register with tracking provider
        provider_record = self.provider.register_shipment(
            shipment_id=clean_shipment_id,
            carrier_id=clean_carrier,
            scenario=scenario,
            base_timestamp=base_timestamp,
            estimated_delivery_at=estimated_delivery_at,
            **kwargs,
        )

        # 6. Persist initial projection and events into SQLite
        return self.repository.create_tracking_record(clean_fo_id, provider_record)

    def ingest_tracking_event(
        self,
        event: ShipmentTrackingEvent,
        *,
        expected_version: Optional[int] = None,
    ) -> ShipmentTrackingRecord:
        """
        Ingests a carrier tracking event, applies domain transition rules,
        and atomically persists the updated tracking record in SQLite.
        """
        if not isinstance(event, ShipmentTrackingEvent):
            raise InvalidTrackingEventError(f"Expected ShipmentTrackingEvent instance, got {type(event)}.")

        current_record = self.repository.get_tracking_record_or_raise(event.shipment_id)
        if event.carrier_id != current_record.carrier_id:
            raise InvalidTrackingEventError(
                f"Carrier ID mismatch for shipment '{event.shipment_id}': event carrier '{event.carrier_id}' "
                f"does not match registered carrier '{current_record.carrier_id}'."
            )

        current_version = self.repository.get_tracking_version(event.shipment_id)

        if expected_version is not None and expected_version != current_version:
            raise StaleUpdateError(
                f"Stale update conflict for shipment '{event.shipment_id}': expected version {expected_version}, "
                f"but persistence is at version {current_version}."
            )

        target_version = current_version if expected_version is None else expected_version
        updated_record = current_record.apply_event(event)

        if updated_record.event_count == current_record.event_count:
            return current_record

        return self.repository.update_tracking_record(
            updated_record,
            new_event=event,
            expected_version=target_version,
        )

    def get_tracking(self, shipment_id: str) -> ShipmentTrackingResult:
        """Retrieves the current persisted tracking projection and history."""
        clean_id = _validate_non_empty_str(shipment_id, "shipment_id")
        record = self.repository.get_tracking_record_or_raise(clean_id)

        return ShipmentTrackingResult(
            shipment_id=record.shipment_id,
            carrier_id=record.carrier_id,
            status=record.current_status,
            record=record,
            queried_at=record.updated_at,
            estimated_delivery_at=record.estimated_delivery_at,
            latest_event=record.latest_event,
            notes=f"Current status: {record.current_status.value}",
        )

    def get_tracking_history(self, shipment_id: str) -> Tuple[ShipmentTrackingEvent, ...]:
        """Retrieves the complete chronological tracking event history for a shipment."""
        clean_id = _validate_non_empty_str(shipment_id, "shipment_id")
        record = self.repository.get_tracking_record_or_raise(clean_id)
        return record.events

    def get_shipments_for_order(self, fo_id: str) -> List[ShipmentTrackingRecord]:
        """Retrieves all tracking records linked to a given FulfilmentOrder."""
        clean_fo_id = _validate_non_empty_str(fo_id, "fo_id")
        return self.repository.get_tracking_records_for_order(clean_fo_id)

    def advance_shipment_tracking(
        self,
        shipment_id: str,
        steps: int = 1,
    ) -> Optional[ShipmentTrackingRecord]:
        """
        Advances a simulated tracking scenario by steps and persists the resulting event(s).
        """
        clean_id = _validate_non_empty_str(shipment_id, "shipment_id")
        record = self.repository.get_tracking_record_or_raise(clean_id)

        if hasattr(self.provider, "has_shipment") and not self.provider.has_shipment(clean_id):
            if hasattr(self.provider, "register_shipment"):
                persisted_scenario = None
                if record.scenario:
                    try:
                        persisted_scenario = TrackingScenario(record.scenario)
                    except ValueError:
                        pass
                self.provider.register_shipment(
                    shipment_id=clean_id,
                    carrier_id=record.carrier_id,
                    scenario=persisted_scenario,
                    base_timestamp=record.created_at,
                    estimated_delivery_at=record.estimated_delivery_at,
                )
            if hasattr(self.provider, "_shipments"):
                self.provider._shipments[clean_id] = record
            if hasattr(self.provider, "_scenario_step_indices"):
                self.provider._scenario_step_indices[clean_id] = record.event_count

        if not hasattr(self.provider, "advance_shipment"):
            raise TrackingError("Injected tracking provider does not support simulated advancement.")

        last_applied = None
        for _ in range(max(1, steps)):
            evt = self.provider.advance_shipment(clean_id, steps=1)
            if evt is None:
                break
            record = self.ingest_tracking_event(evt)
            last_applied = record

        return last_applied

    def advance_to_tracking_status(
        self,
        shipment_id: str,
        target_status: ShipmentTrackingStatus,
    ) -> ShipmentTrackingRecord:
        """
        Advances a simulated tracking scenario until target_status is reached and persisted.
        """
        clean_id = _validate_non_empty_str(shipment_id, "shipment_id")
        record = self.repository.get_tracking_record_or_raise(clean_id)

        if record.current_status == target_status:
            return record

        if hasattr(self.provider, "has_shipment") and not self.provider.has_shipment(clean_id):
            if hasattr(self.provider, "register_shipment"):
                persisted_scenario = None
                if record.scenario:
                    try:
                        persisted_scenario = TrackingScenario(record.scenario)
                    except ValueError:
                        pass
                self.provider.register_shipment(
                    shipment_id=clean_id,
                    carrier_id=record.carrier_id,
                    scenario=persisted_scenario,
                    base_timestamp=record.created_at,
                    estimated_delivery_at=record.estimated_delivery_at,
                )
            if hasattr(self.provider, "_shipments"):
                self.provider._shipments[clean_id] = record
            if hasattr(self.provider, "_scenario_step_indices"):
                self.provider._scenario_step_indices[clean_id] = record.event_count

        if not hasattr(self.provider, "advance_to_status"):
            raise TrackingError("Injected tracking provider does not support advance_to_status.")

        evts = self.provider.advance_to_status(clean_id, target_status=target_status)
        for evt in evts:
            record = self.ingest_tracking_event(evt)

        return record


class FulfilmentService:
    """
    Application service managing the lifecycle and persistence of fulfilment orders.
    Coordinates between callers, the domain state machine, and the persistence repository.
    """

    def __init__(
        self,
        repository: Optional[FulfilmentRepository] = None,
        db_path: Optional[Union[str, Path]] = None,
        gateway: Optional[SupplierGateway] = None,
        tracking_provider: Optional[TrackingProvider] = None,
        inventory_repository: Optional[OperationalInventoryRepository] = None,
    ) -> None:
        if repository is not None:
            self.repository = repository
        else:
            self.repository = FulfilmentRepository(db_path=db_path)
        self.gateway = gateway or SimulatedSupplierAdapter(db_path=self.repository.db_path)
        self.tracking_provider = tracking_provider or SimulatedTrackingProvider()
        self.tracking_service = ShipmentTrackingService(
            repository=self.repository,
            tracking_provider=self.tracking_provider,
        )
        self._inventory_repository: Optional[OperationalInventoryRepository] = inventory_repository

    @property
    def inventory_repository(self) -> OperationalInventoryRepository:
        """Lazily instantiates and returns the OperationalInventoryRepository."""
        if self._inventory_repository is None:
            from src.agents.inventory.repository import OperationalInventoryRepository
            self._inventory_repository = OperationalInventoryRepository(
                db_path=self.repository.db_path,
                fulfilment_repo=self.repository,
            )
        return self._inventory_repository

    @inventory_repository.setter
    def inventory_repository(self, value: Optional[OperationalInventoryRepository]) -> None:
        self._inventory_repository = value

    # -------------------------------------------------------------------------
    # Context Management & Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Closes the underlying repository connections."""
        self.repository.close()
        if self._inventory_repository is not None:
            self._inventory_repository.close()

    def __enter__(self) -> "FulfilmentService":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Helper: Single-Order Transition Orchestration
    # -------------------------------------------------------------------------

    def _execute_transition(
        self,
        fo_id: str,
        transition_fn: Any,
        expected_version: Optional[int] = None,
        key_resolver: Optional[Callable[[FulfilmentOrder], str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> FulfilmentOrder:
        """
        Executes a single-aggregate transition:
        1. Loads latest aggregate and its persistence version.
        2. Verifies expected_version if explicitly supplied.
        3. Resolves idempotency key dynamically via key_resolver if provided.
        4. Calls domain transition function.
        5. Detects idempotent replay: returns without DB write or version increment.
        6. Persists aggregate with optimistic concurrency check.
        """
        clean_id = fo_id.strip() if fo_id else ""
        if not clean_id:
            raise ValueError("Field 'fo_id' must be a non-empty string.")

        # 1. Load latest state and version
        current_fo = self.repository.get_or_raise(clean_id)
        current_version = self.repository.get_version(clean_id)

        # 2. Check optimistic concurrency version
        if expected_version is not None and expected_version != current_version:
            raise StaleUpdateError(
                f"Stale update conflict for '{clean_id}': expected version {expected_version}, "
                f"but database is currently at version {current_version}."
            )

        # Check for unresolved gateway transmissions
        active_txs = self.repository.get_transmissions_for_order(clean_id)
        for tx in active_txs:
            if tx["transmission_status"] in (SupplierTransmissionStatus.AMBIGUOUS_OUTCOME.value, "INITIATED"):
                raise FulfilmentGovernanceError(
                    f"Order '{clean_id}' has an unresolved transmission '{tx['idempotency_key']}' "
                    f"with status '{tx['transmission_status']}'. It must be reconciled before executing lifecycle transitions."
                )

        target_version = current_version if expected_version is None else expected_version

        # 3. Resolve key if resolver supplied
        if key_resolver is not None:
            kwargs["idempotency_key"] = key_resolver(current_fo)

        # 4. Call state machine
        new_fo = transition_fn(current_fo, *args, **kwargs)

        # 5. Handle idempotent replay:
        # In state_machine.py, an idempotent retry returns the existing aggregate without appending a new event
        if len(new_fo.event_history) == len(current_fo.event_history):
            return current_fo

        # 6. Persist updated aggregate with optimistic version condition
        grn_to_persist = kwargs.get("grn")
        self.repository.update(new_fo, expected_version=target_version, grn=grn_to_persist)
        return new_fo

    # -------------------------------------------------------------------------
    # Fulfilment Lifecycle Workflows
    # -------------------------------------------------------------------------

    def create_order(
        self,
        po: PurchaseOrder,
        actor_id: str = "FULFILMENT_AGENT",
        timestamp: Optional[str] = None,
        fo_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        supersedes_fo_id: Optional[str] = None,
        supersedes_po_id: Optional[str] = None,
    ) -> FulfilmentOrder:
        """
        Creates and persists a new FulfilmentOrder aggregate from an approved PurchaseOrder.
        Handles idempotent creation retries cleanly.
        """
        clean_fo_id = fo_id.strip() if fo_id and fo_id.strip() else f"FO-{po.po_id.replace('PO-', '')}"
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"CREATE:{clean_fo_id}:1"
        clean_actor = actor_id.strip() if actor_id and actor_id.strip() else "FULFILMENT_AGENT"
        ts = timestamp if timestamp else po.approval_timestamp or "2024-11-15T11:00:00Z"

        # Check for idempotent retry in repository
        existing = self.repository.get(clean_fo_id)
        if existing:
            if existing.event_history and existing.event_history[0].idempotency_key == clean_key:
                clean_ts = validate_and_normalize_iso_timestamp(ts, "timestamp", error_cls=FulfilmentGovernanceError)
                incoming_payload: Dict[str, Any] = {
                    "action": "CREATE_FULFILMENT_ORDER",
                    "fo_id": clean_fo_id,
                    "po_id": po.po_id,
                    "timestamp": clean_ts,
                    "actor_id": clean_actor,
                }
                if supersedes_fo_id and supersedes_fo_id.strip():
                    incoming_payload["supersedes_fo_id"] = supersedes_fo_id.strip()
                if supersedes_po_id and supersedes_po_id.strip():
                    incoming_payload["supersedes_po_id"] = supersedes_po_id.strip()

                incoming_fp = compute_payload_fingerprint(incoming_payload)
                if existing.event_history[0].payload_fingerprint == incoming_fp:
                    return existing
                raise FulfilmentGovernanceError(
                    f"Idempotency Conflict: key '{clean_key}' presented with differing creation payload fingerprint. "
                    f"Recorded='{existing.event_history[0].payload_fingerprint}', Incoming='{incoming_fp}'."
                )

        fo = create_fulfilment_order(
            po=po,
            actor_id=clean_actor,
            timestamp=ts,
            fo_id=clean_fo_id,
            idempotency_key=clean_key,
            supersedes_fo_id=supersedes_fo_id,
            supersedes_po_id=supersedes_po_id,
        )
        return self.repository.create(fo)

    def transmit_order(
        self,
        fo_id: str,
        timestamp: str,
        actor_id: str = "FULFILMENT_AGENT",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Transmits order to supplier: READY_TO_TRANSMIT -> ISSUED."""
        return self._execute_transition(
            fo_id,
            transmit_fulfilment_order,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="TX",
                event_type=FulfilmentEventType.TRANSMIT_TO_SUPPLIER,
                check_active_status=FulfilmentOrderStatus.ISSUED,
            ),
            transmission_timestamp=timestamp,
            actor_id=actor_id,
        )

    def transmit_purchase_order(
        self,
        fo_id: str,
        po: PurchaseOrder,
        idempotency_key: str,
        timestamp: Optional[str] = None,
        actor_id: str = "FULFILMENT_AGENT",
        expected_version: Optional[int] = None,
        gateway: Optional[SupplierGateway] = None,
        override_scenario: Optional[SupplierSimulationScenario] = None,
        rejection_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[FulfilmentOrder, SupplierTransmissionResult]:
        """
        Transmits an approved Purchase Order through the injected SupplierGateway.

        Guarantees:
        1. Verifies PO matches FO commercial details (po_id, store, supplier, lines, costs).
        2. Preserves Phase 4 human approval governance and PO immutability.
        3. Persists transmission intent before gateway invocation (two-phase commit).
        4. Handles all four transmission outcomes (ACCEPT, REJECT, TRANSMISSION_FAILURE, AMBIGUOUS_OUTCOME).
        5. Atomically commits transmission record and domain aggregate transitions in SQLite.
        6. Replays duplicate requests idempotently without version increments or duplicate events.
        """
        clean_fo_id = _validate_non_empty_str(fo_id, "fo_id")
        clean_key = _validate_non_empty_str(idempotency_key, "idempotency_key")
        clean_actor = _validate_non_empty_str(actor_id, "actor_id")

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

        fo = self.repository.get_or_raise(clean_fo_id)
        _validate_po_matches_fo(po, fo)

        if expected_version is not None:
            current_ver = self.repository.get_version(clean_fo_id)
            if current_ver != expected_version:
                raise StaleUpdateError(
                    f"Stale update conflict for '{clean_fo_id}': expected version {expected_version}, "
                    f"but persistence is at version {current_ver}."
                )

        active_gateway = gateway or self.gateway
        effective_scenario = (
            override_scenario
            or getattr(active_gateway, "_po_scenarios", {}).get(po.po_id)
            or getattr(active_gateway, "_supplier_scenarios", {}).get(po.supplier_id)
            or getattr(active_gateway, "_default_scenario", SupplierSimulationScenario.ACCEPT)
        )

        fp_payload: Dict[str, Any] = {
            "action": "TRANSMIT_PURCHASE_ORDER",
            "fo_id": fo.fo_id,
            "po_id": po.po_id,
            "supplier_id": po.supplier_id,
            "supplier_type": po.supplier_type.value,
            "store_id": po.store_id,
            "created_date": po.created_date,
            "urgency": po.urgency.value,
            "total_order_qty": po.total_order_qty,
            "total_order_cost": po.total_order_cost,
            "scenario": effective_scenario.value if hasattr(effective_scenario, "value") else str(effective_scenario),
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
            fp_payload["rejection_reason"] = rejection_reason.strip()
        incoming_fp = compute_payload_fingerprint(fp_payload)

        # Check existing transmission for clean_key
        existing_tx = self.repository.get_transmission(clean_key)
        if existing_tx:
            if existing_tx["payload_fingerprint"] != incoming_fp or existing_tx["fo_id"] != fo.fo_id:
                raise SupplierGatewayIdempotencyConflictError(
                    f"Idempotency conflict: key '{clean_key}' was previously recorded for fo_id "
                    f"'{existing_tx['fo_id']}' with fingerprint '{existing_tx['payload_fingerprint']}', "
                    f"cannot be reused for fo_id '{fo.fo_id}' with fingerprint '{incoming_fp}'."
                )
            tx_status = existing_tx["transmission_status"]
            if tx_status in (
                SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE.value,
                SupplierTransmissionStatus.CONFIRMED_REJECTION.value,
                SupplierTransmissionStatus.TRANSMISSION_FAILURE.value,
            ):
                result = SupplierTransmissionResult.model_validate_json(existing_tx["result_json"])
                return fo, result
            elif tx_status in (SupplierTransmissionStatus.AMBIGUOUS_OUTCOME.value, "INITIATED"):
                return self.reconcile_transmission(
                    clean_fo_id,
                    po,
                    clean_key,
                    actor_id=clean_actor,
                    expected_version=expected_version,
                    gateway=active_gateway,
                )

        # Fresh transmission attempt
        if fo.status != FulfilmentOrderStatus.READY_TO_TRANSMIT:
            raise InvalidFulfilmentTransitionError(
                f"Cannot transmit FulfilmentOrder '{fo.fo_id}' in status '{fo.status.value}'. "
                "Only orders in READY_TO_TRANSMIT may be transmitted to supplier."
            )

        active_txs = self.repository.get_transmissions_for_order(fo.fo_id)
        for tx in active_txs:
            if tx["transmission_status"] in (SupplierTransmissionStatus.AMBIGUOUS_OUTCOME.value, "INITIATED"):
                raise FulfilmentGovernanceError(
                    f"Cannot initiate fresh transmission for order '{fo.fo_id}': transmission "
                    f"'{tx['idempotency_key']}' has status '{tx['transmission_status']}'. "
                    "It must be reconciled using its original key before initiating a new transmission."
                )

        clean_ts = validate_and_normalize_iso_timestamp(
            timestamp or fo.updated_at, "timestamp"
        )
        key_hash = compute_payload_fingerprint({"k": clean_key})[:12]
        correlation_id = f"CORR-{po.po_id}-{key_hash}"

        # 1. Record Intent (Two-Phase Commit)
        self.repository.record_transmission_intent(
            fo_id=fo.fo_id,
            po_id=po.po_id,
            idempotency_key=clean_key,
            payload_fingerprint=incoming_fp,
            correlation_id=correlation_id,
            timestamp=clean_ts,
        )

        # 2. Invoke Gateway
        try:
            result = active_gateway.transmit_purchase_order(
                po,
                clean_key,
                fo_id=fo.fo_id,
                transmission_timestamp=clean_ts,
                override_scenario=override_scenario,
                rejection_reason=rejection_reason,
                **kwargs,
            )
        except (SupplierGatewayIdempotencyConflictError, InvalidPurchaseOrderError):
            raise
        except Exception as exc:
            # Boundary 3 crash: unexpected error during gateway invocation. Order may have reached vendor.
            # Record AMBIGUOUS_OUTCOME so the order must be reconciled before further action, then re-raise.
            ambiguous_result = SupplierTransmissionResult(
                transmission_id=f"TXN-ERR-{po.po_id}-{key_hash}",
                idempotency_key=clean_key,
                po_id=po.po_id,
                supplier_id=po.supplier_id,
                status=SupplierTransmissionStatus.AMBIGUOUS_OUTCOME,
                transmitted_at=clean_ts,
                correlation_id=correlation_id,
                fo_id=fo.fo_id,
                supplier_reference=None,
                acknowledgment=None,
                error_code="GATEWAY_INVOCATION_EXCEPTION",
                error_message=f"Gateway invocation raised unexpected exception: {exc}",
                details={"exception_type": type(exc).__name__, "error": str(exc)},
            )
            self.repository.record_transmission_outcome(
                clean_key,
                status=SupplierTransmissionStatus.AMBIGUOUS_OUTCOME.value,
                result=ambiguous_result,
                updated_fo=None,
            )
            raise

        # 3. Handle Outcome & Persist
        if result.status == SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE:
            fo_issued = transmit_fulfilment_order(
                fo,
                transmission_timestamp=clean_ts,
                actor_id=clean_actor,
                idempotency_key=f"TXN_TRANS:{clean_key}",
            )
            fo_acked = apply_supplier_acknowledgment(
                fo_issued,
                ack=result.acknowledgment,
                actor_id="SUPPLIER_GATEWAY",
                idempotency_key=f"TXN_ACK:{clean_key}",
            )
            persisted_fo = self.repository.record_transmission_outcome(
                clean_key,
                status=result.status.value,
                result=result,
                updated_fo=fo_acked,
                expected_version=expected_version,
            )
            return persisted_fo, result

        elif result.status == SupplierTransmissionStatus.CONFIRMED_REJECTION:
            fo_issued = transmit_fulfilment_order(
                fo,
                transmission_timestamp=clean_ts,
                actor_id=clean_actor,
                idempotency_key=f"TXN_TRANS:{clean_key}",
            )
            fo_rej = apply_supplier_acknowledgment(
                fo_issued,
                ack=result.acknowledgment,
                actor_id="SUPPLIER_GATEWAY",
                idempotency_key=f"TXN_ACK:{clean_key}",
            )
            persisted_fo = self.repository.record_transmission_outcome(
                clean_key,
                status=result.status.value,
                result=result,
                updated_fo=fo_rej,
                expected_version=expected_version,
            )
            return persisted_fo, result

        elif result.status == SupplierTransmissionStatus.TRANSMISSION_FAILURE:
            # Definite failure before acceptance: aggregate NOT modified
            self.repository.record_transmission_outcome(
                clean_key,
                status=result.status.value,
                result=result,
                updated_fo=None,
            )
            return fo, result

        elif result.status == SupplierTransmissionStatus.AMBIGUOUS_OUTCOME:
            # Supplier may have accepted, but response timed out. Require reconciliation.
            self.repository.record_transmission_outcome(
                clean_key,
                status=result.status.value,
                result=result,
                updated_fo=None,
            )
            return fo, result

        else:
            raise ValueError(f"Unsupported transmission result status: {result.status}")

    # Method alias for clarity
    transmit_order_via_gateway = transmit_purchase_order

    def reconcile_transmission(
        self,
        fo_id: str,
        po: PurchaseOrder,
        idempotency_key: str,
        reconciliation_timestamp: Optional[str] = None,
        actor_id: str = "FULFILMENT_AGENT",
        expected_version: Optional[int] = None,
        gateway: Optional[SupplierGateway] = None,
    ) -> Tuple[FulfilmentOrder, SupplierTransmissionResult]:
        """
        Reconciles an ambiguous, interrupted, or unconfirmed transmission using its original command key.
        Queries the supplier gateway to resolve the actual outcome at the vendor.
        """
        clean_fo_id = _validate_non_empty_str(fo_id, "fo_id")
        clean_key = _validate_non_empty_str(idempotency_key, "idempotency_key")
        clean_actor = _validate_non_empty_str(actor_id, "actor_id")

        if not isinstance(po, PurchaseOrder):
            raise InvalidPurchaseOrderError(f"Expected PurchaseOrder instance, got {type(po)}.")
        if not can_be_dispatched(po) or not verify_approval_evidence(po):
            raise InvalidPurchaseOrderError(
                f"Governance Violation: Purchase Order '{po.po_id}' lacks verified human approval evidence."
            )

        fo = self.repository.get_or_raise(clean_fo_id)
        _validate_po_matches_fo(po, fo)

        tx_rec = self.repository.get_transmission(clean_key)
        if not tx_rec:
            raise ValueError(f"No transmission record found for key '{clean_key}'.")

        active_gateway = gateway or self.gateway

        # If already confirmed, return cached outcome
        if tx_rec["transmission_status"] in (
            SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE.value,
            SupplierTransmissionStatus.CONFIRMED_REJECTION.value,
        ):
            result = SupplierTransmissionResult.model_validate_json(tx_rec["result_json"])
            return fo, result

        clean_ts = validate_and_normalize_iso_timestamp(
            reconciliation_timestamp or fo.updated_at
        )

        # Invoke gateway reconciliation
        reconciled_result = active_gateway.reconcile_transmission(
            clean_key,
            po,
            fo_id=fo.fo_id,
            reconciliation_timestamp=clean_ts,
        )

        if reconciled_result.status == SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE:
            if fo.status == FulfilmentOrderStatus.READY_TO_TRANSMIT:
                fo_issued = transmit_fulfilment_order(
                    fo,
                    transmission_timestamp=clean_ts,
                    actor_id=clean_actor,
                    idempotency_key=f"TXN_TRANS:{clean_key}",
                )
                fo_acked = apply_supplier_acknowledgment(
                    fo_issued,
                    ack=reconciled_result.acknowledgment,
                    actor_id="SUPPLIER_GATEWAY",
                    idempotency_key=f"TXN_ACK:{clean_key}",
                )
                persisted_fo = self.repository.record_transmission_outcome(
                    clean_key,
                    status=reconciled_result.status.value,
                    result=reconciled_result,
                    updated_fo=fo_acked,
                    expected_version=expected_version,
                )
                return persisted_fo, reconciled_result
            else:
                self.repository.record_transmission_outcome(
                    clean_key,
                    status=reconciled_result.status.value,
                    result=reconciled_result,
                    updated_fo=None,
                )
                return fo, reconciled_result

        elif reconciled_result.status == SupplierTransmissionStatus.CONFIRMED_REJECTION:
            if fo.status == FulfilmentOrderStatus.READY_TO_TRANSMIT:
                fo_issued = transmit_fulfilment_order(
                    fo,
                    transmission_timestamp=clean_ts,
                    actor_id=clean_actor,
                    idempotency_key=f"TXN_TRANS:{clean_key}",
                )
                fo_rej = apply_supplier_acknowledgment(
                    fo_issued,
                    ack=reconciled_result.acknowledgment,
                    actor_id="SUPPLIER_GATEWAY",
                    idempotency_key=f"TXN_ACK:{clean_key}",
                )
                persisted_fo = self.repository.record_transmission_outcome(
                    clean_key,
                    status=reconciled_result.status.value,
                    result=reconciled_result,
                    updated_fo=fo_rej,
                    expected_version=expected_version,
                )
                return persisted_fo, reconciled_result
            else:
                self.repository.record_transmission_outcome(
                    clean_key,
                    status=reconciled_result.status.value,
                    result=reconciled_result,
                    updated_fo=None,
                )
                return fo, reconciled_result

        else:
            self.repository.record_transmission_outcome(
                clean_key,
                status=reconciled_result.status.value,
                result=reconciled_result,
                updated_fo=None,
            )
            return fo, reconciled_result

    def get_transmission_record(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a persistent transmission record by idempotency key."""
        return self.repository.get_transmission(idempotency_key)

    def get_transmissions_for_order(self, fo_id: str) -> List[Dict[str, Any]]:
        """Retrieves all persistent transmission records for an order."""
        return self.repository.get_transmissions_for_order(fo_id)

    def acknowledge_order(
        self,
        fo_id: str,
        ack: SupplierAcknowledgment,
        actor_id: str = "SUPPLIER_GATEWAY",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """
        Applies supplier acknowledgment (ACCEPTED, PARTIALLY_ACCEPTED, REJECTED, CHANGE_REQUESTED).
        """
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"ACK:{ack.acknowledgment_id}"
        return self._execute_transition(
            fo_id,
            apply_supplier_acknowledgment,
            expected_version,
            ack=ack,
            actor_id=actor_id,
            idempotency_key=clean_key,
        )

    def dispatch_shipment(
        self,
        fo_id: str,
        asn: AdvanceShipmentNotice,
        actor_id: str = "SUPPLIER_GATEWAY",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Dispatches supplier shipment: -> IN_TRANSIT."""
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"ASN:{asn.shipment_id}"
        return self._execute_transition(
            fo_id,
            apply_shipment_dispatch,
            expected_version,
            asn=asn,
            actor_id=actor_id,
            idempotency_key=clean_key,
        )

    def receive_goods(
        self,
        fo_id: str,
        grn: GoodsReceiptNote,
        actor_id: str = "STORE_RECEIVER",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Conducts dark-store dock receiving: -> PARTIALLY_RECEIVED or RECEIVED."""
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"GRN:{grn.grn_id}"
        return self._execute_transition(
            fo_id,
            apply_goods_receipt,
            expected_version,
            grn=grn,
            actor_id=actor_id,
            idempotency_key=clean_key,
        )

    def process_dock_intake(
        self,
        fo_id: str,
        grn: GoodsReceiptNote,
        po: Optional[PurchaseOrder] = None,
        actor_id: str = "STORE_RECEIVER",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
        verify_carrier_arrival: bool = False,
        auto_post_inventory: bool = True,
    ) -> Tuple[FulfilmentOrder, GoodsReceiptNote]:
        """
        Executes an authorized physical dock intake and inspection workflow.

        Requirements:
        1. Explicit dock-intake workflow through the fulfilment application/service layer.
        2. Requires authorized physical receipt confirmation (actor_id, grn.receiver_id).
        3. Verifies fulfilment order, approved PO (if provided), supplier, store, shipment ID,
           and ASN association.
        4. Supports multiple shipments and multiple GRNs per shipment.
        5. Carrier tracking status DELIVERED may inform the process (e.g. if verify_carrier_arrival=True)
           but never on its own authorizes or creates a GRN.
        6. Captures physical inspection evidence: delivered, accepted, damaged, rejected,
           excess quantities, discrepancy reasons, remaining shelf-life days, batch references.
        7. Enforces quality holds (<60% shelf life, missing shelf life, supervisor signoff > ₹5,000).
        8. Atomically persists GRN, inspection lines, and updated FulfilmentOrder in SQLite.
        9. Durable idempotency: exact replays return (fo, grn) without mutation; conflicts are rejected.
        10. Posts accepted, sellable stock to the operational inventory ledger (auto_post_inventory=True).
            Rejected, damaged, and unresolved quality-held goods are strictly prevented from posting.
        """
        if not isinstance(grn, GoodsReceiptNote):
            raise TypeError(f"Expected GoodsReceiptNote instance, got {type(grn)}.")

        clean_fo_id = _validate_non_empty_str(fo_id, "fo_id")
        clean_actor = _validate_non_empty_str(actor_id, "actor_id")
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"GRN:{grn.grn_id}"

        # 1. Verify FO exists and check governance status
        fo = self.repository.get_or_raise(clean_fo_id)
        if fo.status == FulfilmentOrderStatus.CANCELLED:
            raise FulfilmentGovernanceError(
                f"Cannot process dock intake for cancelled or superseded FulfilmentOrder '{clean_fo_id}'."
            )

        # 2. Verify GRN matches FulfilmentOrder identifiers
        if grn.fo_id != clean_fo_id:
            raise FulfilmentGovernanceError(
                f"GRN fo_id '{grn.fo_id}' does not match target FulfilmentOrder fo_id '{clean_fo_id}'."
            )
        if grn.po_id != fo.po_id:
            raise FulfilmentGovernanceError(
                f"GRN po_id '{grn.po_id}' does not match FulfilmentOrder po_id '{fo.po_id}'."
            )
        if grn.store_id != fo.store_id:
            raise FulfilmentGovernanceError(
                f"GRN store_id '{grn.store_id}' does not match FulfilmentOrder store_id '{fo.store_id}'."
            )
        if grn.supplier_id != fo.supplier_id:
            raise FulfilmentGovernanceError(
                f"GRN supplier_id '{grn.supplier_id}' does not match FulfilmentOrder supplier_id '{fo.supplier_id}'."
            )

        # 3. Verify shipment ID is associated with FO and has ASN dispatch evidence
        if grn.shipment_id not in fo.shipment_ids:
            raise FulfilmentGovernanceError(
                f"Shipment '{grn.shipment_id}' is not associated with FulfilmentOrder '{clean_fo_id}'. "
                f"Active shipment IDs: {fo.shipment_ids}."
            )
        has_asn = any(
            evt.event_type == FulfilmentEventType.SHIPMENT_DISPATCH
            and evt.related_reference_id == grn.shipment_id
            for evt in fo.event_history
        )
        if not has_asn:
            raise FulfilmentGovernanceError(
                f"Shipment '{grn.shipment_id}' has no recorded ASN dispatch on FulfilmentOrder '{clean_fo_id}'."
            )

        # 4. Optional Purchase Order cross-verification
        if po is not None:
            if not isinstance(po, PurchaseOrder):
                raise TypeError(f"Expected PurchaseOrder instance, got {type(po)}.")
            _validate_po_matches_fo(po, fo)
            if grn.po_id != po.po_id:
                raise FulfilmentGovernanceError(
                    f"GRN po_id '{grn.po_id}' does not match PurchaseOrder po_id '{po.po_id}'."
                )

        # 5. Informational carrier tracking verification (Requirement: tracking informs but never creates GRN)
        if verify_carrier_arrival:
            tracking_rec = self.repository.get_tracking_record(grn.shipment_id)
            if tracking_rec and tracking_rec.current_status != ShipmentTrackingStatus.DELIVERED:
                # Informational check when explicit carrier arrival verification is requested
                raise FulfilmentGovernanceError(
                    f"Carrier arrival verification failed for shipment '{grn.shipment_id}': "
                    f"carrier tracking status is '{tracking_rec.current_status.value}', expected 'DELIVERED'."
                )

        # 6. Execute state machine transition and atomic persistence
        updated_fo = self.receive_goods(
            fo_id=clean_fo_id,
            grn=grn,
            actor_id=clean_actor,
            idempotency_key=clean_key,
            expected_version=expected_version,
        )

        # 7. Automatic inventory posting workflow (Phase 5.7.2)
        if auto_post_inventory:
            if grn.total_accepted_quantity > 0:
                posting_key = f"INV_POST:{clean_key}"
                self.post_grn_to_inventory(
                    fo_id=clean_fo_id,
                    grn_id=grn.grn_id,
                    actor_id=clean_actor,
                    idempotency_key=posting_key,
                )
            else:
                self._record_skipped_inventory_posting(grn, clean_key)

        return updated_fo, grn

    def _record_skipped_inventory_posting(self, grn: GoodsReceiptNote, command_key: str) -> None:
        """Records an inventory posting task as SKIPPED_ZERO_ACCEPTED when GRN has no accepted sellable units."""
        task_id = f"POST-TASK-{grn.grn_id}"
        now_utc = dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        task_data = {
            "posting_task_id": task_id,
            "grn_id": grn.grn_id,
            "fo_id": grn.fo_id,
            "po_id": grn.po_id,
            "store_id": grn.store_id,
            "status": "SKIPPED_ZERO_ACCEPTED",
            "idempotency_key": f"INV_POST:{command_key}",
            "payload_fingerprint": compute_payload_fingerprint({
                "grn_id": grn.grn_id,
                "fo_id": grn.fo_id,
                "accepted_qty": 0.0,
            }),
            "total_accepted_quantity": 0.0,
            "posted_quantity": 0.0,
            "error_message": "All goods rejected/damaged; 0 accepted sellable units.",
            "created_at": now_utc,
            "completed_at": now_utc,
            "retry_count": 0,
        }
        self.repository.record_inventory_posting_task(task_data)

    def post_grn_to_inventory(
        self,
        fo_id: str,
        grn_id: str,
        actor_id: str = "STORE_RECEIVER",
        idempotency_key: Optional[str] = None,
    ) -> Tuple[InventoryLedgerEntry, ...]:
        """
        Posts accepted sellable batches from a persisted GoodsReceiptNote to the operational inventory ledger.
        Enforces durable task tracking, exact idempotency replay, and recovery across restarts.
        """
        clean_fo_id = _validate_non_empty_str(fo_id, "fo_id")
        clean_grn_id = _validate_non_empty_str(grn_id, "grn_id")
        clean_actor = _validate_non_empty_str(actor_id, "actor_id")
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"INV_POST:{clean_grn_id}"

        grn = self.repository.get_goods_receipt_note_or_raise(clean_grn_id)
        if grn.fo_id != clean_fo_id:
            raise FulfilmentGovernanceError(
                f"GRN '{clean_grn_id}' belongs to FO '{grn.fo_id}', not '{clean_fo_id}'."
            )

        if grn.total_accepted_quantity <= 0:
            from src.agents.inventory.schemas import InventoryGovernanceError
            raise InventoryGovernanceError(
                f"Goods Receipt Note '{clean_grn_id}' has 0 accepted sellable units. "
                "Rejected, damaged, or unaccepted goods cannot be posted to operational inventory."
            )

        task_id = f"POST-TASK-{clean_grn_id}"
        existing_task = self.repository.get_inventory_posting_task(task_id)

        # Check reconciliation with ledger for idempotent recovery/replay
        recon = self.inventory_repository.reconcile_grn_postings(grn)
        if recon.is_fully_posted and existing_task and existing_task["idempotency_key"] == clean_key:
            entries = self.inventory_repository.get_ledger_history(grn_id=clean_grn_id)
            if existing_task["status"] != "COMPLETED":
                self.repository.update_inventory_posting_task(
                    posting_task_id=task_id,
                    status="COMPLETED",
                    posted_quantity=recon.total_posted_quantity,
                    posting_ids=[e.posting_id for e in entries],
                    completed_at=dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            return entries

        is_already_completed = bool(existing_task and existing_task["status"] == "COMPLETED")
        now_utc = dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not is_already_completed:
            task_data = {
                "posting_task_id": task_id,
                "grn_id": clean_grn_id,
                "fo_id": clean_fo_id,
                "po_id": grn.po_id,
                "store_id": grn.store_id,
                "status": "PENDING",
                "idempotency_key": clean_key,
                "payload_fingerprint": compute_payload_fingerprint({
                    "grn_id": clean_grn_id,
                    "fo_id": clean_fo_id,
                    "accepted_qty": grn.total_accepted_quantity,
                }),
                "total_accepted_quantity": grn.total_accepted_quantity,
                "posted_quantity": 0.0,
                "created_at": existing_task["created_at"] if existing_task else now_utc,
                "retry_count": (existing_task["retry_count"] + 1) if existing_task else 0,
            }
            self.repository.record_inventory_posting_task(task_data)

        from src.agents.inventory.schemas import InventoryPostingReason
        try:
            entries = self.inventory_repository.post_inbound_receipt(
                grn=grn,
                idempotency_key=clean_key,
                actor_id=clean_actor,
                reason=InventoryPostingReason.GOODS_RECEIPT.value,
            )
            if not is_already_completed:
                self.repository.update_inventory_posting_task(
                    posting_task_id=task_id,
                    status="COMPLETED",
                    posted_quantity=grn.total_accepted_quantity,
                    posting_ids=[e.posting_id for e in entries],
                    completed_at=dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            return entries
        except Exception as exc:
            if not is_already_completed:
                self.repository.update_inventory_posting_task(
                    posting_task_id=task_id,
                    status="FAILED",
                    posted_quantity=0.0,
                    posting_ids=[],
                    error_message=str(exc),
                )
            raise

    def post_quality_hold_release_to_inventory(
        self,
        fo_id: str,
        quality_hold_resolution_id: str,
        actor_id: str = "QA_LEAD",
        idempotency_key: Optional[str] = None,
    ) -> Tuple[InventoryLedgerEntry, ...]:
        """
        Posts released stock from an authorized REINSPECTION_RELEASE quality hold resolution
        to the operational inventory ledger.
        """
        clean_fo_id = _validate_non_empty_str(fo_id, "fo_id")
        clean_res_id = _validate_non_empty_str(quality_hold_resolution_id, "quality_hold_resolution_id")
        clean_actor = _validate_non_empty_str(actor_id, "actor_id")
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"INV_REL:{clean_res_id}"

        fo = self.repository.get_or_raise(clean_fo_id)
        qhr = next((r for r in fo.quality_hold_resolutions if r.resolution_id == clean_res_id), None)
        if not qhr:
            raise FulfilmentGovernanceError(
                f"Quality hold resolution '{clean_res_id}' not found on FulfilmentOrder '{clean_fo_id}'."
            )

        if qhr.resolution_type != DiscrepancyResolutionType.REINSPECTION_RELEASE:
            raise FulfilmentGovernanceError(
                f"Cannot post quality hold resolution '{clean_res_id}' with type '{qhr.resolution_type.value}'. "
                "Only REINSPECTION_RELEASE resolutions can be posted to sellable inventory."
            )

        grns = self.repository.get_grns_for_shipment(qhr.shipment_id)
        if not grns:
            grns = self.repository.get_grns_for_order(clean_fo_id)
        if not grns:
            raise FulfilmentGovernanceError(
                f"No GoodsReceiptNote found for shipment '{qhr.shipment_id}' on FO '{clean_fo_id}'."
            )
        grn = next(
            (g for g in grns if any(ln.sku_id == qhr.sku_id and ln.rejected_quantity > 0 for ln in g.inspection_lines)),
            next((g for g in grns if any(ln.sku_id == qhr.sku_id for ln in g.inspection_lines)), grns[0]),
        )

        task_id = f"POST-TASK-QHR-{clean_res_id}"
        existing_task = self.repository.get_inventory_posting_task(task_id)

        # Check existing entries if already completed with exact idempotency key
        if existing_task and existing_task["status"] == "COMPLETED" and existing_task["idempotency_key"] == clean_key:
            return tuple(
                e for e in self.inventory_repository.get_ledger_history(grn_id=grn.grn_id)
                if e.batch_id == f"BATCH-REL-{clean_res_id}"
            )

        is_already_completed = bool(existing_task and existing_task["status"] == "COMPLETED")
        now_utc = dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not is_already_completed:
            task_data = {
                "posting_task_id": task_id,
                "task_type": "QA_RELEASE",
                "resolution_id": clean_res_id,
                "grn_id": grn.grn_id,
                "fo_id": clean_fo_id,
                "po_id": grn.po_id,
                "store_id": grn.store_id,
                "status": "PENDING",
                "idempotency_key": clean_key,
                "payload_fingerprint": compute_payload_fingerprint({
                    "resolution_id": clean_res_id,
                    "fo_id": clean_fo_id,
                    "grn_id": grn.grn_id,
                    "sku_id": qhr.sku_id,
                    "quantity": qhr.quantity,
                }),
                "total_accepted_quantity": qhr.quantity,
                "posted_quantity": 0.0,
                "created_at": existing_task["created_at"] if existing_task else now_utc,
                "retry_count": (existing_task["retry_count"] + 1) if existing_task else 0,
            }
            self.repository.record_inventory_posting_task(task_data)

        batch = FulfilmentBatch(
            batch_id=f"BATCH-REL-{qhr.resolution_id}",
            sku_id=qhr.sku_id,
            units=qhr.quantity,
            remaining_shelf_life_days=qhr.verified_remaining_shelf_life_days if qhr.verified_remaining_shelf_life_days is not None else 0,
            receipt_date=qhr.timestamp[:10],
            fo_id=fo.fo_id,
            grn_id=grn.grn_id,
            supplier_lot_ref=qhr.disposition_reference,
        )

        from src.agents.inventory.schemas import InventoryPostingReason
        try:
            entries = self.inventory_repository.post_inbound_receipt(
                grn=grn,
                batches=(batch,),
                idempotency_key=clean_key,
                actor_id=clean_actor,
                reason=InventoryPostingReason.GOODS_RECEIPT.value,
                notes=f"Released from quality hold {qhr.resolution_id} ({qhr.disposition_reference})",
            )
            if not is_already_completed:
                self.repository.update_inventory_posting_task(
                    posting_task_id=task_id,
                    status="COMPLETED",
                    posted_quantity=qhr.quantity,
                    posting_ids=[e.posting_id for e in entries],
                    completed_at=dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            return entries
        except Exception as exc:
            if not is_already_completed:
                self.repository.update_inventory_posting_task(
                    posting_task_id=task_id,
                    status="FAILED",
                    posted_quantity=0.0,
                    posting_ids=[],
                    error_message=str(exc),
                )
            raise

    def retry_failed_inventory_postings(self) -> List[Dict[str, Any]]:
        """Retries all failed inventory posting tasks (both GRN and QA releases)."""
        tasks = self.repository.get_pending_or_failed_inventory_posting_tasks()
        results = []
        for task in tasks:
            if task["status"] == "FAILED":
                is_qa = (
                    task.get("task_type") == "QA_RELEASE"
                    or bool(task.get("resolution_id"))
                    or task["posting_task_id"].startswith("POST-TASK-QHR-")
                )
                try:
                    if is_qa:
                        res_id = task.get("resolution_id") or task["posting_task_id"].replace("POST-TASK-QHR-", "")
                        entries = self.post_quality_hold_release_to_inventory(
                            fo_id=task["fo_id"],
                            quality_hold_resolution_id=res_id,
                            actor_id="RETRY_WORKER",
                            idempotency_key=task["idempotency_key"],
                        )
                    else:
                        entries = self.post_grn_to_inventory(
                            fo_id=task["fo_id"],
                            grn_id=task["grn_id"],
                            actor_id="RETRY_WORKER",
                            idempotency_key=task["idempotency_key"],
                        )
                    results.append({
                        "posting_task_id": task["posting_task_id"],
                        "task_type": "QA_RELEASE" if is_qa else "GRN_RECEIPT",
                        "grn_id": task["grn_id"],
                        "resolution_id": task.get("resolution_id"),
                        "status": "COMPLETED",
                        "entries_count": len(entries),
                    })
                except Exception as exc:
                    results.append({
                        "posting_task_id": task["posting_task_id"],
                        "task_type": "QA_RELEASE" if is_qa else "GRN_RECEIPT",
                        "grn_id": task["grn_id"],
                        "resolution_id": task.get("resolution_id"),
                        "status": "FAILED",
                        "error": str(exc),
                    })
        return results

    def recover_pending_inventory_postings(self) -> List[Dict[str, Any]]:
        """Recovers both PENDING and FAILED inventory posting tasks after service/system restart."""
        tasks = self.repository.get_pending_or_failed_inventory_posting_tasks()
        results = []
        handled_task_ids = set()

        for task in tasks:
            task_id = task["posting_task_id"]
            handled_task_ids.add(task_id)
            is_qa = (
                task.get("task_type") == "QA_RELEASE"
                or bool(task.get("resolution_id"))
                or task_id.startswith("POST-TASK-QHR-")
            )
            try:
                if is_qa:
                    res_id = task.get("resolution_id") or task_id.replace("POST-TASK-QHR-", "")
                    entries = self.post_quality_hold_release_to_inventory(
                        fo_id=task["fo_id"],
                        quality_hold_resolution_id=res_id,
                        actor_id="RECOVERY_WORKER",
                        idempotency_key=task["idempotency_key"],
                    )
                else:
                    entries = self.post_grn_to_inventory(
                        fo_id=task["fo_id"],
                        grn_id=task["grn_id"],
                        actor_id="RECOVERY_WORKER",
                        idempotency_key=task["idempotency_key"],
                    )
                results.append({
                    "posting_task_id": task_id,
                    "task_type": "QA_RELEASE" if is_qa else "GRN_RECEIPT",
                    "grn_id": task["grn_id"],
                    "resolution_id": task.get("resolution_id"),
                    "status": "COMPLETED",
                    "entries_count": len(entries),
                })
            except Exception as exc:
                results.append({
                    "posting_task_id": task_id,
                    "task_type": "QA_RELEASE" if is_qa else "GRN_RECEIPT",
                    "grn_id": task["grn_id"],
                    "resolution_id": task.get("resolution_id"),
                    "status": "FAILED",
                    "error": str(exc),
                })

        # Also scan orders for any unposted REINSPECTION_RELEASE resolutions that might have missed task creation
        orders = self.repository.list()
        for fo in orders:
            for qhr in fo.quality_hold_resolutions:
                if qhr.resolution_type == DiscrepancyResolutionType.REINSPECTION_RELEASE:
                    qhr_task_id = f"POST-TASK-QHR-{qhr.resolution_id}"
                    if qhr_task_id in handled_task_ids:
                        continue
                    existing = self.repository.get_inventory_posting_task(qhr_task_id)
                    if existing and existing["status"] == "COMPLETED":
                        continue
                    try:
                        entries = self.post_quality_hold_release_to_inventory(
                            fo_id=fo.fo_id,
                            quality_hold_resolution_id=qhr.resolution_id,
                            actor_id="RECOVERY_WORKER",
                            idempotency_key=f"INV_REL:{qhr.resolution_id}",
                        )
                        results.append({
                            "posting_task_id": qhr_task_id,
                            "task_type": "QA_RELEASE",
                            "grn_id": qhr.shipment_id,
                            "resolution_id": qhr.resolution_id,
                            "status": "COMPLETED",
                            "entries_count": len(entries),
                        })
                    except Exception as exc:
                        results.append({
                            "posting_task_id": qhr_task_id,
                            "task_type": "QA_RELEASE",
                            "grn_id": qhr.shipment_id,
                            "resolution_id": qhr.resolution_id,
                            "status": "FAILED",
                            "error": str(exc),
                        })

        return results

    def get_inventory_posting_task(self, posting_task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an inventory posting task by its ID."""
        return self.repository.get_inventory_posting_task(posting_task_id)

    def get_inventory_posting_task_for_grn(self, grn_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the inventory posting task for a GRN."""
        return self.repository.get_inventory_posting_task_for_grn(grn_id)

    def get_inventory_posting_task_for_resolution(self, resolution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the inventory posting task for a quality hold resolution."""
        return self.repository.get_inventory_posting_task_for_resolution(resolution_id)

    def reconcile_inventory_postings(self, grn_id: str) -> GrnPostingReconciliation:
        """Performs line-level audit reconciliation between GRN accepted units and ledger entries."""
        clean_grn = _validate_non_empty_str(grn_id, "grn_id")
        return self.inventory_repository.reconcile_grn_postings(clean_grn)

    def get_inventory_position(
        self,
        store_id: str,
        sku_id: str,
        reserved_stock: float = 0.0,
    ) -> InventoryPosition:
        """
        Derives an authoritative InventoryPosition model for InventoryEngine:
        - on_hand_stock: derived dynamically from operational inventory ledger
        - on_order_stock: derived dynamically from outstanding confirmed inbound pipeline
        - reserved_stock: customer reservations
        - net_available_stock: on_hand_stock - reserved_stock
        """
        clean_store = _validate_non_empty_str(store_id, "store_id")
        clean_sku = _validate_non_empty_str(sku_id, "sku_id")
        on_hand = self.inventory_repository.get_on_hand_balance(clean_store, clean_sku)
        pipeline = self.repository.get_confirmed_pipeline_quantity(store_id=clean_store, sku_id=clean_sku)
        return InventoryPosition(
            on_hand_stock=on_hand,
            reserved_stock=reserved_stock,
            on_order_stock=pipeline,
        )

    def get_active_batches(
        self,
        store_id: str,
        sku_id: str,
        as_of_date: Optional[str] = None,
    ) -> Tuple[InventoryBatch, ...]:
        """
        Retrieves active batches from the operational inventory ledger as core InventoryBatch
        models sorted by FEFO for immediate consumption by InventoryEngine.
        """
        clean_store = _validate_non_empty_str(store_id, "store_id")
        clean_sku = _validate_non_empty_str(sku_id, "sku_id")
        return self.inventory_repository.get_active_inventory_batches(
            clean_store, clean_sku, as_of_date=as_of_date
        )

    def run_inventory_assessment(
        self,
        daily_input: Any,
        reserved_stock: float = 0.0,
        engine: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Executes an end-to-end inventory policy calculation and risk assessment
        using ledger-derived on-hand stock and FEFO active batches.
        """
        store_id = daily_input.store_id
        sku_id = daily_input.sku_id
        current_date = str(daily_input.date)

        inv_pos = self.get_inventory_position(store_id, sku_id, reserved_stock=reserved_stock)
        batches = list(self.get_active_batches(store_id, sku_id, as_of_date=current_date))

        if engine is not None:
            inv_engine = engine
        else:
            from src.agents.inventory.engine import InventoryEngine
            inv_engine = InventoryEngine()
        return inv_engine.calculate_and_assess_risk(
            daily_input=daily_input,
            inventory_position=inv_pos,
            batches=batches,
        )

    def get_goods_receipt_note(self, grn_id: str) -> Optional[GoodsReceiptNote]:
        """Retrieves a persisted GoodsReceiptNote by its identifier."""
        return self.repository.get_goods_receipt_note(grn_id)

    def get_goods_receipt_note_or_raise(self, grn_id: str) -> GoodsReceiptNote:
        """Retrieves a persisted GoodsReceiptNote or raises FulfilmentPersistenceError."""
        return self.repository.get_goods_receipt_note_or_raise(grn_id)

    def get_grns_for_order(self, fo_id: str) -> List[GoodsReceiptNote]:
        """Retrieves all GoodsReceiptNotes recorded for a given FulfilmentOrder."""
        return self.repository.get_grns_for_order(fo_id)

    def get_grns_for_shipment(self, shipment_id: str) -> List[GoodsReceiptNote]:
        """Retrieves all GoodsReceiptNotes recorded for a given shipment."""
        return self.repository.get_grns_for_shipment(shipment_id)

    def get_fulfilment_batches_for_grn(self, grn_id: str) -> Tuple[FulfilmentBatch, ...]:
        """
        Loads the persisted GoodsReceiptNote and mints immutable FulfilmentBatch records
        for all accepted stock.
        """
        grn = self.repository.get_goods_receipt_note_or_raise(grn_id)
        return generate_grn_fulfilment_batches(grn)

    # -------------------------------------------------------------------------
    # Goods Receipt Reconciliation & Discrepancy Resolution (Phase 5.6.2)
    # -------------------------------------------------------------------------

    def reconcile_receipts(self, fo_id: str) -> OrderReceiptReconciliation:
        """
        Calculates authoritative cumulative goods receipt reconciliation for a FulfilmentOrder
        across all persisted GRNs.
        """
        clean_id = _validate_non_empty_str(fo_id, "fo_id")
        fo = self.repository.get_or_raise(clean_id)
        grns = self.repository.get_grns_for_order(clean_id)
        return calculate_order_receipt_reconciliation(fo, grns)

    def get_shipment_reconciliation(self, fo_id: str, shipment_id: str) -> ShipmentReceiptReconciliation:
        """
        Retrieves cumulative receipt and physical custody reconciliation for a specific shipment.
        """
        clean_shipment = _validate_non_empty_str(shipment_id, "shipment_id")
        order_rec = self.reconcile_receipts(fo_id)
        for s_rec in order_rec.shipments:
            if s_rec.shipment_id == clean_shipment:
                return s_rec
        raise ShipmentNotFoundError(f"Shipment '{clean_shipment}' not found on order '{fo_id}'.")

    def resolve_discrepancy(
        self,
        fo_id: str,
        shipment_id: str,
        sku_id: str,
        resolution_type: Union[DiscrepancyResolutionType, str],
        quantity: float,
        discrepancy_reason: Union[DiscrepancyReason, str],
        disposition_reference: str,
        actor_id: str = "OPERATIONS_SUPERVISOR",
        timestamp: str = "2024-11-15T14:00:00Z",
        idempotency_key: Optional[str] = None,
        notes: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> Tuple[FulfilmentOrder, DiscrepancyResolution]:
        """
        Authorizes and records resolution for damaged, excess, short, or rejected goods.
        Persists resolution atomically with the order and audit trail.
        """
        res_type = DiscrepancyResolutionType(resolution_type) if isinstance(resolution_type, str) else resolution_type
        disc_reason = DiscrepancyReason(discrepancy_reason) if isinstance(discrepancy_reason, str) else discrepancy_reason
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"RES:{fo_id}:{shipment_id}:{sku_id}:{res_type.value}"

        updated_fo = self._execute_transition(
            fo_id,
            resolve_goods_receipt_discrepancy,
            expected_version,
            shipment_id=shipment_id,
            sku_id=sku_id,
            resolution_type=res_type,
            quantity=quantity,
            discrepancy_reason=disc_reason,
            disposition_reference=disposition_reference,
            actor_id=actor_id,
            timestamp=timestamp,
            idempotency_key=clean_key,
            notes=notes,
        )

        matching_res = next(
            (r for r in reversed(updated_fo.discrepancy_resolutions)
             if r.shipment_id == shipment_id and r.sku_id == sku_id and r.disposition_reference == disposition_reference),
            updated_fo.discrepancy_resolutions[-1] if updated_fo.discrepancy_resolutions else None
        )
        return updated_fo, matching_res

    def resolve_quality_hold(
        self,
        fo_id: str,
        shipment_id: str,
        sku_id: str,
        resolution_type: Union[DiscrepancyResolutionType, str],
        quantity: float,
        disposition_reference: str,
        inspector_id: str = "QA_LEAD",
        timestamp: str = "2024-11-15T14:30:00Z",
        idempotency_key: Optional[str] = None,
        inspection_notes: str = "Reinspection passed quality standards.",
        verified_remaining_shelf_life_days: Optional[int] = None,
        total_shelf_life_days: Optional[int] = None,
        expected_version: Optional[int] = None,
        auto_post_inventory: bool = True,
    ) -> Tuple[FulfilmentOrder, QualityHoldResolution]:
        """
        Executes an authorized quality hold resolution or reinspection disposition.
        Supports REINSPECTION_RELEASE (with >=60% verified shelf-life ratio),
        CERTIFIED_DISPOSAL, and RETURN_TO_ORIGIN.
        """
        res_type = DiscrepancyResolutionType(resolution_type) if isinstance(resolution_type, str) else resolution_type
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"QHR:{fo_id}:{shipment_id}:{sku_id}:{res_type.value}"

        updated_fo = self._execute_transition(
            fo_id,
            resolve_quality_hold,
            expected_version,
            shipment_id=shipment_id,
            sku_id=sku_id,
            resolution_type=res_type,
            quantity=quantity,
            disposition_reference=disposition_reference,
            inspector_id=inspector_id,
            timestamp=timestamp,
            idempotency_key=clean_key,
            inspection_notes=inspection_notes,
            verified_remaining_shelf_life_days=verified_remaining_shelf_life_days,
            total_shelf_life_days=total_shelf_life_days,
        )

        matching_qhr = next(
            (r for r in reversed(updated_fo.quality_hold_resolutions)
             if r.shipment_id == shipment_id and r.sku_id == sku_id and r.disposition_reference == disposition_reference),
            updated_fo.quality_hold_resolutions[-1] if updated_fo.quality_hold_resolutions else None
        )

        if auto_post_inventory and res_type == DiscrepancyResolutionType.REINSPECTION_RELEASE and matching_qhr:
            self.post_quality_hold_release_to_inventory(
                fo_id=fo_id,
                quality_hold_resolution_id=matching_qhr.resolution_id,
                actor_id=inspector_id,
                idempotency_key=f"INV_REL:{matching_qhr.resolution_id}",
            )

        return updated_fo, matching_qhr

    def resolve_shipment_custody(
        self,
        fo_id: str,
        shipment_id: str,
        disposition_reference: str,
        actor_id: str = "OPERATIONS_SUPERVISOR",
        timestamp: str = "2024-11-15T15:00:00Z",
        idempotency_key: Optional[str] = None,
        notes: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """
        Explicitly records verified physical custody resolution for an entire shipment.
        """
        clean_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"CUSTODY:{fo_id}:{shipment_id}"
        return self._execute_transition(
            fo_id,
            resolve_shipment_custody,
            expected_version,
            shipment_id=shipment_id,
            disposition_reference=disposition_reference,
            actor_id=actor_id,
            timestamp=timestamp,
            idempotency_key=clean_key,
            notes=notes,
        )

    def get_discrepancy_resolutions_for_order(self, fo_id: str) -> List[DiscrepancyResolution]:
        """Retrieves all discrepancy resolutions recorded for a given FulfilmentOrder."""
        return self.repository.get_discrepancy_resolutions_for_order(fo_id)

    def get_discrepancy_resolutions_for_shipment(self, shipment_id: str) -> List[DiscrepancyResolution]:
        """Retrieves all discrepancy resolutions recorded for a given shipment."""
        return self.repository.get_discrepancy_resolutions_for_shipment(shipment_id)

    def get_quality_hold_resolutions_for_order(self, fo_id: str) -> List[QualityHoldResolution]:
        """Retrieves all quality hold resolutions recorded for a given FulfilmentOrder."""
        return self.repository.get_quality_hold_resolutions_for_order(fo_id)

    def get_quality_hold_resolutions_for_shipment(self, shipment_id: str) -> List[QualityHoldResolution]:
        """Retrieves all quality hold resolutions recorded for a given shipment."""
        return self.repository.get_quality_hold_resolutions_for_shipment(shipment_id)

    def record_delivery_exception(
        self,
        fo_id: str,
        exception_reason: str,
        actor_id: str = "STORE_RECEIVER",
        timestamp: str = "2024-11-15T12:00:00Z",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Records active delivery exception: -> DELIVERY_EXCEPTION."""
        return self._execute_transition(
            fo_id,
            record_delivery_exception,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="EXC",
                event_type=FulfilmentEventType.DELIVERY_EXCEPTION,
                check_active_status=FulfilmentOrderStatus.DELIVERY_EXCEPTION,
            ),
            exception_reason=exception_reason,
            actor_id=actor_id,
            timestamp=timestamp,
        )

    def resolve_delivery_exception(
        self,
        fo_id: str,
        resolution_notes: str,
        actor_id: str = "OPERATIONS_SUPERVISOR",
        timestamp: str = "2024-11-15T12:30:00Z",
        idempotency_key: Optional[str] = None,
        target_status: Optional[Union[FulfilmentOrderStatus, str]] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Resolves delivery exception returning to resumable_status or explicitly validated target_status."""
        resolved_target = FulfilmentOrderStatus(target_status) if isinstance(target_status, str) else target_status
        return self._execute_transition(
            fo_id,
            resolve_delivery_exception,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="RES",
                event_type=FulfilmentEventType.RESOLVE_EXCEPTION,
                filter_fn=lambda e: e.details.get("prior_exception") is not None,
            ),
            resolution_notes=resolution_notes,
            actor_id=actor_id,
            timestamp=timestamp,
            target_status=resolved_target,
        )

    def close_short_order(
        self,
        fo_id: str,
        closure_reason: str,
        actor_id: str = "STORE_MANAGER",
        timestamp: str = "2024-11-15T16:00:00Z",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Managerial closure of unfulfilled balances: -> CLOSED_SHORT."""
        return self._execute_transition(
            fo_id,
            close_short_fulfilment_order,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="CLS",
                event_type=FulfilmentEventType.CLOSE_SHORT,
                check_active_status=FulfilmentOrderStatus.CLOSED_SHORT,
            ),
            closure_reason=closure_reason,
            actor_id=actor_id,
            timestamp=timestamp,
        )

    def request_cancellation(
        self,
        fo_id: str,
        reason: str,
        category: CancellationReasonCategory,
        actor_id: str = "OPERATIONS_SUPERVISOR",
        timestamp: str = "2024-11-15T12:00:00Z",
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Initiates cancellation request: -> CANCEL_REQUESTED."""
        return self._execute_transition(
            fo_id,
            request_cancellation,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="REQ_CNC",
                event_type=FulfilmentEventType.REQUEST_CANCEL,
                check_active_status=FulfilmentOrderStatus.CANCEL_REQUESTED,
            ),
            reason=reason,
            category=category,
            actor_id=actor_id,
            timestamp=timestamp,
        )

    def confirm_cancellation(
        self,
        fo_id: str,
        actor_id: str = "OPERATIONS_SUPERVISOR",
        timestamp: str = "2024-11-15T12:30:00Z",
        idempotency_key: Optional[str] = None,
        supplier_cancellation_ref: Optional[str] = None,
        shipment_cancellation_ref: Optional[str] = None,
        shipment_resolutions: Optional[Dict[str, str]] = None,
        order_wide_cancellation_scope: bool = False,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """Finalizes order cancellation: CANCEL_REQUESTED -> CANCELLED."""
        return self._execute_transition(
            fo_id,
            confirm_cancellation,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="CONF_CNC",
                event_type=FulfilmentEventType.CONFIRM_CANCEL,
                check_active_status=FulfilmentOrderStatus.CANCELLED,
            ),
            actor_id=actor_id,
            timestamp=timestamp,
            supplier_cancellation_ref=supplier_cancellation_ref,
            shipment_cancellation_ref=shipment_cancellation_ref,
            shipment_resolutions=shipment_resolutions,
            order_wide_cancellation_scope=order_wide_cancellation_scope,
        )

    def reject_cancellation(
        self,
        fo_id: str,
        rejection_reason: str,
        actor_id: str = "OPERATIONS_SUPERVISOR",
        timestamp: str = "2024-11-15T12:30:00Z",
        idempotency_key: Optional[str] = None,
        target_status: Optional[Union[FulfilmentOrderStatus, str]] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """
        Rejects or withdraws cancellation request: CANCEL_REQUESTED -> prior active status.
        """
        resolved_target = FulfilmentOrderStatus(target_status) if isinstance(target_status, str) else target_status
        return self._execute_transition(
            fo_id,
            reject_cancellation_request,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="REJ_CNC",
                event_type=FulfilmentEventType.REJECT_CANCEL,
            ),
            rejection_reason=rejection_reason,
            actor_id=actor_id,
            timestamp=timestamp,
            target_status=resolved_target,
        )

    def reaffirm_terms(
        self,
        fo_id: str,
        resolution_notes: str,
        actor_id: str = "OPERATIONS_SUPERVISOR",
        timestamp: str = "2024-11-15T12:00:00Z",
        idempotency_key: Optional[str] = None,
        ack: Optional[SupplierAcknowledgment] = None,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """
        Supplier agrees to original PO terms after change request:
        SUPPLIER_CHANGE_PENDING_REVIEW -> ACKNOWLEDGED.
        """
        return self._execute_transition(
            fo_id,
            reaffirm_original_terms,
            expected_version,
            key_resolver=lambda fo: _resolve_idempotency_key(
                fo,
                idempotency_key,
                default_prefix="REAFFIRM",
                event_type=FulfilmentEventType.RESOLVE_EXCEPTION,
                filter_fn=lambda e: e.details.get("terms_status") == "ORIGINAL_TERMS_RETAINED",
            ),
            resolution_notes=resolution_notes,
            actor_id=actor_id,
            timestamp=timestamp,
            ack=ack,
        )

    # -------------------------------------------------------------------------
    # Multi-Aggregate Atomic Supersession
    # -------------------------------------------------------------------------

    def supersede_order(
        self,
        original_fo_id: str,
        replacement_po: PurchaseOrder,
        replacement_fo_id: str,
        actor_id: str,
        timestamp: str,
        idempotency_key: str,
        reason: str,
        supplier_cancellation_ref: Optional[str] = None,
        shipment_cancellation_ref: Optional[str] = None,
        shipment_resolutions: Optional[Dict[str, str]] = None,
        order_wide_cancellation_scope: bool = False,
        expected_version: Optional[int] = None,
    ) -> Tuple[FulfilmentOrder, FulfilmentOrder]:
        """
        Supersedes an existing FulfilmentOrder with a newly approved replacement PurchaseOrder.
        Atomically persists both aggregates (updating superseded order to CANCELLED and creating
        replacement order) within a single SQLite transaction.

        If either operation fails, neither order is committed.
        """
        clean_id = original_fo_id.strip() if original_fo_id else ""
        if not clean_id:
            raise ValueError("Field 'original_fo_id' must be a non-empty string.")

        # 1. Load latest original order and its database version
        current_fo = self.repository.get_or_raise(clean_id)
        current_version = self.repository.get_version(clean_id)

        # 2. Check optimistic concurrency version
        if expected_version is not None and expected_version != current_version:
            raise StaleUpdateError(
                f"Stale update conflict for '{clean_id}': expected version {expected_version}, "
                f"but database is currently at version {current_version}."
            )

        # Check for unresolved gateway transmissions
        active_txs = self.repository.get_transmissions_for_order(clean_id)
        for tx in active_txs:
            if tx["transmission_status"] in (SupplierTransmissionStatus.AMBIGUOUS_OUTCOME.value, "INITIATED"):
                raise FulfilmentGovernanceError(
                    f"Order '{clean_id}' has an unresolved transmission '{tx['idempotency_key']}' "
                    f"with status '{tx['transmission_status']}'. It must be reconciled before superseding."
                )

        target_version = current_version if expected_version is None else expected_version

        # 3. Call domain supersession
        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=current_fo,
            replacement_po=replacement_po,
            replacement_fo_id=replacement_fo_id,
            actor_id=actor_id,
            timestamp=timestamp,
            idempotency_key=idempotency_key,
            reason=reason,
            supplier_cancellation_ref=supplier_cancellation_ref,
            shipment_cancellation_ref=shipment_cancellation_ref,
            shipment_resolutions=shipment_resolutions,
            order_wide_cancellation_scope=order_wide_cancellation_scope,
        )

        # 4. Check for idempotent replay
        if len(superseded_fo.event_history) == len(current_fo.event_history):
            existing_rep = self.repository.get(replacement_fo_id)
            if existing_rep:
                return current_fo, existing_rep
            return current_fo, replacement_fo

        # 5. Atomic multi-aggregate persistence
        self.repository.persist_supersession(
            superseded_fo=superseded_fo,
            replacement_fo=replacement_fo,
            expected_superseded_version=target_version,
        )

        return superseded_fo, replacement_fo

    # -------------------------------------------------------------------------
    # Query & Reporting Operations
    # -------------------------------------------------------------------------

    def get_order(self, fo_id: str) -> Optional[FulfilmentOrder]:
        """Loads order aggregate by ID."""
        return self.repository.get(fo_id)

    def get_order_or_raise(self, fo_id: str) -> FulfilmentOrder:
        """Loads order aggregate by ID or raises FulfilmentOrderNotFoundError."""
        return self.repository.get_or_raise(fo_id)

    def get_order_version(self, fo_id: str) -> Optional[int]:
        """Returns optimistic concurrency version of the order."""
        return self.repository.get_version(fo_id)

    def get_orders_by_po_id(self, po_id: str) -> List[FulfilmentOrder]:
        """Returns all orders associated with a Purchase Order ID."""
        return self.repository.get_by_po_id(po_id)

    def list_orders(
        self,
        store_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        status: Optional[Union[FulfilmentOrderStatus, str]] = None,
        statuses: Optional[Iterable[Union[FulfilmentOrderStatus, str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[FulfilmentOrder]:
        """Lists orders with optional filters and pagination."""
        return self.repository.list(
            store_id=store_id,
            supplier_id=supplier_id,
            status=status,
            statuses=statuses,
            limit=limit,
            offset=offset,
        )

    def count_orders(
        self,
        store_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        status: Optional[Union[FulfilmentOrderStatus, str]] = None,
    ) -> int:
        """Counts matching orders."""
        return self.repository.count(store_id=store_id, supplier_id=supplier_id, status=status)

    def get_order_events(self, fo_id: str) -> List[FulfilmentEvent]:
        """Retrieves verified audit events from the relational log."""
        return self.repository.get_events(fo_id)

    def get_pipeline_quantity(
        self,
        store_id: Optional[str] = None,
        sku_id: Optional[str] = None,
    ) -> float:
        """Derives confirmed outstanding inbound pipeline units directly from persisted aggregates."""
        return self.repository.get_confirmed_pipeline_quantity(store_id=store_id, sku_id=sku_id)

    def get_sku_pipeline_quantities(
        self,
        store_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Returns SKU -> pipeline quantity mapping."""
        return self.repository.get_sku_pipeline_quantities(store_id=store_id)

    def get_pipeline_breakdown(
        self,
        store_id: Optional[str] = None,
        sku_id: Optional[str] = None,
    ) -> Dict[Tuple[str, str], float]:
        """Returns (store_id, sku_id) -> pipeline quantity mapping."""
        return self.repository.get_pipeline_breakdown(store_id=store_id, sku_id=sku_id)

    # -------------------------------------------------------------------------
    # Shipment Tracking & Telemetry (Phase 5.5.2)
    # -------------------------------------------------------------------------

    def register_shipment_tracking(
        self,
        fo_id: str,
        shipment_id: str,
        carrier_id: str,
        *,
        scenario: Optional[TrackingScenario] = None,
        base_timestamp: Optional[str] = None,
        estimated_delivery_at: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingRecord:
        """Registers a dispatched shipment for tracking."""
        return self.tracking_service.register_shipment(
            fo_id=fo_id,
            shipment_id=shipment_id,
            carrier_id=carrier_id,
            scenario=scenario,
            base_timestamp=base_timestamp,
            estimated_delivery_at=estimated_delivery_at,
            **kwargs,
        )

    # Method alias
    register_shipment = register_shipment_tracking
    apply_shipment_dispatch = dispatch_shipment
    apply_goods_receipt = receive_goods
    apply_supplier_acknowledgment = acknowledge_order

    def ingest_tracking_event(
        self,
        event: ShipmentTrackingEvent,
        *,
        expected_version: Optional[int] = None,
    ) -> ShipmentTrackingRecord:
        """Ingests an inbound tracking event."""
        return self.tracking_service.ingest_tracking_event(
            event=event,
            expected_version=expected_version,
        )

    def get_shipment_tracking(self, shipment_id: str) -> ShipmentTrackingResult:
        """Retrieves tracking projection and event history."""
        return self.tracking_service.get_tracking(shipment_id)

    def get_tracking_history(self, shipment_id: str) -> Tuple[ShipmentTrackingEvent, ...]:
        """Retrieves chronological tracking event history."""
        return self.tracking_service.get_tracking_history(shipment_id)

    def get_shipments_for_order(self, fo_id: str) -> List[ShipmentTrackingRecord]:
        """Retrieves all tracking records associated with a given FulfilmentOrder."""
        return self.tracking_service.get_shipments_for_order(fo_id)

    def advance_shipment_tracking(
        self,
        shipment_id: str,
        steps: int = 1,
    ) -> Optional[ShipmentTrackingRecord]:
        """Advances simulated tracking scenario by steps."""
        return self.tracking_service.advance_shipment_tracking(shipment_id, steps=steps)

    def advance_to_tracking_status(
        self,
        shipment_id: str,
        target_status: ShipmentTrackingStatus,
    ) -> ShipmentTrackingRecord:
        """Advances simulated tracking scenario until target_status is reached."""
        return self.tracking_service.advance_to_tracking_status(shipment_id, target_status=target_status)
