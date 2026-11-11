"""
Lead 2 Fulfilment Domain Contracts & Pydantic V2 Schemas
=======================================================
Defines validated, immutable data schemas for supplier-side purchase order fulfilment:
- FulfilmentOrderStatus, SupplierAcknowledgmentStatus, ShipmentStatus, etc.
- FulfilmentOrderLine (tracking ordered, accepted, declined, unconfirmed, shipped,
  delivered, accepted, rejected, outstanding, and remaining quantities)
- FulfilmentOrder (decoupled lifecycle wrapper referencing the immutable approved PurchaseOrder)
- SupplierAcknowledgment & SupplierAcknowledgmentLine
- AdvanceShipmentNotice & ShipmentLineItem
- GoodsReceiptNote & DockInspectionLine
- FulfilmentBatch (extending and linking InventoryBatch with fulfilment provenance)
- FulfilmentEvent (append-only immutable audit trail)

ARCHITECTURAL PRINCIPLES:
1. IMMUTABLE COMMERCIAL ANCHOR:
   The original PurchaseOrder is the commercial contract of record. It remains strictly
   in APPROVED status throughout fulfilment and is NEVER mutated by supplier responses
   or receiving events.
2. DECOUPLED FULFILMENT LIFECYCLE:
   FulfilmentOrder tracks the operational execution (transmission, acknowledgment, dispatch,
   receipt, discrepancies, exceptions) independently of the commercial PurchaseOrder.
3. INBOUND PIPELINE FIDELITY:
   Only supplier-confirmed outstanding units (supplier_accepted - delivered) count as on_order_stock.
   Unacknowledged, supplier-declined, cancelled, or damaged quantities NEVER count as sellable inventory.
4. QUALITY HOLD & DEFECT TRIAGE:
   Perishable deliveries missing shelf-life metrics, arriving below minimum shelf life, or damaged
   in transit are rejected at the dock and excluded from sellable stock. Damaged goods exceeding
   ₹5,000 require mandatory supervisor sign-off.
5. STRICT UTC TIMESTAMPS:
   All timestamps are validated and normalized to UTC ending in 'Z'. Naive datetimes are forbidden.
"""

import hashlib
import json
import math
from datetime import date as dt_date, datetime as dt_datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..common.config import (
    CANONICAL_CATEGORIES,
    CANONICAL_SKUS,
    CANONICAL_STORES,
    SKU_TO_CATEGORY,
)
from ..common.schemas import InventoryBatch
from ..procurement.approval import verify_approval_evidence
from ..procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    SupplierType,
    validate_and_normalize_iso_timestamp,
)


class FulfilmentOrderStatus(str, Enum):
    """
    Operational lifecycle status of a Fulfilment Order.
    Kept strictly separate from PurchaseOrderStatus.
    """
    READY_TO_TRANSMIT = "READY_TO_TRANSMIT"
    ISSUED = "ISSUED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_ACKNOWLEDGED = "PARTIALLY_ACKNOWLEDGED"
    SUPPLIER_CHANGE_PENDING_REVIEW = "SUPPLIER_CHANGE_PENDING_REVIEW"
    REJECTED_BY_SUPPLIER = "REJECTED_BY_SUPPLIER"
    IN_TRANSIT = "IN_TRANSIT"
    PARTIALLY_RECEIVED = "PARTIALLY_RECEIVED"
    RECEIVED = "RECEIVED"
    DELIVERY_EXCEPTION = "DELIVERY_EXCEPTION"
    CLOSED_SHORT = "CLOSED_SHORT"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


class SupplierAcknowledgmentStatus(str, Enum):
    """Classification of a supplier's response to an issued purchase order."""
    ACCEPTED = "ACCEPTED"
    PARTIALLY_ACCEPTED = "PARTIALLY_ACCEPTED"
    REJECTED = "REJECTED"
    CHANGE_REQUESTED = "CHANGE_REQUESTED"


class ShipmentStatus(str, Enum):
    """Operational lifecycle tracking status of an in-transit supplier shipment."""
    LABEL_CREATED = "LABEL_CREATED"
    PICKED_UP = "PICKED_UP"
    DISPATCHED = "DISPATCHED"
    IN_TRANSIT = "IN_TRANSIT"
    AT_HUB = "AT_HUB"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    DELIVERY_EXCEPTION = "DELIVERY_EXCEPTION"
    RETURN_INITIATED = "RETURN_INITIATED"
    RETURNED = "RETURNED"
    CANCELLED = "CANCELLED"


class DiscrepancyReason(str, Enum):
    """Classification of delivery and quality discrepancies discovered at the receiving dock."""
    DAMAGED_TRANSIT = "DAMAGED_TRANSIT"
    EXPIRED_OR_SHORT_SHELF_LIFE = "EXPIRED_OR_SHORT_SHELF_LIFE"
    TEMPERATURE_ABUSE = "TEMPERATURE_ABUSE"
    MISSING_OR_SHORT = "MISSING_OR_SHORT"
    EXCESS_DELIVERY = "EXCESS_DELIVERY"
    WRONG_ITEM = "WRONG_ITEM"
    QUALITY_DEFECT = "QUALITY_DEFECT"


class DiscrepancyResolutionType(str, Enum):
    """Classification of resolution actions for received discrepancies or rejected goods."""
    RETURN_TO_ORIGIN = "RETURN_TO_ORIGIN"
    CERTIFIED_DISPOSAL = "CERTIFIED_DISPOSAL"
    SUPPLIER_CREDIT_NOTE = "SUPPLIER_CREDIT_NOTE"
    REINSPECTION_RELEASE = "REINSPECTION_RELEASE"
    EXCESS_RETURN = "EXCESS_RETURN"
    SHORT_DELIVERY_ACKNOWLEDGED = "SHORT_DELIVERY_ACKNOWLEDGED"


class FulfilmentEventType(str, Enum):
    """Classification of immutable events recorded in the fulfilment audit history."""
    CREATE_FULFILMENT_ORDER = "CREATE_FULFILMENT_ORDER"
    TRANSMIT_TO_SUPPLIER = "TRANSMIT_TO_SUPPLIER"
    SUPPLIER_ACKNOWLEDGE = "SUPPLIER_ACKNOWLEDGE"
    SUPPLIER_REJECT = "SUPPLIER_REJECT"
    SUPPLIER_CHANGE_REQUEST = "SUPPLIER_CHANGE_REQUEST"
    SHIPMENT_DISPATCH = "SHIPMENT_DISPATCH"
    GOODS_RECEIPT = "GOODS_RECEIPT"
    DELIVERY_EXCEPTION = "DELIVERY_EXCEPTION"
    RESOLVE_EXCEPTION = "RESOLVE_EXCEPTION"
    CLOSE_SHORT = "CLOSE_SHORT"
    REQUEST_CANCEL = "REQUEST_CANCEL"
    CONFIRM_CANCEL = "CONFIRM_CANCEL"
    REJECT_CANCEL = "REJECT_CANCEL"
    RESOLVE_DISCREPANCY = "RESOLVE_DISCREPANCY"
    RESOLVE_QUALITY_HOLD = "RESOLVE_QUALITY_HOLD"
    RESOLVE_SHIPMENT_CUSTODY = "RESOLVE_SHIPMENT_CUSTODY"


class FulfilmentActorType(str, Enum):
    """Classification of entities initiating fulfilment state transitions."""
    SYSTEM = "SYSTEM"
    PROCUREMENT_AGENT = "PROCUREMENT_AGENT"
    FULFILMENT_AGENT = "FULFILMENT_AGENT"
    SUPPLIER_GATEWAY = "SUPPLIER_GATEWAY"
    STORE_RECEIVER = "STORE_RECEIVER"
    STORE_MANAGER = "STORE_MANAGER"
    OPERATIONS_SUPERVISOR = "OPERATIONS_SUPERVISOR"


class CancellationReasonCategory(str, Enum):
    """Categorization of purchase order cancellation rationales."""
    SUPPLIER_UNABLE_TO_FULFIL = "SUPPLIER_UNABLE_TO_FULFIL"
    STORE_DEMAND_CHANGED = "STORE_DEMAND_CHANGED"
    COMMERCIAL_TERMS_REJECTED = "COMMERCIAL_TERMS_REJECTED"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    OPERATIONAL_MISTAKE = "OPERATIONAL_MISTAKE"


def _validate_non_empty_str(v: str, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


def validate_timestamp_sequence(t_earlier: str, t_later: str, field_earlier: str, field_later: str) -> None:
    """
    Validates that two normalized ISO 8601 UTC timestamps satisfy t_later >= t_earlier.
    """
    dt_earlier = dt_datetime.fromisoformat(t_earlier.replace("Z", "+00:00"))
    dt_later = dt_datetime.fromisoformat(t_later.replace("Z", "+00:00"))
    if dt_later < dt_earlier:
        raise ValueError(
            f"Temporal ordering violation: '{field_later}' ({t_later}) cannot precede '{field_earlier}' ({t_earlier})."
        )


def compute_payload_fingerprint(payload: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint for a canonicalized payload dictionary.
    Keys are sorted, timestamps are normalized strings, and separators are standardized.
    Does NOT use Python's built-in hash().
    """
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


class FulfilmentOrderLine(BaseModel):
    """
    Line-item fulfilment tracking model.
    Maintains the immutable approved ordered quantity alongside live supplier-confirmed,
    shipped, delivered, accepted, and rejected balances.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sku_id: str = Field(..., description="Canonical product SKU identifier")
    category: str = Field(..., description="Product category (Bakery, Dairy, Meat, Produce)")
    ordered_quantity: float = Field(..., gt=0.0, description="Approved order quantity from commercial PO")
    unit_cost: float = Field(..., ge=0.0, description="Unit wholesale cost in INR from commercial PO")

    supplier_accepted_quantity: float = Field(default=0.0, ge=0.0, description="Units confirmed by supplier")
    supplier_declined_quantity: float = Field(default=0.0, ge=0.0, description="Units declined by supplier")
    supplier_unconfirmed_quantity: float = Field(default=0.0, ge=0.0, description="Units awaiting supplier confirmation")

    shipped_quantity: float = Field(default=0.0, ge=0.0, description="Units dispatched in transit by supplier")
    delivered_quantity: float = Field(default=0.0, ge=0.0, description="Units physically presented at receiving dock")
    accepted_quantity: float = Field(default=0.0, ge=0.0, description="Units passed dock inspection into sellable inventory")
    rejected_quantity: float = Field(default=0.0, ge=0.0, description="Units rejected at dock (damaged, excess, expired)")

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "category")
        if clean not in CANONICAL_CATEGORIES:
            raise ValueError(f"category '{clean}' must be one of {CANONICAL_CATEGORIES}")
        return clean

    @model_validator(mode="after")
    def validate_quantity_invariants(self) -> "FulfilmentOrderLine":
        # Finite number checks
        for q_name in [
            "ordered_quantity", "unit_cost", "supplier_accepted_quantity",
            "supplier_declined_quantity", "supplier_unconfirmed_quantity",
            "shipped_quantity", "delivered_quantity", "accepted_quantity", "rejected_quantity"
        ]:
            val = getattr(self, q_name)
            if not math.isfinite(val):
                raise ValueError(f"Quantity '{q_name}' must be finite; got {val}.")

        # 1. Supplier confirmation reconciliation
        confirmed_sum = self.supplier_accepted_quantity + self.supplier_declined_quantity + self.supplier_unconfirmed_quantity
        if abs(confirmed_sum - self.ordered_quantity) > 1e-4:
            raise ValueError(
                f"[{self.sku_id}] Confirmation mismatch: accepted ({self.supplier_accepted_quantity}) + "
                f"declined ({self.supplier_declined_quantity}) + unconfirmed ({self.supplier_unconfirmed_quantity}) "
                f"= {confirmed_sum}, expected ordered_quantity={self.ordered_quantity}."
            )

        # 2. Supplier bounds
        if self.supplier_accepted_quantity > self.ordered_quantity + 1e-4:
            raise ValueError(
                f"[{self.sku_id}] supplier_accepted_quantity ({self.supplier_accepted_quantity}) "
                f"cannot exceed ordered_quantity ({self.ordered_quantity})."
            )
        if self.supplier_declined_quantity > self.ordered_quantity + 1e-4:
            raise ValueError(
                f"[{self.sku_id}] supplier_declined_quantity ({self.supplier_declined_quantity}) "
                f"cannot exceed ordered_quantity ({self.ordered_quantity})."
            )

        # 3. Shipment bounds
        if self.shipped_quantity > self.supplier_accepted_quantity + 1e-4:
            raise ValueError(
                f"[{self.sku_id}] shipped_quantity ({self.shipped_quantity}) cannot exceed "
                f"supplier_accepted_quantity ({self.supplier_accepted_quantity})."
            )

        # 4. Inspection reconciliation (delivered = accepted + rejected)
        # Note: Physical delivered quantity may exceed ASN quantity or outstanding PO balance
        # to record physical over-delivery discrepancies at the receiving dock.
        inspected_sum = self.accepted_quantity + self.rejected_quantity
        if abs(inspected_sum - self.delivered_quantity) > 1e-4:
            raise ValueError(
                f"[{self.sku_id}] Inspection mismatch: accepted ({self.accepted_quantity}) + "
                f"rejected ({self.rejected_quantity}) = {inspected_sum}, expected delivered={self.delivered_quantity}."
            )

        # 6. Accepted cannot exceed ordered
        if self.accepted_quantity > self.ordered_quantity + 1e-4:
            raise ValueError(
                f"[{self.sku_id}] accepted_quantity ({self.accepted_quantity}) cannot exceed "
                f"ordered_quantity ({self.ordered_quantity})."
            )

        return self

    @property
    def outstanding_confirmed_quantity(self) -> float:
        """
        Units confirmed by supplier still awaiting physical delivery:
        max(0, supplier_accepted_quantity - delivered_quantity).
        """
        return max(0.0, round(self.supplier_accepted_quantity - self.delivered_quantity, 4))

    @property
    def remaining_unfulfilled_quantity(self) -> float:
        """
        Approved order units not yet accepted into sellable inventory:
        max(0, ordered_quantity - accepted_quantity).
        """
        return max(0.0, round(self.ordered_quantity - self.accepted_quantity, 4))


class SupplierAcknowledgmentLine(BaseModel):
    """Line-level quantity confirmation from supplier."""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sku_id: str = Field(..., description="Product SKU identifier")
    accepted_quantity: float = Field(..., ge=0.0, description="Quantity confirmed for fulfilment")
    declined_quantity: float = Field(default=0.0, ge=0.0, description="Quantity supplier cannot supply")
    notes: Optional[str] = Field(default=None, description="Line-level supplier comments")

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("notes")
    @classmethod
    def validate_ack_line_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("Field 'notes', if provided, cannot be empty or whitespace.")
        return clean


class SupplierAcknowledgment(BaseModel):
    """
    Validated supplier response to an issued purchase order.
    A change request or partial acknowledgment NEVER silently alters the approved commercial PO.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    acknowledgment_id: str = Field(..., description="Unique acknowledgment identifier")
    fo_id: str = Field(..., description="Target Fulfilment Order identifier")
    po_id: str = Field(..., description="Original Purchase Order identifier")
    supplier_id: str = Field(..., description="Responding supplier identifier")
    status: SupplierAcknowledgmentStatus = Field(..., description="Supplier response classification")
    acknowledged_at: str = Field(..., description="ISO 8601 UTC timestamp of acknowledgment")

    expected_delivery_at: Optional[str] = Field(
        default=None,
        description="Supplier committed delivery arrival timestamp (ISO 8601 UTC)"
    )
    lines: Tuple[SupplierAcknowledgmentLine, ...] = Field(
        default_factory=tuple,
        description="Line-item quantity acknowledgments"
    )
    rejection_reason: Optional[str] = Field(default=None, description="Mandatory rationale if order is rejected")
    change_request_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured proposed changes when status is CHANGE_REQUESTED"
    )
    supplier_reference: Optional[str] = Field(default=None, description="External vendor order reference")
    notes: Optional[str] = Field(default=None, description="Vendor remarks")

    @field_validator("acknowledgment_id", "fo_id", "po_id", "supplier_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("rejection_reason", "supplier_reference", "notes")
    @classmethod
    def validate_optional_ack_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            if info.field_name == "rejection_reason":
                raise ValueError("rejection_reason is mandatory and cannot be empty or whitespace.")
            raise ValueError(f"Field '{info.field_name}', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("acknowledged_at", "expected_delivery_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v: Any, info) -> Optional[str]:
        if v is None:
            return None
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

    @model_validator(mode="after")
    def validate_acknowledgment_rules(self) -> "SupplierAcknowledgment":
        # Check unique SKUs
        skus_seen = set()
        for line in self.lines:
            if line.sku_id in skus_seen:
                raise ValueError(f"Duplicate SKU '{line.sku_id}' in supplier acknowledgment.")
            skus_seen.add(line.sku_id)

        # Status specific checks
        if self.status == SupplierAcknowledgmentStatus.REJECTED:
            if not self.rejection_reason or not self.rejection_reason.strip():
                raise ValueError("rejection_reason is mandatory when status is REJECTED.")

        if self.status in [SupplierAcknowledgmentStatus.ACCEPTED, SupplierAcknowledgmentStatus.PARTIALLY_ACCEPTED]:
            if not self.expected_delivery_at:
                raise ValueError(f"expected_delivery_at is required when acknowledgment status is '{self.status.value}'.")
            validate_timestamp_sequence(self.acknowledged_at, self.expected_delivery_at, "acknowledged_at", "expected_delivery_at")

        if self.status == SupplierAcknowledgmentStatus.CHANGE_REQUESTED:
            if not self.change_request_details:
                raise ValueError("change_request_details is required when status is CHANGE_REQUESTED.")

        return self


class ShipmentLineItem(BaseModel):
    """Quantity dispatched per SKU within a shipment."""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sku_id: str = Field(..., description="Product SKU identifier")
    shipped_quantity: float = Field(..., gt=0.0, description="Quantity dispatched in this shipment")

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("shipped_quantity")
    @classmethod
    def validate_qty(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"shipped_quantity must be positive and finite; got {v}")
        return v


class AdvanceShipmentNotice(BaseModel):
    """
    Advance Shipment Notice (ASN) emitted when supplier dispatches an in-transit order.
    Supports split shipments against a single FulfilmentOrder.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    shipment_id: str = Field(..., description="Unique shipment identifier")
    fo_id: str = Field(..., description="Target Fulfilment Order identifier")
    po_id: str = Field(..., description="Original Purchase Order identifier")
    supplier_id: str = Field(..., description="Dispatching supplier identifier")
    dispatch_at: str = Field(..., description="ISO 8601 UTC dispatch timestamp")
    expected_arrival_at: str = Field(..., description="ISO 8601 UTC expected dock arrival timestamp")
    lines: Tuple[ShipmentLineItem, ...] = Field(..., min_length=1, description="Shipped line items")
    carrier: Optional[str] = Field(default=None, description="Logistics carrier name")
    tracking_reference: Optional[str] = Field(default=None, description="Waybill or tracking reference")
    notes: Optional[str] = Field(default=None, description="Shipment remarks")

    @field_validator("shipment_id", "fo_id", "po_id", "supplier_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("carrier", "tracking_reference", "notes")
    @classmethod
    def validate_optional_asn_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError(f"Field '{info.field_name}', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("dispatch_at", "expected_arrival_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v: Any, info) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

    @model_validator(mode="after")
    def validate_shipment_integrity(self) -> "AdvanceShipmentNotice":
        # Unique SKUs
        skus_seen = set()
        for item in self.lines:
            if item.sku_id in skus_seen:
                raise ValueError(f"Duplicate SKU '{item.sku_id}' in AdvanceShipmentNotice lines.")
            skus_seen.add(item.sku_id)

        # Arrival >= Dispatch
        validate_timestamp_sequence(self.dispatch_at, self.expected_arrival_at, "dispatch_at", "expected_arrival_at")
        return self


class DockInspectionLine(BaseModel):
    """
    Physical inspection and quality triage recorded at the store receiving dock for a single SKU.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sku_id: str = Field(..., description="Product SKU identifier")
    delivered_quantity: float = Field(..., ge=0.0, description="Total physical units delivered to dock")
    accepted_quantity: float = Field(..., ge=0.0, description="Physical units passing quality triage into sellable stock")
    rejected_quantity: float = Field(..., ge=0.0, description="Units rejected at dock (damaged, excess, expired)")
    unit_cost: float = Field(..., ge=0.0, description="Unit cost in INR for financial discrepancy assessment")

    damaged_quantity: float = Field(default=0.0, ge=0.0, description="Units damaged in transit or cold-chain failure")
    excess_quantity: float = Field(default=0.0, ge=0.0, description="Units delivered above remaining order balance")
    short_shelf_life_quantity: float = Field(default=0.0, ge=0.0, description="Units arriving below acceptable shelf-life threshold")

    remaining_shelf_life_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Remaining shelf-life days verified on delivered batch"
    )
    total_shelf_life_days: Optional[int] = Field(
        default=None,
        gt=0,
        description="Total factory shelf-life days"
    )
    discrepancy_reasons: Tuple[DiscrepancyReason, ...] = Field(
        default_factory=tuple,
        description="Classification of any rejection or delivery issues"
    )
    batch_references: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identifiers of batches minted for accepted stock"
    )
    notes: Optional[str] = Field(default=None, description="Dock inspector remarks")

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("notes")
    @classmethod
    def validate_dock_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("Field 'notes', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("batch_references")
    @classmethod
    def validate_batch_refs(cls, v: Tuple[str, ...]) -> Tuple[str, ...]:
        clean_refs = []
        for ref in v:
            clean = _validate_non_empty_str(ref, "batch_reference")
            clean_refs.append(clean)
        return tuple(clean_refs)

    @model_validator(mode="after")
    def validate_dock_line_integrity(self) -> "DockInspectionLine":
        # Reconciliation: delivered = accepted + rejected
        inspected_sum = self.accepted_quantity + self.rejected_quantity
        if abs(inspected_sum - self.delivered_quantity) > 1e-4:
            raise ValueError(
                f"[{self.sku_id}] Dock inspection quantity mismatch: accepted ({self.accepted_quantity}) + "
                f"rejected ({self.rejected_quantity}) = {inspected_sum}, expected delivered={self.delivered_quantity}."
            )

        # Rejection components check
        rejection_breakdown = self.damaged_quantity + self.excess_quantity + self.short_shelf_life_quantity
        if rejection_breakdown > self.rejected_quantity + 1e-4:
            raise ValueError(
                f"[{self.sku_id}] Rejection breakdown ({rejection_breakdown}) exceeds "
                f"total rejected_quantity ({self.rejected_quantity})."
            )

        # Accepted stock must have batch references
        if self.accepted_quantity > 0 and len(self.batch_references) == 0:
            raise ValueError(
                f"[{self.sku_id}] Accepted quantity ({self.accepted_quantity}) must be backed "
                "by at least one batch reference in batch_references."
            )

        # Perishable quality hold check: missing shelf life cannot be accepted
        if self.accepted_quantity > 0 and self.remaining_shelf_life_days is None:
            raise ValueError(
                f"[{self.sku_id}] Quality Hold Violation: Perishable stock cannot be accepted "
                "without verified remaining_shelf_life_days at dock receiving."
            )

        # Quality hold check: remaining shelf-life ratio must meet minimum threshold (60%)
        if (
            self.accepted_quantity > 0
            and self.remaining_shelf_life_days is not None
            and self.total_shelf_life_days is not None
        ):
            ratio = self.remaining_shelf_life_days / self.total_shelf_life_days
            if ratio < 0.60 - 1e-4:
                raise ValueError(
                    f"[{self.sku_id}] Quality Hold Violation: Remaining shelf-life ratio {ratio:.1%} "
                    f"({self.remaining_shelf_life_days}/{self.total_shelf_life_days} days) is below the "
                    "minimum acceptable threshold of 60.0%. Stock cannot be accepted into sellable inventory."
                )

        return self

    @property
    def rejected_line_cost(self) -> float:
        """Total INR cost of rejected units."""
        return round(self.rejected_quantity * self.unit_cost, 2)

    @property
    def damaged_line_cost(self) -> float:
        """Total INR cost of damaged units."""
        return round(self.damaged_quantity * self.unit_cost, 2)


class GoodsReceiptNote(BaseModel):
    """
    Formal Goods Receipt Note (GRN) generated at dark-store receiving dock.
    Documents physical check-in, discrepancy triage, and supervisor sign-offs.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    grn_id: str = Field(..., description="Unique Goods Receipt Note identifier")
    fo_id: str = Field(..., description="Target Fulfilment Order identifier")
    po_id: str = Field(..., description="Original Purchase Order identifier")
    shipment_id: str = Field(..., description="Associated shipment identifier")
    store_id: str = Field(..., description="Destination dark-store identifier")
    supplier_id: str = Field(..., description="Delivering supplier identifier")
    receipt_at: str = Field(..., description="ISO 8601 UTC receipt timestamp")
    receiver_id: str = Field(..., description="Identifier of receiver conducting dock triage")
    inspection_lines: Tuple[DockInspectionLine, ...] = Field(..., min_length=1, description="Line-level dock inspection results")

    supervisor_signoff_by: Optional[str] = Field(
        default=None,
        description="Supervisor identifier required when damage/rejection exceeds threshold"
    )
    supervisor_signoff_at: Optional[str] = Field(
        default=None,
        description="Supervisor signoff timestamp (ISO 8601 UTC)"
    )
    max_dock_discrepancy_cost_inr: float = Field(
        default=5000.0,
        ge=0.0,
        description="Configurable threshold; line discrepancy strictly exceeding this triggers supervisor sign-off"
    )
    rejected_goods_disposition: Optional[str] = Field(
        default=None,
        description="Documented final disposition (e.g. RTO-xxx, DISP-xxx) for rejected consignments before custody is resolved"
    )
    notes: Optional[str] = Field(default=None, description="Receiving remarks")

    @field_validator("rejected_goods_disposition")
    @classmethod
    def validate_disposition(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("rejected_goods_disposition, if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("supervisor_signoff_by", "notes")
    @classmethod
    def validate_optional_grn_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError(f"Field '{info.field_name}', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("grn_id", "fo_id", "po_id", "shipment_id", "store_id", "supplier_id", "receiver_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("receipt_at", "supervisor_signoff_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v: Any, info) -> Optional[str]:
        if v is None:
            return None
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

    @model_validator(mode="after")
    def validate_grn_integrity(self) -> "GoodsReceiptNote":
        # Check canonical store
        if self.store_id not in CANONICAL_STORES:
            raise ValueError(f"store_id '{self.store_id}' must be one of {CANONICAL_STORES}")

        # Unique SKUs
        skus_seen = set()
        total_damaged_cost = 0.0
        total_rejected_cost = 0.0

        for line in self.inspection_lines:
            if line.sku_id in skus_seen:
                raise ValueError(f"Duplicate SKU '{line.sku_id}' in GoodsReceiptNote inspection lines.")
            skus_seen.add(line.sku_id)
            total_damaged_cost += line.damaged_line_cost
            total_rejected_cost += line.rejected_line_cost

        # Supervisor escalation rule: strictly greater than max_dock_discrepancy_cost_inr
        requires_signoff = False
        for line in self.inspection_lines:
            if line.damaged_line_cost > self.max_dock_discrepancy_cost_inr or line.rejected_line_cost > self.max_dock_discrepancy_cost_inr:
                requires_signoff = True
                break

        if requires_signoff:
            if not self.supervisor_signoff_by or not self.supervisor_signoff_by.strip():
                raise ValueError(
                    f"Governance Violation: GRN '{self.grn_id}' contains damaged/rejected lines exceeding "
                    f"₹{self.max_dock_discrepancy_cost_inr:,.2f} threshold. Mandatory supervisor sign-off ('supervisor_signoff_by') is required."
                )
            if not self.supervisor_signoff_at:
                raise ValueError(
                    f"Governance Violation: GRN '{self.grn_id}' requires 'supervisor_signoff_at' timestamp."
                )
            validate_timestamp_sequence(self.receipt_at, self.supervisor_signoff_at, "receipt_at", "supervisor_signoff_at")
        elif self.supervisor_signoff_by:
            if not self.supervisor_signoff_at:
                raise ValueError(
                    f"Governance Violation: GRN '{self.grn_id}' with supervisor sign-off requires 'supervisor_signoff_at' timestamp."
                )
            validate_timestamp_sequence(self.receipt_at, self.supervisor_signoff_at, "receipt_at", "supervisor_signoff_at")

        return self

    @property
    def total_delivered_quantity(self) -> float:
        return round(sum(line.delivered_quantity for line in self.inspection_lines), 4)

    @property
    def total_accepted_quantity(self) -> float:
        return round(sum(line.accepted_quantity for line in self.inspection_lines), 4)

    @property
    def total_rejected_quantity(self) -> float:
        return round(sum(line.rejected_quantity for line in self.inspection_lines), 4)

    @property
    def total_accepted_cost(self) -> float:
        return round(sum(line.accepted_quantity * line.unit_cost for line in self.inspection_lines), 2)

    @property
    def total_rejected_cost(self) -> float:
        return round(sum(line.rejected_line_cost for line in self.inspection_lines), 2)


class FulfilmentBatch(BaseModel):
    """
    Physical perishable inventory batch minted at goods receipt.
    Extends core InventoryBatch with fulfilment provenance while providing
    lossless conversion to InventoryBatch for inventory engine integration.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    batch_id: str = Field(..., description="Unique batch identifier (e.g. BATCH-{STORE}-{SKU}-{DATE}-{SEQ})")
    sku_id: str = Field(..., description="Product SKU identifier")
    units: float = Field(..., gt=0.0, description="Physical units accepted in this batch")
    remaining_shelf_life_days: int = Field(..., ge=0, description="Verified remaining shelf-life days at receipt")
    receipt_date: str = Field(..., description="Intake date in YYYY-MM-DD format")
    fo_id: str = Field(..., description="Originating Fulfilment Order identifier")
    grn_id: str = Field(..., description="Originating Goods Receipt Note identifier")
    supplier_lot_ref: Optional[str] = Field(default=None, description="External vendor lot reference")

    @field_validator("batch_id", "fo_id", "grn_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("supplier_lot_ref")
    @classmethod
    def validate_lot_ref(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("Field 'supplier_lot_ref', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("receipt_date")
    @classmethod
    def validate_receipt_date_format(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "receipt_date")
        try:
            dt_date.fromisoformat(clean)
        except ValueError as err:
            raise ValueError(f"receipt_date '{clean}' must be in YYYY-MM-DD format: {err}") from err
        if len(clean) != 10 or clean[4] != '-' or clean[7] != '-':
            raise ValueError(f"receipt_date '{clean}' must be in YYYY-MM-DD format.")
        return clean

    def to_inventory_batch(self) -> InventoryBatch:
        """
        Converts to core InventoryBatch representation used by Inventory Engine.
        Preserves backward compatibility without duplicating logic.
        """
        return InventoryBatch(
            batch_id=self.batch_id,
            units=self.units,
            remaining_shelf_life_days=self.remaining_shelf_life_days,
            creation_date=self.receipt_date,
        )


def generate_grn_fulfilment_batches(
    grn: GoodsReceiptNote,
) -> Tuple[FulfilmentBatch, ...]:
    """
    Constructs immutable FulfilmentBatch records for all accepted lines in a GoodsReceiptNote.
    Mints a batch for each reference in batch_references, dividing accepted units equally
    if multiple batches exist for a line.
    """
    batches: List[FulfilmentBatch] = []
    receipt_date = grn.receipt_at[:10]  # YYYY-MM-DD
    for line in grn.inspection_lines:
        if line.accepted_quantity > 0 and line.batch_references:
            units_per_batch = round(line.accepted_quantity / len(line.batch_references), 4)
            for ref in line.batch_references:
                batch = FulfilmentBatch(
                    batch_id=ref,
                    sku_id=line.sku_id,
                    units=units_per_batch,
                    remaining_shelf_life_days=line.remaining_shelf_life_days if line.remaining_shelf_life_days is not None else 0,
                    receipt_date=receipt_date,
                    fo_id=grn.fo_id,
                    grn_id=grn.grn_id,
                )
                batches.append(batch)
    return tuple(batches)


class DiscrepancyResolution(BaseModel):
    """
    Immutable record of an authorized discrepancy or rejected goods resolution.
    Links rejected physical quantities to physical return, disposal, credit note, or release evidence.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    resolution_id: str = Field(..., description="Unique resolution identifier (e.g. RES-FO-001-001)")
    fo_id: str = Field(..., description="Target Fulfilment Order identifier")
    shipment_id: str = Field(..., description="Associated shipment identifier")
    sku_id: str = Field(..., description="Target SKU identifier")
    resolution_type: DiscrepancyResolutionType = Field(..., description="Type of resolution action")
    quantity: float = Field(..., gt=0.0, description="Quantity resolved by this action")
    discrepancy_reason: DiscrepancyReason = Field(..., description="Original discrepancy classification")
    disposition_reference: str = Field(..., description="Physical or commercial disposition reference (e.g. RTO-xxx, DISP-xxx, CRN-xxx)")
    actor_id: str = Field(..., description="Authorized agent or manager initiating resolution")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of resolution execution")
    notes: Optional[str] = Field(default=None, description="Operational notes or disposition evidence")

    @field_validator("resolution_id", "fo_id", "shipment_id", "disposition_reference", "actor_id")
    @classmethod
    def validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("notes")
    @classmethod
    def validate_optional_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("Field 'notes', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name="timestamp", error_cls=ValueError)


class QualityHoldResolution(BaseModel):
    """
    Immutable record of an authorized quality hold resolution or reinspection disposition.
    Documents the outcome of quality review (release after reinspection, certified disposal, or return).
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    resolution_id: str = Field(..., description="Unique quality hold resolution identifier (e.g. QHR-FO-001-001)")
    fo_id: str = Field(..., description="Target Fulfilment Order identifier")
    shipment_id: str = Field(..., description="Associated shipment identifier")
    sku_id: str = Field(..., description="Target SKU identifier")
    resolution_type: DiscrepancyResolutionType = Field(..., description="Resolution action (e.g. REINSPECTION_RELEASE, CERTIFIED_DISPOSAL, RETURN_TO_ORIGIN)")
    quantity: float = Field(..., gt=0.0, description="Quantity resolved by this action")
    disposition_reference: str = Field(..., description="Physical reference or reinspection certificate ID")
    inspector_id: str = Field(..., description="Authorized quality inspector or QA lead ID")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of resolution execution")
    verified_remaining_shelf_life_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Verified remaining shelf life days from secondary inspection"
    )
    total_shelf_life_days: Optional[int] = Field(
        default=None,
        gt=0,
        description="Total factory shelf-life days"
    )
    inspection_notes: str = Field(..., description="Detailed reinspection findings and disposition justification")

    @field_validator("resolution_id", "fo_id", "shipment_id", "disposition_reference", "inspector_id", "inspection_notes")
    @classmethod
    def validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name="timestamp", error_cls=ValueError)

    @model_validator(mode="after")
    def validate_quality_hold_integrity(self) -> "QualityHoldResolution":
        if self.resolution_type == DiscrepancyResolutionType.REINSPECTION_RELEASE:
            if self.verified_remaining_shelf_life_days is None:
                raise ValueError(
                    f"[{self.sku_id}] Quality Hold Violation: Reinspection release requires "
                    "verified_remaining_shelf_life_days."
                )
            if self.total_shelf_life_days is None or self.total_shelf_life_days <= 0:
                raise ValueError(
                    f"[{self.sku_id}] Quality Hold Violation: Reinspection release requires a strictly "
                    f"positive total_shelf_life_days; got {self.total_shelf_life_days}."
                )
            if self.verified_remaining_shelf_life_days < 0:
                raise ValueError(
                    f"[{self.sku_id}] Quality Hold Violation: verified_remaining_shelf_life_days cannot be negative."
                )
            if self.verified_remaining_shelf_life_days > self.total_shelf_life_days:
                raise ValueError(
                    f"[{self.sku_id}] Quality Hold Violation: verified_remaining_shelf_life_days "
                    f"({self.verified_remaining_shelf_life_days}) cannot exceed total_shelf_life_days "
                    f"({self.total_shelf_life_days})."
                )
            ratio = self.verified_remaining_shelf_life_days / self.total_shelf_life_days
            if ratio < 0.60 - 1e-4:
                raise ValueError(
                    f"[{self.sku_id}] Quality Hold Violation: Verified remaining shelf life ratio {ratio:.1%} "
                    f"({self.verified_remaining_shelf_life_days}/{self.total_shelf_life_days} days) is below "
                    "the minimum acceptable threshold of 60.0%. Stock cannot be released into sellable inventory."
                )
        return self


class LineReceiptReconciliation(BaseModel):
    """
    Cumulative receipt balances for a specific SKU line across all receipts.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sku_id: str = Field(..., description="Target SKU identifier")
    ordered_quantity: float = Field(..., ge=0.0, description="Approved ordered quantity from PurchaseOrder")
    supplier_accepted_quantity: float = Field(..., ge=0.0, description="Supplier-confirmed quantity from acknowledgment")
    supplier_declined_quantity: float = Field(..., ge=0.0, description="Supplier-declined quantity")
    supplier_unconfirmed_quantity: float = Field(..., ge=0.0, description="Unconfirmed quantity")
    shipped_quantity: float = Field(..., ge=0.0, description="Quantity dispatched on advance shipment notices")
    delivered_quantity: float = Field(..., ge=0.0, description="Total physical units delivered across all GRNs")
    accepted_quantity: float = Field(..., ge=0.0, description="Physical units accepted into sellable inventory")
    rejected_quantity: float = Field(..., ge=0.0, description="Physical units rejected at receiving dock")
    damaged_quantity: float = Field(..., ge=0.0, description="Physical units damaged")
    excess_quantity: float = Field(..., ge=0.0, description="Physical units delivered in excess of order balance")
    short_shelf_life_quantity: float = Field(..., ge=0.0, description="Physical units rejected due to short shelf life")
    remaining_confirmed_outstanding_quantity: float = Field(..., ge=0.0, description="Confirmed pipeline units yet to be received")
    remaining_unfulfilled_quantity: float = Field(..., ge=0.0, description="Ordered units yet to be accepted into stock")
    is_fully_accepted: bool = Field(..., description="Whether accepted_quantity >= ordered_quantity")

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean


class ShipmentReceiptReconciliation(BaseModel):
    """
    Cumulative receipt and custody reconciliation for a specific shipment across all GRNs.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    shipment_id: str = Field(..., description="Shipment identifier")
    fo_id: str = Field(..., description="Associated Fulfilment Order identifier")
    po_id: str = Field(..., description="Associated Purchase Order identifier")
    grn_ids: Tuple[str, ...] = Field(default_factory=tuple, description="GRNs referencing this shipment")
    shipped_quantity: float = Field(default=0.0, ge=0.0, description="Total physical units shipped on this shipment")
    delivered_quantity: float = Field(..., ge=0.0, description="Total physical units delivered on this shipment")
    accepted_quantity: float = Field(..., ge=0.0, description="Total physical units accepted on this shipment")
    rejected_quantity: float = Field(..., ge=0.0, description="Total physical units rejected on this shipment")
    damaged_quantity: float = Field(..., ge=0.0, description="Total physical units damaged on this shipment")
    excess_quantity: float = Field(..., ge=0.0, description="Total physical units excess on this shipment")
    short_shelf_life_quantity: float = Field(..., ge=0.0, description="Total units short shelf life on this shipment")
    lines: Tuple[LineReceiptReconciliation, ...] = Field(..., description="Per-SKU reconciliation for this shipment")
    is_custody_resolved: bool = Field(..., description="Whether custody of rejected goods or complete delivery is verified")
    disposition_reference: Optional[str] = Field(default=None, description="Documented resolution evidence if resolved")

    @field_validator("shipment_id", "fo_id", "po_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("disposition_reference")
    @classmethod
    def validate_optional_ref(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("disposition_reference, if provided, cannot be empty or whitespace.")
        return clean


class OrderReceiptReconciliation(BaseModel):
    """
    Authoritative order-wide cumulative reconciliation across all lines, shipments, and GRNs.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    fo_id: str = Field(..., description="Fulfilment Order identifier")
    po_id: str = Field(..., description="Purchase Order identifier")
    status: FulfilmentOrderStatus = Field(..., description="Current Fulfilment Order status")
    total_ordered_quantity: float = Field(..., ge=0.0, description="Total approved ordered units")
    total_supplier_confirmed_quantity: float = Field(..., ge=0.0, description="Total supplier accepted units")
    total_supplier_declined_quantity: float = Field(..., ge=0.0, description="Total supplier declined units")
    total_supplier_unconfirmed_quantity: float = Field(..., ge=0.0, description="Total supplier unconfirmed units")
    total_shipped_quantity: float = Field(..., ge=0.0, description="Total units dispatched across all ASNs")
    total_delivered_quantity: float = Field(..., ge=0.0, description="Total physical units delivered across all GRNs")
    total_accepted_quantity: float = Field(..., ge=0.0, description="Total physical units accepted into sellable inventory")
    total_rejected_quantity: float = Field(..., ge=0.0, description="Total physical units rejected at dock")
    total_damaged_quantity: float = Field(..., ge=0.0, description="Total physical units damaged")
    total_excess_quantity: float = Field(..., ge=0.0, description="Total physical units delivered as excess")
    total_short_shelf_life_quantity: float = Field(..., ge=0.0, description="Total physical units short shelf-life")
    total_remaining_confirmed_outstanding_quantity: float = Field(..., ge=0.0, description="Active supplier-confirmed pipeline balance")
    total_remaining_unfulfilled_quantity: float = Field(..., ge=0.0, description="Approved balance unfulfilled")
    lines: Tuple[LineReceiptReconciliation, ...] = Field(..., description="Per-line cumulative reconciliation")
    shipments: Tuple[ShipmentReceiptReconciliation, ...] = Field(..., description="Per-shipment cumulative reconciliation")
    is_fully_received: bool = Field(..., description="Whether all lines are fully accepted (accepted >= ordered)")
    can_close_short: bool = Field(..., description="Whether order is eligible for explicit CLOSED_SHORT")

    @field_validator("fo_id", "po_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)


class FulfilmentEvent(BaseModel):
    """
    Immutable audit record capturing an explicit fulfilment lifecycle event.
    Provides non-repudiation, causality tracking, and idempotency protection.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    event_id: str = Field(..., description="Unique event identifier (e.g. EVT-FO-001-001)")
    idempotency_key: str = Field(..., description="Unique idempotency token preventing duplicate actions")
    fo_id: str = Field(..., description="Target Fulfilment Order identifier")
    event_type: FulfilmentEventType = Field(..., description="Classification of fulfilment event")
    actor_type: FulfilmentActorType = Field(..., description="Classification of acting principal")
    actor_id: str = Field(..., description="Identifier of acting agent or operator")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of event execution")

    previous_status: Optional[FulfilmentOrderStatus] = Field(
        default=None,
        description="Fulfilment status prior to event"
    )
    new_status: FulfilmentOrderStatus = Field(..., description="Fulfilment status resulting from event")
    sequence_number: int = Field(..., ge=1, description="Strictly incrementing 1-based event index")
    related_reference_id: Optional[str] = Field(
        default=None,
        description="Associated PO, Acknowledgment, Shipment, or GRN identifier"
    )
    payload_fingerprint: Optional[str] = Field(
        default=None,
        description="Deterministic SHA-256 fingerprint of canonical payload for strong idempotency verification"
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured contextual event metadata")

    @field_validator("event_id", "idempotency_key", "fo_id", "actor_id")
    @classmethod
    def validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("related_reference_id", "payload_fingerprint")
    @classmethod
    def validate_optional_event_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError(f"Field '{info.field_name}', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_event_timestamp(cls, v: Any) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name="timestamp", error_cls=ValueError)


class FulfilmentOrder(BaseModel):
    """
    Operational fulfilment aggregate root.
    References the immutable approved PurchaseOrder while managing supplier transmission,
    acknowledgment, shipment tracking, goods receipt, discrepancy triage, and audit events.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    fo_id: str = Field(..., description="Unique Fulfilment Order identifier")
    po_id: str = Field(..., description="Original Purchase Order identifier")
    po_snapshot: PurchaseOrder = Field(..., description="Immutable approved PurchaseOrder source of truth")
    store_id: str = Field(..., description="Destination Bangalore dark-store identifier")
    supplier_id: str = Field(..., description="Target supplier identifier")
    supplier_type: SupplierType = Field(..., description="PRIMARY or BACKUP supplier classification")

    status: FulfilmentOrderStatus = Field(
        default=FulfilmentOrderStatus.READY_TO_TRANSMIT,
        description="Operational fulfilment status"
    )
    lines: Tuple[FulfilmentOrderLine, ...] = Field(
        ...,
        min_length=1,
        description="Line-item fulfilment tracking balances"
    )

    created_at: str = Field(..., description="ISO 8601 UTC creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of latest state update")
    expected_delivery_at: Optional[str] = Field(
        default=None,
        description="Supplier-confirmed expected delivery arrival timestamp"
    )

    shipment_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Associated ASN shipment identifiers")
    resolved_shipment_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Shipment identifiers with verified dock receipt, return disposition, or cancellation"
    )
    shipment_resolution_evidence: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of shipment ID to verified resolution or return disposition reference"
    )
    order_wide_cancellation_scope: bool = Field(
        default=False,
        description="Flag indicating whether shipment_cancellation_ref explicitly covers all shipments on the order"
    )
    grn_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Associated Goods Receipt Note identifiers")
    discrepancy_resolutions: Tuple[DiscrepancyResolution, ...] = Field(
        default_factory=tuple,
        description="Recorded resolutions for damaged, rejected, or excess quantities"
    )
    quality_hold_resolutions: Tuple[QualityHoldResolution, ...] = Field(
        default_factory=tuple,
        description="Recorded resolutions for quality holds and reinspections"
    )

    cancellation_reason: Optional[str] = Field(default=None, description="Recorded rationale if order cancellation requested/approved")
    cancellation_category: Optional[CancellationReasonCategory] = Field(default=None, description="Category of cancellation")
    cancelled_by: Optional[str] = Field(default=None, description="Actor requesting or confirming cancellation")
    cancellation_timestamp: Optional[str] = Field(default=None, description="Cancellation timestamp (ISO 8601 UTC)")
    supplier_cancellation_ref: Optional[str] = Field(default=None, description="Supplier acknowledgment of cancellation")
    shipment_cancellation_ref: Optional[str] = Field(default=None, description="Documented cancellation or return disposition for in-transit shipments")

    closed_by: Optional[str] = Field(default=None, description="Actor closing unfulfilled balance")
    closure_timestamp: Optional[str] = Field(default=None, description="Timestamp of order closure (ISO 8601 UTC)")
    closure_reason: Optional[str] = Field(default=None, description="Rationale for closing remaining unfulfilled balance")
    closed_unfulfilled_qty: Optional[float] = Field(default=None, ge=0.0, description="Remaining unfulfilled quantity closed out")

    open_exception_reason: Optional[str] = Field(default=None, description="Active delivery exception explanation")
    open_exception_timestamp: Optional[str] = Field(default=None, description="Timestamp when exception was raised")
    resumable_status: Optional[FulfilmentOrderStatus] = Field(
        default=None,
        description="Status to return to upon resolution of delivery exception"
    )

    event_history: Tuple[FulfilmentEvent, ...] = Field(
        default_factory=tuple,
        description="Append-only immutable audit history of all fulfilment events"
    )
    is_simulation: bool = Field(default=True, description="Certifies this fulfilment order is a simulation object")

    @field_validator("fo_id", "po_id", "supplier_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator(
        "cancellation_reason", "cancelled_by", "supplier_cancellation_ref",
        "shipment_cancellation_ref", "closed_by", "closure_reason", "open_exception_reason"
    )
    @classmethod
    def validate_optional_fo_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError(f"Field '{info.field_name}', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("store_id")
    @classmethod
    def validate_store_id(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "store_id")
        if clean not in CANONICAL_STORES:
            raise ValueError(f"store_id '{clean}' must be one of {CANONICAL_STORES}")
        return clean

    @field_validator("created_at", "updated_at", "expected_delivery_at", "cancellation_timestamp", "open_exception_timestamp", "closure_timestamp", mode="before")
    @classmethod
    def validate_fo_timestamps(cls, v: Any, info) -> Optional[str]:
        if v is None:
            return None
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

    @model_validator(mode="after")
    def validate_fulfilment_order_integrity(self) -> "FulfilmentOrder":
        # 1. Source PO approval and evidence validation
        if self.po_snapshot.status != PurchaseOrderStatus.APPROVED:
            raise ValueError(
                f"Commercial Contract Violation: Source Purchase Order '{self.po_snapshot.po_id}' "
                f"must have status 'APPROVED'; got '{self.po_snapshot.status.value}'."
            )
        if not verify_approval_evidence(self.po_snapshot):
            raise ValueError(
                f"Commercial Governance Violation: Source Purchase Order '{self.po_snapshot.po_id}' "
                "lacks valid, verified human approval audit evidence."
            )

        # 2. Header alignment with PO snapshot
        if self.po_id != self.po_snapshot.po_id:
            raise ValueError(f"po_id mismatch: FulfilmentOrder='{self.po_id}', po_snapshot='{self.po_snapshot.po_id}'")
        if self.store_id != self.po_snapshot.store_id:
            raise ValueError(f"store_id mismatch: FulfilmentOrder='{self.store_id}', po_snapshot='{self.po_snapshot.store_id}'")
        if self.supplier_id != self.po_snapshot.supplier_id:
            raise ValueError(f"supplier_id mismatch: FulfilmentOrder='{self.supplier_id}', po_snapshot='{self.po_snapshot.supplier_id}'")
        if self.supplier_type != self.po_snapshot.supplier_type:
            raise ValueError(f"supplier_type mismatch: FulfilmentOrder='{self.supplier_type}', po_snapshot='{self.po_snapshot.supplier_type}'")

        # 3. Line items alignment with PO snapshot
        po_items_by_sku = {item.sku_id: item for item in self.po_snapshot.items}
        fo_skus_seen = set()

        for line in self.lines:
            if line.sku_id in fo_skus_seen:
                raise ValueError(f"Duplicate SKU '{line.sku_id}' in FulfilmentOrder lines.")
            fo_skus_seen.add(line.sku_id)

        if len(self.lines) != len(self.po_snapshot.items):
            raise ValueError(
                f"Line item count mismatch: FulfilmentOrder has {len(self.lines)} lines, "
                f"po_snapshot has {len(self.po_snapshot.items)} items."
            )

        for line in self.lines:
            if line.sku_id not in po_items_by_sku:
                raise ValueError(f"Line SKU '{line.sku_id}' does not exist on source Purchase Order '{self.po_id}'.")

            po_item = po_items_by_sku[line.sku_id]
            if abs(line.ordered_quantity - po_item.final_order_qty) > 1e-4:
                raise ValueError(
                    f"[{line.sku_id}] ordered_quantity mismatch: FulfilmentOrder={line.ordered_quantity}, "
                    f"po_snapshot.final_order_qty={po_item.final_order_qty}."
                )
            if abs(line.unit_cost - po_item.unit_procurement_cost) > 1e-4:
                raise ValueError(
                    f"[{line.sku_id}] unit_cost mismatch: FulfilmentOrder={line.unit_cost}, "
                    f"po_snapshot.unit_procurement_cost={po_item.unit_procurement_cost}."
                )

        # 4. Temporal alignment
        if self.po_snapshot.approval_timestamp:
            validate_timestamp_sequence(self.po_snapshot.approval_timestamp, self.created_at, "po_snapshot.approval_timestamp", "created_at")
        validate_timestamp_sequence(self.created_at, self.updated_at, "created_at", "updated_at")

        # 5. Cancellation invariants
        if self.status in [FulfilmentOrderStatus.CANCEL_REQUESTED, FulfilmentOrderStatus.CANCELLED]:
            if not self.cancellation_reason or not self.cancellation_reason.strip():
                raise ValueError(f"cancellation_reason is required when status is '{self.status.value}'.")
            if not self.cancelled_by or not self.cancelled_by.strip():
                raise ValueError(f"cancelled_by is required when status is '{self.status.value}'.")
            if not self.cancellation_timestamp:
                raise ValueError(f"cancellation_timestamp is required when status is '{self.status.value}'.")

        if self.status == FulfilmentOrderStatus.CANCELLED:
            if self.total_received_accepted_qty > 0:
                raise ValueError(
                    f"Governance Violation: Cannot mark FulfilmentOrder '{self.fo_id}' as CANCELLED "
                    f"with {self.total_received_accepted_qty} already received sellable units. "
                    "Use CLOSED_SHORT instead."
                )

        # 6. Closure invariants (CLOSED_SHORT)
        if self.status == FulfilmentOrderStatus.CLOSED_SHORT:
            if not self.closed_by or not self.closed_by.strip():
                raise ValueError("closed_by is required when status is CLOSED_SHORT.")
            if not self.closure_reason or not self.closure_reason.strip():
                raise ValueError("closure_reason is required when status is CLOSED_SHORT.")
            if not self.closure_timestamp:
                raise ValueError("closure_timestamp is required when status is CLOSED_SHORT.")
            if self.closed_unfulfilled_qty is None or self.closed_unfulfilled_qty < 0:
                raise ValueError("closed_unfulfilled_qty must be a non-negative float when status is CLOSED_SHORT.")

        # 7. Delivery exception invariants
        if self.status == FulfilmentOrderStatus.DELIVERY_EXCEPTION:
            if not self.open_exception_reason or not self.open_exception_reason.strip():
                raise ValueError("open_exception_reason is required when status is DELIVERY_EXCEPTION.")
            if not self.open_exception_timestamp:
                raise ValueError("open_exception_timestamp is required when status is DELIVERY_EXCEPTION.")

        # 7. Event history integrity & model_copy bypass defense
        if len(self.event_history) > 0:
            # Check latest event matches current status
            latest_event = self.event_history[-1]
            if latest_event.new_status != self.status:
                raise ValueError(
                    f"Audit History Desynchronization: FulfilmentOrder status '{self.status.value}' "
                    f"does not match latest audit event status '{latest_event.new_status.value}'. "
                    "Unvalidated model_copy state mutations are strictly prohibited."
                )

            # Check sequence numbers and timestamps
            seen_idempotency_keys = set()
            for idx, event in enumerate(self.event_history, start=1):
                if event.sequence_number != idx:
                    raise ValueError(
                        f"Event sequence numbering gap: expected sequence_number={idx}, got {event.sequence_number}."
                    )
                if event.idempotency_key in seen_idempotency_keys:
                    raise ValueError(f"Duplicate idempotency_key '{event.idempotency_key}' in event_history.")
                seen_idempotency_keys.add(event.idempotency_key)

                if idx > 1:
                    prev_event = self.event_history[idx - 2]
                    validate_timestamp_sequence(
                        prev_event.timestamp,
                        event.timestamp,
                        f"event[{prev_event.sequence_number}].timestamp",
                        f"event[{event.sequence_number}].timestamp"
                    )

        return self

    @property
    def total_confirmed_pipeline_qty(self) -> float:
        """
        Confirmed units in the active inbound pipeline.
        Only counts while supplier has accepted units and goods are awaiting delivery.
        Strictly zero for unacknowledged, rejected, closed, or cancelled orders.
        """
        if self.status in {
            FulfilmentOrderStatus.ACKNOWLEDGED,
            FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED,
            FulfilmentOrderStatus.IN_TRANSIT,
            FulfilmentOrderStatus.PARTIALLY_RECEIVED,
            FulfilmentOrderStatus.DELIVERY_EXCEPTION,
        }:
            return round(sum(line.outstanding_confirmed_quantity for line in self.lines), 4)
        return 0.0

    @property
    def total_received_accepted_qty(self) -> float:
        """Total units accepted into sellable inventory across all lines."""
        return round(sum(line.accepted_quantity for line in self.lines), 4)

    @property
    def total_remaining_unfulfilled_qty(self) -> float:
        """Total unfulfilled units from the approved order."""
        return round(sum(line.remaining_unfulfilled_quantity for line in self.lines), 4)

    @property
    def active_shipment_ids(self) -> Tuple[str, ...]:
        """
        Shipment identifiers that have been dispatched but are not yet resolved, received, or returned.
        An order-wide cancellation reference resolves multiple shipments only when its scope
        explicitly covers all of them (order_wide_cancellation_scope=True).
        For multiple shipments without order-wide scope, each shipment must have specific resolution evidence.
        """
        if self.shipment_cancellation_ref and self.order_wide_cancellation_scope:
            return ()

        resolved = set(self.resolved_shipment_ids) | set(self.shipment_resolution_evidence.keys())

        # If there is only one shipment on the order and a shipment_cancellation_ref is documented,
        # that cancellation reference applies directly to the single shipment.
        if self.shipment_cancellation_ref and len(self.shipment_ids) == 1:
            resolved.add(self.shipment_ids[0])

        return tuple(s for s in self.shipment_ids if s not in resolved)


def calculate_order_receipt_reconciliation(
    fo: FulfilmentOrder,
    grns: Iterable[GoodsReceiptNote],
    asns: Optional[Iterable[AdvanceShipmentNotice]] = None,
) -> OrderReceiptReconciliation:
    """
    Computes deterministic, authoritative cumulative reconciliation per-PO-line and per-shipment
    across all persisted GRNs.

    Balances computed:
    - Approved ordered quantity
    - Supplier-confirmed quantity
    - Shipped quantity
    - Physically delivered quantity
    - Accepted quantity
    - Rejected, damaged, excess, and short shelf-life quantities
    - Remaining supplier-confirmed outstanding quantity
    - Unconfirmed and declined quantities
    """
    grn_list = [g for g in grns if g.fo_id == fo.fo_id]
    asn_list = [a for a in asns if a.fo_id == fo.fo_id] if asns is not None else []

    # Map shipments to GRNs
    grns_by_shipment: Dict[str, List[GoodsReceiptNote]] = {}
    for s_id in fo.shipment_ids:
        grns_by_shipment[s_id] = [g for g in grn_list if g.shipment_id == s_id]
    for g in grn_list:
        if g.shipment_id not in grns_by_shipment:
            grns_by_shipment[g.shipment_id] = [g]

    # Map shipments to ASNs
    asns_by_shipment: Dict[str, AdvanceShipmentNotice] = {a.shipment_id: a for a in asn_list}

    # Per-line aggregation across all GRNs
    line_reconciliations: List[LineReceiptReconciliation] = []
    for line in fo.lines:
        sku = line.sku_id
        ordered_q = line.ordered_quantity
        supp_acc_q = line.supplier_accepted_quantity
        supp_dec_q = line.supplier_declined_quantity
        supp_unconf_q = line.supplier_unconfirmed_quantity
        shipped_q = line.shipped_quantity

        # Sum quantities across GRNs for this SKU
        sku_grn_lines = [
            insp for g in grn_list for insp in g.inspection_lines if insp.sku_id == sku
        ]
        del_q = round(sum(insp.delivered_quantity for insp in sku_grn_lines), 4)
        dam_q = round(sum(insp.damaged_quantity for insp in sku_grn_lines), 4)
        exc_q = round(sum(insp.excess_quantity for insp in sku_grn_lines), 4)
        ssl_q = round(sum(insp.short_shelf_life_quantity for insp in sku_grn_lines), 4)

        # Use authoritative accepted and rejected from fo.lines (reflects reinspection releases)
        # Fall back to GRN sums if fo.lines has not yet processed the GRN
        acc_q = round(line.accepted_quantity, 4) if line.delivered_quantity > 0 else round(sum(insp.accepted_quantity for insp in sku_grn_lines), 4)
        rej_q = round(line.rejected_quantity, 4) if line.delivered_quantity > 0 else round(sum(insp.rejected_quantity for insp in sku_grn_lines), 4)

        rem_conf_q = max(0.0, round(supp_acc_q - acc_q, 4))
        rem_unfulfilled_q = max(0.0, round(ordered_q - acc_q, 4))
        is_fully_acc = (acc_q >= ordered_q - 1e-4)

        line_rec = LineReceiptReconciliation(
            sku_id=sku,
            ordered_quantity=ordered_q,
            supplier_accepted_quantity=supp_acc_q,
            supplier_declined_quantity=supp_dec_q,
            supplier_unconfirmed_quantity=supp_unconf_q,
            shipped_quantity=shipped_q,
            delivered_quantity=del_q,
            accepted_quantity=acc_q,
            rejected_quantity=rej_q,
            damaged_quantity=dam_q,
            excess_quantity=exc_q,
            short_shelf_life_quantity=ssl_q,
            remaining_confirmed_outstanding_quantity=rem_conf_q,
            remaining_unfulfilled_quantity=rem_unfulfilled_q,
            is_fully_accepted=is_fully_acc,
        )
        line_reconciliations.append(line_rec)

    # Per-shipment aggregation
    shipment_reconciliations: List[ShipmentReceiptReconciliation] = []
    all_shipment_ids = list(fo.shipment_ids)
    for s_id in grns_by_shipment:
        if s_id not in all_shipment_ids:
            all_shipment_ids.append(s_id)

    for s_id in all_shipment_ids:
        s_grns = grns_by_shipment.get(s_id, [])
        s_grn_ids = tuple(g.grn_id for g in s_grns)

        s_lines: List[LineReceiptReconciliation] = []
        for line in fo.lines:
            sku = line.sku_id
            matching_insp = [
                insp for g in s_grns for insp in g.inspection_lines if insp.sku_id == sku
            ]
            del_q = round(sum(insp.delivered_quantity for insp in matching_insp), 4)
            acc_q = round(sum(insp.accepted_quantity for insp in matching_insp), 4)
            rej_q = round(sum(insp.rejected_quantity for insp in matching_insp), 4)
            dam_q = round(sum(insp.damaged_quantity for insp in matching_insp), 4)
            exc_q = round(sum(insp.excess_quantity for insp in matching_insp), 4)
            ssl_q = round(sum(insp.short_shelf_life_quantity for insp in matching_insp), 4)

            # Check if this shipment line had reinspection releases
            released_qty = sum(
                r.quantity for r in fo.quality_hold_resolutions
                if r.shipment_id == s_id and r.sku_id == sku and r.resolution_type == DiscrepancyResolutionType.REINSPECTION_RELEASE
            )
            if released_qty > 0:
                acc_q = round(acc_q + released_qty, 4)
                rej_q = max(0.0, round(rej_q - released_qty, 4))

            shipped_q = 0.0
            if s_id in asns_by_shipment:
                asn_match = next((l for l in asns_by_shipment[s_id].lines if l.sku_id == sku), None)
                if asn_match:
                    shipped_q = asn_match.dispatched_quantity

            s_line_rec = LineReceiptReconciliation(
                sku_id=sku,
                ordered_quantity=line.ordered_quantity,
                supplier_accepted_quantity=line.supplier_accepted_quantity,
                supplier_declined_quantity=line.supplier_declined_quantity,
                supplier_unconfirmed_quantity=line.supplier_unconfirmed_quantity,
                shipped_quantity=shipped_q,
                delivered_quantity=del_q,
                accepted_quantity=acc_q,
                rejected_quantity=rej_q,
                damaged_quantity=dam_q,
                excess_quantity=exc_q,
                short_shelf_life_quantity=ssl_q,
                remaining_confirmed_outstanding_quantity=max(0.0, round(line.supplier_accepted_quantity - acc_q, 4)),
                remaining_unfulfilled_quantity=max(0.0, round(line.ordered_quantity - acc_q, 4)),
                is_fully_accepted=(acc_q >= line.ordered_quantity - 1e-4),
            )
            s_lines.append(s_line_rec)

        s_tot_shipped = round(sum(l.shipped_quantity for l in s_lines), 4)
        s_tot_del = round(sum(l.delivered_quantity for l in s_lines), 4)
        s_tot_acc = round(sum(l.accepted_quantity for l in s_lines), 4)
        s_tot_rej = round(sum(l.rejected_quantity for l in s_lines), 4)
        s_tot_dam = round(sum(l.damaged_quantity for l in s_lines), 4)
        s_tot_exc = round(sum(l.excess_quantity for l in s_lines), 4)
        s_tot_ssl = round(sum(l.short_shelf_life_quantity for l in s_lines), 4)

        # Physical custody resolution check:
        # Commercial settlements (SUPPLIER_CREDIT_NOTE, SHORT_DELIVERY_ACKNOWLEDGED) do not prove
        # physical goods have left store custody.
        AUTHORIZED_DISPOSITION_TYPES = {
            DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
            DiscrepancyResolutionType.SUPPLIER_CREDIT_NOTE,
            DiscrepancyResolutionType.EXCESS_RETURN,
            DiscrepancyResolutionType.REINSPECTION_RELEASE,
        }
        disposed_qty = sum(
            r.quantity for r in fo.discrepancy_resolutions
            if r.shipment_id == s_id and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
        ) + sum(
            r.quantity for r in fo.quality_hold_resolutions
            if r.shipment_id == s_id and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
        )
        is_custody_res = False
        if s_id in fo.resolved_shipment_ids:
            is_custody_res = True
        elif s_tot_rej <= 1e-4 and s_tot_acc > 0:
            is_custody_res = True
        elif s_tot_rej > 0 and disposed_qty >= s_tot_rej - 1e-4:
            is_custody_res = True

        evidence_ref = fo.shipment_resolution_evidence.get(s_id)
        if not evidence_ref:
            res_refs = [
                r.disposition_reference for r in fo.discrepancy_resolutions
                if r.shipment_id == s_id and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
            ] + [
                r.disposition_reference for r in fo.quality_hold_resolutions
                if r.shipment_id == s_id and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
            ]
            if res_refs:
                evidence_ref = res_refs[-1]

        shipment_rec = ShipmentReceiptReconciliation(
            shipment_id=s_id,
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            grn_ids=s_grn_ids,
            shipped_quantity=s_tot_shipped,
            delivered_quantity=s_tot_del,
            accepted_quantity=s_tot_acc,
            rejected_quantity=s_tot_rej,
            damaged_quantity=s_tot_dam,
            excess_quantity=s_tot_exc,
            short_shelf_life_quantity=s_tot_ssl,
            lines=tuple(s_lines),
            is_custody_resolved=is_custody_res,
            disposition_reference=evidence_ref,
        )
        shipment_reconciliations.append(shipment_rec)

    # Order totals
    tot_ord = round(sum(l.ordered_quantity for l in line_reconciliations), 4)
    tot_supp_acc = round(sum(l.supplier_accepted_quantity for l in line_reconciliations), 4)
    tot_supp_dec = round(sum(l.supplier_declined_quantity for l in line_reconciliations), 4)
    tot_supp_unconf = round(sum(l.supplier_unconfirmed_quantity for l in line_reconciliations), 4)
    tot_shipped = round(sum(l.shipped_quantity for l in line_reconciliations), 4)
    tot_del = round(sum(l.delivered_quantity for l in line_reconciliations), 4)
    tot_acc = round(sum(l.accepted_quantity for l in line_reconciliations), 4)
    tot_rej = round(sum(l.rejected_quantity for l in line_reconciliations), 4)
    tot_dam = round(sum(l.damaged_quantity for l in line_reconciliations), 4)
    tot_exc = round(sum(l.excess_quantity for l in line_reconciliations), 4)
    tot_ssl = round(sum(l.short_shelf_life_quantity for l in line_reconciliations), 4)
    tot_rem_conf = round(sum(l.remaining_confirmed_outstanding_quantity for l in line_reconciliations), 4)
    tot_rem_unf = round(sum(l.remaining_unfulfilled_quantity for l in line_reconciliations), 4)

    is_fully_rec = all(l.is_fully_accepted for l in line_reconciliations)
    can_cls_short = (
        fo.status in {FulfilmentOrderStatus.PARTIALLY_RECEIVED, FulfilmentOrderStatus.DELIVERY_EXCEPTION}
        and tot_rem_unf > 0
    )

    return OrderReceiptReconciliation(
        fo_id=fo.fo_id,
        po_id=fo.po_id,
        status=fo.status,
        total_ordered_quantity=tot_ord,
        total_supplier_confirmed_quantity=tot_supp_acc,
        total_supplier_declined_quantity=tot_supp_dec,
        total_supplier_unconfirmed_quantity=tot_supp_unconf,
        total_shipped_quantity=tot_shipped,
        total_delivered_quantity=tot_del,
        total_accepted_quantity=tot_acc,
        total_rejected_quantity=tot_rej,
        total_damaged_quantity=tot_dam,
        total_excess_quantity=tot_exc,
        total_short_shelf_life_quantity=tot_ssl,
        total_remaining_confirmed_outstanding_quantity=tot_rem_conf,
        total_remaining_unfulfilled_quantity=tot_rem_unf,
        lines=tuple(line_reconciliations),
        shipments=tuple(shipment_reconciliations),
        is_fully_received=is_fully_rec,
        can_close_short=can_cls_short,
    )
