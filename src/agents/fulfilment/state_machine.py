"""
Lead 2 Fulfilment State Machine & Transition Engine
==================================================
Implements pure, validated state machine transitions for quick-commerce FulfilmentOrders.

CORE PRINCIPLES & GOVERNANCE:
1. IMMUTABLE ORIGINAL PURCHASE ORDER:
   The original PurchaseOrder is the commercial contract of record. It remains strictly
   in APPROVED status throughout fulfilment and is NEVER mutated by supplier responses
   or receiving events.
2. FULL RECONSTRUCTION (NO MODEL_COPY BYPASS):
   All state transitions execute through dictionary reconstruction and model_validate().
   model_copy(update=...) is strictly defended against at workflow boundaries.
3. STRONG IDEMPOTENCY PAYLOAD INTEGRITY:
   Every transition accepts an idempotency key. Duplicate events with the same key
   and identical canonical SHA-256 payload fingerprint return the existing state safely.
   Any conflicting event type or altered payload raises a FulfilmentGovernanceError.
4. AUDIT EVENT TRACEABILITY:
   Every lifecycle transition appends an immutable FulfilmentEvent with incrementing
   sequence numbers and validated non-decreasing UTC timestamps.
5. QUALITY, PHYSICAL DISCREPANCY & CANCELLATION POLICIES:
   - Physical delivered quantity may exceed ASN quantity or outstanding PO balance
     so the discrepancy can be recorded; excess is rejected with EXCESS_DELIVERY.
   - Orders with already received sellable stock cannot be cancelled (must use close_short).
   - Once transmitted, cancellation requires external supplier cancellation reference.
   - If goods are in transit, cancellation requires documented shipment return disposition.
   - Terminal states (RECEIVED, CLOSED_SHORT, CANCELLED, REJECTED_BY_SUPPLIER) are immutable.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import ValidationError

from ..procurement.approval import can_be_dispatched, verify_approval_evidence
from ..procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderStatus,
    validate_and_normalize_iso_timestamp,
)
from .schemas import (
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DiscrepancyReason,
    DiscrepancyResolution,
    DiscrepancyResolutionType,
    DockInspectionLine,
    FulfilmentActorType,
    FulfilmentEvent,
    FulfilmentEventType,
    FulfilmentOrder,
    FulfilmentOrderLine,
    FulfilmentOrderStatus,
    GoodsReceiptNote,
    QualityHoldResolution,
    SupplierAcknowledgment,
    SupplierAcknowledgmentStatus,
    compute_payload_fingerprint,
    validate_timestamp_sequence,
)


class InvalidFulfilmentTransitionError(ValueError):
    """Raised when an illegal, out-of-order, or forbidden status transition is attempted."""
    pass


class FulfilmentGovernanceError(ValueError):
    """Raised when operational governance, approval evidence, or authorization rules are violated."""
    pass


def _validate_non_empty_str(v: Any, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


# Explicit transition map defining all allowed forward and exception paths
ALLOWED_TRANSITIONS: Dict[FulfilmentOrderStatus, Set[FulfilmentOrderStatus]] = {
    FulfilmentOrderStatus.READY_TO_TRANSMIT: {
        FulfilmentOrderStatus.ISSUED,
        FulfilmentOrderStatus.CANCEL_REQUESTED,
    },
    FulfilmentOrderStatus.ISSUED: {
        FulfilmentOrderStatus.ACKNOWLEDGED,
        FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED,
        FulfilmentOrderStatus.REJECTED_BY_SUPPLIER,
        FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW,
        FulfilmentOrderStatus.CANCEL_REQUESTED,
    },
    FulfilmentOrderStatus.ACKNOWLEDGED: {
        FulfilmentOrderStatus.IN_TRANSIT,
        FulfilmentOrderStatus.CANCEL_REQUESTED,
        FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW,
    },
    FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED: {
        FulfilmentOrderStatus.IN_TRANSIT,
        FulfilmentOrderStatus.CANCEL_REQUESTED,
        FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW,
    },
    FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW: {
        FulfilmentOrderStatus.ACKNOWLEDGED,
        FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED,
        FulfilmentOrderStatus.REJECTED_BY_SUPPLIER,
        FulfilmentOrderStatus.CANCEL_REQUESTED,
        FulfilmentOrderStatus.CANCELLED,  # When superseded by newly approved replacement PO
    },
    FulfilmentOrderStatus.IN_TRANSIT: {
        FulfilmentOrderStatus.PARTIALLY_RECEIVED,
        FulfilmentOrderStatus.RECEIVED,
        FulfilmentOrderStatus.DELIVERY_EXCEPTION,
        FulfilmentOrderStatus.CANCEL_REQUESTED,
    },
    FulfilmentOrderStatus.PARTIALLY_RECEIVED: {
        FulfilmentOrderStatus.PARTIALLY_RECEIVED,  # repeated partial receipt
        FulfilmentOrderStatus.RECEIVED,
        FulfilmentOrderStatus.DELIVERY_EXCEPTION,
        FulfilmentOrderStatus.CLOSED_SHORT,
        FulfilmentOrderStatus.IN_TRANSIT,  # subsequent split shipment dispatch
    },
    FulfilmentOrderStatus.DELIVERY_EXCEPTION: {
        FulfilmentOrderStatus.IN_TRANSIT,
        FulfilmentOrderStatus.PARTIALLY_RECEIVED,
        FulfilmentOrderStatus.RECEIVED,
        FulfilmentOrderStatus.CLOSED_SHORT,
        FulfilmentOrderStatus.CANCEL_REQUESTED,
    },
    FulfilmentOrderStatus.CANCEL_REQUESTED: {
        FulfilmentOrderStatus.CANCELLED,
        # Rejection of cancellation returns to prior active status
        FulfilmentOrderStatus.READY_TO_TRANSMIT,
        FulfilmentOrderStatus.ISSUED,
        FulfilmentOrderStatus.ACKNOWLEDGED,
        FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED,
        FulfilmentOrderStatus.IN_TRANSIT,
        FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW,
        FulfilmentOrderStatus.DELIVERY_EXCEPTION,
    },
    # Terminal states
    FulfilmentOrderStatus.RECEIVED: set(),
    FulfilmentOrderStatus.CLOSED_SHORT: set(),
    FulfilmentOrderStatus.CANCELLED: set(),
    FulfilmentOrderStatus.REJECTED_BY_SUPPLIER: set(),
}

TERMINAL_STATUSES: Set[FulfilmentOrderStatus] = {
    FulfilmentOrderStatus.RECEIVED,
    FulfilmentOrderStatus.CLOSED_SHORT,
    FulfilmentOrderStatus.CANCELLED,
    FulfilmentOrderStatus.REJECTED_BY_SUPPLIER,
}


def validate_inbound_fulfilment_order(fo: FulfilmentOrder) -> FulfilmentOrder:
    """
    Validates incoming FulfilmentOrder at workflow boundaries.
    Defends against unvalidated model_copy(update=...) mutations.
    """
    if not isinstance(fo, FulfilmentOrder):
        raise FulfilmentGovernanceError(f"Expected FulfilmentOrder instance, got {type(fo)}.")

    try:
        return FulfilmentOrder.model_validate(fo.model_dump())
    except ValidationError as err:
        raise FulfilmentGovernanceError(
            f"Workflow Boundary Rejection: FulfilmentOrder '{fo.fo_id}' failed validation: {err}"
        ) from err


def is_transition_allowed(current_status: FulfilmentOrderStatus, target_status: FulfilmentOrderStatus) -> bool:
    """Checks whether a state machine transition is formally permitted."""
    allowed = ALLOWED_TRANSITIONS.get(current_status, set())
    return target_status in allowed


def _verify_idempotency(
    event_history: Tuple[FulfilmentEvent, ...],
    idempotency_key: str,
    expected_event_type: FulfilmentEventType,
    payload_fingerprint: str,
) -> Optional[FulfilmentEvent]:
    """
    Helper to enforce strong idempotency:
    1. Returns matching event if key, event_type, and payload fingerprint all match.
    2. Raises FulfilmentGovernanceError if key matches but event_type or payload differs.
    3. Returns None if key has not been seen before.
    """
    for evt in event_history:
        if evt.idempotency_key == idempotency_key:
            if evt.event_type != expected_event_type:
                raise FulfilmentGovernanceError(
                    f"Idempotency Conflict: key '{idempotency_key}' was previously used for event "
                    f"type '{evt.event_type.value}', cannot be reused for '{expected_event_type.value}'."
                )
            if evt.payload_fingerprint and evt.payload_fingerprint != payload_fingerprint:
                raise FulfilmentGovernanceError(
                    f"Idempotency Conflict: key '{idempotency_key}' presented with differing payload fingerprint. "
                    f"Recorded='{evt.payload_fingerprint}', Incoming='{payload_fingerprint}'."
                )
            return evt
    return None


def create_fulfilment_order(
    po: PurchaseOrder,
    actor_id: str,
    timestamp: str,
    fo_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    supersedes_fo_id: Optional[str] = None,
    supersedes_po_id: Optional[str] = None,
) -> FulfilmentOrder:
    """
    Initial handoff: creates a new FulfilmentOrder in READY_TO_TRANSMIT status from
    an approved PurchaseOrder.
    """
    if not can_be_dispatched(po):
        raise FulfilmentGovernanceError(
            f"Commercial Governance Violation: Purchase Order '{po.po_id}' cannot enter fulfilment. "
            "Valid, verified human approval evidence is strictly required."
        )

    clean_actor = actor_id.strip() if actor_id and actor_id.strip() else "FULFILMENT_AGENT"
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp", error_cls=FulfilmentGovernanceError)
    assigned_fo_id = fo_id.strip() if fo_id and fo_id.strip() else f"FO-{po.po_id.replace('PO-', '')}"
    assigned_idempotency_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else f"CREATE:{assigned_fo_id}:1"
    clean_supersedes_fo = supersedes_fo_id.strip() if supersedes_fo_id and supersedes_fo_id.strip() else None
    clean_supersedes_po = supersedes_po_id.strip() if supersedes_po_id and supersedes_po_id.strip() else None

    lines: List[Dict[str, Any]] = []
    for item in po.items:
        lines.append({
            "sku_id": item.sku_id,
            "category": item.category,
            "ordered_quantity": item.final_order_qty,
            "unit_cost": item.unit_procurement_cost,
            "supplier_accepted_quantity": 0.0,
            "supplier_declined_quantity": 0.0,
            "supplier_unconfirmed_quantity": item.final_order_qty,
            "shipped_quantity": 0.0,
            "delivered_quantity": 0.0,
            "accepted_quantity": 0.0,
            "rejected_quantity": 0.0,
        })

    payload = {
        "action": "CREATE_FULFILMENT_ORDER",
        "fo_id": assigned_fo_id,
        "po_id": po.po_id,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    if clean_supersedes_fo:
        payload["supersedes_fo_id"] = clean_supersedes_fo
    if clean_supersedes_po:
        payload["supersedes_po_id"] = clean_supersedes_po
    fingerprint = compute_payload_fingerprint(payload)

    event_details: Dict[str, Any] = {
        "source_po_cost": po.total_order_cost,
        "item_count": len(po.items),
    }
    if clean_supersedes_fo:
        event_details["supersedes_fo_id"] = clean_supersedes_fo
    if clean_supersedes_po:
        event_details["supersedes_po_id"] = clean_supersedes_po

    initial_event = FulfilmentEvent(
        event_id=f"EVT-{assigned_fo_id}-001",
        idempotency_key=assigned_idempotency_key,
        fo_id=assigned_fo_id,
        event_type=FulfilmentEventType.CREATE_FULFILMENT_ORDER,
        actor_type=FulfilmentActorType.FULFILMENT_AGENT,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=None,
        new_status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
        sequence_number=1,
        related_reference_id=po.po_id,
        payload_fingerprint=fingerprint,
        details=event_details,
    )

    data: Dict[str, Any] = {
        "fo_id": assigned_fo_id,
        "po_id": po.po_id,
        "po_snapshot": po.model_dump(),
        "store_id": po.store_id,
        "supplier_id": po.supplier_id,
        "supplier_type": po.supplier_type.value,
        "status": FulfilmentOrderStatus.READY_TO_TRANSMIT.value,
        "lines": lines,
        "created_at": clean_timestamp,
        "updated_at": clean_timestamp,
        "event_history": [initial_event.model_dump()],
        "is_simulation": True,
    }

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Failed to construct FulfilmentOrder: {err}") from err


def transmit_fulfilment_order(
    fo: FulfilmentOrder,
    transmission_timestamp: str,
    actor_id: str,
    idempotency_key: str,
) -> FulfilmentOrder:
    """
    Transmission transition: READY_TO_TRANSMIT -> ISSUED.
    Enforces strong payload-fingerprint idempotency.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(transmission_timestamp, "transmission_timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()

    payload = {
        "action": "TRANSMIT_TO_SUPPLIER",
        "fo_id": validated_fo.fo_id,
        "po_id": validated_fo.po_id,
        "transmission_timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    # Strong idempotency check
    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.TRANSMIT_TO_SUPPLIER, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in TERMINAL_STATUSES:
        raise InvalidFulfilmentTransitionError(
            f"Cannot transmit FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    if validated_fo.status != FulfilmentOrderStatus.READY_TO_TRANSMIT:
        raise InvalidFulfilmentTransitionError(
            f"Cannot transmit FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'. "
            "Only orders in READY_TO_TRANSMIT may be issued to supplier."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "transmission_timestamp")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.TRANSMIT_TO_SUPPLIER,
        actor_type=FulfilmentActorType.FULFILMENT_AGENT,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=FulfilmentOrderStatus.ISSUED,
        sequence_number=new_seq,
        related_reference_id=validated_fo.po_id,
        payload_fingerprint=fingerprint,
    )

    data = validated_fo.model_dump()
    data["status"] = FulfilmentOrderStatus.ISSUED.value
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Transmission transition failed: {err}") from err


def apply_supplier_acknowledgment(
    fo: FulfilmentOrder,
    ack: SupplierAcknowledgment,
    actor_id: str,
    idempotency_key: str,
) -> FulfilmentOrder:
    """
    Processes supplier response:
    - ACCEPTED -> ACKNOWLEDGED
    - PARTIALLY_ACCEPTED -> PARTIALLY_ACKNOWLEDGED
    - REJECTED -> REJECTED_BY_SUPPLIER (terminal)
    - CHANGE_REQUESTED -> SUPPLIER_CHANGE_PENDING_REVIEW
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()

    payload = {
        "action": "SUPPLIER_ACKNOWLEDGE",
        "acknowledgment_id": ack.acknowledgment_id,
        "fo_id": ack.fo_id,
        "po_id": ack.po_id,
        "supplier_id": ack.supplier_id,
        "status": ack.status.value,
        "acknowledged_at": ack.acknowledged_at,
        "expected_delivery_at": ack.expected_delivery_at,
        "lines": [
            {"sku_id": l.sku_id, "accepted_quantity": l.accepted_quantity, "declined_quantity": l.declined_quantity}
            for l in ack.lines
        ],
        "rejection_reason": ack.rejection_reason,
        "change_request_details": ack.change_request_details,
    }
    fingerprint = compute_payload_fingerprint(payload)

    # Strong idempotency check
    expected_type = (
        FulfilmentEventType.SUPPLIER_REJECT if ack.status == SupplierAcknowledgmentStatus.REJECTED
        else FulfilmentEventType.SUPPLIER_CHANGE_REQUEST if ack.status == SupplierAcknowledgmentStatus.CHANGE_REQUESTED
        else FulfilmentEventType.SUPPLIER_ACKNOWLEDGE
    )
    existing = _verify_idempotency(validated_fo.event_history, clean_key, expected_type, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in TERMINAL_STATUSES:
        raise InvalidFulfilmentTransitionError(
            f"Cannot apply acknowledgment to FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    valid_ack_statuses = {
        FulfilmentOrderStatus.ISSUED,
        FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW,
        FulfilmentOrderStatus.ACKNOWLEDGED,
        FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED,
    }
    if validated_fo.status not in valid_ack_statuses:
        raise InvalidFulfilmentTransitionError(
            f"Cannot apply acknowledgment to FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'. "
            "Order must be in ISSUED, SUPPLIER_CHANGE_PENDING_REVIEW, ACKNOWLEDGED, or PARTIALLY_ACKNOWLEDGED."
        )

    if ack.fo_id != validated_fo.fo_id or ack.po_id != validated_fo.po_id or ack.supplier_id != validated_fo.supplier_id:
        raise FulfilmentGovernanceError(
            f"Acknowledgment identifier mismatch: ack={(ack.fo_id, ack.po_id, ack.supplier_id)}, "
            f"order={(validated_fo.fo_id, validated_fo.po_id, validated_fo.supplier_id)}."
        )

    validate_timestamp_sequence(validated_fo.updated_at, ack.acknowledged_at, "updated_at", "acknowledged_at")

    updated_lines: List[Dict[str, Any]] = []
    ack_lines_by_sku = {line.sku_id: line for line in ack.lines}

    for sku in ack_lines_by_sku:
        if not any(l.sku_id == sku for l in validated_fo.lines):
            raise FulfilmentGovernanceError(f"Acknowledgment contains unknown SKU '{sku}' not on order.")

    if ack.status == SupplierAcknowledgmentStatus.REJECTED:
        target_status = FulfilmentOrderStatus.REJECTED_BY_SUPPLIER
        event_type = FulfilmentEventType.SUPPLIER_REJECT
        for line in validated_fo.lines:
            line_dict = line.model_dump()
            line_dict["supplier_accepted_quantity"] = 0.0
            line_dict["supplier_declined_quantity"] = line.ordered_quantity
            line_dict["supplier_unconfirmed_quantity"] = 0.0
            updated_lines.append(line_dict)

    elif ack.status == SupplierAcknowledgmentStatus.ACCEPTED:
        target_status = FulfilmentOrderStatus.ACKNOWLEDGED
        event_type = FulfilmentEventType.SUPPLIER_ACKNOWLEDGE
        for line in validated_fo.lines:
            line_dict = line.model_dump()
            if line.sku_id in ack_lines_by_sku:
                ack_l = ack_lines_by_sku[line.sku_id]
                if ack_l.accepted_quantity > line.ordered_quantity + 1e-4:
                    raise FulfilmentGovernanceError(
                        f"[{line.sku_id}] Acknowledged quantity ({ack_l.accepted_quantity}) "
                        f"cannot exceed ordered quantity ({line.ordered_quantity})."
                    )
                if ack_l.accepted_quantity + ack_l.declined_quantity > line.ordered_quantity + 1e-4:
                    raise FulfilmentGovernanceError(
                        f"[{line.sku_id}] Total acknowledged quantity (accepted {ack_l.accepted_quantity} + declined {ack_l.declined_quantity}) "
                        f"cannot exceed ordered quantity ({line.ordered_quantity})."
                    )
                line_dict["supplier_accepted_quantity"] = ack_l.accepted_quantity
                line_dict["supplier_declined_quantity"] = ack_l.declined_quantity
                line_dict["supplier_unconfirmed_quantity"] = max(
                    0.0, round(line.ordered_quantity - ack_l.accepted_quantity - ack_l.declined_quantity, 4)
                )
            else:
                line_dict["supplier_accepted_quantity"] = line.ordered_quantity
                line_dict["supplier_declined_quantity"] = 0.0
                line_dict["supplier_unconfirmed_quantity"] = 0.0
            updated_lines.append(line_dict)

    elif ack.status == SupplierAcknowledgmentStatus.PARTIALLY_ACCEPTED:
        target_status = FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED
        event_type = FulfilmentEventType.SUPPLIER_ACKNOWLEDGE
        for line in validated_fo.lines:
            line_dict = line.model_dump()
            if line.sku_id in ack_lines_by_sku:
                ack_l = ack_lines_by_sku[line.sku_id]
                if ack_l.accepted_quantity > line.ordered_quantity + 1e-4:
                    raise FulfilmentGovernanceError(
                        f"[{line.sku_id}] Acknowledged quantity ({ack_l.accepted_quantity}) "
                        f"cannot exceed ordered quantity ({line.ordered_quantity})."
                    )
                if ack_l.accepted_quantity + ack_l.declined_quantity > line.ordered_quantity + 1e-4:
                    raise FulfilmentGovernanceError(
                        f"[{line.sku_id}] Total acknowledged quantity (accepted {ack_l.accepted_quantity} + declined {ack_l.declined_quantity}) "
                        f"cannot exceed ordered quantity ({line.ordered_quantity})."
                    )
                line_dict["supplier_accepted_quantity"] = ack_l.accepted_quantity
                line_dict["supplier_declined_quantity"] = ack_l.declined_quantity
                line_dict["supplier_unconfirmed_quantity"] = max(
                    0.0, round(line.ordered_quantity - ack_l.accepted_quantity - ack_l.declined_quantity, 4)
                )
            else:
                line_dict["supplier_accepted_quantity"] = 0.0
                line_dict["supplier_declined_quantity"] = line.ordered_quantity
                line_dict["supplier_unconfirmed_quantity"] = 0.0
            updated_lines.append(line_dict)

    elif ack.status == SupplierAcknowledgmentStatus.CHANGE_REQUESTED:
        target_status = FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW
        event_type = FulfilmentEventType.SUPPLIER_CHANGE_REQUEST
        updated_lines = [l.model_dump() for l in validated_fo.lines]
    else:
        raise InvalidFulfilmentTransitionError(f"Unsupported acknowledgment status: {ack.status}")

    if not is_transition_allowed(validated_fo.status, target_status):
        raise InvalidFulfilmentTransitionError(
            f"Cannot transition FulfilmentOrder '{validated_fo.fo_id}' from '{validated_fo.status.value}' "
            f"to '{target_status.value}' via supplier acknowledgment."
        )

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=event_type,
        actor_type=FulfilmentActorType.SUPPLIER_GATEWAY,
        actor_id=clean_actor,
        timestamp=ack.acknowledged_at,
        previous_status=validated_fo.status,
        new_status=target_status,
        sequence_number=new_seq,
        related_reference_id=ack.acknowledgment_id,
        payload_fingerprint=fingerprint,
        details={
            "ack_status": ack.status.value,
            "rejection_reason": ack.rejection_reason,
            "change_request_details": ack.change_request_details,
        },
    )

    data = validated_fo.model_dump()
    data["status"] = target_status.value
    data["lines"] = updated_lines
    data["updated_at"] = ack.acknowledged_at
    if ack.expected_delivery_at:
        data["expected_delivery_at"] = ack.expected_delivery_at
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Acknowledgment transition failed: {err}") from err


def apply_shipment_dispatch(
    fo: FulfilmentOrder,
    asn: AdvanceShipmentNotice,
    actor_id: str,
    idempotency_key: str,
) -> FulfilmentOrder:
    """
    Shipment dispatch transition: ACKNOWLEDGED / PARTIALLY_ACKNOWLEDGED -> IN_TRANSIT.
    Supports split/multiple shipments.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()

    payload = {
        "action": "SHIPMENT_DISPATCH",
        "shipment_id": asn.shipment_id,
        "fo_id": asn.fo_id,
        "po_id": asn.po_id,
        "supplier_id": asn.supplier_id,
        "dispatch_at": asn.dispatch_at,
        "expected_arrival_at": asn.expected_arrival_at,
        "lines": [{"sku_id": l.sku_id, "shipped_quantity": l.shipped_quantity} for l in asn.lines],
        "carrier": asn.carrier,
        "tracking_reference": asn.tracking_reference,
    }
    fingerprint = compute_payload_fingerprint(payload)

    # Strong idempotency check
    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.SHIPMENT_DISPATCH, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in TERMINAL_STATUSES:
        raise InvalidFulfilmentTransitionError(
            f"Cannot dispatch shipment for FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    if validated_fo.status not in {
        FulfilmentOrderStatus.ACKNOWLEDGED,
        FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED,
        FulfilmentOrderStatus.IN_TRANSIT,
        FulfilmentOrderStatus.PARTIALLY_RECEIVED,
    }:
        raise InvalidFulfilmentTransitionError(
            f"Cannot dispatch shipment for FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'."
        )

    if asn.fo_id != validated_fo.fo_id or asn.po_id != validated_fo.po_id or asn.supplier_id != validated_fo.supplier_id:
        raise FulfilmentGovernanceError(
            f"Shipment identifier mismatch: asn={(asn.fo_id, asn.po_id, asn.supplier_id)}, "
            f"order={(validated_fo.fo_id, validated_fo.po_id, validated_fo.supplier_id)}."
        )

    validate_timestamp_sequence(validated_fo.updated_at, asn.dispatch_at, "updated_at", "dispatch_at")

    if asn.shipment_id in validated_fo.shipment_ids:
        raise FulfilmentGovernanceError(f"Duplicate shipment ID '{asn.shipment_id}' has already been processed.")

    asn_lines_by_sku = {line.sku_id: line.shipped_quantity for line in asn.lines}
    updated_lines: List[Dict[str, Any]] = []

    for line in validated_fo.lines:
        line_dict = line.model_dump()
        if line.sku_id in asn_lines_by_sku:
            inc_shipped = asn_lines_by_sku[line.sku_id]
            new_shipped = round(line.shipped_quantity + inc_shipped, 4)
            if new_shipped > line.supplier_accepted_quantity + 1e-4:
                raise FulfilmentGovernanceError(
                    f"[{line.sku_id}] Cumulative shipped quantity ({new_shipped}) "
                    f"exceeds supplier accepted quantity ({line.supplier_accepted_quantity})."
                )
            line_dict["shipped_quantity"] = new_shipped
        updated_lines.append(line_dict)

    for sku in asn_lines_by_sku:
        if not any(l.sku_id == sku for l in validated_fo.lines):
            raise FulfilmentGovernanceError(f"Shipment contains unknown SKU '{sku}' not on order.")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.SHIPMENT_DISPATCH,
        actor_type=FulfilmentActorType.SUPPLIER_GATEWAY,
        actor_id=clean_actor,
        timestamp=asn.dispatch_at,
        previous_status=validated_fo.status,
        new_status=FulfilmentOrderStatus.IN_TRANSIT,
        sequence_number=new_seq,
        related_reference_id=asn.shipment_id,
        payload_fingerprint=fingerprint,
        details={"carrier": asn.carrier, "tracking_ref": asn.tracking_reference},
    )

    data = validated_fo.model_dump()
    data["status"] = FulfilmentOrderStatus.IN_TRANSIT.value
    data["lines"] = updated_lines
    data["updated_at"] = asn.dispatch_at
    data["shipment_ids"] = list(validated_fo.shipment_ids) + [asn.shipment_id]
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Shipment dispatch transition failed: {err}") from err


def apply_goods_receipt(
    fo: FulfilmentOrder,
    grn: GoodsReceiptNote,
    actor_id: str,
    idempotency_key: str,
) -> FulfilmentOrder:
    """
    Receiving transition: IN_TRANSIT / PARTIALLY_RECEIVED -> RECEIVED / PARTIALLY_RECEIVED / DELIVERY_EXCEPTION.

    PHYSICAL OVER-DELIVERY SUPPORT:
    - Physical delivered quantity may exceed ASN quantity or outstanding PO balance
      so physical discrepancies can be triaged and audited at dock receiving.
    - Accepted quantity CANNOT exceed outstanding approved order quantity.
    - Excess units are recorded as rejected discrepancies with EXCESS_DELIVERY.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()

    payload = {
        "action": "GOODS_RECEIPT",
        "grn_id": grn.grn_id,
        "fo_id": grn.fo_id,
        "po_id": grn.po_id,
        "shipment_id": grn.shipment_id,
        "store_id": grn.store_id,
        "receipt_at": grn.receipt_at,
        "receiver_id": grn.receiver_id,
        "supervisor_signoff_by": grn.supervisor_signoff_by,
        "supervisor_signoff_at": grn.supervisor_signoff_at,
        "rejected_goods_disposition": grn.rejected_goods_disposition,
        "inspection_lines": [
            {
                "sku_id": l.sku_id,
                "delivered_quantity": l.delivered_quantity,
                "accepted_quantity": l.accepted_quantity,
                "rejected_quantity": l.rejected_quantity,
                "damaged_quantity": l.damaged_quantity,
                "excess_quantity": l.excess_quantity,
                "short_shelf_life_quantity": l.short_shelf_life_quantity,
                "remaining_shelf_life_days": l.remaining_shelf_life_days,
                "total_shelf_life_days": l.total_shelf_life_days,
                "batch_references": list(l.batch_references),
                "discrepancy_reasons": [r.value for r in l.discrepancy_reasons],
            }
            for l in grn.inspection_lines
        ],
    }
    fingerprint = compute_payload_fingerprint(payload)

    # Strong idempotency check
    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.GOODS_RECEIPT, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in TERMINAL_STATUSES:
        raise InvalidFulfilmentTransitionError(
            f"Cannot receive goods for FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    if validated_fo.status not in {
        FulfilmentOrderStatus.IN_TRANSIT,
        FulfilmentOrderStatus.PARTIALLY_RECEIVED,
        FulfilmentOrderStatus.DELIVERY_EXCEPTION,
    }:
        raise InvalidFulfilmentTransitionError(
            f"Cannot receive goods for FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'."
        )

    if (
        grn.fo_id != validated_fo.fo_id
        or grn.po_id != validated_fo.po_id
        or grn.store_id != validated_fo.store_id
        or grn.supplier_id != validated_fo.supplier_id
    ):
        raise FulfilmentGovernanceError(
            f"GRN identifier mismatch: grn={(grn.fo_id, grn.po_id, grn.store_id, grn.supplier_id)}, "
            f"order={(validated_fo.fo_id, validated_fo.po_id, validated_fo.store_id, validated_fo.supplier_id)}."
        )

    if grn.shipment_id not in validated_fo.shipment_ids:
        raise FulfilmentGovernanceError(
            f"GRN references unknown or undispatched shipment '{grn.shipment_id}'. "
            f"Order active shipment IDs: {validated_fo.shipment_ids}."
        )

    validate_timestamp_sequence(validated_fo.updated_at, grn.receipt_at, "updated_at", "receipt_at")

    if grn.grn_id in validated_fo.grn_ids:
        raise FulfilmentGovernanceError(f"Duplicate GRN submission: GRN ID '{grn.grn_id}' has already been processed.")

    grn_lines_by_sku = {line.sku_id: line for line in grn.inspection_lines}
    updated_lines: List[Dict[str, Any]] = []

    for line in validated_fo.lines:
        line_dict = line.model_dump()
        if line.sku_id in grn_lines_by_sku:
            insp = grn_lines_by_sku[line.sku_id]
            new_del = round(line.delivered_quantity + insp.delivered_quantity, 4)
            new_acc = round(line.accepted_quantity + insp.accepted_quantity, 4)
            new_rej = round(line.rejected_quantity + insp.rejected_quantity, 4)

            # Cumulative accepted quantity cannot exceed ordered quantity
            if new_acc > line.ordered_quantity + 1e-4:
                raise FulfilmentGovernanceError(
                    f"[{line.sku_id}] Cumulative accepted quantity ({new_acc}) "
                    f"exceeds approved ordered quantity ({line.ordered_quantity})."
                )

            line_dict["delivered_quantity"] = new_del
            line_dict["accepted_quantity"] = new_acc
            line_dict["rejected_quantity"] = new_rej
        updated_lines.append(line_dict)

    for sku in grn_lines_by_sku:
        if not any(l.sku_id == sku for l in validated_fo.lines):
            raise FulfilmentGovernanceError(f"GRN contains unknown SKU '{sku}' not on order.")

    all_fully_accepted = all(
        abs(l["accepted_quantity"] - l["ordered_quantity"]) < 1e-4 for l in updated_lines
    )
    if all_fully_accepted:
        target_status = FulfilmentOrderStatus.RECEIVED
    else:
        target_status = FulfilmentOrderStatus.PARTIALLY_RECEIVED

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.GOODS_RECEIPT,
        actor_type=FulfilmentActorType.STORE_RECEIVER,
        actor_id=clean_actor,
        timestamp=grn.receipt_at,
        previous_status=validated_fo.status,
        new_status=target_status,
        sequence_number=new_seq,
        related_reference_id=grn.grn_id,
        payload_fingerprint=fingerprint,
        details={
            "grn_id": grn.grn_id,
            "shipment_id": grn.shipment_id,
            "receiving_dock": getattr(grn, "receiving_dock", None) or grn.store_id,
            "accepted_units": grn.total_accepted_quantity,
            "rejected_units": grn.total_rejected_quantity,
            "total_accepted_cost": grn.total_accepted_cost,
            "rejected_goods_disposition": grn.rejected_goods_disposition,
            "inspection_lines": [
                {
                    "sku_id": line.sku_id,
                    "delivered_quantity": line.delivered_quantity,
                    "accepted_quantity": line.accepted_quantity,
                    "rejected_quantity": line.rejected_quantity,
                    "discrepancy_reasons": [r.value for r in line.discrepancy_reasons],
                    "damaged_quantity": line.damaged_quantity,
                    "short_shelf_life_quantity": line.short_shelf_life_quantity,
                    "excess_quantity": line.excess_quantity,
                }
                for line in grn.inspection_lines
            ],
        },
    )

    new_resolved = set(validated_fo.resolved_shipment_ids)
    new_evidence = dict(validated_fo.shipment_resolution_evidence)

    # Calculate cumulative rejected units across past GOODS_RECEIPT events + current GRN for this shipment
    prior_grn_rejected = sum(
        e.details.get("rejected_units", 0.0)
        for e in validated_fo.event_history
        if e.event_type == FulfilmentEventType.GOODS_RECEIPT and e.details.get("shipment_id") == grn.shipment_id
    )
    cum_shipment_rejected = prior_grn_rejected + grn.total_rejected_quantity

    if cum_shipment_rejected < 1e-4:
        # Clean delivery on this shipment with zero rejected units
        new_resolved.add(grn.shipment_id)
    else:
        # Shipment has rejected goods. Custody requires explicit authorized physical disposition.
        if grn.rejected_goods_disposition and grn.rejected_goods_disposition.strip():
            clean_disp = grn.rejected_goods_disposition.strip()
            new_resolved.add(grn.shipment_id)
            new_evidence[grn.shipment_id] = clean_disp
        else:
            # Check if prior resolutions already fully covered all rejected units
            AUTHORIZED_DISPOSITION_TYPES = {
                DiscrepancyResolutionType.RETURN_TO_ORIGIN,
                DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
                DiscrepancyResolutionType.SUPPLIER_CREDIT_NOTE,
                DiscrepancyResolutionType.EXCESS_RETURN,
                DiscrepancyResolutionType.REINSPECTION_RELEASE,
            }
            prior_disp = sum(
                r.quantity for r in validated_fo.discrepancy_resolutions
                if r.shipment_id == grn.shipment_id and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
            ) + sum(
                r.quantity for r in validated_fo.quality_hold_resolutions
                if r.shipment_id == grn.shipment_id and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
            )
            if prior_disp >= cum_shipment_rejected - 1e-4:
                new_resolved.add(grn.shipment_id)
            else:
                # Unresolved rejected inventory remains at store dock
                new_resolved.discard(grn.shipment_id)

    data = validated_fo.model_dump()
    data["status"] = target_status.value
    data["lines"] = updated_lines
    data["updated_at"] = grn.receipt_at
    data["grn_ids"] = list(validated_fo.grn_ids) + [grn.grn_id]
    data["resolved_shipment_ids"] = tuple(sorted(list(new_resolved)))
    data["shipment_resolution_evidence"] = new_evidence
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Goods receipt transition failed: {err}") from err


def record_delivery_exception(
    fo: FulfilmentOrder,
    exception_reason: str,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
) -> FulfilmentOrder:
    """
    Transitions an active order into DELIVERY_EXCEPTION status.
    Preserves resumable_status for resumption upon resolution.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_reason = _validate_non_empty_str(exception_reason, "exception_reason")

    payload = {
        "action": "DELIVERY_EXCEPTION",
        "fo_id": validated_fo.fo_id,
        "exception_reason": clean_reason,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.DELIVERY_EXCEPTION, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in TERMINAL_STATUSES:
        raise InvalidFulfilmentTransitionError(
            f"Cannot raise delivery exception on FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    if validated_fo.status not in {FulfilmentOrderStatus.IN_TRANSIT, FulfilmentOrderStatus.PARTIALLY_RECEIVED}:
        raise InvalidFulfilmentTransitionError(
            f"Cannot raise delivery exception from status '{validated_fo.status.value}'. Must be IN_TRANSIT or PARTIALLY_RECEIVED."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.DELIVERY_EXCEPTION,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=FulfilmentOrderStatus.DELIVERY_EXCEPTION,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={"exception_reason": clean_reason},
    )

    data = validated_fo.model_dump()
    data["status"] = FulfilmentOrderStatus.DELIVERY_EXCEPTION.value
    data["open_exception_reason"] = clean_reason
    data["open_exception_timestamp"] = clean_timestamp
    data["resumable_status"] = validated_fo.status.value
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Delivery exception transition failed: {err}") from err


def resolve_delivery_exception(
    fo: FulfilmentOrder,
    resolution_notes: str,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
    target_status: Optional[FulfilmentOrderStatus] = None,
) -> FulfilmentOrder:
    """
    Resolves an active DELIVERY_EXCEPTION and returns to the recorded resumable status
    or an explicitly validated allowed forward status.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_notes = _validate_non_empty_str(resolution_notes, "resolution_notes")

    # If retrying a past resolution, preserve the status it resolved to so fingerprint matches
    existing_evt = next((e for e in validated_fo.event_history if e.idempotency_key == clean_key), None)
    if existing_evt is not None:
        resolved_status = target_status or existing_evt.new_status
    else:
        resolved_status = target_status or validated_fo.resumable_status or FulfilmentOrderStatus.IN_TRANSIT

    payload = {
        "action": "RESOLVE_EXCEPTION",
        "fo_id": validated_fo.fo_id,
        "resolution_notes": clean_notes,
        "target_status": resolved_status.value,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.RESOLVE_EXCEPTION, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status != FulfilmentOrderStatus.DELIVERY_EXCEPTION:
        raise InvalidFulfilmentTransitionError(
            f"Cannot resolve exception on FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'. "
            "Order must currently be in DELIVERY_EXCEPTION."
        )

    if not is_transition_allowed(FulfilmentOrderStatus.DELIVERY_EXCEPTION, resolved_status):
        raise InvalidFulfilmentTransitionError(
            f"Cannot resolve DELIVERY_EXCEPTION to '{resolved_status.value}'."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.RESOLVE_EXCEPTION,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=resolved_status,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={"resolution_notes": clean_notes, "prior_exception": validated_fo.open_exception_reason},
    )

    data = validated_fo.model_dump()
    data["status"] = resolved_status.value
    data["open_exception_reason"] = None
    data["open_exception_timestamp"] = None
    data["resumable_status"] = None
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Exception resolution failed: {err}") from err


def close_short_fulfilment_order(
    fo: FulfilmentOrder,
    closure_reason: str,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
) -> FulfilmentOrder:
    """
    Managerial terminal closure of unfulfilled balance: PARTIALLY_RECEIVED -> CLOSED_SHORT.
    Enforces that unfulfilled quantities are closed exactly once.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_reason = _validate_non_empty_str(closure_reason, "closure_reason")

    payload = {
        "action": "CLOSE_SHORT",
        "fo_id": validated_fo.fo_id,
        "closure_reason": clean_reason,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    # Strong idempotency check
    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.CLOSE_SHORT, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status == FulfilmentOrderStatus.CLOSED_SHORT:
        raise InvalidFulfilmentTransitionError(
            f"FulfilmentOrder '{validated_fo.fo_id}' is already CLOSED_SHORT. Remaining quantities must be closed exactly once."
        )

    if validated_fo.status in TERMINAL_STATUSES:
        raise InvalidFulfilmentTransitionError(
            f"Cannot close short FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    if validated_fo.status not in {FulfilmentOrderStatus.PARTIALLY_RECEIVED, FulfilmentOrderStatus.DELIVERY_EXCEPTION}:
        raise InvalidFulfilmentTransitionError(
            f"Cannot close short from status '{validated_fo.status.value}'. Must be PARTIALLY_RECEIVED or DELIVERY_EXCEPTION."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    remaining_qty = validated_fo.total_remaining_unfulfilled_qty
    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.CLOSE_SHORT,
        actor_type=FulfilmentActorType.STORE_MANAGER,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=FulfilmentOrderStatus.CLOSED_SHORT,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={"closure_reason": clean_reason, "closed_unfulfilled_qty": remaining_qty},
    )

    data = validated_fo.model_dump()
    data["status"] = FulfilmentOrderStatus.CLOSED_SHORT.value
    data["closed_by"] = clean_actor
    data["closure_timestamp"] = clean_timestamp
    data["closure_reason"] = clean_reason
    data["closed_unfulfilled_qty"] = remaining_qty
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Close short transition failed: {err}") from err


def request_cancellation(
    fo: FulfilmentOrder,
    reason: str,
    category: CancellationReasonCategory,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
) -> FulfilmentOrder:
    """
    Initiates order cancellation request: -> CANCEL_REQUESTED.

    GOVERNANCE RULES:
    1. A PO with already received sellable stock CANNOT be cancelled (must use close_short).
    2. Terminal orders cannot be cancelled.
    3. Partially received orders cannot be cancelled.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_reason = _validate_non_empty_str(reason, "reason")

    payload = {
        "action": "REQUEST_CANCEL",
        "fo_id": validated_fo.fo_id,
        "cancellation_reason": clean_reason,
        "category": category.value,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.REQUEST_CANCEL, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in TERMINAL_STATUSES:
        raise InvalidFulfilmentTransitionError(
            f"Cannot cancel FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    # Invariant: PO with received inventory cannot be cancelled
    if validated_fo.total_received_accepted_qty > 0 or validated_fo.status == FulfilmentOrderStatus.PARTIALLY_RECEIVED:
        raise FulfilmentGovernanceError(
            f"Governance Violation: Cannot cancel FulfilmentOrder '{validated_fo.fo_id}' with "
            f"{validated_fo.total_received_accepted_qty} already received sellable units. "
            "Use close_short_fulfilment_order() instead."
        )

    if not is_transition_allowed(validated_fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED):
        raise InvalidFulfilmentTransitionError(
            f"Cannot request cancellation for order in status '{validated_fo.status.value}'."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.REQUEST_CANCEL,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=FulfilmentOrderStatus.CANCEL_REQUESTED,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={"cancellation_reason": clean_reason, "category": category.value},
    )

    data = validated_fo.model_dump()
    data["status"] = FulfilmentOrderStatus.CANCEL_REQUESTED.value
    data["cancellation_reason"] = clean_reason
    data["cancellation_category"] = category.value
    data["cancelled_by"] = clean_actor
    data["cancellation_timestamp"] = clean_timestamp
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Cancellation request transition failed: {err}") from err


def confirm_cancellation(
    fo: FulfilmentOrder,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
    supplier_cancellation_ref: Optional[str] = None,
    shipment_cancellation_ref: Optional[str] = None,
    shipment_resolutions: Optional[Dict[str, str]] = None,
    order_wide_cancellation_scope: bool = False,
) -> FulfilmentOrder:
    """
    Finalizes cancellation: CANCEL_REQUESTED -> CANCELLED (terminal).

    GOVERNANCE:
    1. If order was transmitted (ISSUED or beyond), supplier_cancellation_ref is required.
    2. If goods are in transit, documented shipment cancellation or return disposition
       ('shipment_cancellation_ref') is required before cancellation can be finalized.
    3. Multiple shipments: An order-wide cancellation reference resolves multiple shipments only
       when its scope explicitly covers all of them (order_wide_cancellation_scope=True).
       Otherwise, resolution evidence must be associated with specific shipment IDs.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_supplier_ref = supplier_cancellation_ref.strip() if supplier_cancellation_ref else None
    clean_shipment_ref = shipment_cancellation_ref.strip() if shipment_cancellation_ref else None
    is_order_wide = order_wide_cancellation_scope or validated_fo.order_wide_cancellation_scope

    payload = {
        "action": "CONFIRM_CANCEL",
        "fo_id": validated_fo.fo_id,
        "supplier_cancellation_ref": clean_supplier_ref,
        "shipment_cancellation_ref": clean_shipment_ref,
        "order_wide_cancellation_scope": is_order_wide,
        "shipment_resolutions": shipment_resolutions,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.CONFIRM_CANCEL, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status != FulfilmentOrderStatus.CANCEL_REQUESTED:
        raise InvalidFulfilmentTransitionError(
            f"Cannot confirm cancellation on FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'. "
            "Order must currently be in CANCEL_REQUESTED."
        )

    # Check transmission status in event history
    was_transmitted = any(
        evt.new_status in {
            FulfilmentOrderStatus.ISSUED,
            FulfilmentOrderStatus.ACKNOWLEDGED,
            FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED,
            FulfilmentOrderStatus.IN_TRANSIT,
        }
        for evt in validated_fo.event_history
    )
    if was_transmitted and not clean_supplier_ref:
        raise FulfilmentGovernanceError(
            f"Governance Violation: Order '{validated_fo.fo_id}' was already transmitted to supplier. "
            "Explicit supplier cancellation confirmation ('supplier_cancellation_ref') is required."
        )

    # Resolve shipments and check coverage
    combined_evidence: Dict[str, str] = dict(validated_fo.shipment_resolution_evidence)
    if shipment_resolutions:
        for s_id, disp_ref in shipment_resolutions.items():
            clean_s_id = _validate_non_empty_str(s_id, "shipment_id in shipment_resolutions")
            clean_disp = _validate_non_empty_str(disp_ref, f"disposition reference for {clean_s_id}")
            if clean_s_id not in validated_fo.shipment_ids:
                raise FulfilmentGovernanceError(
                    f"Shipment resolution provided for unknown shipment ID '{clean_s_id}'."
                )
            combined_evidence[clean_s_id] = clean_disp

    resolved_set = set(validated_fo.resolved_shipment_ids) | set(combined_evidence.keys())
    all_shipment_ids = list(validated_fo.shipment_ids)

    if len(all_shipment_ids) > 1 and clean_shipment_ref and not is_order_wide and not shipment_resolutions:
        raise FulfilmentGovernanceError(
            f"Ambiguous Shipment Resolution: FulfilmentOrder '{validated_fo.fo_id}' has multiple shipments "
            f"{all_shipment_ids}. A single 'shipment_cancellation_ref' cannot be applied without explicit "
            "'order_wide_cancellation_scope=True' or shipment-specific 'shipment_resolutions'."
        )

    if is_order_wide and clean_shipment_ref:
        for s_id in all_shipment_ids:
            if s_id not in combined_evidence:
                combined_evidence[s_id] = clean_shipment_ref
            resolved_set.add(s_id)
    elif len(all_shipment_ids) == 1 and clean_shipment_ref:
        single_id = all_shipment_ids[0]
        if single_id not in combined_evidence:
            combined_evidence[single_id] = clean_shipment_ref
        resolved_set.add(single_id)

    unresolved_shipments = [s_id for s_id in all_shipment_ids if s_id not in resolved_set]

    # 1. Check if any shipments received physical rejected dock inventory without disposition
    rejected_shipments = {
        e.details.get("shipment_id")
        for e in validated_fo.event_history
        if e.event_type == FulfilmentEventType.GOODS_RECEIPT and e.details.get("rejected_units", 0.0) > 0
    }
    unresolved_rejected_shipments = [s_id for s_id in rejected_shipments if s_id not in resolved_set]
    if unresolved_rejected_shipments:
        raise FulfilmentGovernanceError(
            f"Governance Violation: Order '{validated_fo.fo_id}' has physical rejected inventory at store dock "
            f"for shipment(s): {unresolved_rejected_shipments}. Documented physical return or disposal disposition is required before cancellation."
        )

    # 2. Check remaining unresolved in-transit shipments (not yet received at dock)
    received_shipments = {
        e.details.get("shipment_id")
        for e in validated_fo.event_history
        if e.event_type == FulfilmentEventType.GOODS_RECEIPT
    }
    has_in_transit_shipments = any(
        evt.new_status == FulfilmentOrderStatus.IN_TRANSIT or len(validated_fo.shipment_ids) > 0
        for evt in validated_fo.event_history
    )
    in_transit_unresolved = [s_id for s_id in unresolved_shipments if s_id not in received_shipments]
    if has_in_transit_shipments and in_transit_unresolved:
        raise FulfilmentGovernanceError(
            f"Governance Violation: Order '{validated_fo.fo_id}' has active in-transit shipments: {in_transit_unresolved}. "
            "Documented shipment cancellation or return disposition ('shipment_cancellation_ref') is required."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.CONFIRM_CANCEL,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=FulfilmentOrderStatus.CANCELLED,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={
            "supplier_cancellation_ref": clean_supplier_ref,
            "shipment_cancellation_ref": clean_shipment_ref,
            "order_wide_cancellation_scope": is_order_wide,
            "shipment_resolution_evidence": combined_evidence,
        },
    )

    data = validated_fo.model_dump()
    data["status"] = FulfilmentOrderStatus.CANCELLED.value
    data["supplier_cancellation_ref"] = clean_supplier_ref
    data["shipment_cancellation_ref"] = clean_shipment_ref
    data["order_wide_cancellation_scope"] = is_order_wide
    data["shipment_resolution_evidence"] = combined_evidence
    data["resolved_shipment_ids"] = tuple(sorted(list(resolved_set)))
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Cancellation confirmation failed: {err}") from err


def reject_cancellation_request(
    fo: FulfilmentOrder,
    rejection_reason: str,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
    target_status: Optional[FulfilmentOrderStatus] = None,
) -> FulfilmentOrder:
    """
    Transitions order out of CANCEL_REQUESTED back to its previous active status
    (e.g., READY_TO_TRANSMIT, ISSUED, ACKNOWLEDGED, PARTIALLY_ACKNOWLEDGED, IN_TRANSIT, DELIVERY_EXCEPTION)
    when supplier or operations supervisor rejects or withdraws a cancellation request.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_reason = _validate_non_empty_str(rejection_reason, "rejection_reason")

    # If retrying a past rejection, preserve the status it restored to so fingerprint matches
    existing_evt = next((e for e in validated_fo.event_history if e.idempotency_key == clean_key), None)
    if existing_evt is not None:
        restored_status = target_status or existing_evt.new_status
    else:
        # Determine restored status: explicit target_status, or the previous status recorded on the REQUEST_CANCEL event
        restored_status = target_status
        if restored_status is None:
            for evt in reversed(validated_fo.event_history):
                if evt.event_type == FulfilmentEventType.REQUEST_CANCEL and evt.previous_status is not None:
                    restored_status = evt.previous_status
                    break
        if restored_status is None:
            restored_status = FulfilmentOrderStatus.IN_TRANSIT if validated_fo.shipment_ids else FulfilmentOrderStatus.ACKNOWLEDGED

    payload = {
        "action": "REJECT_CANCELLATION",
        "fo_id": validated_fo.fo_id,
        "rejection_reason": clean_reason,
        "target_status": restored_status.value,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    # Strong idempotency check executed BEFORE status validation to ensure durable idempotency on replay
    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.REJECT_CANCEL, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status != FulfilmentOrderStatus.CANCEL_REQUESTED:
        raise InvalidFulfilmentTransitionError(
            f"Cannot reject cancellation on FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'. "
            "Order must currently be in CANCEL_REQUESTED."
        )

    if not is_transition_allowed(FulfilmentOrderStatus.CANCEL_REQUESTED, restored_status):
        raise InvalidFulfilmentTransitionError(
            f"Cannot restore CANCEL_REQUESTED to status '{restored_status.value}'."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.REJECT_CANCEL,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=restored_status,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={
            "rejection_reason": clean_reason,
            "rejection_actor": clean_actor,
            "rejection_timestamp": clean_timestamp,
            "restored_status": restored_status.value,
            "original_cancellation_reason": validated_fo.cancellation_reason,
            "original_cancellation_category": (
                validated_fo.cancellation_category.value if validated_fo.cancellation_category else None
            ),
            "original_cancelled_by": validated_fo.cancelled_by,
            "original_cancellation_timestamp": validated_fo.cancellation_timestamp,
        },
    )

    data = validated_fo.model_dump()
    data["status"] = restored_status.value
    data["cancellation_reason"] = None
    data["cancellation_category"] = None
    data["cancelled_by"] = None
    data["cancellation_timestamp"] = None
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Cancellation rejection transition failed: {err}") from err


def reaffirm_original_terms(
    fo: FulfilmentOrder,
    resolution_notes: str,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
    ack: Optional[SupplierAcknowledgment] = None,
) -> FulfilmentOrder:
    """
    Scenario 1: Original commercial terms retained.
    Supplier confirms it will fulfill the original approved PO despite prior change request.
    Transitions SUPPLIER_CHANGE_PENDING_REVIEW -> ACKNOWLEDGED.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_notes = _validate_non_empty_str(resolution_notes, "resolution_notes")

    payload = {
        "action": "REAFFIRM_ORIGINAL_TERMS",
        "fo_id": validated_fo.fo_id,
        "resolution_notes": clean_notes,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.RESOLVE_EXCEPTION, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status != FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW:
        raise InvalidFulfilmentTransitionError(
            f"Cannot reaffirm terms on FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'. "
            "Order must be in SUPPLIER_CHANGE_PENDING_REVIEW."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    target_status = FulfilmentOrderStatus.ACKNOWLEDGED
    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.RESOLVE_EXCEPTION,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=target_status,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={"resolution_notes": clean_notes, "terms_status": "ORIGINAL_TERMS_RETAINED"},
    )

    updated_lines: List[Dict[str, Any]] = []
    for line in validated_fo.lines:
        line_dict = line.model_dump()
        line_dict["supplier_accepted_quantity"] = line.ordered_quantity
        line_dict["supplier_declined_quantity"] = 0.0
        line_dict["supplier_unconfirmed_quantity"] = 0.0
        updated_lines.append(line_dict)

    data = validated_fo.model_dump()
    data["status"] = target_status.value
    data["lines"] = updated_lines
    data["updated_at"] = clean_timestamp
    if ack and ack.expected_delivery_at:
        data["expected_delivery_at"] = ack.expected_delivery_at
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Reaffirming original terms failed: {err}") from err


def supersede_fulfilment_order(
    original_fo: FulfilmentOrder,
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
) -> Tuple[FulfilmentOrder, FulfilmentOrder]:
    """
    Scenario 2: Commercial terms changed.
    Supplier proposed revised terms (price, quantity, SKU, etc.).

    GOVERNANCE RULES & SAFEGUARDS:
    1. IMMUTABLE COMMERCIAL ANCHOR:
       Original PO is NEVER mutated. Modified terms are NOT accepted under the original PO.
    2. REQUIRED COMMERCIAL APPROVAL:
       Requires newly approved commercial PO ('replacement_po') with verified approval evidence.
    3. MANDATORY SUPPLIER REFUSAL/CANCELLATION EVIDENCE:
       An issued fulfilment order cannot be cancelled and replaced merely because the supplier
       requested changed commercial terms. Explicit auditable supplier confirmation that the
       original terms will not be fulfilled ('supplier_cancellation_ref') is strictly required.
    4. RESOLVED SHIPMENTS & CUSTODY DISPOSITION:
       Block supersession while any shipment is in transit, awaiting receipt, or awaiting return disposition.
       Multiple shipments: A cancellation reference for one shipment must not resolve other active shipments.
       Associate resolution evidence with the specific shipment ID. An order-wide cancellation reference
       may resolve multiple shipments only when its scope explicitly covers all of them (order_wide_cancellation_scope=True).
    5. REJECTED GOODS DISPOSITION:
       A GRN with zero accepted quantity must not automatically mean physical custody is resolved.
       Require documented return, disposal, or another authorized final disposition of rejected goods
       before treating the shipment as fully resolved for supersession.
    6. ZERO SELLABLE RECEIVED STOCK:
       Permit supersession only when NO sellable goods have been accepted into store inventory.
    7. AUDIT TRAIL PRESERVATION:
       Historical shipment IDs, event history, lines, and receipt records are never erased.
       Original FulfilmentOrder is closed as CANCELLED (superseded). A new FulfilmentOrder is
       minted for the replacement PO with full bidirectional cross-audit linkage.
    """
    validated_fo = validate_inbound_fulfilment_order(original_fo)
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_reason = _validate_non_empty_str(reason, "reason")
    clean_supplier_ref = supplier_cancellation_ref.strip() if supplier_cancellation_ref and supplier_cancellation_ref.strip() else None
    clean_shipment_ref = (
        shipment_cancellation_ref.strip()
        if shipment_cancellation_ref and shipment_cancellation_ref.strip()
        else (validated_fo.shipment_cancellation_ref if validated_fo.order_wide_cancellation_scope else None)
    )
    is_order_wide = order_wide_cancellation_scope or validated_fo.order_wide_cancellation_scope

    payload = {
        "action": "SUPERSEDE_ORDER",
        "original_fo_id": validated_fo.fo_id,
        "replacement_po_id": replacement_po.po_id,
        "replacement_fo_id": replacement_fo_id,
        "supplier_cancellation_ref": clean_supplier_ref,
        "shipment_cancellation_ref": clean_shipment_ref,
        "order_wide_cancellation_scope": is_order_wide,
        "shipment_resolutions": shipment_resolutions,
        "reason": clean_reason,
        "timestamp": clean_timestamp,
        "actor_id": clean_actor,
    }
    fingerprint = compute_payload_fingerprint(payload)

    # Strong idempotency check
    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.CONFIRM_CANCEL, fingerprint)
    if existing:
        replacement_fo = create_fulfilment_order(
            po=replacement_po,
            actor_id=clean_actor,
            timestamp=clean_timestamp,
            fo_id=replacement_fo_id,
            idempotency_key=f"CREATE:{replacement_fo_id}:1",
            supersedes_fo_id=validated_fo.fo_id,
            supersedes_po_id=validated_fo.po_id,
        )
        return validated_fo, replacement_fo

    if validated_fo.status != FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW:
        raise InvalidFulfilmentTransitionError(
            f"Cannot supersede FulfilmentOrder '{validated_fo.fo_id}' in status '{validated_fo.status.value}'. "
            "Order must be in SUPPLIER_CHANGE_PENDING_REVIEW."
        )

    # SAFEGUARD 1: Require auditable supplier confirmation that original terms will not be fulfilled
    if not clean_supplier_ref:
        raise FulfilmentGovernanceError(
            f"Governance Violation: Cannot supersede FulfilmentOrder '{validated_fo.fo_id}'. "
            "An issued fulfilment order cannot be cancelled and replaced merely because the supplier "
            "requested changed commercial terms. Auditable supplier confirmation that the original "
            "terms will not be fulfilled ('supplier_cancellation_ref') is strictly required."
        )

    # SAFEGUARD 2: Zero sellable received stock invariant
    # Once sellable inventory has been accepted into dark-store stock, supersession is permanently forbidden.
    if validated_fo.total_received_accepted_qty > 0:
        raise FulfilmentGovernanceError(
            f"Governance Violation: Cannot supersede FulfilmentOrder '{validated_fo.fo_id}' with "
            f"{validated_fo.total_received_accepted_qty} already received sellable units. "
            "Orders with received sellable inventory cannot be cancelled; use close_short_fulfilment_order() instead."
        )

    # SAFEGUARD 3: Distinguish active (unresolved) shipments from historical resolved/returned shipments
    combined_evidence: Dict[str, str] = dict(validated_fo.shipment_resolution_evidence)

    if shipment_resolutions:
        for s_id, disp_ref in shipment_resolutions.items():
            clean_s_id = _validate_non_empty_str(s_id, "shipment_id in shipment_resolutions")
            clean_disp = _validate_non_empty_str(disp_ref, f"disposition reference for {clean_s_id}")
            if clean_s_id not in validated_fo.shipment_ids:
                raise FulfilmentGovernanceError(
                    f"Shipment resolution provided for unknown shipment ID '{clean_s_id}'."
                )
            combined_evidence[clean_s_id] = clean_disp

    resolved_set = set(validated_fo.resolved_shipment_ids) | set(combined_evidence.keys())
    all_shipment_ids = list(validated_fo.shipment_ids)

    # Ambiguous multiple shipments guard:
    if len(all_shipment_ids) > 1 and clean_shipment_ref and not is_order_wide and not shipment_resolutions:
        raise FulfilmentGovernanceError(
            f"Ambiguous Shipment Resolution: FulfilmentOrder '{validated_fo.fo_id}' has multiple shipments "
            f"{all_shipment_ids}. A single 'shipment_cancellation_ref' cannot be applied without explicit "
            "'order_wide_cancellation_scope=True' or shipment-specific 'shipment_resolutions'."
        )

    # Multiple shipments vs single shipment scoping:
    # 1. Order-wide cancellation reference: explicitly covers all shipments on the order
    if is_order_wide and clean_shipment_ref:
        for s_id in all_shipment_ids:
            if s_id not in combined_evidence:
                combined_evidence[s_id] = clean_shipment_ref
            resolved_set.add(s_id)
    # 2. Single shipment: cancellation reference applies to that specific shipment
    elif len(all_shipment_ids) == 1 and clean_shipment_ref:
        single_id = all_shipment_ids[0]
        if single_id not in combined_evidence:
            combined_evidence[single_id] = clean_shipment_ref
        resolved_set.add(single_id)
    # 3. Multiple shipments without order-wide scope:
    # A cancellation reference for one shipment must not resolve other active shipments.
    # Associate resolution evidence with specific shipment IDs.

    unresolved_shipments = [s_id for s_id in all_shipment_ids if s_id not in resolved_set]
    if unresolved_shipments:
        raise FulfilmentGovernanceError(
            f"Governance Violation: Cannot supersede FulfilmentOrder '{validated_fo.fo_id}'. "
            f"Shipment(s) {unresolved_shipments} are currently in transit, awaiting receipt, or awaiting "
            "return disposition. Active shipments must have verified dock receipt or documented return "
            "disposition ('shipment_cancellation_ref') before the order can be superseded."
        )

    # SAFEGUARD 4: Rejected goods custody disposition check
    # A GRN with zero accepted quantity must not automatically mean physical custody is resolved.
    # Require documented return, disposal, or another authorized final disposition of rejected goods.
    if any(line.rejected_quantity > 0 for line in validated_fo.lines):
        for s_id in all_shipment_ids:
            if s_id not in combined_evidence:
                raise FulfilmentGovernanceError(
                    f"Governance Violation: Cannot supersede FulfilmentOrder '{validated_fo.fo_id}'. "
                    f"Shipment '{s_id}' has rejected goods at dock receiving without documented return, "
                    "disposal, or authorized final disposition."
                )

    if not can_be_dispatched(replacement_po):
        raise FulfilmentGovernanceError(
            f"Replacement Purchase Order '{replacement_po.po_id}' must be legally APPROVED with verified approval evidence."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    # 1. Supersede and cancel original fulfilment order
    new_seq = len(validated_fo.event_history) + 1
    cancel_event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.CONFIRM_CANCEL,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=FulfilmentOrderStatus.CANCELLED,
        sequence_number=new_seq,
        payload_fingerprint=fingerprint,
        details={
            "superseded_by_po_id": replacement_po.po_id,
            "superseded_by_fo_id": replacement_fo_id,
            "supplier_cancellation_ref": clean_supplier_ref,
            "shipment_cancellation_ref": clean_shipment_ref,
            "order_wide_cancellation_scope": is_order_wide,
            "shipment_resolution_evidence": combined_evidence,
            "reason": clean_reason,
        },
    )

    orig_data = validated_fo.model_dump()
    orig_data["status"] = FulfilmentOrderStatus.CANCELLED.value
    orig_data["cancellation_reason"] = f"Superseded by replacement PO '{replacement_po.po_id}': {clean_reason}"
    orig_data["cancellation_category"] = CancellationReasonCategory.COMMERCIAL_TERMS_REJECTED.value
    orig_data["cancelled_by"] = clean_actor
    orig_data["cancellation_timestamp"] = clean_timestamp
    orig_data["supplier_cancellation_ref"] = clean_supplier_ref
    orig_data["shipment_cancellation_ref"] = clean_shipment_ref
    orig_data["order_wide_cancellation_scope"] = is_order_wide
    orig_data["shipment_resolution_evidence"] = combined_evidence
    orig_data["resolved_shipment_ids"] = tuple(sorted(list(resolved_set)))
    orig_data["updated_at"] = clean_timestamp
    orig_data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [cancel_event.model_dump()]

    superseded_fo = FulfilmentOrder.model_validate(orig_data)

    # 2. Mint new replacement FulfilmentOrder with cross-linkage
    replacement_fo = create_fulfilment_order(
        po=replacement_po,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        fo_id=replacement_fo_id,
        idempotency_key=f"CREATE:{replacement_fo_id}:1",
        supersedes_fo_id=validated_fo.fo_id,
        supersedes_po_id=validated_fo.po_id,
    )

    return superseded_fo, replacement_fo


def resolve_goods_receipt_discrepancy(
    fo: FulfilmentOrder,
    shipment_id: str,
    sku_id: str,
    resolution_type: DiscrepancyResolutionType,
    quantity: float,
    discrepancy_reason: DiscrepancyReason,
    disposition_reference: str,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
    notes: Optional[str] = None,
) -> FulfilmentOrder:
    """
    Records an authorized discrepancy or rejected-goods resolution for a specific shipment and SKU.
    Supports return-to-origin (RTO), certified disposal, supplier credit notes, excess return,
    and acknowledged short delivery.

    Governance Rules:
    1. Rejects terminal orders (CANCELLED, REJECTED_BY_SUPPLIER).
    2. Verifies shipment_id is associated with the order.
    3. Verifies sku_id belongs to the order lines.
    4. Verified quantity > 0.
    5. Prevents double-resolution: sum of resolved quantities for this (shipment_id, sku_id)
       cannot exceed the recorded rejected quantity for that line.
    6. Verifies mandatory disposition_reference.
    7. Idempotent: exact retries with identical payload fingerprint return fo unmodified.
    8. Appends immutable RESOLVE_DISCREPANCY audit event.
    9. If all rejected units on the shipment are now resolved, marks shipment custody as resolved.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_shipment = shipment_id.strip()
    clean_sku = sku_id.strip()
    clean_disp = disposition_reference.strip()
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_notes = notes.strip() if notes and notes.strip() else None

    if not clean_disp:
        raise FulfilmentGovernanceError("disposition_reference must be a non-empty string.")
    if quantity <= 0.0:
        raise FulfilmentGovernanceError(f"Resolved quantity must be strictly positive; got {quantity}.")

    payload = {
        "action": "RESOLVE_DISCREPANCY",
        "fo_id": validated_fo.fo_id,
        "shipment_id": clean_shipment,
        "sku_id": clean_sku,
        "resolution_type": resolution_type.value,
        "quantity": quantity,
        "discrepancy_reason": discrepancy_reason.value,
        "disposition_reference": clean_disp,
        "actor_id": clean_actor,
        "timestamp": clean_timestamp,
        "notes": clean_notes,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.RESOLVE_DISCREPANCY, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in {FulfilmentOrderStatus.CANCELLED, FulfilmentOrderStatus.REJECTED_BY_SUPPLIER}:
        raise InvalidFulfilmentTransitionError(
            f"Cannot resolve discrepancy on FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    if clean_shipment not in validated_fo.shipment_ids:
        raise FulfilmentGovernanceError(
            f"Shipment '{clean_shipment}' is not associated with FulfilmentOrder '{validated_fo.fo_id}'. "
            f"Order shipment IDs: {validated_fo.shipment_ids}."
        )

    line = next((l for l in validated_fo.lines if l.sku_id == clean_sku), None)
    if not line:
        raise FulfilmentGovernanceError(f"SKU '{clean_sku}' does not exist on FulfilmentOrder '{validated_fo.fo_id}'.")

    AUTHORIZED_DISPOSITION_TYPES = {
        DiscrepancyResolutionType.RETURN_TO_ORIGIN,
        DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
        DiscrepancyResolutionType.SUPPLIER_CREDIT_NOTE,
        DiscrepancyResolutionType.EXCESS_RETURN,
        DiscrepancyResolutionType.REINSPECTION_RELEASE,
    }

    # Double-resolution defense: prevent over-resolution within the respective category
    if resolution_type in AUTHORIZED_DISPOSITION_TYPES:
        prior_resolved_qty = sum(
            r.quantity for r in validated_fo.discrepancy_resolutions
            if r.shipment_id == clean_shipment and r.sku_id == clean_sku and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
        ) + sum(
            r.quantity for r in validated_fo.quality_hold_resolutions
            if r.shipment_id == clean_shipment and r.sku_id == clean_sku and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
        )
        if round(prior_resolved_qty + quantity, 4) > line.rejected_quantity + 1e-4:
            raise FulfilmentGovernanceError(
                f"Double-Resolution Violation: Cannot resolve {quantity} units for SKU '{clean_sku}' "
                f"on shipment '{clean_shipment}'. Prior resolved: {prior_resolved_qty}, total rejected: {line.rejected_quantity}."
            )
    elif resolution_type == DiscrepancyResolutionType.SHORT_DELIVERY_ACKNOWLEDGED:
        prior_comm_qty = sum(
            r.quantity for r in validated_fo.discrepancy_resolutions
            if r.shipment_id == clean_shipment and r.sku_id == clean_sku and r.resolution_type == DiscrepancyResolutionType.SHORT_DELIVERY_ACKNOWLEDGED
        )
        max_short = max(0.0, line.ordered_quantity - line.delivered_quantity)
        if round(prior_comm_qty + quantity, 4) > max_short + 1e-4:
            raise FulfilmentGovernanceError(
                f"Double-Resolution Violation: Cannot acknowledge short delivery of {quantity} units for SKU '{clean_sku}'. "
                f"Prior short acknowledged: {prior_comm_qty}, total short quantity: {max_short}."
            )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    resolution_id = f"RES-{validated_fo.fo_id}-{len(validated_fo.discrepancy_resolutions) + 1:03d}"
    resolution = DiscrepancyResolution(
        resolution_id=resolution_id,
        fo_id=validated_fo.fo_id,
        shipment_id=clean_shipment,
        sku_id=clean_sku,
        resolution_type=resolution_type,
        quantity=quantity,
        discrepancy_reason=discrepancy_reason,
        disposition_reference=clean_disp,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        notes=clean_notes,
    )

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.RESOLVE_DISCREPANCY,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=validated_fo.status,
        sequence_number=new_seq,
        related_reference_id=clean_disp,
        payload_fingerprint=fingerprint,
        details={
            "resolution_id": resolution_id,
            "shipment_id": clean_shipment,
            "sku_id": clean_sku,
            "resolution_type": resolution_type.value,
            "quantity": quantity,
            "disposition_reference": clean_disp,
        },
    )

    new_resolutions = list(validated_fo.discrepancy_resolutions) + [resolution]
    new_resolved_shipments = set(validated_fo.resolved_shipment_ids)
    new_evidence = dict(validated_fo.shipment_resolution_evidence)

    # Custody / discrepancy resolution calculation:
    shipment_total_rejected = sum(
        e.details.get("rejected_units", 0.0)
        for e in validated_fo.event_history
        if e.event_type == FulfilmentEventType.GOODS_RECEIPT and e.details.get("shipment_id") == clean_shipment
    )

    all_resolutions = new_resolutions + list(validated_fo.quality_hold_resolutions)
    shipment_disp_qty = sum(
        r.quantity for r in all_resolutions
        if r.shipment_id == clean_shipment and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
    )

    if shipment_total_rejected < 1e-4:
        new_resolved_shipments.add(clean_shipment)
    elif shipment_disp_qty >= shipment_total_rejected - 1e-4:
        new_resolved_shipments.add(clean_shipment)
        new_evidence[clean_shipment] = f"Custody resolved: {clean_disp}"
    else:
        # Partial disposition or unresolved rejected goods do not resolve store dock custody
        new_resolved_shipments.discard(clean_shipment)

    data = validated_fo.model_dump()
    data["discrepancy_resolutions"] = [r.model_dump() for r in new_resolutions]
    data["resolved_shipment_ids"] = tuple(sorted(list(new_resolved_shipments)))
    data["shipment_resolution_evidence"] = new_evidence
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Discrepancy resolution failed: {err}") from err


def resolve_quality_hold(
    fo: FulfilmentOrder,
    shipment_id: str,
    sku_id: str,
    resolution_type: DiscrepancyResolutionType,
    quantity: float,
    disposition_reference: str,
    inspector_id: str,
    timestamp: str,
    idempotency_key: str,
    inspection_notes: str,
    verified_remaining_shelf_life_days: Optional[int] = None,
    total_shelf_life_days: Optional[int] = None,
) -> FulfilmentOrder:
    """
    Executes an authorized quality hold resolution or reinspection disposition.
    Supports REINSPECTION_RELEASE (releasing held stock into sellable inventory),
    CERTIFIED_DISPOSAL, and RETURN_TO_ORIGIN.

    Governance Rules:
    1. REINSPECTION_RELEASE requires verified_remaining_shelf_life_days.
    2. If total_shelf_life_days is provided, verified_remaining_shelf_life_days / total_shelf_life_days >= 0.60 (60%).
       Missing or failing shelf-life ratio strictly raises FulfilmentGovernanceError.
    3. Releasing stock increments line accepted_quantity and decrements line rejected_quantity.
       If all lines become fully accepted, order transitions to RECEIVED.
    4. Prevents double-resolution: cannot resolve more than line.rejected_quantity.
    5. Appends immutable RESOLVE_QUALITY_HOLD audit event.
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_key = idempotency_key.strip()
    clean_inspector = inspector_id.strip()
    clean_shipment = shipment_id.strip()
    clean_sku = sku_id.strip()
    clean_disp = disposition_reference.strip()
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_notes = inspection_notes.strip()

    if not clean_disp:
        raise FulfilmentGovernanceError("disposition_reference must be a non-empty string.")
    if not clean_notes:
        raise FulfilmentGovernanceError("inspection_notes must be a non-empty string.")
    if quantity <= 0.0:
        raise FulfilmentGovernanceError(f"Resolved quantity must be strictly positive; got {quantity}.")

    payload = {
        "action": "RESOLVE_QUALITY_HOLD",
        "fo_id": validated_fo.fo_id,
        "shipment_id": clean_shipment,
        "sku_id": clean_sku,
        "resolution_type": resolution_type.value,
        "quantity": quantity,
        "disposition_reference": clean_disp,
        "inspector_id": clean_inspector,
        "timestamp": clean_timestamp,
        "verified_remaining_shelf_life_days": verified_remaining_shelf_life_days,
        "total_shelf_life_days": total_shelf_life_days,
        "inspection_notes": clean_notes,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.RESOLVE_QUALITY_HOLD, fingerprint)
    if existing:
        return validated_fo

    if validated_fo.status in {FulfilmentOrderStatus.CANCELLED, FulfilmentOrderStatus.REJECTED_BY_SUPPLIER}:
        raise InvalidFulfilmentTransitionError(
            f"Cannot resolve quality hold on FulfilmentOrder '{validated_fo.fo_id}' in terminal status '{validated_fo.status.value}'."
        )

    if clean_shipment not in validated_fo.shipment_ids:
        raise FulfilmentGovernanceError(
            f"Shipment '{clean_shipment}' is not associated with FulfilmentOrder '{validated_fo.fo_id}'."
        )

    line = next((l for l in validated_fo.lines if l.sku_id == clean_sku), None)
    if not line:
        raise FulfilmentGovernanceError(f"SKU '{clean_sku}' does not exist on FulfilmentOrder '{validated_fo.fo_id}'.")

    AUTHORIZED_DISPOSITION_TYPES = {
        DiscrepancyResolutionType.RETURN_TO_ORIGIN,
        DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
        DiscrepancyResolutionType.SUPPLIER_CREDIT_NOTE,
        DiscrepancyResolutionType.EXCESS_RETURN,
        DiscrepancyResolutionType.REINSPECTION_RELEASE,
    }

    # Double-resolution check: cannot resolve more than line.rejected_quantity
    prior_resolved_qty = sum(
        r.quantity for r in validated_fo.discrepancy_resolutions
        if r.shipment_id == clean_shipment and r.sku_id == clean_sku and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
    ) + sum(
        r.quantity for r in validated_fo.quality_hold_resolutions
        if r.shipment_id == clean_shipment and r.sku_id == clean_sku and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
    )

    if round(prior_resolved_qty + quantity, 4) > line.rejected_quantity + 1e-4:
        raise FulfilmentGovernanceError(
            f"Double-Resolution Violation: Cannot resolve {quantity} units for SKU '{clean_sku}' "
            f"on shipment '{clean_shipment}'. Prior resolutions: {prior_resolved_qty}, total rejected: {line.rejected_quantity}."
        )

    # Shelf-life and quality-hold verification for REINSPECTION_RELEASE
    if resolution_type == DiscrepancyResolutionType.REINSPECTION_RELEASE:
        if verified_remaining_shelf_life_days is None:
            raise FulfilmentGovernanceError(
                f"[{clean_sku}] Quality Hold Violation: Reinspection release requires verified_remaining_shelf_life_days."
            )
        if total_shelf_life_days is not None:
            if total_shelf_life_days <= 0:
                raise FulfilmentGovernanceError(
                    f"[{clean_sku}] Quality Hold Violation: total_shelf_life_days must be strictly positive; got {total_shelf_life_days}."
                )
            if verified_remaining_shelf_life_days < 0:
                raise FulfilmentGovernanceError(
                    f"[{clean_sku}] Quality Hold Violation: verified_remaining_shelf_life_days cannot be negative; got {verified_remaining_shelf_life_days}."
                )
            if verified_remaining_shelf_life_days > total_shelf_life_days:
                raise FulfilmentGovernanceError(
                    f"[{clean_sku}] Quality Hold Violation: verified_remaining_shelf_life_days ({verified_remaining_shelf_life_days}) "
                    f"cannot exceed total_shelf_life_days ({total_shelf_life_days})."
                )
            ratio = verified_remaining_shelf_life_days / total_shelf_life_days
            if ratio < 0.60 - 1e-4:
                raise FulfilmentGovernanceError(
                    f"[{clean_sku}] Quality Hold Violation: Remaining shelf-life ratio {ratio:.1%} "
                    f"({verified_remaining_shelf_life_days}/{total_shelf_life_days} days) is below the minimum "
                    "acceptable threshold of 60.0%. Stock cannot be released into sellable inventory."
                )

        # Inspect event history to verify quantity held on quality/shelf-life hold (transit damage and excess cannot be released)
        has_inspection_breakdown = False
        eligible_quality_qty = 0.0
        for e in validated_fo.event_history:
            if e.event_type == FulfilmentEventType.GOODS_RECEIPT and e.details.get("shipment_id") == clean_shipment:
                ilines = e.details.get("inspection_lines", [])
                if ilines:
                    has_inspection_breakdown = True
                    for il in ilines:
                        if il.get("sku_id") == clean_sku:
                            dmg = il.get("damaged_quantity", 0.0)
                            exc = il.get("excess_quantity", 0.0)
                            rej = il.get("rejected_quantity", 0.0)
                            ssl = il.get("short_shelf_life_quantity", 0.0)
                            reasons = il.get("discrepancy_reasons", [])
                            if ssl > 0:
                                eligible_quality_qty += ssl
                            elif any(r in ("EXPIRED_OR_SHORT_SHELF_LIFE", "QUALITY_DEFECT") for r in reasons):
                                eligible_quality_qty += max(0.0, rej - (dmg + exc))

        if has_inspection_breakdown:
            prior_released = sum(
                r.quantity for r in validated_fo.quality_hold_resolutions
                if r.shipment_id == clean_shipment and r.sku_id == clean_sku and r.resolution_type == DiscrepancyResolutionType.REINSPECTION_RELEASE
            )
            if round(prior_released + quantity, 4) > eligible_quality_qty + 1e-4:
                raise FulfilmentGovernanceError(
                    f"[{clean_sku}] Reinspection Release Violation: Cannot release {quantity} units into sellable inventory. "
                    f"Only {eligible_quality_qty} units were held on quality/shelf-life hold (transit damage and unapproved excess cannot be released). "
                    f"Prior released: {prior_released}."
                )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    resolution_id = f"QHR-{validated_fo.fo_id}-{len(validated_fo.quality_hold_resolutions) + 1:03d}"
    try:
        q_res = QualityHoldResolution(
            resolution_id=resolution_id,
            fo_id=validated_fo.fo_id,
            shipment_id=clean_shipment,
            sku_id=clean_sku,
            resolution_type=resolution_type,
            quantity=quantity,
            disposition_reference=clean_disp,
            inspector_id=clean_inspector,
            timestamp=clean_timestamp,
            verified_remaining_shelf_life_days=verified_remaining_shelf_life_days,
            total_shelf_life_days=total_shelf_life_days,
            inspection_notes=clean_notes,
        )
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Quality hold resolution validation failed: {err}") from err

    updated_lines = []
    target_status = validated_fo.status
    for l in validated_fo.lines:
        l_dict = l.model_dump()
        if l.sku_id == clean_sku:
            if resolution_type == DiscrepancyResolutionType.REINSPECTION_RELEASE:
                new_acc = round(l.accepted_quantity + quantity, 4)
                new_rej = max(0.0, round(l.rejected_quantity - quantity, 4))
                if new_acc > l.ordered_quantity + 1e-4:
                    raise FulfilmentGovernanceError(
                        f"[{clean_sku}] Reinspection release would cause accepted quantity ({new_acc}) "
                        f"to exceed ordered quantity ({l.ordered_quantity})."
                    )
                l_dict["accepted_quantity"] = new_acc
                l_dict["rejected_quantity"] = new_rej
        updated_lines.append(l_dict)

    if resolution_type == DiscrepancyResolutionType.REINSPECTION_RELEASE:
        all_fully_accepted = all(
            abs(l["accepted_quantity"] - l["ordered_quantity"]) < 1e-4 for l in updated_lines
        )
        if all_fully_accepted and validated_fo.status == FulfilmentOrderStatus.PARTIALLY_RECEIVED:
            target_status = FulfilmentOrderStatus.RECEIVED

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.RESOLVE_QUALITY_HOLD,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_inspector,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=target_status,
        sequence_number=new_seq,
        related_reference_id=clean_disp,
        payload_fingerprint=fingerprint,
        details={
            "resolution_id": resolution_id,
            "shipment_id": clean_shipment,
            "sku_id": clean_sku,
            "resolution_type": resolution_type.value,
            "quantity": quantity,
            "disposition_reference": clean_disp,
            "verified_shelf_life_days": verified_remaining_shelf_life_days,
        },
    )

    new_qhrs = list(validated_fo.quality_hold_resolutions) + [q_res]
    new_resolved_shipments = set(validated_fo.resolved_shipment_ids)
    new_evidence = dict(validated_fo.shipment_resolution_evidence)

    # Custody / discrepancy resolution calculation:
    shipment_total_rejected = sum(
        e.details.get("rejected_units", 0.0)
        for e in validated_fo.event_history
        if e.event_type == FulfilmentEventType.GOODS_RECEIPT and e.details.get("shipment_id") == clean_shipment
    )

    all_resolutions = list(validated_fo.discrepancy_resolutions) + new_qhrs
    shipment_disp_qty = sum(
        r.quantity for r in all_resolutions
        if r.shipment_id == clean_shipment and r.resolution_type in AUTHORIZED_DISPOSITION_TYPES
    )

    if shipment_total_rejected < 1e-4:
        new_resolved_shipments.add(clean_shipment)
    elif shipment_disp_qty >= shipment_total_rejected - 1e-4:
        new_resolved_shipments.add(clean_shipment)
        new_evidence[clean_shipment] = f"Custody resolved: {clean_disp}"
    else:
        new_resolved_shipments.discard(clean_shipment)

    data = validated_fo.model_dump()
    data["status"] = target_status.value
    data["lines"] = updated_lines
    data["quality_hold_resolutions"] = [r.model_dump() for r in new_qhrs]
    data["resolved_shipment_ids"] = tuple(sorted(list(new_resolved_shipments)))
    data["shipment_resolution_evidence"] = new_evidence
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Quality hold resolution failed: {err}") from err


def resolve_shipment_custody(
    fo: FulfilmentOrder,
    shipment_id: str,
    disposition_reference: str,
    actor_id: str,
    timestamp: str,
    idempotency_key: str,
    notes: Optional[str] = None,
) -> FulfilmentOrder:
    """
    Explicitly records physical custody resolution for an entire shipment.
    Requires verifiable disposition reference (e.g. carrier bill of lading, return receipt).
    """
    validated_fo = validate_inbound_fulfilment_order(fo)
    clean_key = idempotency_key.strip()
    clean_actor = actor_id.strip()
    clean_shipment = shipment_id.strip()
    clean_disp = disposition_reference.strip()
    clean_timestamp = validate_and_normalize_iso_timestamp(timestamp, "timestamp")
    clean_notes = notes.strip() if notes and notes.strip() else None

    if not clean_disp:
        raise FulfilmentGovernanceError("disposition_reference must be a non-empty string.")

    payload = {
        "action": "RESOLVE_SHIPMENT_CUSTODY",
        "fo_id": validated_fo.fo_id,
        "shipment_id": clean_shipment,
        "disposition_reference": clean_disp,
        "actor_id": clean_actor,
        "timestamp": clean_timestamp,
        "notes": clean_notes,
    }
    fingerprint = compute_payload_fingerprint(payload)

    existing = _verify_idempotency(validated_fo.event_history, clean_key, FulfilmentEventType.RESOLVE_SHIPMENT_CUSTODY, fingerprint)
    if existing:
        return validated_fo

    if clean_shipment not in validated_fo.shipment_ids:
        raise FulfilmentGovernanceError(
            f"Shipment '{clean_shipment}' is not associated with FulfilmentOrder '{validated_fo.fo_id}'."
        )

    validate_timestamp_sequence(validated_fo.updated_at, clean_timestamp, "updated_at", "timestamp")

    new_seq = len(validated_fo.event_history) + 1
    event = FulfilmentEvent(
        event_id=f"EVT-{validated_fo.fo_id}-{new_seq:03d}",
        idempotency_key=clean_key,
        fo_id=validated_fo.fo_id,
        event_type=FulfilmentEventType.RESOLVE_SHIPMENT_CUSTODY,
        actor_type=FulfilmentActorType.OPERATIONS_SUPERVISOR,
        actor_id=clean_actor,
        timestamp=clean_timestamp,
        previous_status=validated_fo.status,
        new_status=validated_fo.status,
        sequence_number=new_seq,
        related_reference_id=clean_disp,
        payload_fingerprint=fingerprint,
        details={
            "shipment_id": clean_shipment,
            "disposition_reference": clean_disp,
            "notes": clean_notes,
        },
    )

    new_resolved = set(validated_fo.resolved_shipment_ids) | {clean_shipment}
    new_evidence = dict(validated_fo.shipment_resolution_evidence)
    new_evidence[clean_shipment] = clean_disp

    data = validated_fo.model_dump()
    data["resolved_shipment_ids"] = tuple(sorted(list(new_resolved)))
    data["shipment_resolution_evidence"] = new_evidence
    data["updated_at"] = clean_timestamp
    data["event_history"] = [e.model_dump() for e in validated_fo.event_history] + [event.model_dump()]

    try:
        return FulfilmentOrder.model_validate(data)
    except ValidationError as err:
        raise FulfilmentGovernanceError(f"Shipment custody resolution failed: {err}") from err
