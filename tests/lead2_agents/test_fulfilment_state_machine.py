"""
Unit & Integration Tests: Fulfilment State Machine & Transition Engine
=====================================================================
Phase 5.2 Verification Tests:
39. Valid transitions across the complete lifecycle.
40. Unapproved transmission rejected.
41. Invalid direct status transitions rejected.
42. model_copy(update=...) bypass rejected at workflow boundaries.
43. Duplicate transmission event prevented by idempotency contract.
44. Repeated partial receipt handled without duplicate quantity posting.
45. Terminal state immutability.
46. Invalid event ordering rejected.
47. Cancellation without reason rejected.
48. Cancellation after receipt rejected.
49. Cancellation after transmission without supplier confirmation rejected.
50. Delivery exception resolution requires matching open exception.
51. Supplier rejection does not trigger automatic backup selection.
52. Supplier change request cannot alter original approved PO.
53. Audit event sequence and timestamp integrity.
54. Close short terminal transition from PARTIALLY_RECEIVED.
55. Exception resumption to recorded resumable status.
"""

import unittest
from pydantic import ValidationError

from src.agents.fulfilment import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DiscrepancyReason,
    DockInspectionLine,
    FulfilmentActorType,
    FulfilmentGovernanceError,
    FulfilmentOrder,
    FulfilmentOrderStatus,
    GoodsReceiptNote,
    InvalidFulfilmentTransitionError,
    ShipmentLineItem,
    SupplierAcknowledgment,
    SupplierAcknowledgmentLine,
    SupplierAcknowledgmentStatus,
    apply_goods_receipt,
    apply_shipment_dispatch,
    apply_supplier_acknowledgment,
    close_short_fulfilment_order,
    confirm_cancellation,
    create_fulfilment_order,
    is_transition_allowed,
    reaffirm_original_terms,
    record_delivery_exception,
    request_cancellation,
    resolve_delivery_exception,
    supersede_fulfilment_order,
    transmit_fulfilment_order,
    validate_inbound_fulfilment_order,
)
from src.agents.procurement.approval import approve_purchase_order
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
)


def _make_approved_po(
    store_id: str = "DS_BLR_KORAMANGALA",
    sku_id: str = "SKU_MILK_1L",
    category: str = "Dairy",
    final_order_qty: float = 24.0,
    unit_procurement_cost: float = 30.0,
    po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
    approval_timestamp: str = "2024-11-15T10:30:00Z",
) -> PurchaseOrder:
    """Constructs a valid APPROVED PurchaseOrder with verified approval evidence."""
    item = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id=sku_id,
        category=category,
        raw_suggested_qty=20.0,
        case_pack_rounded_qty=24.0,
        final_order_qty=final_order_qty,
        case_pack_size=12,
        moq=10.0,
        unit_retail_price=50.0,
        unit_procurement_cost=unit_procurement_cost,
        total_line_cost=round(final_order_qty * unit_procurement_cost, 2),
        requires_escalation=False,
        escalation_reasons=[],
    )
    pending_po = PurchaseOrder(
        po_id=po_id,
        created_date="2024-11-15",
        store_id=store_id,
        supplier_id="SIM_SUPP_DAIRY_PRIMARY",
        supplier_type=SupplierType.PRIMARY,
        items=[item],
        total_order_qty=final_order_qty,
        total_order_cost=round(final_order_qty * unit_procurement_cost, 2),
        urgency=PurchaseOrderUrgency.STANDARD,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        requires_escalation=False,
        escalation_reasons=[],
        audit_trail=(),
    )
    return approve_purchase_order(
        po=pending_po,
        approved_by="OPERATOR_DEEPANSHU",
        approval_timestamp=approval_timestamp,
        approval_reason="Routine replenishment approved.",
    )


class TestFulfilmentStateMachine(unittest.TestCase):
    def setUp(self):
        self.approved_po = _make_approved_po()
        self.fo_id = "FO-20241115-DS_BLR_KORAMANGALA-DAIRY-01"

    # 39. Valid transitions across the complete lifecycle
    def test_39_complete_lifecycle_transitions(self):
        # 1. Handoff to Fulfilment
        fo = create_fulfilment_order(
            po=self.approved_po,
            actor_id="AGENT_FULFILMENT",
            timestamp="2024-11-15T11:00:00Z",
            fo_id=self.fo_id,
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)
        self.assertEqual(len(fo.event_history), 1)

        # 2. Transmit to supplier
        fo = transmit_fulfilment_order(
            fo=fo,
            transmission_timestamp="2024-11-15T11:15:00Z",
            actor_id="AGENT_FULFILMENT",
            idempotency_key=f"TX:{fo.fo_id}:1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(len(fo.event_history), 2)

        # 3. Supplier acknowledges full order
        ack_line = SupplierAcknowledgmentLine(sku_id="SKU_MILK_1L", accepted_quantity=24.0)
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-001",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
            lines=(ack_line,),
        )
        fo = apply_supplier_acknowledgment(
            fo=fo,
            ack=ack,
            actor_id="SIM_GATEWAY",
            idempotency_key=f"ACK:{fo.fo_id}:1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(fo.total_confirmed_pipeline_qty, 24.0)
        self.assertEqual(len(fo.event_history), 3)

        # 4. Supplier dispatches shipment
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-001",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(
            fo=fo,
            asn=asn,
            actor_id="SIM_GATEWAY",
            idempotency_key=f"ASN:{fo.fo_id}:1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(len(fo.event_history), 4)

        # 5. Store receives full order at dock
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            batch_references=("BATCH-001",),
        )
        grn = GoodsReceiptNote(
            grn_id="GRN-001",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-001",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
        )
        fo = apply_goods_receipt(
            fo=fo,
            grn=grn,
            actor_id="RECEIVER_ARUN",
            idempotency_key=f"GRN:{fo.fo_id}:1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(fo.total_confirmed_pipeline_qty, 0.0)  # Drained on full receipt
        self.assertEqual(fo.total_received_accepted_qty, 24.0)
        self.assertEqual(fo.total_remaining_unfulfilled_qty, 0.0)
        self.assertEqual(len(fo.event_history), 5)

    # 40. Unapproved transmission rejected
    def test_40_unapproved_transmission_rejected(self):
        unapproved_po = PurchaseOrder(
            po_id="PO-UNAPPROVED",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            supplier_type=SupplierType.PRIMARY,
            items=self.approved_po.items,
            total_order_qty=24.0,
            total_order_cost=720.0,
            status=PurchaseOrderStatus.PENDING_APPROVAL,
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            create_fulfilment_order(
                po=unapproved_po,
                actor_id="AGENT",
                timestamp="2024-11-15T11:00:00Z",
            )
        self.assertIn("cannot enter fulfilment", str(ctx.exception))

    # 41. Invalid direct status transitions rejected
    def test_41_invalid_direct_status_transitions_rejected(self):
        fo = create_fulfilment_order(
            po=self.approved_po,
            actor_id="AGENT",
            timestamp="2024-11-15T11:00:00Z",
        )
        # Cannot jump from READY_TO_TRANSMIT to IN_TRANSIT
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-001",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        with self.assertRaises(InvalidFulfilmentTransitionError) as ctx:
            apply_shipment_dispatch(fo, asn, "GATEWAY", "IDEMP-1")
        self.assertIn("Cannot dispatch shipment", str(ctx.exception))

    # 42. model_copy(update=...) bypass rejected at workflow boundaries
    def test_42_model_copy_bypass_rejected(self):
        fo = create_fulfilment_order(
            po=self.approved_po,
            actor_id="AGENT",
            timestamp="2024-11-15T11:00:00Z",
        )
        # Malicious or buggy model_copy bypassing state machine
        tampered_fo = fo.model_copy(update={"status": FulfilmentOrderStatus.RECEIVED})
        # Boundary validation detects status mismatch with event history
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            validate_inbound_fulfilment_order(tampered_fo)
        self.assertIn("Audit History Desynchronization", str(ctx.exception))

    # 43. Duplicate transmission event prevented by idempotency contract
    def test_43_duplicate_transmission_idempotency(self):
        fo = create_fulfilment_order(
            po=self.approved_po,
            actor_id="AGENT",
            timestamp="2024-11-15T11:00:00Z",
        )
        key = "IDEMP-TX-123"
        fo1 = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", key)
        self.assertEqual(fo1.status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(len(fo1.event_history), 2)

        # Retry with identical idempotency key returns fo1 safely without duplicate events
        fo2 = transmit_fulfilment_order(fo1, "2024-11-15T11:15:00Z", "AGENT", key)
        self.assertEqual(fo2.status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(len(fo2.event_history), 2)

    # 44. Repeated partial receipt handled without duplicate quantity posting
    def test_44_repeated_partial_receipt(self):
        # Setup order up to IN_TRANSIT with 24 units shipped
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")

        # First partial delivery: 10 units
        grn1 = GoodsReceiptNote(
            grn_id="GRN-P1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn1, "RECEIVER_ARUN", "GRN-1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(fo.lines[0].accepted_quantity, 10.0)
        self.assertEqual(fo.lines[0].outstanding_confirmed_quantity, 14.0)

        # Second delivery: remaining 14 units
        grn2 = GoodsReceiptNote(
            grn_id="GRN-P2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T14:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=14.0,
                    accepted_quantity=14.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-002",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn2, "RECEIVER_ARUN", "GRN-2")
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(fo.lines[0].accepted_quantity, 24.0)
        self.assertEqual(fo.lines[0].outstanding_confirmed_quantity, 0.0)

    # 45. Terminal state immutability
    def test_45_terminal_state_immutability(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        # Supplier rejects
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-REJ",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.REJECTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            rejection_reason="Supplier factory maintenance shutdown.",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-REJ")
        self.assertEqual(fo.status, FulfilmentOrderStatus.REJECTED_BY_SUPPLIER)
        self.assertIn(fo.status, TERMINAL_STATUSES)

        # Attempting further transitions raises error
        with self.assertRaises(InvalidFulfilmentTransitionError):
            transmit_fulfilment_order(fo, "2024-11-15T12:00:00Z", "AGENT", "TX-NEW")

    # 46. Invalid event ordering rejected
    def test_46_invalid_event_ordering_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        # Moving backward in time from 11:00 to 10:00
        with self.assertRaises(ValueError) as ctx:
            transmit_fulfilment_order(fo, "2024-11-15T10:00:00Z", "AGENT", "TX-BACKWARD")
        self.assertIn("cannot precede", str(ctx.exception))

    # 47. Cancellation without reason rejected
    def test_47_cancellation_without_reason_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        with self.assertRaises(ValueError) as ctx:
            request_cancellation(
                fo=fo,
                reason="",  # Empty
                category=CancellationReasonCategory.STORE_DEMAND_CHANGED,
                actor_id="MGR_SUNIL",
                timestamp="2024-11-15T11:30:00Z",
                idempotency_key="CNC-1",
            )
        self.assertIn("reason", str(ctx.exception))

    # 48. Cancellation after receipt rejected
    def test_48_cancellation_after_receipt_rejected(self):
        # Setup order with partial receipt
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")
        grn = GoodsReceiptNote(
            grn_id="GRN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn, "RECEIVER_ARUN", "GRN-1")

        # Attempting cancellation on order with accepted inventory must be rejected
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            request_cancellation(
                fo=fo,
                reason="Demand dropped suddenly.",
                category=CancellationReasonCategory.STORE_DEMAND_CHANGED,
                actor_id="MGR_SUNIL",
                timestamp="2024-11-16T09:00:00Z",
                idempotency_key="CNC-RECV",
            )
        self.assertIn("already received sellable units", str(ctx.exception))

    # 49. Cancellation after transmission without supplier confirmation rejected
    def test_49_cancellation_after_transmission_requires_supplier_ref(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        fo = request_cancellation(
            fo=fo,
            reason="Duplicate order entered by mistake.",
            category=CancellationReasonCategory.DUPLICATE_ORDER,
            actor_id="MGR_SUNIL",
            timestamp="2024-11-15T11:30:00Z",
            idempotency_key="CNC-REQ",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED)

        # Confirming without supplier_cancellation_ref fails
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            confirm_cancellation(
                fo=fo,
                actor_id="MGR_SUNIL",
                timestamp="2024-11-15T11:45:00Z",
                idempotency_key="CNC-CONF",
                supplier_cancellation_ref=None,
            )
        self.assertIn("supplier cancellation confirmation", str(ctx.exception))

        # Confirming with supplier ref succeeds
        fo_cancelled = confirm_cancellation(
            fo=fo,
            actor_id="MGR_SUNIL",
            timestamp="2024-11-15T11:45:00Z",
            idempotency_key="CNC-CONF",
            supplier_cancellation_ref="SUPP-CNC-9988",
        )
        self.assertEqual(fo_cancelled.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(fo_cancelled.supplier_cancellation_ref, "SUPP-CNC-9988")

    # 50. Delivery exception resolution requires matching open exception
    def test_50_delivery_exception_resolution(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")

        # Delivery exception raised in transit
        fo = record_delivery_exception(
            fo=fo,
            exception_reason="Refrigeration unit failure in transit truck.",
            actor_id="SUPERVISOR",
            timestamp="2024-11-16T07:00:00Z",
            idempotency_key="EXC-1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.DELIVERY_EXCEPTION)
        self.assertEqual(fo.resumable_status, FulfilmentOrderStatus.IN_TRANSIT)

        # Resolve exception
        fo = resolve_delivery_exception(
            fo=fo,
            resolution_notes="Backup refrigerated van deployed; goods cleared for dock inspection.",
            actor_id="SUPERVISOR",
            timestamp="2024-11-16T08:00:00Z",
            idempotency_key="RES-1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertIsNone(fo.open_exception_reason)

    # 51. Supplier rejection does not trigger automatic backup selection
    def test_51_supplier_rejection_no_automatic_backup(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-REJ",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.REJECTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            rejection_reason="Out of stock.",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-REJ")
        self.assertEqual(fo.status, FulfilmentOrderStatus.REJECTED_BY_SUPPLIER)
        # Supplier remains unchanged as PRIMARY; no automatic mutation to backup vendor
        self.assertEqual(fo.supplier_type, SupplierType.PRIMARY)
        self.assertEqual(fo.supplier_id, "SIM_SUPP_DAIRY_PRIMARY")

    # 52. Supplier change request cannot alter original approved PO
    def test_52_supplier_change_request_preserves_po(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)
        # Commercial PO remains completely unmutated
        self.assertEqual(fo.po_snapshot.items[0].unit_procurement_cost, 30.0)
        self.assertEqual(fo.po_snapshot.items[0].final_order_qty, 24.0)

    # 53. Audit event sequence and timestamp integrity
    def test_53_audit_event_sequence_and_integrity(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")

        self.assertEqual(len(fo.event_history), 3)
        for idx, event in enumerate(fo.event_history, start=1):
            self.assertEqual(event.sequence_number, idx)
            self.assertEqual(event.fo_id, fo.fo_id)
            self.assertTrue(event.timestamp.endswith("Z"))

    # 54. Close short terminal transition from PARTIALLY_RECEIVED
    def test_54_close_short_terminal_transition(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")
        grn = GoodsReceiptNote(
            grn_id="GRN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn, "RECEIVER_ARUN", "GRN-1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Store Manager decides to cancel unfulfilled balance via CLOSE_SHORT
        fo_closed = close_short_fulfilment_order(
            fo=fo,
            closure_reason="Vendor unable to deliver remaining 12 units due to stockout.",
            actor_id="MGR_SUNIL",
            timestamp="2024-11-16T12:00:00Z",
            idempotency_key="CS-1",
        )
        self.assertEqual(fo_closed.status, FulfilmentOrderStatus.CLOSED_SHORT)
        self.assertIn(fo_closed.status, TERMINAL_STATUSES)
        self.assertEqual(fo_closed.total_confirmed_pipeline_qty, 0.0)  # Pipeline relieved

    # 55. Exception resumption to recorded resumable status (PARTIALLY_RECEIVED)
    def test_55_exception_resumption_to_recorded_resumable_status(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")
        grn = GoodsReceiptNote(
            grn_id="GRN-P1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn, "RECEIVER_ARUN", "GRN-P1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Exception occurs while in PARTIALLY_RECEIVED
        fo_exc = record_delivery_exception(
            fo=fo,
            exception_reason="Dock chiller failure during unloading of second batch.",
            actor_id="SUPERVISOR",
            timestamp="2024-11-16T09:00:00Z",
            idempotency_key="EXC-DOCK-1",
        )
        self.assertEqual(fo_exc.status, FulfilmentOrderStatus.DELIVERY_EXCEPTION)
        self.assertEqual(fo_exc.resumable_status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Resolve exception returns order to PARTIALLY_RECEIVED
        fo_res = resolve_delivery_exception(
            fo=fo_exc,
            resolution_notes="Dock chiller repaired and temperature re-stabilized.",
            actor_id="SUPERVISOR",
            timestamp="2024-11-16T09:30:00Z",
            idempotency_key="RES-DOCK-1",
        )
        self.assertEqual(fo_res.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertIsNone(fo_res.open_exception_reason)
        self.assertIsNone(fo_res.resumable_status)

    # -------------------------------------------------------------------------
    # HARDENING TESTS: FIX B — Strong Idempotency Payload Integrity
    # -------------------------------------------------------------------------

    def test_fix_b_idempotency_exact_duplicate_transmission_returns_same_fo(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        key = "IDEMP-TX-EXACT"
        fo1 = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", key)
        self.assertEqual(fo1.status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(len(fo1.event_history), 2)

        # Retry with identical key and identical parameters returns fo1 without appending events
        fo2 = transmit_fulfilment_order(fo1, "2024-11-15T11:15:00Z", "AGENT", key)
        self.assertEqual(fo2.status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(len(fo2.event_history), 2)
        self.assertEqual(fo1.event_history[-1].payload_fingerprint, fo2.event_history[-1].payload_fingerprint)

    def test_fix_b_idempotency_changed_accepted_quantity_raises_governance_error(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        key = "IDEMP-ACK-QTY-CONFLICT"
        ack1 = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
            lines=(SupplierAcknowledgmentLine(sku_id="SKU_MILK_1L", accepted_quantity=24.0, declined_quantity=0.0),),
        )
        fo_ack = apply_supplier_acknowledgment(fo, ack1, "GATEWAY", key)
        self.assertEqual(fo_ack.status, FulfilmentOrderStatus.ACKNOWLEDGED)

        # Conflicting retry: same key, but different accepted quantity (12.0)
        ack2 = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo_ack.fo_id,
            po_id=fo_ack.po_id,
            supplier_id=fo_ack.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
            lines=(SupplierAcknowledgmentLine(sku_id="SKU_MILK_1L", accepted_quantity=12.0, declined_quantity=12.0),),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            apply_supplier_acknowledgment(fo_ack, ack2, "GATEWAY", key)
        self.assertIn("differing payload fingerprint", str(ctx.exception))

    def test_fix_b_idempotency_changed_supplier_id_raises_governance_error(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        key = "IDEMP-ACK-SUPP-CONFLICT"
        ack1 = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo_ack = apply_supplier_acknowledgment(fo, ack1, "GATEWAY", key)

        # Conflicting retry: same key, but different supplier_id
        ack2 = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo_ack.fo_id,
            po_id=fo_ack.po_id,
            supplier_id="DIFFERENT_SUPPLIER",
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            apply_supplier_acknowledgment(fo_ack, ack2, "GATEWAY", key)
        self.assertIn("differing payload fingerprint", str(ctx.exception))

    def test_fix_b_idempotency_changed_event_type_raises_governance_error(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        shared_key = "REUSED-KEY-123"
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", shared_key)
        self.assertEqual(fo.status, FulfilmentOrderStatus.ISSUED)

        # Attempt to reuse the transmission idempotency key for supplier acknowledgment
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            apply_supplier_acknowledgment(fo, ack, "GATEWAY", shared_key)
        self.assertIn("was previously used for event type 'TRANSMIT_TO_SUPPLIER'", str(ctx.exception))

    def test_fix_b_idempotency_timezone_offsets_produce_identical_fingerprint(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        key = "IDEMP-TZ-MATCH"

        # 17:00:00+05:30 is exactly 11:30:00Z in UTC
        ack_ist = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T17:00:00+05:30",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo_ack1 = apply_supplier_acknowledgment(fo, ack_ist, "GATEWAY", key)

        # Retry with identical UTC representation
        ack_utc = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        # Should match identical payload fingerprint and return existing fo_ack1 without exception
        fo_ack2 = apply_supplier_acknowledgment(fo_ack1, ack_utc, "GATEWAY", key)
        self.assertEqual(fo_ack1.event_history[-1].payload_fingerprint, fo_ack2.event_history[-1].payload_fingerprint)
        self.assertEqual(len(fo_ack2.event_history), len(fo_ack1.event_history))

    def test_fix_b_idempotency_different_keys_partial_receipts_succeed(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")

        # First partial receipt with Key 1
        grn1 = GoodsReceiptNote(
            grn_id="GRN-PARTIAL-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_1",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn1, "RECEIVER_1", "IDEMP-GRN-1")
        self.assertEqual(fo.total_received_accepted_qty, 10.0)

        # Second partial receipt with Key 2
        grn2 = GoodsReceiptNote(
            grn_id="GRN-PARTIAL-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T14:00:00Z",
            receiver_id="RECEIVER_1",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=14.0,
                    accepted_quantity=14.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-002",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn2, "RECEIVER_1", "IDEMP-GRN-2")
        self.assertEqual(fo.total_received_accepted_qty, 24.0)
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)

    def test_fix_b_idempotency_duplicate_grn_id_under_different_key_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")

        grn = GoodsReceiptNote(
            grn_id="GRN-DUP-TEST",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_1",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn, "RECEIVER_1", "KEY-ORIGINAL")
        self.assertEqual(fo.total_received_accepted_qty, 10.0)

        # Presenting the same GRN ID under a different idempotency key
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            apply_goods_receipt(fo, grn, "RECEIVER_1", "KEY-DIFFERENT")
        self.assertIn("Duplicate GRN submission", str(ctx.exception))
        # Cumulative accepted quantity is NOT double counted
        self.assertEqual(fo.total_received_accepted_qty, 10.0)

    # -------------------------------------------------------------------------
    # HARDENING TESTS: FIX C — Cancellation & Partial Receipt Lifecycle
    # -------------------------------------------------------------------------

    def test_fix_c_cancellation_before_transmission_succeeds(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        self.assertEqual(fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

        # Request cancellation prior to transmission
        fo = request_cancellation(
            fo=fo,
            reason="Commercial review found redundant stock.",
            category=CancellationReasonCategory.STORE_DEMAND_CHANGED,
            actor_id="MGR_SUNIL",
            timestamp="2024-11-15T11:10:00Z",
            idempotency_key="CNC-PRE-TX-REQ",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED)

        # Confirm cancellation: since not transmitted, supplier_cancellation_ref is NOT required
        fo_cancelled = confirm_cancellation(
            fo=fo,
            actor_id="MGR_SUNIL",
            timestamp="2024-11-15T11:15:00Z",
            idempotency_key="CNC-PRE-TX-CONF",
            supplier_cancellation_ref=None,
            shipment_cancellation_ref=None,
        )
        self.assertEqual(fo_cancelled.status, FulfilmentOrderStatus.CANCELLED)
        self.assertIn(fo_cancelled.status, TERMINAL_STATUSES)
        self.assertEqual(fo_cancelled.total_confirmed_pipeline_qty, 0.0)

    def test_fix_c_cancellation_in_transit_requires_shipment_cancellation_ref(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        fo = request_cancellation(
            fo=fo,
            reason="Store cold room flooded; unable to receive any shipment.",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T15:00:00Z",
            idempotency_key="CNC-INT-REQ",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED)

        # Confirming without shipment_cancellation_ref must be rejected
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            confirm_cancellation(
                fo=fo,
                actor_id="SUPERVISOR",
                timestamp="2024-11-15T15:30:00Z",
                idempotency_key="CNC-INT-CONF-FAIL",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_cancellation_ref=None,
            )
        self.assertIn("shipment cancellation or return disposition", str(ctx.exception))

        # Confirming with both supplier and shipment cancellation refs succeeds
        fo_cancelled = confirm_cancellation(
            fo=fo,
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T15:30:00Z",
            idempotency_key="CNC-INT-CONF-OK",
            supplier_cancellation_ref="SUPP-CNC-01",
            shipment_cancellation_ref="TRUCK-RTO-RETURN-001",
        )
        self.assertEqual(fo_cancelled.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(fo_cancelled.shipment_cancellation_ref, "TRUCK-RTO-RETURN-001")
        self.assertEqual(fo_cancelled.supplier_cancellation_ref, "SUPP-CNC-01")

    def test_fix_c_close_short_lifecycle_and_duplicate_rejection(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")
        grn = GoodsReceiptNote(
            grn_id="GRN-P1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn, "RECEIVER_ARUN", "GRN-P1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(fo.total_received_accepted_qty, 10.0)

        # Attempt to cancel partially received order -> rejected
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            request_cancellation(
                fo=fo,
                reason="Cancelling remainder.",
                category=CancellationReasonCategory.STORE_DEMAND_CHANGED,
                actor_id="MGR_SUNIL",
                timestamp="2024-11-16T09:00:00Z",
                idempotency_key="CNC-FORBIDDEN",
            )
        self.assertIn("already received sellable units", str(ctx.exception))

        # Close short unfulfilled balance of 14 units
        closed_fo = close_short_fulfilment_order(
            fo=fo,
            closure_reason="Vendor out of stock for remainder; closing unfulfilled balance.",
            actor_id="MGR_SUNIL",
            timestamp="2024-11-16T09:30:00Z",
            idempotency_key="CS-KEY-1",
        )
        self.assertEqual(closed_fo.status, FulfilmentOrderStatus.CLOSED_SHORT)
        self.assertEqual(closed_fo.closed_by, "MGR_SUNIL")
        self.assertEqual(closed_fo.closure_reason, "Vendor out of stock for remainder; closing unfulfilled balance.")
        self.assertEqual(closed_fo.closure_timestamp, "2024-11-16T09:30:00Z")
        self.assertEqual(closed_fo.closed_unfulfilled_qty, 14.0)
        self.assertEqual(closed_fo.total_received_accepted_qty, 10.0)
        self.assertEqual(closed_fo.total_confirmed_pipeline_qty, 0.0)  # Zero phantom pipeline

        # Retry with identical key returns closed_fo idempotently
        closed_retry = close_short_fulfilment_order(
            fo=closed_fo,
            closure_reason="Vendor out of stock for remainder; closing unfulfilled balance.",
            actor_id="MGR_SUNIL",
            timestamp="2024-11-16T09:30:00Z",
            idempotency_key="CS-KEY-1",
        )
        self.assertEqual(closed_retry.status, FulfilmentOrderStatus.CLOSED_SHORT)

        # Attempt duplicate close short under a different key -> raises error
        with self.assertRaises(InvalidFulfilmentTransitionError) as ctx:
            close_short_fulfilment_order(
                fo=closed_fo,
                closure_reason="Another attempt",
                actor_id="MGR_SUNIL",
                timestamp="2024-11-16T10:00:00Z",
                idempotency_key="CS-KEY-2",
            )
        self.assertIn("already CLOSED_SHORT", str(ctx.exception))

    # -------------------------------------------------------------------------
    # HARDENING TESTS: FIX D — Supplier Commercial Change Requests
    # -------------------------------------------------------------------------

    def test_fix_d_scenario_1_reaffirm_original_terms(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)
        # Commercial PO remains unmutated
        self.assertEqual(fo.po_snapshot.items[0].unit_procurement_cost, 30.0)

        # Scenario 1: Reaffirm original terms
        fo_reaffirmed = reaffirm_original_terms(
            fo=fo,
            resolution_notes="Supplier agreed to honor contracted rate and quantity.",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:00:00Z",
            idempotency_key="REAFFIRM-1",
        )
        self.assertEqual(fo_reaffirmed.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(fo_reaffirmed.po_snapshot.items[0].unit_procurement_cost, 30.0)
        self.assertEqual(fo_reaffirmed.event_history[-1].details["terms_status"], "ORIGINAL_TERMS_RETAINED")

        # Retry with identical key is idempotent
        fo_reaffirmed_retry = reaffirm_original_terms(
            fo=fo_reaffirmed,
            resolution_notes="Supplier agreed to honor contracted rate and quantity.",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:00:00Z",
            idempotency_key="REAFFIRM-1",
        )
        self.assertEqual(fo_reaffirmed_retry.status, FulfilmentOrderStatus.ACKNOWLEDGED)

    def test_fix_d_scenario_2_supersede_fulfilment_order(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)

        # Create newly approved commercial PO with revised terms
        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REVISED-DAIRY-99",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:30:00Z",
            idempotency_key="SUP-KEY-1",
            reason="Commercial price increase approved by procurement authority.",
            supplier_cancellation_ref="SUPP-CNC-REV-99",
        )

        # Invariant checks on original (superseded) FO:
        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(superseded_fo.cancellation_category, CancellationReasonCategory.COMMERCIAL_TERMS_REJECTED)
        self.assertEqual(superseded_fo.supplier_cancellation_ref, "SUPP-CNC-REV-99")
        self.assertIn("PO-REVISED-DAIRY-99", superseded_fo.cancellation_reason)
        # Original PO snapshot in superseded FO is strictly unchanged
        self.assertEqual(superseded_fo.po_snapshot.items[0].unit_procurement_cost, 30.0)
        self.assertEqual(superseded_fo.po_snapshot.items[0].final_order_qty, 24.0)
        self.assertEqual(superseded_fo.event_history[-1].details["superseded_by_po_id"], "PO-REVISED-DAIRY-99")
        self.assertEqual(superseded_fo.event_history[-1].details["superseded_by_fo_id"], "FO-REVISED-DAIRY-99")
        self.assertEqual(superseded_fo.event_history[-1].details["supplier_cancellation_ref"], "SUPP-CNC-REV-99")

        # Invariant checks on new replacement FO:
        self.assertEqual(replacement_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)
        self.assertEqual(replacement_fo.fo_id, "FO-REVISED-DAIRY-99")
        self.assertEqual(replacement_fo.po_id, "PO-REVISED-DAIRY-99")
        self.assertEqual(replacement_fo.lines[0].ordered_quantity, 36.0)
        self.assertEqual(replacement_fo.lines[0].unit_cost, 35.0)

    def test_fix_d_supersede_with_unapproved_replacement_po_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        unapproved_po = PurchaseOrder(
            po_id="PO-UNAPPROVED-REVISED",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            supplier_type=SupplierType.PRIMARY,
            items=self.approved_po.items,
            total_order_qty=24.0,
            total_order_cost=720.0,
            status=PurchaseOrderStatus.PENDING_APPROVAL,
        )

        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo,
                replacement_po=unapproved_po,
                replacement_fo_id="FO-NEW",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:00:00Z",
                idempotency_key="SUP-FAIL",
                reason="Attempting to supersede with unapproved PO.",
                supplier_cancellation_ref="SUPP-CNC-FAIL",
            )
        self.assertIn("must be legally APPROVED", str(ctx.exception))

    def test_missing_supplier_cancellation_evidence_blocks_supersession(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # 1. Call without supplier_cancellation_ref (omitted / None)
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REVISED-DAIRY-99",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:30:00Z",
                idempotency_key="SUP-NO-REF-1",
                reason="Supplier requested changed price.",
                supplier_cancellation_ref=None,
            )
        self.assertIn("Auditable supplier confirmation that the original terms will not be fulfilled", str(ctx.exception))

        # 2. Call with empty or whitespace supplier_cancellation_ref
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REVISED-DAIRY-99",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:30:00Z",
                idempotency_key="SUP-NO-REF-2",
                reason="Supplier requested changed price.",
                supplier_cancellation_ref="   ",
            )
        self.assertIn("Auditable supplier confirmation that the original terms will not be fulfilled", str(ctx.exception))

    def test_active_shipment_blocks_supersession(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        # Simulate that a shipment was dispatched and is unresolved / active
        fo_with_active_shipment = fo.model_copy(update={"shipment_ids": ("ASN-ACTIVE-01",), "resolved_shipment_ids": ()})

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo_with_active_shipment,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REVISED-DAIRY-99",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:30:00Z",
                idempotency_key="SUP-SHIPMENT-FAIL",
                reason="Price change",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_cancellation_ref=None,
            )
        self.assertIn("currently in transit, awaiting receipt, or awaiting return disposition", str(ctx.exception))
        self.assertIn("ASN-ACTIVE-01", str(ctx.exception))

    def test_resolved_returned_shipment_permits_supersession_via_return_ref(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        # Historical shipment exists, but has documented return disposition
        fo_with_shipment = fo.model_copy(update={"shipment_ids": ("ASN-HIST-01",)})

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo_with_shipment,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REVISED-DAIRY-99",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:30:00Z",
            idempotency_key="SUP-RESOLVED-1",
            reason="Price change with return of prior consignment.",
            supplier_cancellation_ref="SUPP-CNC-01",
            shipment_cancellation_ref="CARRIER-RTO-7788",
        )

        # Historical shipment ID is NOT erased
        self.assertEqual(superseded_fo.shipment_ids, ("ASN-HIST-01",))
        self.assertIn("ASN-HIST-01", superseded_fo.resolved_shipment_ids)
        self.assertEqual(superseded_fo.shipment_cancellation_ref, "CARRIER-RTO-7788")
        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(replacement_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

    def test_resolved_returned_shipment_permits_supersession_via_dock_rejection_grn(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        # Historical shipment was fully received at dock, but 100% rejected (0 accepted sellable units)
        modified_line = fo.lines[0].model_copy(update={
            "supplier_accepted_quantity": 24.0,
            "supplier_unconfirmed_quantity": 0.0,
            "shipped_quantity": 24.0,
            "delivered_quantity": 24.0,
            "accepted_quantity": 0.0,
            "rejected_quantity": 24.0,
        })
        fo_with_grn_rejection = fo.model_copy(update={
            "shipment_ids": ("ASN-HIST-02",),
            "grn_ids": ("GRN-REJECT-01",),
            "resolved_shipment_ids": ("ASN-HIST-02",),
            "shipment_resolution_evidence": {"ASN-HIST-02": "DOCK-RTO-DISP-01"},
            "lines": (modified_line,),
        })
        self.assertEqual(fo_with_grn_rejection.total_received_accepted_qty, 0.0)

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo_with_grn_rejection,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REVISED-DAIRY-99",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:30:00Z",
            idempotency_key="SUP-RESOLVED-DOCK-1",
            reason="Shipment arrived spoiled; 100% rejected at dock. Vendor to replace with new PO.",
            supplier_cancellation_ref="SUPP-CNC-01",
            shipment_cancellation_ref=None,
        )

        # Historical shipment and GRN records preserved
        self.assertEqual(superseded_fo.shipment_ids, ("ASN-HIST-02",))
        self.assertEqual(superseded_fo.grn_ids, ("GRN-REJECT-01",))
        self.assertEqual(superseded_fo.resolved_shipment_ids, ("ASN-HIST-02",))
        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(replacement_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

    def test_previously_received_sellable_goods_still_block_supersession(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        # Simulate received sellable units on a line (10.0 accepted units)
        modified_line = fo.lines[0].model_copy(update={"accepted_quantity": 10.0, "delivered_quantity": 10.0})
        fo_with_receipt = fo.model_copy(update={
            "shipment_ids": ("ASN-01",),
            "resolved_shipment_ids": ("ASN-01",),
            "lines": (modified_line,),
        })

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo_with_receipt,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REVISED-DAIRY-99",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:30:00Z",
                idempotency_key="SUP-RECV-FAIL",
                reason="Price change",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_cancellation_ref="RTO-01",
            )
        self.assertIn("already received sellable units", str(ctx.exception))
        self.assertIn("close_short_fulfilment_order", str(ctx.exception))

    def test_multiple_shipments_cancellation_ref_for_one_does_not_resolve_other(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        # Two split shipments dispatched: ASN-SPLIT-01 and ASN-SPLIT-02
        fo_multi_shipment = fo.model_copy(update={
            "shipment_ids": ("ASN-SPLIT-01", "ASN-SPLIT-02"),
        })
        self.assertEqual(len(fo_multi_shipment.active_shipment_ids), 2)

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # Providing resolution evidence for ASN-SPLIT-01 ONLY must NOT resolve ASN-SPLIT-02
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo_multi_shipment,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REVISED-DAIRY-99",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:30:00Z",
                idempotency_key="SUP-MULTI-PARTIAL",
                reason="Terms changed",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_resolutions={"ASN-SPLIT-01": "RTO-CARRIER-01"},
                order_wide_cancellation_scope=False,
            )
        self.assertIn("ASN-SPLIT-02", str(ctx.exception))
        self.assertIn("awaiting return disposition", str(ctx.exception))

    def test_multiple_shipments_order_wide_cancellation_scope_resolves_all(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        fo_multi_shipment = fo.model_copy(update={
            "shipment_ids": ("ASN-SPLIT-01", "ASN-SPLIT-02"),
        })

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # When order_wide_cancellation_scope is True, an order-wide cancellation reference resolves all shipments
        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo_multi_shipment,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REVISED-DAIRY-99",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:30:00Z",
            idempotency_key="SUP-MULTI-ORDER-WIDE",
            reason="Terms changed, all consignments recalled.",
            supplier_cancellation_ref="SUPP-CNC-01",
            shipment_cancellation_ref="ORDER-WIDE-RECALL-999",
            order_wide_cancellation_scope=True,
        )

        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(superseded_fo.shipment_ids, ("ASN-SPLIT-01", "ASN-SPLIT-02"))
        self.assertEqual(set(superseded_fo.resolved_shipment_ids), {"ASN-SPLIT-01", "ASN-SPLIT-02"})
        self.assertEqual(superseded_fo.active_shipment_ids, ())
        self.assertTrue(superseded_fo.order_wide_cancellation_scope)
        self.assertEqual(replacement_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

    def test_multiple_shipments_shipment_specific_resolutions_resolve_all(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        fo_multi_shipment = fo.model_copy(update={
            "shipment_ids": ("ASN-SPLIT-01", "ASN-SPLIT-02"),
        })

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-DAIRY-99",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # Providing specific resolutions for every shipment allows supersession without order-wide override
        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo_multi_shipment,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REVISED-DAIRY-99",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:30:00Z",
            idempotency_key="SUP-MULTI-SPECIFIC",
            reason="Terms changed, individual return notes collected.",
            supplier_cancellation_ref="SUPP-CNC-01",
            shipment_resolutions={"ASN-SPLIT-01": "RTO-TRUCK-1", "ASN-SPLIT-02": "RTO-TRUCK-2"},
            order_wide_cancellation_scope=False,
        )

        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(superseded_fo.active_shipment_ids, ())
        self.assertEqual(set(superseded_fo.resolved_shipment_ids), {"ASN-SPLIT-01", "ASN-SPLIT-02"})
        self.assertEqual(superseded_fo.shipment_resolution_evidence["ASN-SPLIT-01"], "RTO-TRUCK-1")
        self.assertEqual(superseded_fo.shipment_resolution_evidence["ASN-SPLIT-02"], "RTO-TRUCK-2")
        self.assertEqual(replacement_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

    def test_grn_zero_accepted_without_disposition_blocks_supersession(self):
        # 1. Verify apply_goods_receipt with 0 accepted and no disposition does NOT resolve custody
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-15T12:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-SPOILED-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T11:45:00Z",
            expected_arrival_at="2024-11-15T12:00:00Z",
            lines=(
                ShipmentLineItem(sku_id=fo.lines[0].sku_id, shipped_quantity=fo.lines[0].ordered_quantity),
            ),
            carrier="ColdChain Express",
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-TX-1")
        grn = GoodsReceiptNote(
            grn_id="GRN-SPOILED-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id=asn.shipment_id,
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-15T12:15:00Z",
            receiver_id="STORE_RECEIVER_RAMESH",
            inspection_lines=(
                DockInspectionLine(
                    sku_id=fo.lines[0].sku_id,
                    delivered_quantity=fo.lines[0].ordered_quantity,
                    accepted_quantity=0.0,
                    rejected_quantity=fo.lines[0].ordered_quantity,
                    damaged_quantity=fo.lines[0].ordered_quantity,
                    unit_cost=fo.lines[0].unit_cost,
                    remaining_shelf_life_days=0,
                    total_shelf_life_days=5,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                ),
            ),
            rejected_goods_disposition=None,
        )
        fo_received = apply_goods_receipt(fo, grn, "STORE_RECEIVER_RAMESH", "GRN-TX-1")
        self.assertNotIn(asn.shipment_id, fo_received.resolved_shipment_ids)
        self.assertIn(asn.shipment_id, fo_received.active_shipment_ids)
        self.assertEqual(fo_received.total_received_accepted_qty, 0.0)

        # 2. On an order in SUPPLIER_CHANGE_PENDING_REVIEW with rejected goods and no disposition, supersession is blocked
        fo_change = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo_change = transmit_fulfilment_order(fo_change, "2024-11-15T11:15:00Z", "AGENT", "TX-2")
        ack_change = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo_change.fo_id,
            po_id=fo_change.po_id,
            supplier_id=fo_change.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo_change = apply_supplier_acknowledgment(fo_change, ack_change, "GATEWAY", "ACK-CHG")
        modified_line = fo_change.lines[0].model_copy(update={
            "supplier_accepted_quantity": 24.0,
            "supplier_unconfirmed_quantity": 0.0,
            "shipped_quantity": 24.0,
            "delivered_quantity": 24.0,
            "accepted_quantity": 0.0,
            "rejected_quantity": 24.0,
        })
        fo_with_rejection_no_disp = fo_change.model_copy(update={
            "shipment_ids": ("ASN-SPOILED-01",),
            "grn_ids": ("GRN-SPOILED-01",),
            "lines": (modified_line,),
        })

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-REPLACE-1",
            approval_timestamp="2024-11-15T12:30:00Z",
        )

        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo_with_rejection_no_disp,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REVISED-REPLACE-1",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:45:00Z",
                idempotency_key="SUP-SPOILED-NODISP",
                reason="Shipment damaged; replacing.",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_cancellation_ref=None,
            )
        self.assertIn("ASN-SPOILED-01", str(ctx.exception))
        self.assertIn("awaiting return disposition", str(ctx.exception))

        # 3. Even if shipment was placed in resolved_shipment_ids without disposition evidence, Safeguard 4 blocks it
        fo_resolved_without_evidence = fo_with_rejection_no_disp.model_copy(update={
            "resolved_shipment_ids": ("ASN-SPOILED-01",),
            "shipment_resolution_evidence": {},
        })
        with self.assertRaises(FulfilmentGovernanceError) as ctx2:
            supersede_fulfilment_order(
                original_fo=fo_resolved_without_evidence,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REVISED-REPLACE-1",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:45:00Z",
                idempotency_key="SUP-SPOILED-NODISP-2",
                reason="Shipment damaged; replacing.",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_cancellation_ref=None,
            )
        self.assertIn("has rejected goods at dock receiving without documented return, disposal, or authorized final disposition", str(ctx2.exception))

    def test_grn_zero_accepted_with_disposition_permits_supersession(self):
        # 1. Verify apply_goods_receipt with 0 accepted WITH disposition DOES resolve custody
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-15T12:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-SPOILED-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T11:45:00Z",
            expected_arrival_at="2024-11-15T12:00:00Z",
            lines=(
                ShipmentLineItem(sku_id=fo.lines[0].sku_id, shipped_quantity=fo.lines[0].ordered_quantity),
            ),
            carrier="ColdChain Express",
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-TX-2")
        grn = GoodsReceiptNote(
            grn_id="GRN-SPOILED-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id=asn.shipment_id,
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-15T12:15:00Z",
            receiver_id="STORE_RECEIVER_RAMESH",
            inspection_lines=(
                DockInspectionLine(
                    sku_id=fo.lines[0].sku_id,
                    delivered_quantity=fo.lines[0].ordered_quantity,
                    accepted_quantity=0.0,
                    rejected_quantity=fo.lines[0].ordered_quantity,
                    damaged_quantity=fo.lines[0].ordered_quantity,
                    unit_cost=fo.lines[0].unit_cost,
                    remaining_shelf_life_days=0,
                    total_shelf_life_days=5,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                ),
            ),
            rejected_goods_disposition="RTO-CARRIER-RETURN-CHALLAN-8899",
        )
        fo_received = apply_goods_receipt(fo, grn, "STORE_RECEIVER_RAMESH", "GRN-TX-2")
        self.assertIn(asn.shipment_id, fo_received.resolved_shipment_ids)
        self.assertEqual(fo_received.active_shipment_ids, ())
        self.assertEqual(fo_received.shipment_resolution_evidence[asn.shipment_id], "RTO-CARRIER-RETURN-CHALLAN-8899")

        # 2. When disposition is documented, supersession is permitted
        fo_change = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo_change = transmit_fulfilment_order(fo_change, "2024-11-15T11:15:00Z", "AGENT", "TX-2")
        ack_change = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo_change.fo_id,
            po_id=fo_change.po_id,
            supplier_id=fo_change.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0, "suggested_qty": 36.0},
        )
        fo_change = apply_supplier_acknowledgment(fo_change, ack_change, "GATEWAY", "ACK-CHG")
        modified_line = fo_change.lines[0].model_copy(update={
            "supplier_accepted_quantity": 24.0,
            "supplier_unconfirmed_quantity": 0.0,
            "shipped_quantity": 24.0,
            "delivered_quantity": 24.0,
            "accepted_quantity": 0.0,
            "rejected_quantity": 24.0,
        })
        fo_with_rejection_and_disp = fo_change.model_copy(update={
            "shipment_ids": ("ASN-SPOILED-02",),
            "grn_ids": ("GRN-SPOILED-02",),
            "resolved_shipment_ids": ("ASN-SPOILED-02",),
            "shipment_resolution_evidence": {"ASN-SPOILED-02": "RTO-CARRIER-RETURN-CHALLAN-8899"},
            "lines": (modified_line,),
        })

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REVISED-REPLACE-2",
            approval_timestamp="2024-11-15T12:30:00Z",
        )

        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo_with_rejection_and_disp,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REVISED-REPLACE-2",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:45:00Z",
            idempotency_key="SUP-SPOILED-WITHDISP",
            reason="Shipment damaged, returned to carrier on arrival; replacing.",
            supplier_cancellation_ref="SUPP-CNC-01",
            shipment_cancellation_ref=None,
        )

        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(superseded_fo.shipment_ids, ("ASN-SPOILED-02",))
        self.assertEqual(superseded_fo.grn_ids, ("GRN-SPOILED-02",))
        self.assertIn("ASN-SPOILED-02", superseded_fo.resolved_shipment_ids)
        self.assertEqual(superseded_fo.active_shipment_ids, ())
        self.assertEqual(superseded_fo.shipment_resolution_evidence["ASN-SPOILED-02"], "RTO-CARRIER-RETURN-CHALLAN-8899")
        self.assertEqual(replacement_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

    def test_reaffirm_original_terms_restores_pipeline_confirmed_quantity(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack_change = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price_inr": 35.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack_change, "GATEWAY", "ACK-CHG-1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)
        # In change pending review, confirmed pipeline quantity is 0
        self.assertEqual(fo.total_confirmed_pipeline_qty, 0.0)

        # Reaffirm terms
        reaffirm_ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-REAFFIRM-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:45:00Z",
            expected_delivery_at="2024-11-16T10:00:00Z",
        )
        fo_reaffirmed = reaffirm_original_terms(
            fo=fo,
            resolution_notes="Supplier conceded to original approved commercial PO terms.",
            actor_id="SUPERVISOR_PRIYA",
            timestamp="2024-11-15T12:00:00Z",
            idempotency_key="REAFFIRM-KEY-01",
            ack=reaffirm_ack,
        )

        self.assertEqual(fo_reaffirmed.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(fo_reaffirmed.expected_delivery_at, "2024-11-16T10:00:00Z")
        self.assertEqual(fo_reaffirmed.lines[0].supplier_accepted_quantity, 24.0)
        self.assertEqual(fo_reaffirmed.lines[0].supplier_declined_quantity, 0.0)
        self.assertEqual(fo_reaffirmed.lines[0].supplier_unconfirmed_quantity, 0.0)
        self.assertEqual(fo_reaffirmed.total_confirmed_pipeline_qty, 24.0)

    def test_ack_exceeding_ordered_quantity_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")

        # accepted (20) + declined (10) = 30 > ordered (24)
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-OVER",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.PARTIALLY_ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T10:00:00Z",
            lines=(
                SupplierAcknowledgmentLine(
                    sku_id="SKU_MILK_1L",
                    accepted_quantity=20.0,
                    declined_quantity=10.0,
                ),
            ),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-OVER-1")
        self.assertIn("Total acknowledged quantity", str(ctx.exception))

    def test_goods_receipt_unknown_shipment_or_mismatched_supplier_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-VALID-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")

        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            batch_references=("BATCH-001",),
        )

        # 1. Unknown / undispatched shipment
        grn_unknown_shipment = GoodsReceiptNote(
            grn_id="GRN-UNKNOWN-SHIP",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-PHANTOM-99",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            apply_goods_receipt(fo, grn_unknown_shipment, "RECEIVER_ARUN", "GRN-KEY-1")
        self.assertIn("unknown or undispatched shipment", str(ctx.exception))

        # 2. Mismatched supplier
        grn_wrong_supplier = GoodsReceiptNote(
            grn_id="GRN-WRONG-SUPP",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-VALID-01",
            store_id=fo.store_id,
            supplier_id="SIM_SUPP_BAKERY_PRIMARY",
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            apply_goods_receipt(fo, grn_wrong_supplier, "RECEIVER_ARUN", "GRN-KEY-2")
        self.assertIn("GRN identifier mismatch", str(ctx.exception))

    def test_split_dispatch_from_partially_received_to_in_transit(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-1")

        # Split 1: 10 units
        asn1 = AdvanceShipmentNotice(
            shipment_id="ASN-SPLIT-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=10.0),),
        )
        fo = apply_shipment_dispatch(fo, asn1, "GATEWAY", "ASN-KEY-1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        # Receive Split 1
        grn1 = GoodsReceiptNote(
            grn_id="GRN-SPLIT-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SPLIT-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn1, "RECEIVER_ARUN", "GRN-KEY-1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Split 2: 14 units dispatched while order is PARTIALLY_RECEIVED -> transitions to IN_TRANSIT
        asn2 = AdvanceShipmentNotice(
            shipment_id="ASN-SPLIT-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-16T10:00:00Z",
            expected_arrival_at="2024-11-16T14:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=14.0),),
        )
        fo = apply_shipment_dispatch(fo, asn2, "GATEWAY", "ASN-KEY-2")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        # Receive Split 2 -> order is now fully RECEIVED
        grn2 = GoodsReceiptNote(
            grn_id="GRN-SPLIT-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SPLIT-2",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T14:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=14.0,
                    accepted_quantity=14.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-002",),
                ),
            ),
        )
        fo = apply_goods_receipt(fo, grn2, "RECEIVER_ARUN", "GRN-KEY-2")
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(fo.total_received_accepted_qty, 24.0)

    def test_supersede_order_idempotency_conflict_and_replay(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REPLACE-IDEMP",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # First call: succeeds
        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REPLACE-IDEMP",
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T12:15:00Z",
            idempotency_key="SUPERSEDE-IDEMP-KEY-1",
            reason="Supplier price updated; commercial renegotiation completed.",
            supplier_cancellation_ref="SUPP-CNC-IDEMP",
        )
        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)

        # Idempotent replay: identical payload returns exact same result
        superseded_fo2, replacement_fo2 = supersede_fulfilment_order(
            original_fo=fo,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REPLACE-IDEMP",
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T12:15:00Z",
            idempotency_key="SUPERSEDE-IDEMP-KEY-1",
            reason="Supplier price updated; commercial renegotiation completed.",
            supplier_cancellation_ref="SUPP-CNC-IDEMP",
        )
        self.assertEqual(superseded_fo2.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(replacement_fo2.fo_id, "FO-REPLACE-IDEMP")

        # Conflict: same key reused for a different event type on fo raises conflict
        conflict_key = "TX-1"  # already used for TRANSMIT_TO_SUPPLIER
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REPLACE-IDEMP",
                actor_id="SUPERVISOR",
                timestamp="2024-11-15T12:15:00Z",
                idempotency_key=conflict_key,
                reason="Supplier price updated; commercial renegotiation completed.",
                supplier_cancellation_ref="SUPP-CNC-IDEMP",
            )
        self.assertIn("Idempotency Conflict", str(ctx.exception))

    def test_supersede_order_ambiguous_shipment_resolution_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        # Mock 2 shipments in shipment_ids
        fo_multi = fo.model_copy(update={"shipment_ids": ("ASN-01", "ASN-02")})
        new_approved_po = _make_approved_po(
            final_order_qty=24.0,
            unit_procurement_cost=35.0,
            po_id="PO-REPLACE-MULTI",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # Single shipment_cancellation_ref without order_wide scope or shipment_resolutions
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            supersede_fulfilment_order(
                original_fo=fo_multi,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REPLACE-MULTI",
                actor_id="SUPERVISOR",
                timestamp="2024-11-15T12:15:00Z",
                idempotency_key="SUPERSEDE-AMBIG",
                reason="Terms changed",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_cancellation_ref="RTO-SINGLE",
                order_wide_cancellation_scope=False,
            )
        self.assertIn("Ambiguous Shipment Resolution", str(ctx.exception))

    def test_supersede_order_bidirectional_audit_linkage(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = apply_supplier_acknowledgment(fo, ack, "GATEWAY", "ACK-CHG")

        new_approved_po = _make_approved_po(
            final_order_qty=36.0,
            unit_procurement_cost=35.0,
            po_id="PO-REPLACE-LINK",
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        superseded_fo, replacement_fo = supersede_fulfilment_order(
            original_fo=fo,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REPLACE-LINK",
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T12:15:00Z",
            idempotency_key="SUPERSEDE-LINK-KEY",
            reason="Supplier price updated; commercial renegotiation completed.",
            supplier_cancellation_ref="SUPP-CNC-LINK",
        )

        # Check superseded FO points to replacement
        cancel_evt = superseded_fo.event_history[-1]
        self.assertEqual(cancel_evt.details["superseded_by_po_id"], new_approved_po.po_id)
        self.assertEqual(cancel_evt.details["superseded_by_fo_id"], replacement_fo.fo_id)

        # Check replacement FO points to superseded FO
        create_evt = replacement_fo.event_history[0]
        self.assertEqual(create_evt.details["supersedes_fo_id"], superseded_fo.fo_id)
        self.assertEqual(create_evt.details["supersedes_po_id"], superseded_fo.po_id)

    def test_confirm_cancellation_ambiguous_shipment_resolution_rejected(self):
        fo = create_fulfilment_order(self.approved_po, "AGENT", "2024-11-15T11:00:00Z")
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        # Mock 2 shipments on order
        fo_multi = fo.model_copy(update={
            "shipment_ids": ("ASN-01", "ASN-02"),
        })
        fo_multi = request_cancellation(
            fo=fo_multi,
            reason="Store warehouse damaged; cancelling order.",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T12:00:00Z",
            idempotency_key="REQ-CNC-MULTI",
        )

        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            confirm_cancellation(
                fo=fo_multi,
                actor_id="SUPERVISOR",
                timestamp="2024-11-15T12:15:00Z",
                idempotency_key="CONF-CNC-AMBIG",
                supplier_cancellation_ref="SUPP-CNC-01",
                shipment_cancellation_ref="RTO-SINGLE",
                order_wide_cancellation_scope=False,
            )
        self.assertIn("Ambiguous Shipment Resolution", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
