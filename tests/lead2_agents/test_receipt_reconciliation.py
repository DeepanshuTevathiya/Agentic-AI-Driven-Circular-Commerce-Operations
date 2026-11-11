"""
Integration & Verification Test Suite: Cumulative Goods Receipt Reconciliation & Discrepancy Resolution (Phase 5.6.2)
=====================================================================================================================
Tests:
1. Cumulative receipt reconciliation per-PO-line and per-shipment across all persisted GRNs.
2. Multi-GRN reconciliation against single shipment and multi-shipment reconciliation against single PO.
3. Quantity conservation: physical excess recorded as rejected excess, never inflating approved order quantities.
4. Partial delivery & active inbound pipeline tracking: order remains PARTIALLY_RECEIVED while units remain outstanding.
5. Carrier tracking DELIVERED informs process but never authorizes or completes receipt.
6. Explicit CLOSED_SHORT workflow recording actor, timestamp, reason, and remaining balance without mutating history.
7. Discrepancy and rejected-goods resolution (RTO, certified disposal, supplier credit note, excess return, short delivery).
8. Physical custody resolution requiring shipment-specific evidence (no silent resolution via GRN alone).
9. Quality-hold reinspection release: blocking missing shelf-life, blocking <60% ratio, allowing >=60% ratio.
10. Quality-hold release stock adjustment: increments accepted, decrements rejected, transitions to RECEIVED if complete.
11. Double-resolution prevention: cannot resolve more than rejected quantity across resolutions.
12. Durable idempotency: exact replays are no-ops; conflicting resolution payloads are rejected.
13. Optimistic concurrency and transactional integrity with rollback.
14. Persistence across process restart: resolutions and reconciliation balances survive restart.
15. Strict scope boundaries: 0 inventory ledger writes, 0 PurchaseOrder mutations.
"""

from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from typing import Tuple

from src.agents.fulfilment import (
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DiscrepancyReason,
    DiscrepancyResolution,
    DiscrepancyResolutionType,
    DockInspectionLine,
    FulfilmentActorType,
    FulfilmentBatch,
    FulfilmentEvent,
    FulfilmentEventType,
    FulfilmentGovernanceError,
    FulfilmentOrder,
    FulfilmentOrderStatus,
    FulfilmentPersistenceError,
    FulfilmentRepository,
    FulfilmentService,
    GoodsReceiptNote,
    IdempotencyConflictError,
    LineReceiptReconciliation,
    OrderReceiptReconciliation,
    QualityHoldResolution,
    ShipmentLineItem,
    ShipmentReceiptReconciliation,
    ShipmentTrackingEvent,
    ShipmentTrackingStatus,
    StaleUpdateError,
    SupplierAcknowledgment,
    SupplierAcknowledgmentStatus,
    calculate_order_receipt_reconciliation,
)
from src.agents.procurement.approval import approve_purchase_order
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
)
from tests.lead2_agents.test_fulfilment_state_machine import _make_approved_po


def _setup_in_transit_order(
    service: FulfilmentService,
    po: PurchaseOrder,
    shipment_id: str = "ASN-01",
    dispatched_qty: float = 24.0,
    carrier: str = "DELHIVERY",
) -> FulfilmentOrder:
    """Helper to set up an order in IN_TRANSIT status ready for dock intake."""
    fo = service.create_order(po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
    fo = service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT")
    ack = SupplierAcknowledgment(
        acknowledgment_id=f"ACK-{fo.fo_id}",
        fo_id=fo.fo_id,
        po_id=fo.po_id,
        supplier_id=fo.supplier_id,
        status=SupplierAcknowledgmentStatus.ACCEPTED,
        acknowledged_at="2024-11-15T11:30:00Z",
        expected_delivery_at="2024-11-16T08:00:00Z",
    )
    fo = service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")
    asn = AdvanceShipmentNotice(
        shipment_id=shipment_id,
        fo_id=fo.fo_id,
        po_id=fo.po_id,
        supplier_id=fo.supplier_id,
        carrier=carrier,
        tracking_reference=f"TRK-{shipment_id}",
        dispatch_at="2024-11-15T14:00:00Z",
        expected_arrival_at="2024-11-16T08:00:00Z",
        lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=dispatched_qty),),
    )
    fo = service.dispatch_shipment(fo.fo_id, asn, actor_id="GATEWAY")
    return fo


def _make_multi_sku_approved_po() -> PurchaseOrder:
    """Helper to create an approved PO with two distinct SKUs."""
    store_id = "DS_BLR_KORAMANGALA"
    item1 = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id="SKU_MILK_1L",
        category="Dairy",
        raw_suggested_qty=20.0,
        case_pack_rounded_qty=24.0,
        final_order_qty=24.0,
        case_pack_size=12,
        moq=10.0,
        unit_retail_price=50.0,
        unit_procurement_cost=30.0,
        total_line_cost=720.0,
        requires_escalation=False,
        escalation_reasons=[],
    )
    item2 = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id="SKU_CURD_400G",
        category="Dairy",
        raw_suggested_qty=10.0,
        case_pack_rounded_qty=12.0,
        final_order_qty=12.0,
        case_pack_size=6,
        moq=6.0,
        unit_retail_price=40.0,
        unit_procurement_cost=25.0,
        total_line_cost=300.0,
        requires_escalation=False,
        escalation_reasons=[],
    )
    pending_po = PurchaseOrder(
        po_id="PO-TEST-MULTI-SKU",
        created_date="2024-11-15",
        store_id=store_id,
        supplier_id="SIM_SUPP_DAIRY_PRIMARY",
        supplier_type=SupplierType.PRIMARY,
        items=[item1, item2],
        total_order_qty=36.0,
        total_order_cost=1020.0,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        urgency=PurchaseOrderUrgency.STANDARD,
    )
    return approve_purchase_order(
        po=pending_po,
        approved_by="TEST_LEAD",
        approval_timestamp="2024-11-15T10:30:00Z",
        approval_reason="Approved multi-sku test order",
    )


class TestReceiptReconciliationAndDiscrepancies(unittest.TestCase):
    """Comprehensive test suite for Phase 5.6.2 receipt reconciliation & discrepancy resolution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_fulfilment.db"
        self.service = FulfilmentService(db_path=self.db_path)
        self.po = _make_approved_po()

    def tearDown(self) -> None:
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Cumulative Receipt Reconciliation Tests
    # -------------------------------------------------------------------------

    def test_single_shipment_multi_grn_reconciliation(self) -> None:
        """Tests cumulative balances across multiple GRNs for a single shipment."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-01", dispatched_qty=24.0)

        # GRN 1: delivers 10 units (all accepted)
        grn1 = GoodsReceiptNote(
            grn_id="GRN-01-A",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("BATCH-01-A",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn1)
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Intermediate reconciliation check
        rec1 = self.service.reconcile_receipts(fo.fo_id)
        self.assertEqual(rec1.total_ordered_quantity, 24.0)
        self.assertEqual(rec1.total_delivered_quantity, 10.0)
        self.assertEqual(rec1.total_accepted_quantity, 10.0)
        self.assertEqual(rec1.total_remaining_confirmed_outstanding_quantity, 14.0)
        self.assertEqual(rec1.total_remaining_unfulfilled_quantity, 14.0)
        self.assertFalse(rec1.is_fully_received)
        self.assertTrue(rec1.can_close_short)

        # GRN 2: delivers remaining 14 units (all accepted)
        grn2 = GoodsReceiptNote(
            grn_id="GRN-01-B",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T11:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=14.0,
                    accepted_quantity=14.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("BATCH-01-B",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn2)
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)

        # Final reconciliation check
        rec2 = self.service.reconcile_receipts(fo.fo_id)
        self.assertEqual(rec2.total_ordered_quantity, 24.0)
        self.assertEqual(rec2.total_delivered_quantity, 24.0)
        self.assertEqual(rec2.total_accepted_quantity, 24.0)
        self.assertEqual(rec2.total_rejected_quantity, 0.0)
        self.assertEqual(rec2.total_remaining_confirmed_outstanding_quantity, 0.0)
        self.assertEqual(rec2.total_remaining_unfulfilled_quantity, 0.0)
        self.assertTrue(rec2.is_fully_received)
        self.assertFalse(rec2.can_close_short)

        # Shipment level check
        s_rec = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-01")
        self.assertEqual(s_rec.delivered_quantity, 24.0)
        self.assertEqual(s_rec.accepted_quantity, 24.0)
        self.assertTrue(s_rec.is_custody_resolved)

    def test_multi_shipment_multi_sku_reconciliation(self) -> None:
        """Tests multi-shipment split delivery against a multi-SKU purchase order."""
        multi_po = _make_multi_sku_approved_po()
        fo = self.service.create_order(multi_po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id=f"ACK-{fo.fo_id}",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")

        # Split Shipment 1: 12 Milk, 6 Curd
        asn1 = AdvanceShipmentNotice(
            shipment_id="SHIP-MULTI-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            carrier="DELHIVERY",
            tracking_reference="TRK-SM-1",
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(
                ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=12.0),
                ShipmentLineItem(sku_id="SKU_CURD_400G", shipped_quantity=6.0),
            ),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn1, actor_id="GATEWAY")

        # Split Shipment 2: 12 Milk, 6 Curd
        asn2 = AdvanceShipmentNotice(
            shipment_id="SHIP-MULTI-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            carrier="BLUEDART",
            tracking_reference="TRK-SM-2",
            dispatch_at="2024-11-15T15:00:00Z",
            expected_arrival_at="2024-11-16T12:00:00Z",
            lines=(
                ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=12.0),
                ShipmentLineItem(sku_id="SKU_CURD_400G", shipped_quantity=6.0),
            ),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn2, actor_id="GATEWAY")

        # Dock receipt for Shipment 1
        grn1 = GoodsReceiptNote(
            grn_id="GRN-SM-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-MULTI-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("B-M-1",),
                ),
                DockInspectionLine(
                    sku_id="SKU_CURD_400G",
                    delivered_quantity=6.0,
                    accepted_quantity=6.0,
                    rejected_quantity=0.0,
                    unit_cost=25.0,
                    remaining_shelf_life_days=10,
                    total_shelf_life_days=15,
                    batch_references=("B-C-1",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn1)

        # Dock receipt for Shipment 2
        grn2 = GoodsReceiptNote(
            grn_id="GRN-SM-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-MULTI-2",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T13:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("B-M-2",),
                ),
                DockInspectionLine(
                    sku_id="SKU_CURD_400G",
                    delivered_quantity=6.0,
                    accepted_quantity=6.0,
                    rejected_quantity=0.0,
                    unit_cost=25.0,
                    remaining_shelf_life_days=10,
                    total_shelf_life_days=15,
                    batch_references=("B-C-2",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn2)
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)

        # Reconcile across both shipments and all lines
        rec = self.service.reconcile_receipts(fo.fo_id)
        self.assertEqual(rec.total_ordered_quantity, 36.0)
        self.assertEqual(rec.total_delivered_quantity, 36.0)
        self.assertEqual(rec.total_accepted_quantity, 36.0)
        self.assertEqual(len(rec.shipments), 2)
        self.assertEqual(len(rec.lines), 2)
        self.assertTrue(rec.is_fully_received)

    def test_physical_excess_recorded_as_rejected_excess_does_not_inflate_po(self) -> None:
        """Physical excess delivered at dock must remain recorded as rejected excess and never increase approved PO quantity."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-EXCESS", dispatched_qty=24.0)

        # Supplier physical over-delivery: delivers 30 units (6 excess)
        grn = GoodsReceiptNote(
            grn_id="GRN-EXCESS-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-EXCESS",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            rejected_goods_disposition="RTO-EXCESS-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=30.0,
                    accepted_quantity=24.0,  # Max allowable accepted
                    rejected_quantity=6.0,
                    excess_quantity=6.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.EXCESS_DELIVERY,),
                    batch_references=("BATCH-OK",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)

        rec = self.service.reconcile_receipts(fo.fo_id)
        self.assertEqual(rec.total_ordered_quantity, 24.0)  # PO balance unchanged!
        self.assertEqual(rec.total_delivered_quantity, 30.0)  # Physical intake captured
        self.assertEqual(rec.total_accepted_quantity, 24.0)  # Sellable stock capped
        self.assertEqual(rec.total_rejected_quantity, 6.0)
        self.assertEqual(rec.total_excess_quantity, 6.0)
        self.assertEqual(rec.total_remaining_unfulfilled_quantity, 0.0)

    # -------------------------------------------------------------------------
    # 2. Partial Delivery & Closed-Short Handling Tests
    # -------------------------------------------------------------------------

    def test_active_pipeline_maintained_during_partial_delivery(self) -> None:
        """Partially received order keeps remaining confirmed units in active inbound pipeline."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-PARTIAL", dispatched_qty=24.0)

        # Deliver 8 units out of 24
        grn = GoodsReceiptNote(
            grn_id="GRN-PARTIAL-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-PARTIAL",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=8.0,
                    accepted_quantity=8.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("BATCH-PARTIAL-1",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Active pipeline quantity derived from order aggregate
        self.assertEqual(fo.total_confirmed_pipeline_qty, 16.0)
        pipe_qty = self.service.get_pipeline_quantity(store_id=fo.store_id, sku_id="SKU_MILK_1L")
        self.assertEqual(pipe_qty, 16.0)

    def test_carrier_delivered_does_not_authorize_or_complete_receipt(self) -> None:
        """Carrier tracking status DELIVERED must never authorize or complete goods receipt without physical dock triage."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-CARRIER", dispatched_qty=24.0)

        # Advance tracking to DELIVERED
        self.service.register_shipment(fo.fo_id, "SHIP-CARRIER", "DELHIVERY")
        self.service.tracking_service.advance_to_tracking_status("SHIP-CARRIER", ShipmentTrackingStatus.DELIVERED)

        # Verify tracking is DELIVERED
        trk = self.service.get_shipment_tracking("SHIP-CARRIER")
        self.assertEqual(trk.status, ShipmentTrackingStatus.DELIVERED)

        # Fulfilment order MUST still be IN_TRANSIT with 0 delivered and 24 pipeline
        fo_reloaded = self.service.get_order_or_raise(fo.fo_id)
        self.assertEqual(fo_reloaded.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(fo_reloaded.total_received_accepted_qty, 0.0)
        self.assertEqual(fo_reloaded.total_confirmed_pipeline_qty, 24.0)

    def test_closed_short_workflow_preserves_historical_records(self) -> None:
        """Authorized CLOSED_SHORT terminates active unfulfilled balance without mutating historical events or accepted goods."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-CS", dispatched_qty=24.0)

        # Partial receipt of 10 units
        grn = GoodsReceiptNote(
            grn_id="GRN-CS-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-CS",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("BATCH-CS-1",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Store manager executes CLOSE_SHORT
        fo_closed = self.service.close_short_order(
            fo_id=fo.fo_id,
            closure_reason="Supplier production breakdown, remaining 14 units unavailable",
            actor_id="STORE_MANAGER_01",
            timestamp="2024-11-16T14:00:00Z",
            idempotency_key="CLS:TEST:001",
        )
        self.assertEqual(fo_closed.status, FulfilmentOrderStatus.CLOSED_SHORT)
        self.assertEqual(fo_closed.closed_by, "STORE_MANAGER_01")
        self.assertEqual(fo_closed.closed_unfulfilled_qty, 14.0)
        self.assertEqual(fo_closed.closure_reason, "Supplier production breakdown, remaining 14 units unavailable")

        # Pipeline is cleared
        self.assertEqual(fo_closed.total_confirmed_pipeline_qty, 0.0)

        # Accepted stock remains preserved
        self.assertEqual(fo_closed.total_received_accepted_qty, 10.0)

        # Reconciliation reflects closure
        rec = self.service.reconcile_receipts(fo.fo_id)
        self.assertEqual(rec.status, FulfilmentOrderStatus.CLOSED_SHORT)
        self.assertFalse(rec.can_close_short)
        self.assertEqual(rec.total_accepted_quantity, 10.0)
        self.assertEqual(rec.total_remaining_unfulfilled_quantity, 14.0)

        # Attempting receipt after CLOSED_SHORT is rejected
        grn_after = GoodsReceiptNote(
            grn_id="GRN-CS-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-CS",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T15:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=5.0,
                    accepted_quantity=5.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("BATCH-CS-2",),
                ),
            ),
        )
        with self.assertRaises(ValueError):
            self.service.process_dock_intake(fo.fo_id, grn_after)

    # -------------------------------------------------------------------------
    # 3. Discrepancy & Rejected-Goods Resolution Tests
    # -------------------------------------------------------------------------

    def test_discrepancy_resolution_rto_and_credit_note(self) -> None:
        """Tests explicit resolution of transit-damaged and missing goods with verifiable disposition references."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-DISC", dispatched_qty=24.0)

        # GRN with 18 accepted and 6 damaged
        grn = GoodsReceiptNote(
            grn_id="GRN-DISC-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-DISC",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=18.0,
                    rejected_quantity=6.0,
                    damaged_quantity=6.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-OK-18",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # Resolve 4 units via Return-To-Origin (RTO)
        fo, res1 = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-DISC",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=4.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-BOL-7701",
            actor_id="OPS_SUPERVISOR_1",
            timestamp="2024-11-16T12:00:00Z",
            idempotency_key="RES:RTO:001",
        )
        self.assertEqual(res1.resolution_type, DiscrepancyResolutionType.RETURN_TO_ORIGIN)
        self.assertEqual(res1.quantity, 4.0)
        self.assertEqual(res1.disposition_reference, "RTO-BOL-7701")

        # Resolve remaining 2 units via Supplier Credit Note
        fo, res2 = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-DISC",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.SUPPLIER_CREDIT_NOTE,
            quantity=2.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="CRN-VEND-8820",
            actor_id="OPS_SUPERVISOR_1",
            timestamp="2024-11-16T12:30:00Z",
            idempotency_key="RES:CRN:001",
        )
        self.assertEqual(res2.resolution_type, DiscrepancyResolutionType.SUPPLIER_CREDIT_NOTE)
        self.assertEqual(res2.quantity, 2.0)

        # Verify query methods in repository
        resolutions = self.service.get_discrepancy_resolutions_for_order(fo.fo_id)
        self.assertEqual(len(resolutions), 2)

        # Verify shipment custody is resolved
        s_rec = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-DISC")
        self.assertTrue(s_rec.is_custody_resolved)

    def test_prevent_double_resolution(self) -> None:
        """The system must prevent resolving more than the recorded rejected quantity."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-DBL", dispatched_qty=24.0)

        # 4 units rejected
        grn = GoodsReceiptNote(
            grn_id="GRN-DBL-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-DBL",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=20.0,
                    rejected_quantity=4.0,
                    damaged_quantity=4.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-DBL",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # Resolve 3 units
        fo, _ = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-DBL",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=3.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-01",
            idempotency_key="KEY-RTO-1",
            timestamp="2024-11-16T12:00:00Z",
        )

        # Attempt to resolve 2 more units (3 + 2 = 5 > 4 rejected) -> must fail!
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.resolve_discrepancy(
                fo_id=fo.fo_id,
                shipment_id="SHIP-DBL",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
                quantity=2.0,
                discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
                disposition_reference="DISP-01",
                idempotency_key="KEY-DISP-1",
                timestamp="2024-11-16T12:30:00Z",
            )
        self.assertIn("Double-Resolution Violation", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Quality-Hold Resolution Tests
    # -------------------------------------------------------------------------

    def test_quality_hold_reinspection_release_rules(self) -> None:
        """Quality hold reinspection release must enforce minimum 60% shelf life ratio and correctly update stock."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-QH", dispatched_qty=24.0)

        # GRN where 6 units were held/rejected due to temperature abuse/quality defect
        grn = GoodsReceiptNote(
            grn_id="GRN-QH-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-QH",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=18.0,
                    rejected_quantity=6.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.QUALITY_DEFECT,),
                    batch_references=("BATCH-OK-QH",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # 1. Missing shelf life must be blocked
        with self.assertRaises(FulfilmentGovernanceError) as ctx1:
            self.service.resolve_quality_hold(
                fo_id=fo.fo_id,
                shipment_id="SHIP-QH",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
                quantity=6.0,
                disposition_reference="QA-RELEASE-001",
                verified_remaining_shelf_life_days=None,  # Missing!
                total_shelf_life_days=7,
                timestamp="2024-11-16T12:00:00Z",
                idempotency_key="QHR-FAIL-1",
            )
        self.assertIn("Quality Hold Violation", str(ctx1.exception))

        # 2. Ratio < 60% (e.g. 2 days / 7 days = 28.5%) must be blocked
        with self.assertRaises(FulfilmentGovernanceError) as ctx2:
            self.service.resolve_quality_hold(
                fo_id=fo.fo_id,
                shipment_id="SHIP-QH",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
                quantity=6.0,
                disposition_reference="QA-RELEASE-002",
                verified_remaining_shelf_life_days=2,
                total_shelf_life_days=7,
                timestamp="2024-11-16T12:00:00Z",
                idempotency_key="QHR-FAIL-2",
            )
        self.assertIn("below the minimum acceptable threshold of 60.0%", str(ctx2.exception))

        # 3. Ratio >= 60% (e.g. 5 days / 7 days = 71.4%) succeeds and releases stock!
        fo_released, qhr = self.service.resolve_quality_hold(
            fo_id=fo.fo_id,
            shipment_id="SHIP-QH",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
            quantity=6.0,
            disposition_reference="QA-RELEASE-003",
            verified_remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            timestamp="2024-11-16T12:00:00Z",
            idempotency_key="QHR-PASS-1",
            inspection_notes="Secondary temperature lab test cleared batch. Organoleptic testing passed.",
        )
        self.assertEqual(qhr.resolution_type, DiscrepancyResolutionType.REINSPECTION_RELEASE)
        self.assertEqual(qhr.quantity, 6.0)

        # Order must now be fully accepted (18 + 6 = 24) and transition to RECEIVED!
        self.assertEqual(fo_released.status, FulfilmentOrderStatus.RECEIVED)
        line = fo_released.lines[0]
        self.assertEqual(line.accepted_quantity, 24.0)
        self.assertEqual(line.rejected_quantity, 0.0)
        self.assertEqual(line.delivered_quantity, 24.0)

        # Audit event RESOLVE_QUALITY_HOLD must be in event history
        last_evt = fo_released.event_history[-1]
        self.assertEqual(last_evt.event_type, FulfilmentEventType.RESOLVE_QUALITY_HOLD)

    def test_quality_hold_certified_disposal(self) -> None:
        """Quality hold certified disposal resolves hold without releasing stock into inventory."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-DISP", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-DISP-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-DISP",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=18.0,
                    rejected_quantity=6.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.TEMPERATURE_ABUSE,),
                    batch_references=("BATCH-OK-DISP",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # Certified disposal of 6 units
        fo, qhr = self.service.resolve_quality_hold(
            fo_id=fo.fo_id,
            shipment_id="SHIP-DISP",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
            quantity=6.0,
            disposition_reference="CERT-BIO-WASTE-109",
            inspector_id="QA_SUPERVISOR",
            timestamp="2024-11-16T12:00:00Z",
            idempotency_key="QHR-DISP-1",
            inspection_notes="Severe temperature abuse detected. Bio-waste certified disposal scheduled.",
        )
        self.assertEqual(qhr.resolution_type, DiscrepancyResolutionType.CERTIFIED_DISPOSAL)
        # Order remains PARTIALLY_RECEIVED (accepted is still 18)
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(fo.lines[0].accepted_quantity, 18.0)
        self.assertEqual(fo.lines[0].rejected_quantity, 6.0)

        # Check repository query
        qhrs = self.service.get_quality_hold_resolutions_for_order(fo.fo_id)
        self.assertEqual(len(qhrs), 1)

    # -------------------------------------------------------------------------
    # 5. Custody & Explicit Shipment Resolution Tests
    # -------------------------------------------------------------------------

    def test_zero_accepted_grn_requires_documented_disposition_for_custody(self) -> None:
        """A GRN with zero accepted units must not silently resolve custody without verified disposition evidence."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-REJ-ALL", dispatched_qty=24.0)

        # Entire consignment rejected at dock, NO disposition specified yet
        grn = GoodsReceiptNote(
            grn_id="GRN-REJ-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-REJ-ALL",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            rejected_goods_disposition=None,  # No disposition yet
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=0.0,
                    rejected_quantity=24.0,
                    damaged_quantity=24.0,
                    unit_cost=30.0,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    notes="Vehicle collision; total consignment loss.",
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # Shipment custody must NOT be resolved!
        self.assertNotIn("SHIP-REJ-ALL", fo.resolved_shipment_ids)
        s_rec = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-REJ-ALL")
        self.assertFalse(s_rec.is_custody_resolved)

        # Now explicitly resolve custody with RTO waybill
        fo_custody = self.service.resolve_shipment_custody(
            fo_id=fo.fo_id,
            shipment_id="SHIP-REJ-ALL",
            disposition_reference="RTO-WAYBILL-COL-99",
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-16T12:00:00Z",
            idempotency_key="CUSTODY:REJ:1",
            notes="Carrier picked up damaged crates for insurer inspection.",
        )
        self.assertIn("SHIP-REJ-ALL", fo_custody.resolved_shipment_ids)
        s_rec_after = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-REJ-ALL")
        self.assertTrue(s_rec_after.is_custody_resolved)
        self.assertEqual(s_rec_after.disposition_reference, "RTO-WAYBILL-COL-99")

    # -------------------------------------------------------------------------
    # 6. Concurrency, Idempotency & Persistence Tests
    # -------------------------------------------------------------------------

    def test_durable_idempotency_for_resolution_operations(self) -> None:
        """Exact replays of resolution commands must be no-ops; conflicting payloads must be rejected."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-IDEMP", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-IDEMP-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-IDEMP",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=20.0,
                    rejected_quantity=4.0,
                    damaged_quantity=4.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-IDEMP",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # 1. First execution
        cmd_key = "IDEMP-RES-CMD-001"
        fo_res, _ = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-IDEMP",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=2.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-IDEMP-1",
            timestamp="2024-11-16T12:00:00Z",
            idempotency_key=cmd_key,
        )
        event_count = len(fo_res.event_history)

        # 2. Exact retry: should be no-op with identical event count
        fo_retry, _ = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-IDEMP",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=2.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-IDEMP-1",
            timestamp="2024-11-16T12:00:00Z",
            idempotency_key=cmd_key,
        )
        self.assertEqual(len(fo_retry.event_history), event_count)

        # 3. Conflicting payload: same key with different quantity (e.g. 3.0 instead of 2.0)
        with self.assertRaises((FulfilmentGovernanceError, IdempotencyConflictError)):
            self.service.resolve_discrepancy(
                fo_id=fo.fo_id,
                shipment_id="SHIP-IDEMP",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
                quantity=3.0,  # Conflict!
                discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
                disposition_reference="RTO-IDEMP-1",
                timestamp="2024-11-16T12:00:00Z",
                idempotency_key=cmd_key,
            )

    def test_optimistic_concurrency_control(self) -> None:
        """Supplying a stale expected_version must raise StaleUpdateError and prevent writes."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-OCC", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-OCC-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-OCC",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=20.0,
                    rejected_quantity=4.0,
                    damaged_quantity=4.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-OCC",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        current_version = self.service.get_order_version(fo.fo_id)

        # Call with stale expected_version
        with self.assertRaises(StaleUpdateError):
            self.service.resolve_discrepancy(
                fo_id=fo.fo_id,
                shipment_id="SHIP-OCC",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
                quantity=2.0,
                discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
                disposition_reference="RTO-OCC-1",
                expected_version=current_version - 1,  # Stale!
            )

    def test_persistence_across_process_restart(self) -> None:
        """All resolutions, GRNs, and reconciliation balances must survive process restart."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-RESTART", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-RESTART-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-RESTART",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=20.0,
                    rejected_quantity=4.0,
                    damaged_quantity=4.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-RESTART",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        fo, _ = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-RESTART",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=4.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-RESTART-BOL",
            timestamp="2024-11-16T12:00:00Z",
        )

        # Simulate complete restart by closing current service & opening fresh instance
        self.service.close()
        new_service = FulfilmentService(db_path=self.db_path)
        try:
            reloaded_fo = new_service.get_order_or_raise(fo.fo_id)
            self.assertEqual(len(reloaded_fo.discrepancy_resolutions), 1)
            self.assertEqual(reloaded_fo.discrepancy_resolutions[0].disposition_reference, "RTO-RESTART-BOL")

            # Check reconciliation via new service
            rec = new_service.reconcile_receipts(fo.fo_id)
            self.assertEqual(rec.total_delivered_quantity, 24.0)
            self.assertEqual(rec.total_accepted_quantity, 20.0)
            self.assertEqual(rec.total_rejected_quantity, 4.0)
            self.assertEqual(rec.total_damaged_quantity, 4.0)

            # Check relational query via new repository
            res_list = new_service.get_discrepancy_resolutions_for_order(fo.fo_id)
            self.assertEqual(len(res_list), 1)
            self.assertEqual(res_list[0].resolution_id, reloaded_fo.discrepancy_resolutions[0].resolution_id)
        finally:
            new_service.close()

    # -------------------------------------------------------------------------
    # 7. Strict Scope Boundaries Tests
    # -------------------------------------------------------------------------

    def test_strict_scope_boundaries_no_inventory_or_po_mutations(self) -> None:
        """Reconciliation and discrepancy resolution must not mutate the commercial PO or write to inventory tables."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-SCOPE", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-SCOPE-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-SCOPE",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("BATCH-SCOPE",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        rec = self.service.reconcile_receipts(fo.fo_id)

        # 1. PurchaseOrder remains strictly in APPROVED status with original quantities
        self.assertEqual(self.po.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(self.po.items[0].final_order_qty, 24.0)
        self.assertEqual(fo.po_snapshot.status, PurchaseOrderStatus.APPROVED)

        # 2. Database contains zero inventory ledger tables
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            for forbidden_table in ["inventory_ledger", "stock_allocation", "on_hand_stock", "replenishment_orders"]:
                self.assertNotIn(forbidden_table, tables)
        finally:
            conn.close()


class TestAdversarialPhase56Hardening(unittest.TestCase):
    """
    Comprehensive adversarial regression tests verifying Phase 5.6 hardening:
    1. Partial rejection keeps shipment custody unresolved until all rejected units are disposed.
    2. Undelivered shortage acknowledgment does not resolve dock custody of physical rejected goods.
    3. Reinspection release cannot release transit-damaged goods or unapproved excess.
    4. Strict validation of shelf-life bounds and ratios on reinspection release.
    5. Cancellation safeguard prevents cancelling an order with unresolved rejected dock inventory.
    6. Dock intake rejects shipments without recorded ASN dispatch events.
    7. Idempotency conflict detection on resolution ID collision with differing attributes.
    8. Audit event details include receiving dock and full inspection lines breakdown.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "adversarial_test.db"
        self.service = FulfilmentService(db_path=self.db_path)
        self.po = _make_approved_po()

    def tearDown(self) -> None:
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_partial_rejection_keeps_custody_unresolved_until_fully_disposed(self) -> None:
        """A GRN with partial rejection must NOT mark shipment custody resolved until all rejected units are disposed."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-PART-REJ", dispatched_qty=24.0)
        grn = GoodsReceiptNote(
            grn_id="GRN-PART-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-PART-REJ",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=14.0,
                    rejected_quantity=10.0,
                    damaged_quantity=10.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-14",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        s_rec = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-PART-REJ")
        self.assertFalse(s_rec.is_custody_resolved, "Custody must not be resolved when 10 units are damaged and unresolved.")

        # Partial resolution of 6 units out of 10
        fo, _ = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-PART-REJ",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=6.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-PART-01",
            idempotency_key="KEY-RTO-P1",
            timestamp="2024-11-16T12:00:00Z",
        )
        s_rec = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-PART-REJ")
        self.assertFalse(s_rec.is_custody_resolved, "Custody must remain unresolved when 4 units are still in limbo.")

        # Final resolution of remaining 4 units
        fo, _ = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-PART-REJ",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
            quantity=4.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="DISP-PART-02",
            idempotency_key="KEY-DISP-P2",
            timestamp="2024-11-16T12:30:00Z",
        )
        s_rec = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-PART-REJ")
        self.assertTrue(s_rec.is_custody_resolved, "Custody must now be resolved after all 10 units are disposed.")

    def test_short_delivery_acknowledgement_does_not_resolve_dock_rejected_goods(self) -> None:
        """Acknowledging undelivered short quantity does not resolve damaged units sitting on the dock."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-SHORT-DAM", dispatched_qty=20.0)
        grn = GoodsReceiptNote(
            grn_id="GRN-SHORT-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-SHORT-DAM",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=20.0,
                    accepted_quantity=15.0,
                    rejected_quantity=5.0,
                    damaged_quantity=5.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-15",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # Acknowledge the 4 undelivered units (24 ordered - 20 delivered = 4 short)
        fo, _ = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-SHORT-DAM",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.SHORT_DELIVERY_ACKNOWLEDGED,
            quantity=4.0,
            discrepancy_reason=DiscrepancyReason.MISSING_OR_SHORT,
            disposition_reference="SHORT-ACK-01",
            idempotency_key="KEY-SHORT-ACK",
            timestamp="2024-11-16T12:00:00Z",
        )
        s_rec = self.service.get_shipment_reconciliation(fo.fo_id, "SHIP-SHORT-DAM")
        self.assertFalse(s_rec.is_custody_resolved, "Shipment custody must remain unresolved because 5 damaged units remain on dock.")

    def test_reinspection_release_cannot_release_transit_damage(self) -> None:
        """Transit damage cannot be released into sellable stock via reinspection release."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-REINSP", dispatched_qty=24.0)
        grn = GoodsReceiptNote(
            grn_id="GRN-REINSP-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-REINSP",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=14.0,
                    rejected_quantity=10.0,
                    damaged_quantity=6.0,
                    short_shelf_life_quantity=4.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT, DiscrepancyReason.EXPIRED_OR_SHORT_SHELF_LIFE),
                    batch_references=("BATCH-14",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # Reinspection release of 4 shelf-life held units with verified 5 days (5/7 = 71.4% >= 60%) succeeds
        fo, qh1 = self.service.resolve_quality_hold(
            fo_id=fo.fo_id,
            shipment_id="SHIP-REINSP",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
            quantity=4.0,
            disposition_reference="QA-RELEASE-01",
            inspector_id="QA-INSP-01",
            inspection_notes="Laboratory re-test confirmed satisfactory shelf stability",
            verified_remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            idempotency_key="KEY-QH-01",
            timestamp="2024-11-16T12:00:00Z",
        )
        self.assertEqual(qh1.quantity, 4.0)

        # Attempting to release any more units (transit damaged stock) must fail
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.resolve_quality_hold(
                fo_id=fo.fo_id,
                shipment_id="SHIP-REINSP",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
                quantity=2.0,
                disposition_reference="QA-RELEASE-02",
                inspector_id="QA-INSP-01",
                inspection_notes="Attempting to release crushed bottles",
                verified_remaining_shelf_life_days=5,
                total_shelf_life_days=7,
                idempotency_key="KEY-QH-02",
                timestamp="2024-11-16T12:30:00Z",
            )
        self.assertIn("Reinspection Release Violation", str(ctx.exception))

    def test_reinspection_release_strict_shelf_life_validation_boundaries(self) -> None:
        """Enforces boundary checks on shelf life: ratio 59.9% fails, 60.0% passes, invalid days fail."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-BOUND", dispatched_qty=24.0)
        grn = GoodsReceiptNote(
            grn_id="GRN-BOUND-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-BOUND",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=14.0,
                    rejected_quantity=10.0,
                    short_shelf_life_quantity=10.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    discrepancy_reasons=(DiscrepancyReason.EXPIRED_OR_SHORT_SHELF_LIFE,),
                    batch_references=("BATCH-14",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # 1. Total shelf life <= 0 fails
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.resolve_quality_hold(
                fo_id=fo.fo_id,
                shipment_id="SHIP-BOUND",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
                quantity=2.0,
                disposition_reference="QA-BOUND-01",
                inspector_id="QA-INSP",
                inspection_notes="Negative total",
                verified_remaining_shelf_life_days=5,
                total_shelf_life_days=0,
                idempotency_key="KEY-BND-01",
                timestamp="2024-11-16T12:00:00Z",
            )
        self.assertIn("must be strictly positive", str(ctx.exception))

        # 2. Remaining shelf life > total shelf life fails
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.resolve_quality_hold(
                fo_id=fo.fo_id,
                shipment_id="SHIP-BOUND",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
                quantity=2.0,
                disposition_reference="QA-BOUND-02",
                inspector_id="QA-INSP",
                inspection_notes="Remaining exceeds total",
                verified_remaining_shelf_life_days=15,
                total_shelf_life_days=10,
                idempotency_key="KEY-BND-02",
                timestamp="2024-11-16T12:00:00Z",
            )
        self.assertIn("cannot exceed total_shelf_life_days", str(ctx.exception))

        # 3. Ratio < 60% (e.g. 5 days out of 10 = 50%) fails
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.resolve_quality_hold(
                fo_id=fo.fo_id,
                shipment_id="SHIP-BOUND",
                sku_id="SKU_MILK_1L",
                resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
                quantity=2.0,
                disposition_reference="QA-BOUND-03",
                inspector_id="QA-INSP",
                inspection_notes="50% ratio",
                verified_remaining_shelf_life_days=5,
                total_shelf_life_days=10,
                idempotency_key="KEY-BND-03",
                timestamp="2024-11-16T12:00:00Z",
            )
        self.assertIn("below the minimum acceptable threshold of 60.0%", str(ctx.exception))

        # 4. Ratio exactly 60.0% (6 days out of 10 = 60.0%) passes
        fo, qh = self.service.resolve_quality_hold(
            fo_id=fo.fo_id,
            shipment_id="SHIP-BOUND",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
            quantity=2.0,
            disposition_reference="QA-BOUND-04",
            inspector_id="QA-INSP",
            inspection_notes="Exact 60.0% ratio passes",
            verified_remaining_shelf_life_days=6,
            total_shelf_life_days=10,
            idempotency_key="KEY-BND-04",
            timestamp="2024-11-16T12:00:00Z",
        )
        self.assertEqual(qh.quantity, 2.0)

    def test_cancellation_safeguard_blocks_orders_with_unresolved_in_transit_shipments(self) -> None:
        """Cancellation confirmation is blocked if shipments remain in-transit without documented disposition."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-CNC", dispatched_qty=24.0)

        # Request cancellation while order is in transit
        fo = self.service.request_cancellation(
            fo_id=fo.fo_id,
            reason="Supplier contract dispute",
            category=CancellationReasonCategory.COMMERCIAL_TERMS_REJECTED,
            timestamp="2024-11-16T10:00:00Z",
            idempotency_key="REQ-CNC-01",
        )

        # Attempt to confirm cancellation with supplier ref but without shipment disposition evidence
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.confirm_cancellation(
                fo_id=fo.fo_id,
                supplier_cancellation_ref="SUPP-CNC-999",
                timestamp="2024-11-16T10:30:00Z",
                idempotency_key="CONF-CNC-FAIL",
            )
        self.assertIn("has active in-transit shipments", str(ctx.exception))

        # Providing documented shipment disposition evidence in shipment_resolutions allows cancellation to complete
        fo = self.service.confirm_cancellation(
            fo_id=fo.fo_id,
            supplier_cancellation_ref="SUPP-CNC-999",
            shipment_resolutions={"SHIP-CNC": "RTO-FULL-RETURN-BOL-990"},
            timestamp="2024-11-16T10:30:00Z",
            idempotency_key="CONF-CNC-PASS",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCELLED)

    def test_process_dock_intake_fails_when_shipment_has_no_asn_dispatch(self) -> None:
        """process_dock_intake fails if the shipment has no recorded ASN dispatch event on the order."""
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT")
        # Manually associate shipment without dispatching ASN event
        fo_dict = fo.model_dump()
        fo_dict["shipment_ids"] = ["SHIP-NO-ASN"]
        fo = self.service.repository.update(FulfilmentOrder.model_validate(fo_dict))

        grn = GoodsReceiptNote(
            grn_id="GRN-NO-ASN",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-NO-ASN",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("BATCH-OK",),
                ),
            ),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo.fo_id, grn)
        self.assertIn("has no recorded ASN dispatch", str(ctx.exception))

    def test_repository_idempotency_conflict_detection_for_resolutions(self) -> None:
        """Repository raises IdempotencyConflictError if resolution_id is reused with differing attributes."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-RES-CONF", dispatched_qty=24.0)
        grn = GoodsReceiptNote(
            grn_id="GRN-RES-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-RES-CONF",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=14.0,
                    rejected_quantity=10.0,
                    damaged_quantity=10.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-14",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)

        # 1. Resolve 4 units
        fo, res1 = self.service.resolve_discrepancy(
            fo_id=fo.fo_id,
            shipment_id="SHIP-RES-CONF",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=4.0,
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-ORIG-01",
            idempotency_key="KEY-RES-01",
            timestamp="2024-11-16T12:00:00Z",
        )

        # 2. Forge an order aggregate reusing resolution_id with different quantity
        fo_dict = fo.model_dump()
        conflicting_res = DiscrepancyResolution(
            resolution_id=res1.resolution_id,
            fo_id=fo.fo_id,
            shipment_id="SHIP-RES-CONF",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.RETURN_TO_ORIGIN,
            quantity=8.0,  # Changed from 4.0 to 8.0!
            discrepancy_reason=DiscrepancyReason.DAMAGED_TRANSIT,
            disposition_reference="RTO-ORIG-01",
            actor_id="OPS_SUPERVISOR_1",
            timestamp="2024-11-16T12:00:00Z",
        )
        fo_dict["discrepancy_resolutions"] = [conflicting_res.model_dump()]
        forged_fo = FulfilmentOrder.model_validate(fo_dict)

        with self.assertRaises(IdempotencyConflictError) as ctx:
            self.service.repository.update(forged_fo)
        self.assertIn("already exists with differing attributes", str(ctx.exception))

    def test_reconciliation_audit_event_traceability(self) -> None:
        """GRN audit events must contain receiving_dock and inspected_lines breakdown in details."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="SHIP-AUDIT", dispatched_qty=24.0)
        grn = GoodsReceiptNote(
            grn_id="GRN-AUDIT-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="SHIP-AUDIT",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RCV-01",
            notes="DOCK_BAY_NORTH_3",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=20.0,
                    rejected_quantity=4.0,
                    damaged_quantity=4.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                    batch_references=("BATCH-AUDIT",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        grn_event = next(e for e in fo.event_history if e.event_type == FulfilmentEventType.GOODS_RECEIPT)

        self.assertIn("receiving_dock", grn_event.details)
        self.assertIn("inspection_lines", grn_event.details)
        self.assertEqual(len(grn_event.details["inspection_lines"]), 1)
        line_detail = grn_event.details["inspection_lines"][0]
        self.assertEqual(line_detail["sku_id"], "SKU_MILK_1L")
        self.assertEqual(line_detail["delivered_quantity"], 24.0)
        self.assertEqual(line_detail["accepted_quantity"], 20.0)
        self.assertEqual(line_detail["rejected_quantity"], 4.0)
        self.assertEqual(line_detail["damaged_quantity"], 4.0)
        self.assertEqual(line_detail["discrepancy_reasons"], ["DAMAGED_TRANSIT"])


if __name__ == "__main__":
    unittest.main()
