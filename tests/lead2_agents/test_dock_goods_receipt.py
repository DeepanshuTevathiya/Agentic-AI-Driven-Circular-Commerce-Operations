"""
Integration & Verification Test Suite: Physical Goods Receipt Intake and Dock Inspection (Phase 5.6.1)
=====================================================================================================
Tests:
1. Physical dock intake workflow through FulfilmentService (process_dock_intake, receive_goods).
2. Authorized receiver validation and cross-verification of FO, approved PO, supplier, store, shipment, ASN.
3. Support for multiple shipments and multiple GRNs per shipment.
4. Carrier tracking status DELIVERED informs process (verify_carrier_arrival=True) but never creates GRN on its own.
5. Dock inspection evidence capture: delivered, accepted, damaged, rejected, excess quantities, discrepancy reasons, remaining shelf-life days, batch references.
6. Quality holds: missing shelf-life rejection, <60% remaining shelf-life ratio rejection, >=60% ratio acceptance.
7. Supervisor sign-off strictly > ₹5,000 threshold enforcement.
8. Quantity conservation preventing excess physical deliveries from increasing approved PO quantities or sellable inventory.
9. Physical custody and rejected goods disposition for zero accepted units before shipment resolution.
10. Atomic SQLite persistence of GoodsReceiptNote, inspection lines, and aggregate changes.
11. Durable idempotency: exact replays are no-ops; conflicting replays are rejected.
12. FulfilmentBatch minting for accepted stock and conversion to InventoryBatch.
13. Strict scope boundary enforcement: 0 inventory postings, 0 PO mutations.
"""

import json
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
    ShipmentLineItem,
    ShipmentTrackingEvent,
    ShipmentTrackingStatus,
    StaleUpdateError,
    SupplierAcknowledgment,
    SupplierAcknowledgmentStatus,
    TrackingLocation,
    generate_grn_fulfilment_batches,
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
        urgency=PurchaseOrderUrgency.STANDARD,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        requires_escalation=False,
        escalation_reasons=[],
        audit_trail=(),
    )
    return approve_purchase_order(
        po=pending_po,
        approved_by="OPERATOR_DEEPANSHU",
        approval_timestamp="2024-11-15T10:30:00Z",
        approval_reason="Routine multi-sku replenishment approved.",
    )


class TestPhysicalDockIntakeWorkflow(unittest.TestCase):
    """Test suite covering the dock-intake workflow, verification, and governance."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_dock_intake.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.service = FulfilmentService(repository=self.repo)
        self.po = _make_approved_po()

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_clean_dock_intake_single_sku_full_receipt(self):
        """Clean intake of full delivery transitions order to RECEIVED and persists GRN."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 4)

        grn = GoodsReceiptNote(
            grn_id="GRN-001",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-MILK-001",),
                ),
            ),
        )

        updated_fo, returned_grn = self.service.process_dock_intake(
            fo_id=fo.fo_id,
            grn=grn,
            po=self.po,
            actor_id="RECEIVER_ARUN",
        )

        # 1. State machine transition check
        self.assertEqual(updated_fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 5)
        self.assertIn("ASN-01", updated_fo.resolved_shipment_ids)
        self.assertIn("GRN-001", updated_fo.grn_ids)
        self.assertEqual(updated_fo.lines[0].accepted_quantity, 24.0)

        # 2. Relational persistence check
        persisted_grn = self.service.get_goods_receipt_note("GRN-001")
        self.assertIsNotNone(persisted_grn)
        self.assertEqual(persisted_grn.grn_id, "GRN-001")
        self.assertEqual(persisted_grn.total_accepted_quantity, 24.0)
        self.assertEqual(persisted_grn.total_accepted_cost, 720.0)

        # 3. Query methods check
        grns_for_fo = self.service.get_grns_for_order(fo.fo_id)
        self.assertEqual(len(grns_for_fo), 1)
        self.assertEqual(grns_for_fo[0].grn_id, "GRN-001")

        grns_for_shipment = self.service.get_grns_for_shipment("ASN-01")
        self.assertEqual(len(grns_for_shipment), 1)
        self.assertEqual(grns_for_shipment[0].grn_id, "GRN-001")

    def test_02_clean_dock_intake_multiple_skus(self):
        """Clean intake of multiple SKUs verifies and updates each line balance."""
        multi_po = _make_multi_sku_approved_po()
        fo = self.service.create_order(multi_po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-MULTI",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-MULTI",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(
                ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),
                ShipmentLineItem(sku_id="SKU_CURD_400G", shipped_quantity=12.0),
            ),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, actor_id="GATEWAY")

        grn = GoodsReceiptNote(
            grn_id="GRN-MULTI",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-MULTI",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-M-01",),
                ),
                DockInspectionLine(
                    sku_id="SKU_CURD_400G",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=25.0,
                    remaining_shelf_life_days=15,
                    total_shelf_life_days=20,
                    batch_references=("BATCH-C-01",),
                ),
            ),
        )

        updated_fo, _ = self.service.process_dock_intake(
            fo_id=fo.fo_id,
            grn=grn,
            po=multi_po,
            actor_id="RECEIVER_ARUN",
        )

        self.assertEqual(updated_fo.status, FulfilmentOrderStatus.RECEIVED)
        lines_by_sku = {l.sku_id: l for l in updated_fo.lines}
        self.assertEqual(lines_by_sku["SKU_MILK_1L"].accepted_quantity, 24.0)
        self.assertEqual(lines_by_sku["SKU_CURD_400G"].accepted_quantity, 12.0)

        # Verify inspection lines in database
        persisted = self.service.get_goods_receipt_note("GRN-MULTI")
        self.assertEqual(len(persisted.inspection_lines), 2)

    def test_03_commercial_and_entity_mismatch_rejections(self):
        """Rejects GRN if fo_id, po_id, store_id, or supplier_id does not match target order."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        def make_valid_line():
            return DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=24.0,
                rejected_quantity=0.0,
                unit_cost=30.0,
                remaining_shelf_life_days=7,
                total_shelf_life_days=10,
                batch_references=("BATCH-001",),
            )

        # 1. fo_id mismatch
        grn_bad_fo = GoodsReceiptNote(
            grn_id="GRN-BAD-FO",
            fo_id="FO-NONEXISTENT",
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(make_valid_line(),),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo_id=fo.fo_id, grn=grn_bad_fo)
        self.assertIn("GRN fo_id", str(ctx.exception))

        # 2. po_id mismatch
        grn_bad_po = GoodsReceiptNote(
            grn_id="GRN-BAD-PO",
            fo_id=fo.fo_id,
            po_id="PO-TAMPERED",
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(make_valid_line(),),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo_id=fo.fo_id, grn=grn_bad_po)
        self.assertIn("GRN po_id", str(ctx.exception))

        # 3. store_id mismatch (using another valid canonical store)
        grn_bad_store = GoodsReceiptNote(
            grn_id="GRN-BAD-STORE",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id="DS_BLR_INDIRANAGAR",  # Order is at DS_BLR_KORAMANGALA
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(make_valid_line(),),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo_id=fo.fo_id, grn=grn_bad_store)
        self.assertIn("GRN store_id", str(ctx.exception))

        # 4. supplier_id mismatch
        grn_bad_supp = GoodsReceiptNote(
            grn_id="GRN-BAD-SUPP",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id="SIM_SUPP_DAIRY_SECONDARY",  # Order is with SIM_SUPP_DAIRY_PRIMARY
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(make_valid_line(),),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo_id=fo.fo_id, grn=grn_bad_supp)
        self.assertIn("GRN supplier_id", str(ctx.exception))

    def test_04_unknown_or_undispatched_shipment_rejection(self):
        """Rejects GRN referencing a shipment ID that was not dispatched on the order."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-UNKNOWN-SHIP",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-GHOST-999",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )

        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo_id=fo.fo_id, grn=grn)
        self.assertIn("Shipment 'ASN-GHOST-999' is not associated", str(ctx.exception))

    def test_05_cancelled_or_superseded_order_rejection(self):
        """Rejects dock intake if order is cancelled or superseded."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)
        # Cancel order
        fo = self.service.request_cancellation(
            fo_id=fo.fo_id,
            reason="Dock access cancelled due to severe flood warning.",
            category=CancellationReasonCategory.STORE_DEMAND_CHANGED,
            actor_id="DISPATCHER_VEERA",
            timestamp="2024-11-15T15:00:00Z",
        )
        fo = self.service.confirm_cancellation(
            fo_id=fo.fo_id,
            actor_id="DISPATCHER_VEERA",
            timestamp="2024-11-15T15:15:00Z",
            supplier_cancellation_ref="SUPP-CNC-001",
            shipment_cancellation_ref="SHIP-RTO-001",
            shipment_resolutions={"ASN-01": "CANCELLED_TRANSIT"},
            order_wide_cancellation_scope=True,
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCELLED)

        grn = GoodsReceiptNote(
            grn_id="GRN-ON-CANCELLED",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )

        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo_id=fo.fo_id, grn=grn)
        self.assertIn("Cannot process dock intake for cancelled", str(ctx.exception))


class TestMultipleShipmentsAndGRNs(unittest.TestCase):
    """Test suite covering multiple shipments and multiple GRNs per shipment."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_multi_shipments.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.service = FulfilmentService(repository=self.repo)
        self.po = _make_approved_po(final_order_qty=48.0, po_id="PO-TEST-48-UNITS")

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_multiple_shipments_separate_grns(self):
        """Split dispatches across 2 shipments: receives GRN 1 on Shipment 1, then GRN 2 on Shipment 2."""
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-48",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")

        # Dispatch Shipment 1 (24 units)
        asn1 = AdvanceShipmentNotice(
            shipment_id="ASN-SPLIT-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn1, actor_id="GATEWAY")

        # Dispatch Shipment 2 (remaining 24 units)
        asn2 = AdvanceShipmentNotice(
            shipment_id="ASN-SPLIT-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T13:00:00Z",
            expected_arrival_at="2024-11-16T09:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn2, actor_id="GATEWAY")
        self.assertEqual(len(fo.shipment_ids), 2)

        # Intake GRN 1 against Shipment 1
        grn1 = GoodsReceiptNote(
            grn_id="GRN-SPLIT-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SPLIT-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-01",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn1)
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertIn("ASN-SPLIT-01", fo.resolved_shipment_ids)
        self.assertNotIn("ASN-SPLIT-02", fo.resolved_shipment_ids)

        # Intake GRN 2 against Shipment 2
        grn2 = GoodsReceiptNote(
            grn_id="GRN-SPLIT-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SPLIT-02",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T09:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-02",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn2)
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertIn("ASN-SPLIT-02", fo.resolved_shipment_ids)
        self.assertEqual(len(fo.grn_ids), 2)

        # Verify querying by order and shipment
        all_grns = self.service.get_grns_for_order(fo.fo_id)
        self.assertEqual(len(all_grns), 2)
        self.assertEqual([g.grn_id for g in all_grns], ["GRN-SPLIT-01", "GRN-SPLIT-02"])

        grns_ship1 = self.service.get_grns_for_shipment("ASN-SPLIT-01")
        self.assertEqual(len(grns_ship1), 1)
        self.assertEqual(grns_ship1[0].grn_id, "GRN-SPLIT-01")

        grns_ship2 = self.service.get_grns_for_shipment("ASN-SPLIT-02")
        self.assertEqual(len(grns_ship2), 1)
        self.assertEqual(grns_ship2[0].grn_id, "GRN-SPLIT-02")

    def test_02_multiple_grns_per_same_shipment(self):
        """Single shipment delivered in 2 truck unload batches: 2 distinct GRNs against the same shipment."""
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-48-SINGLE",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")

        asn = AdvanceShipmentNotice(
            shipment_id="ASN-SINGLE-48",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=48.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, actor_id="GATEWAY")

        # First intake triage (pallet 1: 24 units)
        grn1 = GoodsReceiptNote(
            grn_id="GRN-PALLET-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SINGLE-48",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-PALLET-1",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn1)
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(fo.lines[0].accepted_quantity, 24.0)

        # Second intake triage on the SAME shipment (pallet 2: 24 units)
        grn2 = GoodsReceiptNote(
            grn_id="GRN-PALLET-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SINGLE-48",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:45:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-PALLET-2",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn2)
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(fo.lines[0].accepted_quantity, 48.0)

        # Verify that get_grns_for_shipment returns BOTH GRNs
        shipment_grns = self.service.get_grns_for_shipment("ASN-SINGLE-48")
        self.assertEqual(len(shipment_grns), 2)
        self.assertEqual([g.grn_id for g in shipment_grns], ["GRN-PALLET-01", "GRN-PALLET-02"])


class TestDockInspectionRules(unittest.TestCase):
    """Test suite covering quality holds, supervisor escalation, and quantity conservation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_dock_inspection.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.service = FulfilmentService(repository=self.repo)
        self.po = _make_approved_po()

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_quality_hold_missing_shelf_life_rejected(self):
        """Accepted perishable stock without verified remaining_shelf_life_days is rejected."""
        with self.assertRaises(ValueError) as ctx:
            DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=24.0,
                rejected_quantity=0.0,
                unit_cost=30.0,
                remaining_shelf_life_days=None,  # Missing!
                batch_references=("BATCH-001",),
            )
        self.assertIn("Quality Hold Violation", str(ctx.exception))
        self.assertIn("without verified remaining_shelf_life_days", str(ctx.exception))

    def test_02_quality_hold_failing_shelf_life_ratio_rejected(self):
        """Accepted perishable stock below 60% shelf life ratio is rejected."""
        # 5 out of 10 days = 50% < 60%
        with self.assertRaises(ValueError) as ctx:
            DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=24.0,
                rejected_quantity=0.0,
                unit_cost=30.0,
                remaining_shelf_life_days=5,
                total_shelf_life_days=10,  # 5/10 = 50%
                batch_references=("BATCH-001",),
            )
        self.assertIn("Quality Hold Violation", str(ctx.exception))
        self.assertIn("below the minimum acceptable threshold of 60.0%", str(ctx.exception))

    def test_03_quality_hold_satisfying_shelf_life_ratio_accepted(self):
        """Accepted perishable stock with >= 60% shelf life ratio is accepted."""
        # 6 out of 10 days = 60.0% >= 60%
        line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=6,
            total_shelf_life_days=10,
            batch_references=("BATCH-001",),
        )
        self.assertEqual(line.accepted_quantity, 24.0)

        # 8 out of 10 days = 80.0%
        line2 = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=8,
            total_shelf_life_days=10,
            batch_references=("BATCH-001",),
        )
        self.assertEqual(line2.accepted_quantity, 24.0)

    def test_04_supervisor_signoff_strictly_greater_than_5000_threshold(self):
        """Discrepancy strictly exceeding ₹5,000 requires supervisor sign-off; at or below ₹5,000 does not."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        # 1. Exactly ₹5,000 discrepancy: unit_cost=100.0, damaged=50 -> 50 * 100 = ₹5,000
        # Does NOT require signoff
        line_at_5000 = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=50.0,
            accepted_quantity=0.0,
            rejected_quantity=50.0,
            damaged_quantity=50.0,
            unit_cost=100.0,
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        self.assertEqual(line_at_5000.damaged_line_cost, 5000.0)

        grn_5000 = GoodsReceiptNote(
            grn_id="GRN-EXACT-5000",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(line_at_5000,),
            supervisor_signoff_by=None,  # Not required!
            supervisor_signoff_at=None,
        )
        self.assertIsNone(grn_5000.supervisor_signoff_by)

        # 2. ₹5,100 discrepancy: unit_cost=100.0, damaged=51 -> 51 * 100 = ₹5,100 (> ₹5,000)
        # Without sign-off: MUST fail
        line_above_5000 = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=51.0,
            accepted_quantity=0.0,
            rejected_quantity=51.0,
            damaged_quantity=51.0,
            unit_cost=100.0,
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        self.assertEqual(line_above_5000.damaged_line_cost, 5100.0)

        with self.assertRaises(ValueError) as ctx:
            GoodsReceiptNote(
                grn_id="GRN-ABOVE-5000-FAIL",
                fo_id=fo.fo_id,
                po_id=fo.po_id,
                shipment_id="ASN-01",
                store_id=fo.store_id,
                supplier_id=fo.supplier_id,
                receipt_at="2024-11-16T08:30:00Z",
                receiver_id="RECEIVER_ARUN",
                inspection_lines=(line_above_5000,),
                supervisor_signoff_by=None,
            )
        self.assertIn("Governance Violation", str(ctx.exception))
        self.assertIn("Mandatory supervisor sign-off", str(ctx.exception))

        # 3. ₹5,100 with supervisor sign-off: MUST succeed
        grn_above_5000_ok = GoodsReceiptNote(
            grn_id="GRN-ABOVE-5000-OK",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(line_above_5000,),
            supervisor_signoff_by="SUP_MEENA",
            supervisor_signoff_at="2024-11-16T08:45:00Z",
        )
        self.assertEqual(grn_above_5000_ok.supervisor_signoff_by, "SUP_MEENA")

    def test_05_quantity_conservation_excess_deliveries(self):
        """Physical excess delivery cannot increase accepted quantity beyond approved PO quantity."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        # 1. Receiver records physical excess: delivered 30, accepted 24, excess 6, rejected 6
        line_with_excess = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=30.0,
            accepted_quantity=24.0,
            rejected_quantity=6.0,
            excess_quantity=6.0,
            unit_cost=30.0,
            remaining_shelf_life_days=7,
            total_shelf_life_days=10,
            batch_references=("BATCH-001",),
            discrepancy_reasons=(DiscrepancyReason.EXCESS_DELIVERY,),
        )
        self.assertEqual(line_with_excess.accepted_quantity, 24.0)
        self.assertEqual(line_with_excess.rejected_quantity, 6.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-EXCESS",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(line_with_excess,),
        )

        updated_fo, _ = self.service.process_dock_intake(fo.fo_id, grn)
        self.assertEqual(updated_fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(updated_fo.lines[0].ordered_quantity, 24.0)
        self.assertEqual(updated_fo.lines[0].accepted_quantity, 24.0)  # Exactly 24 accepted!
        self.assertEqual(updated_fo.lines[0].delivered_quantity, 30.0)  # 30 physically arrived
        self.assertEqual(updated_fo.lines[0].rejected_quantity, 6.0)  # 6 rejected as excess

        # 2. Attempting to accept cumulative quantity exceeding ordered quantity on a fresh in-transit order must fail
        po2 = _make_approved_po(po_id="PO-CONSERVE-TEST-02")
        fo2 = _setup_in_transit_order(self.service, po2, shipment_id="ASN-02", dispatched_qty=24.0)

        grn_over_accept = GoodsReceiptNote(
            grn_id="GRN-OVER-ACCEPT",
            fo_id=fo2.fo_id,
            po_id=fo2.po_id,
            shipment_id="ASN-02",
            store_id=fo2.store_id,
            supplier_id=fo2.supplier_id,
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=30.0,
                    accepted_quantity=30.0,  # 30 > 24 ordered!
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-EXTRA",),
                ),
            ),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo2.fo_id, grn_over_accept)
        self.assertIn("Cumulative accepted quantity", str(ctx.exception))
        self.assertIn("exceeds approved ordered quantity", str(ctx.exception))

    def test_06_zero_accepted_custody_resolution(self):
        """When 0 units are accepted, shipment is only resolved if rejected_goods_disposition is documented."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        # Line where all 24 units are damaged
        damaged_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=0.0,
            rejected_quantity=24.0,
            damaged_quantity=24.0,
            unit_cost=30.0,
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )

        # 1. Without rejected_goods_disposition: custody remains UNRESOLVED
        grn_no_disp = GoodsReceiptNote(
            grn_id="GRN-NO-DISP",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(damaged_line,),
            rejected_goods_disposition=None,
        )
        updated_fo, _ = self.service.process_dock_intake(fo.fo_id, grn_no_disp)
        self.assertEqual(updated_fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertNotIn("ASN-01", updated_fo.resolved_shipment_ids)
        self.assertNotIn("ASN-01", updated_fo.shipment_resolution_evidence)

        # 2. With documented rejected_goods_disposition: custody IS RESOLVED with evidence
        po2 = _make_approved_po(po_id="PO-ZERO-DISP-02")
        fo2 = _setup_in_transit_order(self.service, po2, shipment_id="ASN-02", dispatched_qty=24.0)
        grn_with_disp = GoodsReceiptNote(
            grn_id="GRN-WITH-DISP",
            fo_id=fo2.fo_id,
            po_id=fo2.po_id,
            shipment_id="ASN-02",
            store_id=fo2.store_id,
            supplier_id=fo2.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(damaged_line,),
            rejected_goods_disposition="RTO-VENDOR-REJECT-001",
        )
        updated_fo2, _ = self.service.process_dock_intake(fo2.fo_id, grn_with_disp)
        self.assertIn("ASN-02", updated_fo2.resolved_shipment_ids)
        self.assertEqual(updated_fo2.shipment_resolution_evidence.get("ASN-02"), "RTO-VENDOR-REJECT-001")


class TestFulfilmentBatchMinting(unittest.TestCase):
    """Test suite covering immutable FulfilmentBatch minting and InventoryBatch compatibility."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_batches.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.service = FulfilmentService(repository=self.repo)
        self.po = _make_approved_po()

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_batch_minting_for_accepted_stock(self):
        """Mints FulfilmentBatch records with divided quantities across multiple batch references."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-BATCH-TEST",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-LOT-A", "BATCH-LOT-B"),
                ),
            ),
        )
        self.service.process_dock_intake(fo.fo_id, grn)

        # Retrieve batches via service
        batches = self.service.get_fulfilment_batches_for_grn("GRN-BATCH-TEST")
        self.assertEqual(len(batches), 2)

        b1, b2 = batches
        self.assertEqual(b1.batch_id, "BATCH-LOT-A")
        self.assertEqual(b1.sku_id, "SKU_MILK_1L")
        self.assertEqual(b1.units, 12.0)
        self.assertEqual(b1.remaining_shelf_life_days, 7)
        self.assertEqual(b1.receipt_date, "2024-11-16")
        self.assertEqual(b1.fo_id, fo.fo_id)
        self.assertEqual(b1.grn_id, "GRN-BATCH-TEST")

        self.assertEqual(b2.batch_id, "BATCH-LOT-B")
        self.assertEqual(b2.units, 12.0)

        # Verify lossless conversion to core InventoryBatch
        inv_batch = b1.to_inventory_batch()
        self.assertEqual(inv_batch.batch_id, "BATCH-LOT-A")
        self.assertEqual(inv_batch.units, 12.0)
        self.assertEqual(inv_batch.remaining_shelf_life_days, 7)
        self.assertEqual(inv_batch.creation_date, "2024-11-16")

    def test_02_zero_batches_for_fully_rejected_stock(self):
        """Zero batches minted when accepted quantity is zero."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-ZERO-ACCEPTED",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=0.0,
                    rejected_quantity=24.0,
                    damaged_quantity=24.0,
                    unit_cost=30.0,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                ),
            ),
            rejected_goods_disposition="RTO-DAMAGE-001",
        )
        self.service.process_dock_intake(fo.fo_id, grn)

        batches = self.service.get_fulfilment_batches_for_grn("GRN-ZERO-ACCEPTED")
        self.assertEqual(len(batches), 0)


class TestCarrierTrackingIntegration(unittest.TestCase):
    """Test suite covering tracking verification informing intake without authorizing GRN."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_tracking_intake.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.service = FulfilmentService(repository=self.repo)
        self.po = _make_approved_po()

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_carrier_status_alone_never_creates_grn(self):
        """Carrier reaching DELIVERED status does not create or authorize a GRN."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-TRK-01", dispatched_qty=24.0)

        # Register and advance shipment tracking to DELIVERED
        self.service.tracking_service.register_shipment(
            fo_id=fo.fo_id,
            shipment_id="ASN-TRK-01",
            carrier_id="DELHIVERY",
        )
        tracking_rec = self.service.tracking_service.advance_to_tracking_status(
            "ASN-TRK-01",
            ShipmentTrackingStatus.DELIVERED,
        )
        self.assertEqual(tracking_rec.current_status, ShipmentTrackingStatus.DELIVERED)

        # Fulfilment order MUST still be IN_TRANSIT with 0 GRNs!
        fo_reloaded = self.service.get_order_or_raise(fo.fo_id)
        self.assertEqual(fo_reloaded.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(len(fo_reloaded.grn_ids), 0)
        self.assertEqual(len(self.service.get_grns_for_order(fo.fo_id)), 0)

    def test_02_verify_carrier_arrival_flag_enforcement(self):
        """When verify_carrier_arrival=True, rejects intake if carrier has not reached DELIVERED."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-TRK-02", dispatched_qty=24.0)
        self.service.tracking_service.register_shipment(
            fo_id=fo.fo_id,
            shipment_id="ASN-TRK-02",
            carrier_id="DELHIVERY",
        )
        # Advance only to IN_TRANSIT
        self.service.tracking_service.advance_to_tracking_status(
            "ASN-TRK-02",
            ShipmentTrackingStatus.IN_TRANSIT,
        )

        grn = GoodsReceiptNote(
            grn_id="GRN-TRK-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-TRK-02",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )

        # Fails with verify_carrier_arrival=True
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(
                fo.fo_id,
                grn,
                verify_carrier_arrival=True,
            )
        self.assertIn("Carrier arrival verification failed", str(ctx.exception))
        self.assertIn("expected 'DELIVERED'", str(ctx.exception))

        # Advance to DELIVERED and retry: now succeeds!
        self.service.tracking_service.advance_to_tracking_status(
            "ASN-TRK-02",
            ShipmentTrackingStatus.DELIVERED,
        )
        updated_fo, _ = self.service.process_dock_intake(
            fo.fo_id,
            grn,
            verify_carrier_arrival=True,
        )
        self.assertEqual(updated_fo.status, FulfilmentOrderStatus.RECEIVED)


class TestPersistenceIdempotencyAndRollback(unittest.TestCase):
    """Test suite covering durable idempotency, optimistic locking, and atomic rollback."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_persistence.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.service = FulfilmentService(repository=self.repo)
        self.po = _make_approved_po()

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_durable_idempotency_exact_replay_no_op(self):
        """Exact retry of process_dock_intake returns without appending new events or bumping version."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-IDEMPOTENT-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )

        fo_first, _ = self.service.process_dock_intake(
            fo.fo_id,
            grn,
            idempotency_key="CMD-INTAKE-001",
        )
        version_first = self.service.get_order_version(fo.fo_id)
        events_first = len(fo_first.event_history)
        self.assertEqual(version_first, 5)

        # Exact retry
        fo_retry, _ = self.service.process_dock_intake(
            fo.fo_id,
            grn,
            idempotency_key="CMD-INTAKE-001",
        )
        version_retry = self.service.get_order_version(fo.fo_id)
        events_retry = len(fo_retry.event_history)

        self.assertEqual(version_retry, version_first)
        self.assertEqual(events_retry, events_first)

    def test_02_idempotency_conflict_rejection_different_payload(self):
        """Reusing the same idempotency key with differing inspection payload is rejected on active order."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        # First intake: partial receipt (12 units) -> status remains PARTIALLY_RECEIVED
        grn1 = GoodsReceiptNote(
            grn_id="GRN-CONF-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        self.service.process_dock_intake(fo.fo_id, grn1, idempotency_key="KEY-COLLIDE")

        # Second intake: DIFFERENT GRN payload with SAME key
        grn2 = GoodsReceiptNote(
            grn_id="GRN-CONF-02",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=10.0,  # Changed!
                    rejected_quantity=2.0,
                    damaged_quantity=2.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
                ),
            ),
        )
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo.fo_id, grn2, idempotency_key="KEY-COLLIDE")
        self.assertIn("Idempotency Conflict", str(ctx.exception))

    def test_03_duplicate_grn_id_rejection(self):
        """Duplicate submission of the same grn_id is strictly rejected on active order."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        # Partial receipt: 12 units -> PARTIALLY_RECEIVED
        grn = GoodsReceiptNote(
            grn_id="GRN-DUP-TEST",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        self.service.process_dock_intake(fo.fo_id, grn)

        # Attempting to submit again with different idempotency key
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.process_dock_intake(fo.fo_id, grn, idempotency_key="KEY-DIFFERENT")
        self.assertIn("Duplicate GRN submission", str(ctx.exception))

    def test_04_optimistic_concurrency_stale_update_rejected(self):
        """Rejects update when expected_version does not match current database version."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)
        current_ver = self.service.get_order_version(fo.fo_id)
        self.assertEqual(current_ver, 4)

        grn = GoodsReceiptNote(
            grn_id="GRN-STALE-LOCK",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )

        with self.assertRaises(StaleUpdateError) as ctx:
            self.service.process_dock_intake(
                fo.fo_id,
                grn,
                expected_version=current_ver - 1,  # Stale version!
            )
        self.assertIn("Stale update conflict", str(ctx.exception))

    def test_05_strict_scope_boundaries_po_immutability(self):
        """Verifies approved PurchaseOrder remains strictly immutable across dock intake."""
        original_po_dump = self.po.model_dump()
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-PO-IMMUTABLE",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        self.service.process_dock_intake(fo.fo_id, grn, po=self.po)

        # Assert PO model is 100% identical and unchanged
        self.assertEqual(self.po.model_dump(), original_po_dump)

    def test_06_atomic_rollback_on_database_failure(self):
        """Simulates an error during GRN insertion; ensures FO and GRN roll back atomically."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)
        version_before = self.service.get_order_version(fo.fo_id)
        events_before = len(fo.event_history)

        grn = GoodsReceiptNote(
            grn_id="GRN-ROLLBACK-TEST",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )

        # Force a database failure during _save_grn_internal by monkeypatching
        original_save_grn = self.repo._save_grn_internal

        def failing_save_grn(conn, g):
            raise sqlite3.OperationalError("Simulated disk I/O failure during GRN insert.")

        self.repo._save_grn_internal = failing_save_grn
        try:
            with self.assertRaises(sqlite3.OperationalError):
                self.service.process_dock_intake(fo.fo_id, grn)
        finally:
            self.repo._save_grn_internal = original_save_grn

        # Verify neither FO nor GRN was committed
        fo_after = self.service.get_order_or_raise(fo.fo_id)
        self.assertEqual(fo_after.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(self.service.get_order_version(fo.fo_id), version_before)
        self.assertEqual(len(fo_after.event_history), events_before)
        self.assertIsNone(self.service.get_goods_receipt_note("GRN-ROLLBACK-TEST"))

    def test_07_strict_scope_boundaries_no_inventory_postings(self):
        """Verifies Phase 5.6.1 strictly records receipt evidence without mutating inventory ledger."""
        fo = _setup_in_transit_order(self.service, self.po, shipment_id="ASN-01", dispatched_qty=24.0)

        grn = GoodsReceiptNote(
            grn_id="GRN-SCOPE-CHECK",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=10,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        updated_fo, processed_grn = self.service.process_dock_intake(
            fo.fo_id,
            grn,
            auto_post_inventory=False,
        )

        # 1. Pipeline quantity of received order drops to 0 (since it is no longer in active inbound pipeline)
        self.assertEqual(self.service.get_pipeline_quantity(sku_id="SKU_MILK_1L"), 0.0)

        # 2. Batches minted are domain representations (FulfilmentBatch), NOT posted to an inventory table
        batches = self.service.get_fulfilment_batches_for_grn("GRN-SCOPE-CHECK")
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].units, 24.0)

        # 3. Verify SQLite DB has no inventory tables created or touched
        conn = self.repo._get_connection()
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%inventory%';")
        self.assertEqual(len(cur.fetchall()), 0)


if __name__ == "__main__":
    unittest.main()
