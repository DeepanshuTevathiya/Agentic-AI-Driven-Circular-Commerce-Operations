"""
Unit & Integration Tests: Fulfilment SQLite Persistence & Repository
===================================================================
Phase 5.3.1 Verification Tests:
1. Persistence and reload after closing and reopening database.
2. Complete Pydantic model round-trip across all fields, nested objects, and types.
3. Duplicate IDs and idempotency conflicts.
4. Atomic rollback when a write fails (zero side-effects).
5. Stale concurrent update rejection (optimistic locking).
6. Append-only event history enforcement and non-repudiation.
7. Pipeline calculations across multiple orders, partial acknowledgments, split shipments, and GRNs.
8. Dynamic query filtering (store_id, supplier_id, status, pagination).
9. Original approved PurchaseOrder immutability.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List

from src.agents.fulfilment import (
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DockInspectionLine,
    DuplicateFulfilmentOrderError,
    FulfilmentEvent,
    FulfilmentOrder,
    FulfilmentOrderNotFoundError,
    FulfilmentOrderStatus,
    FulfilmentPersistenceError,
    FulfilmentRepository,
    GoodsReceiptNote,
    IdempotencyConflictError,
    ShipmentLineItem,
    StaleUpdateError,
    SupplierAcknowledgment,
    SupplierAcknowledgmentLine,
    SupplierAcknowledgmentStatus,
    apply_goods_receipt,
    apply_shipment_dispatch,
    apply_supplier_acknowledgment,
    close_short_fulfilment_order,
    create_fulfilment_order,
    record_delivery_exception,
    request_cancellation,
    confirm_cancellation,
    transmit_fulfilment_order,
)
from src.agents.procurement.approval import approve_purchase_order
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    SupplierType,
)


from tests.lead2_agents.test_fulfilment_state_machine import _make_approved_po


def _build_test_approved_po(
    po_id: str = "PO-20241115-TEST-01",
    store_id: str = "DS_BLR_KORAMANGALA",
    sku_id: str = "SKU_MILK_1L",
    category: str = "Dairy",
    final_order_qty: float = 24.0,
    unit_cost: float = 30.0,
    approval_timestamp: str = "2024-11-15T10:30:00Z",
) -> PurchaseOrder:
    return _make_approved_po(
        store_id=store_id,
        sku_id=sku_id,
        category=category,
        final_order_qty=final_order_qty,
        unit_procurement_cost=unit_cost,
        po_id=po_id,
        approval_timestamp=approval_timestamp,
    )


class TestFulfilmentRepository(unittest.TestCase):
    """Test suite for FulfilmentRepository SQLite persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_fulfilment.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.po = _build_test_approved_po()
        self.fo = create_fulfilment_order(self.po, "AGENT", "2024-11-15T11:00:00Z")

    def tearDown(self):
        self.repo.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Persistence & Reload After Closing / Reopening
    # -------------------------------------------------------------------------

    def test_01_persistence_and_reload_after_close_and_reopen(self):
        saved = self.repo.create(self.fo)
        self.assertEqual(saved.fo_id, self.fo.fo_id)

        # Explicitly close the connection
        self.repo.close()

        # Reopen with a completely fresh repository instance pointing to the same file
        reopened_repo = FulfilmentRepository(self.db_path)
        try:
            loaded = reopened_repo.get(self.fo.fo_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.fo_id, self.fo.fo_id)
            self.assertEqual(loaded.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)
            self.assertEqual(len(loaded.lines), 1)
            self.assertEqual(loaded.lines[0].sku_id, "SKU_MILK_1L")
            self.assertEqual(loaded.lines[0].ordered_quantity, 24.0)
            self.assertEqual(len(loaded.event_history), 1)
            self.assertEqual(loaded.po_snapshot.po_id, self.po.po_id)
            # Full model equality
            self.assertEqual(loaded, self.fo)
        finally:
            reopened_repo.close()

    # -------------------------------------------------------------------------
    # 2. Complete Pydantic Model Round-Trip
    # -------------------------------------------------------------------------

    def test_02_complete_pydantic_model_round_trip_through_transitions(self):
        # Progress order through multiple transitions to populate complex fields
        fo = self.repo.create(self.fo)

        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        fo = self.repo.update(fo)

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
        fo = self.repo.update(fo)

        asn = AdvanceShipmentNotice(
            shipment_id="ASN-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
            carrier="ColdChain Logistics",
            tracking_reference="TRK-12345",
        )
        fo = apply_shipment_dispatch(fo, asn, "GATEWAY", "ASN-1")
        fo = self.repo.update(fo)

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
                    delivered_quantity=24.0,
                    accepted_quantity=20.0,
                    rejected_quantity=4.0,
                    damaged_quantity=4.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=4,
                    batch_references=("BATCH-001",),
                ),
            ),
            rejected_goods_disposition="RTO-DISP-01",
        )
        fo = apply_goods_receipt(fo, grn, "RECEIVER_ARUN", "GRN-1")
        fo = self.repo.update(fo)

        # Retrieve and verify complete model roundtrip
        loaded = self.repo.get(fo.fo_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded, fo)
        self.assertEqual(loaded.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(loaded.shipment_ids, ("ASN-1",))
        self.assertEqual(loaded.resolved_shipment_ids, ("ASN-1",))
        self.assertEqual(loaded.shipment_resolution_evidence, fo.shipment_resolution_evidence)
        self.assertEqual(loaded.grn_ids, ("GRN-1",))
        self.assertEqual(len(loaded.event_history), 5)
        # Original PO snapshot remains approved and unmodified
        self.assertEqual(loaded.po_snapshot.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(loaded.po_snapshot.approved_by, "OPERATOR_DEEPANSHU")

    # -------------------------------------------------------------------------
    # 3. Duplicate IDs & Idempotency Conflicts
    # -------------------------------------------------------------------------

    def test_03a_duplicate_fo_id_rejected(self):
        self.repo.create(self.fo)
        with self.assertRaises(DuplicateFulfilmentOrderError) as ctx:
            self.repo.create(self.fo)
        self.assertIn("already exists", str(ctx.exception))

    def test_03b_idempotency_conflict_across_orders_rejected(self):
        self.repo.create(self.fo)

        # Create second order attempting to reuse the same idempotency key
        po2 = _build_test_approved_po(po_id="PO-20241115-TEST-02", final_order_qty=12.0)
        fo2 = create_fulfilment_order(
            po2,
            "AGENT",
            "2024-11-15T11:00:00Z",
            fo_id="FO-20241115-TEST-02",
            idempotency_key=self.fo.event_history[0].idempotency_key,  # Reused key
        )

        with self.assertRaises(IdempotencyConflictError) as ctx:
            self.repo.create(fo2)
        self.assertIn("Idempotency conflict", str(ctx.exception))

    def test_03c_save_dispatches_cleanly(self):
        # Initial save creates
        self.assertIsNone(self.repo.get(self.fo.fo_id))
        self.repo.save(self.fo)
        self.assertIsNotNone(self.repo.get(self.fo.fo_id))
        self.assertEqual(self.repo.get_version(self.fo.fo_id), 1)

        # Subsequent save updates
        fo_tx = transmit_fulfilment_order(self.fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        self.repo.save(fo_tx)
        self.assertEqual(self.repo.get(self.fo.fo_id).status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(self.repo.get_version(self.fo.fo_id), 2)

    # -------------------------------------------------------------------------
    # 4. Atomic Rollback on Write Failure
    # -------------------------------------------------------------------------

    def test_04_atomic_rollback_on_write_failure(self):
        # Corrupt order with illegal foreign key / tampered structure inside update
        self.repo.create(self.fo)
        initial_version = self.repo.get_version(self.fo.fo_id)
        self.assertEqual(initial_version, 1)

        # Attempt an update that violates non-repudiation (event history sequence altered)
        tampered_event = self.fo.event_history[0].model_copy(update={"event_id": "EVT-TAMPERED-999"})
        tampered_fo = self.fo.model_copy(update={"event_history": (tampered_event,)})

        with self.assertRaises(FulfilmentPersistenceError) as ctx:
            self.repo.update(tampered_fo)
        self.assertIn("Audit History Tampering", str(ctx.exception))

        # Assert database was rolled back: original version and events are intact
        current = self.repo.get(self.fo.fo_id)
        self.assertEqual(current.event_history[0].event_id, self.fo.event_history[0].event_id)
        self.assertEqual(self.repo.get_version(self.fo.fo_id), 1)

    # -------------------------------------------------------------------------
    # 5. Stale Concurrent Update Rejection (Optimistic Locking)
    # -------------------------------------------------------------------------

    def test_05_stale_concurrent_update_rejection(self):
        self.repo.create(self.fo)

        # Worker 1 and Worker 2 both load version 1
        worker1_fo = self.repo.get(self.fo.fo_id)
        worker2_fo = self.repo.get(self.fo.fo_id)
        self.assertEqual(self.repo.get_version(self.fo.fo_id), 1)

        # Worker 2 successfully updates first -> version advances to 2
        worker2_fo_tx = transmit_fulfilment_order(worker2_fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        self.repo.update(worker2_fo_tx, expected_version=1)
        self.assertEqual(self.repo.get_version(self.fo.fo_id), 2)

        # Worker 1 tries to update using stale version 1 -> must be rejected
        worker1_fo_tx = transmit_fulfilment_order(worker1_fo, "2024-11-15T11:15:00Z", "AGENT", "TX-STALE")
        with self.assertRaises(StaleUpdateError) as ctx:
            self.repo.update(worker1_fo_tx, expected_version=1)
        self.assertIn("Stale update conflict", str(ctx.exception))

        # Worker 1 also fails with event count regression check if expected_version is omitted
        with self.assertRaises(StaleUpdateError) as ctx:
            self.repo.update(worker1_fo)  # 1 event, but DB already has 2
        self.assertIn("Stale update conflict", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 6. Append-Only Event History
    # -------------------------------------------------------------------------

    def test_06_append_only_event_history_enforcement(self):
        fo = self.repo.create(self.fo)
        fo = transmit_fulfilment_order(fo, "2024-11-15T11:15:00Z", "AGENT", "TX-1")
        fo = self.repo.update(fo)

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
        fo = self.repo.update(fo)

        # Verify relational event logs
        events = self.repo.get_events(fo.fo_id)
        self.assertEqual(len(events), 3)
        self.assertEqual([e.sequence_number for e in events], [1, 2, 3])
        self.assertEqual(events[0].event_type.value, "CREATE_FULFILMENT_ORDER")
        self.assertEqual(events[1].event_type.value, "TRANSMIT_TO_SUPPLIER")
        self.assertEqual(events[2].event_type.value, "SUPPLIER_ACKNOWLEDGE")

    # -------------------------------------------------------------------------
    # 7. Pipeline Calculation Across Multiple Orders & States
    # -------------------------------------------------------------------------

    def test_07_pipeline_quantities_across_lifecycle_states(self):
        """
        Comprehensive test of dynamic pipeline query logic:
        Only active, supplier-confirmed outstanding units count.
        Unconfirmed, declined, rejected, cancelled, and received units do NOT count.
        """
        # Order 1: READY_TO_TRANSMIT (qty=24) -> Pipeline = 0.0
        po1 = _build_test_approved_po(po_id="PO-P-01", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo1 = create_fulfilment_order(po1, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-01")
        self.repo.create(fo1)

        # Order 2: ISSUED (qty=24) -> Pipeline = 0.0
        po2 = _build_test_approved_po(po_id="PO-P-02", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo2 = create_fulfilment_order(po2, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-02")
        fo2 = transmit_fulfilment_order(fo2, "2024-11-15T11:15:00Z", "AGENT", "TX-P2")
        self.repo.create(fo2)

        # Order 3: ACKNOWLEDGED (accepted=24) -> Pipeline = 24.0
        po3 = _build_test_approved_po(po_id="PO-P-03", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo3 = create_fulfilment_order(po3, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-03")
        fo3 = transmit_fulfilment_order(fo3, "2024-11-15T11:15:00Z", "AGENT", "TX-P3")
        ack3 = SupplierAcknowledgment(
            acknowledgment_id="ACK-P3",
            fo_id=fo3.fo_id,
            po_id=fo3.po_id,
            supplier_id=fo3.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo3 = apply_supplier_acknowledgment(fo3, ack3, "GATEWAY", "ACK-P3")
        self.repo.create(fo3)

        # Order 4: PARTIALLY_ACKNOWLEDGED (ordered=24, accepted=15, declined=9) -> Pipeline = 15.0
        po4 = _build_test_approved_po(po_id="PO-P-04", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo4 = create_fulfilment_order(po4, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-04")
        fo4 = transmit_fulfilment_order(fo4, "2024-11-15T11:15:00Z", "AGENT", "TX-P4")
        ack4 = SupplierAcknowledgment(
            acknowledgment_id="ACK-P4",
            fo_id=fo4.fo_id,
            po_id=fo4.po_id,
            supplier_id=fo4.supplier_id,
            status=SupplierAcknowledgmentStatus.PARTIALLY_ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
            lines=(
                SupplierAcknowledgmentLine(
                    sku_id="SKU_MILK_1L",
                    accepted_quantity=15.0,
                    declined_quantity=9.0,
                ),
            ),
        )
        fo4 = apply_supplier_acknowledgment(fo4, ack4, "GATEWAY", "ACK-P4")
        self.repo.create(fo4)

        # Order 5: PARTIALLY_RECEIVED (ordered=24, accepted=24, delivered=10, accepted_at_dock=10)
        # Outstanding pipeline = 24 - 10 = 14.0
        po5 = _build_test_approved_po(po_id="PO-P-05", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo5 = create_fulfilment_order(po5, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-05")
        fo5 = transmit_fulfilment_order(fo5, "2024-11-15T11:15:00Z", "AGENT", "TX-P5")
        ack5 = SupplierAcknowledgment(
            acknowledgment_id="ACK-P5",
            fo_id=fo5.fo_id,
            po_id=fo5.po_id,
            supplier_id=fo5.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo5 = apply_supplier_acknowledgment(fo5, ack5, "GATEWAY", "ACK-P5")
        asn5 = AdvanceShipmentNotice(
            shipment_id="ASN-P5-1",
            fo_id=fo5.fo_id,
            po_id=fo5.po_id,
            supplier_id=fo5.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=10.0),),
        )
        fo5 = apply_shipment_dispatch(fo5, asn5, "GATEWAY", "ASN-P5-1")
        grn5 = GoodsReceiptNote(
            grn_id="GRN-P5-1",
            fo_id=fo5.fo_id,
            po_id=fo5.po_id,
            shipment_id="ASN-P5-1",
            store_id=fo5.store_id,
            supplier_id=fo5.supplier_id,
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
                    batch_references=("BATCH-005",),
                ),
            ),
        )
        fo5 = apply_goods_receipt(fo5, grn5, "RECEIVER_ARUN", "GRN-P5-1")
        self.repo.create(fo5)

        # Order 6: RECEIVED (all 24 received) -> Pipeline = 0.0
        po6 = _build_test_approved_po(po_id="PO-P-06", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo6 = create_fulfilment_order(po6, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-06")
        fo6 = transmit_fulfilment_order(fo6, "2024-11-15T11:15:00Z", "AGENT", "TX-P6")
        ack6 = SupplierAcknowledgment(
            acknowledgment_id="ACK-P6",
            fo_id=fo6.fo_id,
            po_id=fo6.po_id,
            supplier_id=fo6.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo6 = apply_supplier_acknowledgment(fo6, ack6, "GATEWAY", "ACK-P6")
        asn6 = AdvanceShipmentNotice(
            shipment_id="ASN-P6-1",
            fo_id=fo6.fo_id,
            po_id=fo6.po_id,
            supplier_id=fo6.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo6 = apply_shipment_dispatch(fo6, asn6, "GATEWAY", "ASN-P6-1")
        grn6 = GoodsReceiptNote(
            grn_id="GRN-P6-1",
            fo_id=fo6.fo_id,
            po_id=fo6.po_id,
            shipment_id="ASN-P6-1",
            store_id=fo6.store_id,
            supplier_id=fo6.supplier_id,
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-006",),
                ),
            ),
        )
        fo6 = apply_goods_receipt(fo6, grn6, "RECEIVER_ARUN", "GRN-P6-1")
        self.repo.create(fo6)

        # Order 7: CLOSED_SHORT -> Pipeline = 0.0
        po7 = _build_test_approved_po(po_id="PO-P-07", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo7 = create_fulfilment_order(po7, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-07")
        fo7 = transmit_fulfilment_order(fo7, "2024-11-15T11:15:00Z", "AGENT", "TX-P7")
        ack7 = SupplierAcknowledgment(
            acknowledgment_id="ACK-P7",
            fo_id=fo7.fo_id,
            po_id=fo7.po_id,
            supplier_id=fo7.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo7 = apply_supplier_acknowledgment(fo7, ack7, "GATEWAY", "ACK-P7")
        asn7 = AdvanceShipmentNotice(
            shipment_id="ASN-P7-1",
            fo_id=fo7.fo_id,
            po_id=fo7.po_id,
            supplier_id=fo7.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=10.0),),
        )
        fo7 = apply_shipment_dispatch(fo7, asn7, "GATEWAY", "ASN-P7-1")
        grn7 = GoodsReceiptNote(
            grn_id="GRN-P7-1",
            fo_id=fo7.fo_id,
            po_id=fo7.po_id,
            shipment_id="ASN-P7-1",
            store_id=fo7.store_id,
            supplier_id=fo7.supplier_id,
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
                    batch_references=("BATCH-007",),
                ),
            ),
        )
        fo7 = apply_goods_receipt(fo7, grn7, "RECEIVER_ARUN", "GRN-P7-1")
        fo7 = close_short_fulfilment_order(
            fo7, "Supplier informed stockout; closing short", "MGR_PRIYA", "2024-11-16T10:00:00Z", "CLS-P7"
        )
        self.repo.create(fo7)

        # Order 8: CANCELLED -> Pipeline = 0.0
        po8 = _build_test_approved_po(po_id="PO-P-08", sku_id="SKU_MILK_1L", final_order_qty=24.0)
        fo8 = create_fulfilment_order(po8, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-08")
        fo8 = request_cancellation(
            fo8, "Accidental duplicate order", CancellationReasonCategory.OPERATIONAL_MISTAKE, "SUP", "2024-11-15T11:15:00Z", "REQ-CNC-8"
        )
        fo8 = confirm_cancellation(fo8, "SUP", "2024-11-15T11:30:00Z", "CONF-CNC-8")
        self.repo.create(fo8)

        # Order 9: A different SKU in a different store (Bread in Indiranagar) -> Pipeline = 24.0
        po9 = _build_test_approved_po(
            po_id="PO-P-09",
            store_id="DS_BLR_INDIRANAGAR",
            sku_id="SKU_BREAD_WHITE",
            category="Bakery",
            final_order_qty=24.0,
            unit_cost=25.0,
        )
        fo9 = create_fulfilment_order(po9, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-P-09")
        fo9 = transmit_fulfilment_order(fo9, "2024-11-15T11:15:00Z", "AGENT", "TX-P9")
        ack9 = SupplierAcknowledgment(
            acknowledgment_id="ACK-P9",
            fo_id=fo9.fo_id,
            po_id=fo9.po_id,
            supplier_id=fo9.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo9 = apply_supplier_acknowledgment(fo9, ack9, "GATEWAY", "ACK-P9")
        self.repo.create(fo9)

        # ---------------------------------------------------------------------
        # Verify Pipeline Queries
        # ---------------------------------------------------------------------
        # Expected SKU_MILK_1L in Koramangala:
        # Order 3 (24.0) + Order 4 (15.0) + Order 5 (14.0) = 53.0 units
        milk_koramangala_pipeline = self.repo.get_confirmed_pipeline_quantity(
            store_id="DS_BLR_KORAMANGALA", sku_id="SKU_MILK_1L"
        )
        self.assertEqual(milk_koramangala_pipeline, 53.0)

        # Bread in Indiranagar:
        bread_indiranagar_pipeline = self.repo.get_confirmed_pipeline_quantity(
            store_id="DS_BLR_INDIRANAGAR", sku_id="SKU_BREAD_WHITE"
        )
        self.assertEqual(bread_indiranagar_pipeline, 24.0)

        # Total across all stores & SKUs: 53.0 + 24.0 = 77.0
        total_pipeline = self.repo.get_confirmed_pipeline_quantity()
        self.assertEqual(total_pipeline, 77.0)

        # Filter by SKU across all stores
        milk_total = self.repo.get_confirmed_pipeline_quantity(sku_id="SKU_MILK_1L")
        self.assertEqual(milk_total, 53.0)

        # SKU breakdown for Koramangala
        sku_map = self.repo.get_sku_pipeline_quantities(store_id="DS_BLR_KORAMANGALA")
        self.assertEqual(sku_map, {"SKU_MILK_1L": 53.0})

        # Pipeline breakdown mapping
        breakdown = self.repo.get_pipeline_breakdown()
        self.assertEqual(
            breakdown,
            {
                ("DS_BLR_KORAMANGALA", "SKU_MILK_1L"): 53.0,
                ("DS_BLR_INDIRANAGAR", "SKU_BREAD_WHITE"): 24.0,
            },
        )

    # -------------------------------------------------------------------------
    # 8. Querying & Listing Operations
    # -------------------------------------------------------------------------

    def test_08_listing_filtering_and_pagination(self):
        self.repo.create(self.fo)

        po2 = _build_test_approved_po(po_id="PO-L-02", store_id="DS_BLR_INDIRANAGAR")
        fo2 = create_fulfilment_order(po2, "AGENT", "2024-11-15T11:00:00Z", fo_id="FO-L-02")
        self.repo.create(fo2)

        # Count total
        self.assertEqual(self.repo.count(), 2)
        self.assertEqual(self.repo.count(store_id="DS_BLR_KORAMANGALA"), 1)
        self.assertEqual(self.repo.count(store_id="DS_BLR_INDIRANAGAR"), 1)

        # List with filter
        k_orders = self.repo.list(store_id="DS_BLR_KORAMANGALA")
        self.assertEqual(len(k_orders), 1)
        self.assertEqual(k_orders[0].fo_id, self.fo.fo_id)

        # Pagination: limit=1, offset=1
        paged = self.repo.list(limit=1, offset=1)
        self.assertEqual(len(paged), 1)
        self.assertEqual(paged[0].fo_id, fo2.fo_id)

        # Get or raise
        self.assertEqual(self.repo.get_or_raise(self.fo.fo_id).fo_id, self.fo.fo_id)
        with self.assertRaises(FulfilmentOrderNotFoundError):
            self.repo.get_or_raise("FO-NONEXISTENT")

        # Get by PO ID
        po_orders = self.repo.get_by_po_id(self.po.po_id)
        self.assertEqual(len(po_orders), 1)
        self.assertEqual(po_orders[0].fo_id, self.fo.fo_id)

    # -------------------------------------------------------------------------
    # 9. Context Manager Support
    # -------------------------------------------------------------------------

    def test_09_context_manager_support(self):
        with FulfilmentRepository(self.db_path) as ctx_repo:
            ctx_repo.create(self.fo)
            retrieved = ctx_repo.get(self.fo.fo_id)
            self.assertIsNotNone(retrieved)

        # Connection was cleanly closed on exit
        # Reopening works
        fresh = FulfilmentRepository(self.db_path)
        try:
            self.assertEqual(fresh.count(), 1)
        finally:
            fresh.close()


if __name__ == "__main__":
    unittest.main()
