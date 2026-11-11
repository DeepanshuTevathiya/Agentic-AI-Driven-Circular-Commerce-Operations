"""
Integration Tests: Fulfilment Application Service Layer
======================================================
Phase 5.3.2 Verification Tests:
1. Successful end-to-end lifecycle execution, persistence, and reload.
2. Idempotent retries without duplicate events or unnecessary version increments.
3. Stale concurrent update rejection (optimistic locking).
4. Multi-aggregate supersession atomicity and rollback on failure.
5. Delivery exception recording and resumption through service layer.
6. Reaffirmation and pipeline restoration through service layer.
7. Close short managerial closure through service layer.
8. Dynamic pipeline reporting through service layer.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.agents.fulfilment import (
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DockInspectionLine,
    DuplicateFulfilmentOrderError,
    FulfilmentEventType,
    FulfilmentGovernanceError,
    FulfilmentOrder,
    FulfilmentOrderNotFoundError,
    FulfilmentOrderStatus,
    FulfilmentRepository,
    FulfilmentService,
    GoodsReceiptNote,
    IdempotencyConflictError,
    ShipmentLineItem,
    StaleUpdateError,
    SupplierAcknowledgment,
    SupplierAcknowledgmentLine,
    SupplierAcknowledgmentStatus,
)
from src.agents.procurement.schemas import PurchaseOrderStatus
from tests.lead2_agents.test_fulfilment_state_machine import _make_approved_po


class TestFulfilmentService(unittest.TestCase):
    """Integration test suite for FulfilmentService."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_service.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.service = FulfilmentService(repository=self.repo)
        self.po = _make_approved_po()

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Successful Persistence & Reload Through Service Workflows
    # -------------------------------------------------------------------------

    def test_01_end_to_end_lifecycle_persistence_and_reload(self):
        # 1. Create order
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        self.assertEqual(fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 1)

        # 2. Transmit order
        fo = self.service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT")
        self.assertEqual(fo.status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 2)

        # 3. Acknowledge order
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 3)

        # 4. Dispatch shipment
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, actor_id="GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 4)

        # 5. Receive goods at dark-store dock
        grn = GoodsReceiptNote(
            grn_id="GRN-01",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-01",
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
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-001",),
                ),
            ),
        )
        fo = self.service.receive_goods(fo.fo_id, grn, actor_id="RECEIVER_ARUN")
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 5)

        # Verify reloaded state matches in-memory aggregate
        loaded = self.service.get_order_or_raise(fo.fo_id)
        self.assertEqual(loaded, fo)
        self.assertEqual(len(loaded.event_history), 5)
        self.assertEqual(loaded.total_received_accepted_qty, 24.0)
        self.assertEqual(self.service.get_pipeline_quantity(), 0.0)  # Received order has 0 pipeline

    # -------------------------------------------------------------------------
    # 2. Idempotent Retries (No Duplicate Events, No Version Increments)
    # -------------------------------------------------------------------------

    def test_02_idempotent_retries_without_duplicate_events_or_version_increments(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        self.assertEqual(self.service.get_order_version(fo.fo_id), 1)
        self.assertEqual(len(fo.event_history), 1)

        # 1. Idempotent creation retry with same fo_id and idempotency_key
        key_create = fo.event_history[0].idempotency_key
        fo_create_retry = self.service.create_order(
            self.po,
            actor_id="AGENT",
            timestamp="2024-11-15T11:00:00Z",
            fo_id=fo.fo_id,
            idempotency_key=key_create,
        )
        self.assertEqual(fo_create_retry.fo_id, fo.fo_id)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 1)
        self.assertEqual(len(fo_create_retry.event_history), 1)

        # 2. Transmit order
        fo = self.service.transmit_order(
            fo.fo_id,
            timestamp="2024-11-15T11:15:00Z",
            actor_id="AGENT",
            idempotency_key="TX-EXACT-01",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ISSUED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), 2)
        self.assertEqual(len(fo.event_history), 2)

        # 3. Idempotent transmission retry with exact same idempotency_key & parameters
        fo_tx_retry = self.service.transmit_order(
            fo.fo_id,
            timestamp="2024-11-15T11:15:00Z",
            actor_id="AGENT",
            idempotency_key="TX-EXACT-01",
        )
        self.assertEqual(fo_tx_retry.status, FulfilmentOrderStatus.ISSUED)
        # Database version must remain strictly at 2
        self.assertEqual(self.service.get_order_version(fo.fo_id), 2)
        # Event history must not have any duplicate events appended
        self.assertEqual(len(fo_tx_retry.event_history), 2)
        events = self.service.get_order_events(fo.fo_id)
        self.assertEqual(len(events), 2)

    # -------------------------------------------------------------------------
    # 3. Stale Concurrent Updates (Optimistic Concurrency)
    # -------------------------------------------------------------------------

    def test_03_stale_concurrent_update_rejection(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        initial_version = self.service.get_order_version(fo.fo_id)
        self.assertEqual(initial_version, 1)

        # Simulating Worker 2 updating the order concurrently
        worker2_service = FulfilmentService(repository=self.repo)
        worker2_service.transmit_order(fo.fo_id, timestamp="2024-11-15T11:15:00Z", actor_id="AGENT_2")
        self.assertEqual(self.service.get_order_version(fo.fo_id), 2)

        # Worker 1 tries to transmit with stale expected_version=1 -> rejected
        with self.assertRaises(StaleUpdateError) as ctx:
            self.service.transmit_order(
                fo.fo_id,
                timestamp="2024-11-15T11:15:00Z",
                actor_id="AGENT_1",
                expected_version=1,
            )
        self.assertIn("Stale update conflict", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Multi-Aggregate Supersession Atomicity & Rollback
    # -------------------------------------------------------------------------

    def test_04a_successful_supersession_atomicity(self):
        # Setup order in SUPPLIER_CHANGE_PENDING_REVIEW
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)

        new_approved_po = _make_approved_po(
            po_id="PO-REPLACE-SUCCESS",
            final_order_qty=24.0,
            unit_procurement_cost=35.0,
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        superseded_fo, replacement_fo = self.service.supersede_order(
            original_fo_id=fo.fo_id,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REPLACE-SUCCESS",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:15:00Z",
            idempotency_key="SUP-TX-SUCCESS",
            reason="Supplier price updated; renegotiation completed.",
            supplier_cancellation_ref="SUPP-CNC-01",
        )

        # Assert in-memory states
        self.assertEqual(superseded_fo.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(replacement_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

        # Assert persisted database states
        loaded_old = self.service.get_order_or_raise(fo.fo_id)
        loaded_new = self.service.get_order_or_raise(replacement_fo.fo_id)
        self.assertEqual(loaded_old.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(loaded_new.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

        # Cross-audit linkage verified in database
        self.assertEqual(loaded_old.event_history[-1].details["superseded_by_fo_id"], loaded_new.fo_id)
        self.assertEqual(loaded_new.event_history[0].details["supersedes_fo_id"], loaded_old.fo_id)

    def test_04b_supersession_atomic_rollback_on_replacement_id_collision(self):
        # Pre-seed an order with the ID intended for replacement to force an integrity collision
        existing_po = _make_approved_po(po_id="PO-COLLISION-PRESEED")
        self.service.create_order(
            existing_po,
            actor_id="AGENT",
            timestamp="2024-11-15T11:00:00Z",
            fo_id="FO-COLLISION-ID",
        )
        self.assertIsNotNone(self.service.get_order("FO-COLLISION-ID"))

        # Setup original order in SUPPLIER_CHANGE_PENDING_REVIEW
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")
        initial_old_version = self.service.get_order_version(fo.fo_id)

        new_approved_po = _make_approved_po(
            po_id="PO-REPLACE-COLLIDE",
            final_order_qty=24.0,
            unit_procurement_cost=35.0,
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # Attempt supersession specifying colliding replacement ID -> must fail atomically
        with self.assertRaises(DuplicateFulfilmentOrderError) as ctx:
            self.service.supersede_order(
                original_fo_id=fo.fo_id,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-COLLISION-ID",  # Collision!
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:15:00Z",
                idempotency_key="SUP-COLLIDE-KEY",
                reason="Terms renegotiated",
                supplier_cancellation_ref="SUPP-CNC-01",
            )
        self.assertIn("already exists", str(ctx.exception))

        # Verify atomic rollback: original order was NOT cancelled and its version did not change
        reloaded_old = self.service.get_order_or_raise(fo.fo_id)
        self.assertEqual(reloaded_old.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)
        self.assertEqual(self.service.get_order_version(fo.fo_id), initial_old_version)

    def test_04c_supersession_atomic_rollback_on_stale_expected_version(self):
        # Setup original order in SUPPLIER_CHANGE_PENDING_REVIEW
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG-3",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")
        current_version = self.service.get_order_version(fo.fo_id)

        new_approved_po = _make_approved_po(
            po_id="PO-REPLACE-STALE",
            final_order_qty=24.0,
            unit_procurement_cost=35.0,
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        # Call with stale expected_version
        with self.assertRaises(StaleUpdateError) as ctx:
            self.service.supersede_order(
                original_fo_id=fo.fo_id,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REPLACE-STALE",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:15:00Z",
                idempotency_key="SUP-STALE-KEY",
                reason="Terms renegotiated",
                supplier_cancellation_ref="SUPP-CNC-01",
                expected_version=current_version - 1,  # Stale!
            )
        self.assertIn("Stale update conflict", str(ctx.exception))

        # Verify replacement order was never created
        self.assertIsNone(self.service.get_order("FO-REPLACE-STALE"))

    # -------------------------------------------------------------------------
    # 5. Delivery Exception & Resumption
    # -------------------------------------------------------------------------

    def test_05_delivery_exception_and_resumption_workflow(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-EXC",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-EXC-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, "GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        # Record delivery exception
        fo = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Delivery truck delayed due to highway obstruction.",
            actor_id="STORE_RECEIVER",
            timestamp="2024-11-15T16:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.DELIVERY_EXCEPTION)
        self.assertEqual(fo.resumable_status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(self.service.get_pipeline_quantity(), 24.0)

        # Resolve delivery exception -> resumes IN_TRANSIT
        fo = self.service.resolve_delivery_exception(
            fo.fo_id,
            resolution_notes="Truck resumed transit; arrival expected in 2 hours.",
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-15T17:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(self.service.get_order_or_raise(fo.fo_id).status, FulfilmentOrderStatus.IN_TRANSIT)

    # -------------------------------------------------------------------------
    # 6. Reaffirmation & Pipeline Restoration
    # -------------------------------------------------------------------------

    def test_06_reaffirm_terms_and_pipeline_restoration(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG-RF",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)
        self.assertEqual(self.service.get_pipeline_quantity(), 0.0)

        # Reaffirm terms
        reaffirm_ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-RF-CONFIRM",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T12:00:00Z",
            expected_delivery_at="2024-11-16T10:00:00Z",
        )
        fo = self.service.reaffirm_terms(
            fo.fo_id,
            resolution_notes="Supplier conceded to original commercial terms.",
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-15T12:15:00Z",
            ack=reaffirm_ack,
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(fo.lines[0].supplier_accepted_quantity, 24.0)
        # Pipeline confirmed quantity restored in persistence
        self.assertEqual(self.service.get_pipeline_quantity(), 24.0)

    # -------------------------------------------------------------------------
    # 7. Close Short & Cancellation Workflows
    # -------------------------------------------------------------------------

    def test_07_close_short_and_cancellation_workflows(self):
        # 1. Close Short Workflow
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CLS",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-CLS-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=10.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, "GATEWAY")
        grn = GoodsReceiptNote(
            grn_id="GRN-CLS-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-CLS-1",
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
        fo = self.service.receive_goods(fo.fo_id, grn, "RECEIVER_ARUN")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(self.service.get_pipeline_quantity(), 14.0)

        # Close short remaining 14 units
        fo = self.service.close_short_order(
            fo.fo_id,
            closure_reason="Vendor out of stock for remainder; closing out.",
            actor_id="STORE_MANAGER",
            timestamp="2024-11-16T12:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CLOSED_SHORT)
        self.assertEqual(fo.closed_unfulfilled_qty, 14.0)
        self.assertEqual(self.service.get_pipeline_quantity(), 0.0)

        # 2. Cancellation Workflow
        po_cnc = _make_approved_po(po_id="PO-CNC-TEST")
        fo_cnc = self.service.create_order(po_cnc, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo_cnc = self.service.transmit_order(fo_cnc.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        fo_cnc = self.service.request_cancellation(
            fo_cnc.fo_id,
            reason="Store inventory miscount; not required.",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-15T11:30:00Z",
        )
        self.assertEqual(fo_cnc.status, FulfilmentOrderStatus.CANCEL_REQUESTED)

        fo_cnc = self.service.confirm_cancellation(
            fo_cnc.fo_id,
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-15T12:00:00Z",
            supplier_cancellation_ref="SUPP-CNC-AGREED",
        )
        self.assertEqual(fo_cnc.status, FulfilmentOrderStatus.CANCELLED)
        self.assertEqual(self.service.get_order_or_raise(fo_cnc.fo_id).status, FulfilmentOrderStatus.CANCELLED)

    # -------------------------------------------------------------------------
    # 8. Supersession Idempotent Retry
    # -------------------------------------------------------------------------

    def test_08_supersession_idempotent_retry(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG-IDEMP",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")

        new_approved_po = _make_approved_po(
            po_id="PO-REPLACE-IDEMP",
            final_order_qty=24.0,
            unit_procurement_cost=35.0,
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        old_order, rep_order = self.service.supersede_order(
            original_fo_id=fo.fo_id,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REPLACE-IDEMP",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:15:00Z",
            idempotency_key="SUP-IDEMP-KEY",
            reason="Terms renegotiated",
            supplier_cancellation_ref="SUPP-CNC-01",
        )
        old_v1 = self.service.get_order_version(old_order.fo_id)
        rep_v1 = self.service.get_order_version(rep_order.fo_id)

        # Re-run supersession with exact same parameters and idempotency key
        old_order_retry, rep_order_retry = self.service.supersede_order(
            original_fo_id=fo.fo_id,
            replacement_po=new_approved_po,
            replacement_fo_id="FO-REPLACE-IDEMP",
            actor_id="BUYER_PRIYA",
            timestamp="2024-11-15T12:15:00Z",
            idempotency_key="SUP-IDEMP-KEY",
            reason="Terms renegotiated",
            supplier_cancellation_ref="SUPP-CNC-01",
        )

        self.assertEqual(old_order_retry.fo_id, old_order.fo_id)
        self.assertEqual(rep_order_retry.fo_id, rep_order.fo_id)
        self.assertEqual(self.service.get_order_version(old_order.fo_id), old_v1)
        self.assertEqual(self.service.get_order_version(rep_order.fo_id), rep_v1)

    # -------------------------------------------------------------------------
    # 9. Idempotency Conflict on Divergent Payload
    # -------------------------------------------------------------------------

    def test_09_idempotency_conflict_error_on_divergent_payload(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        self.service.transmit_order(
            fo.fo_id,
            timestamp="2024-11-15T11:15:00Z",
            actor_id="AGENT",
            idempotency_key="TX-CONFLICT-TEST",
        )

        # Attempt retransmission with same idempotency key but different timestamp
        with self.assertRaises((FulfilmentGovernanceError, IdempotencyConflictError)) as ctx:
            self.service.transmit_order(
                fo.fo_id,
                timestamp="2024-11-15T11:20:00Z",  # Different timestamp!
                actor_id="AGENT",
                idempotency_key="TX-CONFLICT-TEST",
            )
        self.assertIn("Idempotency Conflict", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 10. Query Operations, Filtering, and Pagination
    # -------------------------------------------------------------------------

    def test_10_query_operations_filtering_and_pagination(self):
        # Create orders for 2 different POs with canonical store IDs
        po1 = _make_approved_po(po_id="PO-QUERY-1", store_id="DS_BLR_KORAMANGALA")
        po2 = _make_approved_po(po_id="PO-QUERY-2", store_id="DS_BLR_INDIRANAGAR")

        fo1 = self.service.create_order(po1, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo2 = self.service.create_order(po2, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")

        # Query by PO ID
        orders_po1 = self.service.get_orders_by_po_id("PO-QUERY-1")
        self.assertEqual(len(orders_po1), 1)
        self.assertEqual(orders_po1[0].fo_id, fo1.fo_id)

        # Filtering by store
        store1_orders = self.service.list_orders(store_id="DS_BLR_KORAMANGALA")
        self.assertEqual(len(store1_orders), 1)
        self.assertEqual(store1_orders[0].fo_id, fo1.fo_id)

        # Filtering by status
        ready_orders = self.service.list_orders(status=FulfilmentOrderStatus.READY_TO_TRANSMIT)
        self.assertEqual(len(ready_orders), 2)

        # Count
        self.assertEqual(self.service.count_orders(store_id="DS_BLR_KORAMANGALA"), 1)
        self.assertEqual(self.service.count_orders(), 2)

        # Pagination: limit 1, offset 1
        paginated = self.service.list_orders(limit=1, offset=1)
        self.assertEqual(len(paginated), 1)

        # Not found raises FulfilmentOrderNotFoundError
        with self.assertRaises(FulfilmentOrderNotFoundError):
            self.service.get_order_or_raise("NON_EXISTENT_FO")

        # Pipeline breakdown queries
        breakdown = self.service.get_pipeline_breakdown()
        self.assertIsInstance(breakdown, dict)
        sku_quantities = self.service.get_sku_pipeline_quantities()
        self.assertIsInstance(sku_quantities, dict)

    # -------------------------------------------------------------------------
    # 11. Repeated Exception & Resolution Cycles Without Key Collision
    # -------------------------------------------------------------------------

    def test_11_repeated_exception_and_resolution_cycles_without_key_collision(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CYCLE",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-CYCLE-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T14:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, "GATEWAY")

        # Cycle 1: Exception 1 (default key -> EXC:fo_id:1)
        fo = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Traffic jam delay",
            timestamp="2024-11-15T15:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.DELIVERY_EXCEPTION)
        self.assertEqual(fo.event_history[-1].idempotency_key, f"EXC:{fo.fo_id}:1")

        # Exact retry of Exception 1 (no-op)
        fo_retry = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Traffic jam delay",
            timestamp="2024-11-15T15:00:00Z",
        )
        self.assertEqual(len(fo_retry.event_history), len(fo.event_history))

        # Resolution 1 (default key -> RES:fo_id:1)
        fo = self.service.resolve_delivery_exception(
            fo.fo_id,
            resolution_notes="Traffic cleared",
            timestamp="2024-11-15T15:30:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(fo.event_history[-1].idempotency_key, f"RES:{fo.fo_id}:1")

        # Cycle 2: Exception 2 (derived key -> EXC:fo_id:2) - NO KEY COLLISION!
        fo = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Flat tire breakdown",
            timestamp="2024-11-15T16:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.DELIVERY_EXCEPTION)
        self.assertEqual(fo.event_history[-1].idempotency_key, f"EXC:{fo.fo_id}:2")

        # Resolution 2 (derived key -> RES:fo_id:2) - NO KEY COLLISION!
        fo = self.service.resolve_delivery_exception(
            fo.fo_id,
            resolution_notes="Tire replaced, vehicle resumed transit",
            timestamp="2024-11-15T16:30:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(fo.event_history[-1].idempotency_key, f"RES:{fo.fo_id}:2")

        # Cycle 3 with explicit command identifiers
        fo = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Dock gate blocked",
            timestamp="2024-11-15T17:00:00Z",
            idempotency_key="CMD-EXC-003",
        )
        self.assertEqual(fo.event_history[-1].idempotency_key, "CMD-EXC-003")

        # Divergent retry of CMD-EXC-003 must be rejected
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.record_delivery_exception(
                fo.fo_id,
                exception_reason="Different reason entirely",
                timestamp="2024-11-15T17:00:00Z",
                idempotency_key="CMD-EXC-003",
            )
        self.assertIn("Idempotency Conflict", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 12. Split Dispatches After Partial Receipt Lifecycle Path
    # -------------------------------------------------------------------------

    def test_12_split_dispatches_after_partial_receipt(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-SPLIT",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")

        # Shipment 1: 10 units
        asn1 = AdvanceShipmentNotice(
            shipment_id="ASN-SPLIT-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-15T14:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=10.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn1, "GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        # Receipt 1: 10 units accepted -> PARTIALLY_RECEIVED
        grn1 = GoodsReceiptNote(
            grn_id="GRN-SPLIT-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SPLIT-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-15T14:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-SPLIT-1",),
                ),
            ),
        )
        fo = self.service.receive_goods(fo.fo_id, grn1, "RECEIVER_ARUN")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(fo.total_received_accepted_qty, 10.0)

        # Shipment 2: 14 units dispatched from PARTIALLY_RECEIVED -> IN_TRANSIT
        asn2 = AdvanceShipmentNotice(
            shipment_id="ASN-SPLIT-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T16:00:00Z",
            expected_arrival_at="2024-11-15T18:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=14.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn2, "GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        # Receipt 2: 14 units accepted -> RECEIVED (terminal)
        grn2 = GoodsReceiptNote(
            grn_id="GRN-SPLIT-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-SPLIT-2",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-15T18:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=14.0,
                    accepted_quantity=14.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-SPLIT-2",),
                ),
            ),
        )
        fo = self.service.receive_goods(fo.fo_id, grn2, "RECEIVER_ARUN")
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(fo.total_received_accepted_qty, 24.0)

    # -------------------------------------------------------------------------
    # 13. Repeated Supplier Change & Reaffirmation Cycles
    # -------------------------------------------------------------------------

    def test_13_repeated_supplier_change_and_reaffirmation_cycles(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")

        # Cycle 1: Change request 1 -> review
        ack1 = SupplierAcknowledgment(
            acknowledgment_id="ACK-CYCLE-CHG-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 32.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack1, "GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)

        # Cycle 1: Reaffirmation 1 -> ACKNOWLEDGED
        fo = self.service.reaffirm_terms(
            fo.fo_id,
            resolution_notes="Buyer reaffirmed original price 30.0",
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T12:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(fo.event_history[-1].idempotency_key, f"REAFFIRM:{fo.fo_id}:1")

        # Cycle 2: Supplier requests change AGAIN from ACKNOWLEDGED status before shipping
        ack2 = SupplierAcknowledgment(
            acknowledgment_id="ACK-CYCLE-CHG-2",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T12:30:00Z",
            change_request_details={"suggested_price": 31.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack2, "GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)

        # Cycle 2: Reaffirmation 2 -> ACKNOWLEDGED (uses derived key REAFFIRM:fo_id:2 without collision)
        fo = self.service.reaffirm_terms(
            fo.fo_id,
            resolution_notes="Supplier accepted after second negotiation",
            actor_id="SUPERVISOR",
            timestamp="2024-11-15T13:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(fo.event_history[-1].idempotency_key, f"REAFFIRM:{fo.fo_id}:2")

    # -------------------------------------------------------------------------
    # 14. Cancellation Rejection & Lifecycle Recovery
    # -------------------------------------------------------------------------

    def test_14_cancellation_rejection_and_lifecycle_recovery(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CAN-REJ",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")

        # 1. Request cancellation
        fo = self.service.request_cancellation(
            fo.fo_id,
            reason="Mistaken duplicate reorder",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            timestamp="2024-11-15T12:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED)

        # 2. Reject cancellation -> Restores order back to ACKNOWLEDGED with dedicated REJECT_CANCEL event
        fo = self.service.reject_cancellation(
            fo.fo_id,
            rejection_reason="Supplier already loaded truck, cancellation rejected",
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-15T12:30:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertIsNone(fo.cancellation_reason)

        # Audit event classification and evidence preservation verification
        rej_event = fo.event_history[-1]
        self.assertEqual(rej_event.event_type, FulfilmentEventType.REJECT_CANCEL)
        self.assertEqual(rej_event.new_status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(rej_event.previous_status, FulfilmentOrderStatus.CANCEL_REQUESTED)
        self.assertEqual(rej_event.actor_id, "OPERATIONS_SUPERVISOR")
        self.assertEqual(rej_event.details["rejection_reason"], "Supplier already loaded truck, cancellation rejected")
        self.assertEqual(rej_event.details["rejection_actor"], "OPERATIONS_SUPERVISOR")
        self.assertEqual(rej_event.details["rejection_timestamp"], "2024-11-15T12:30:00Z")
        self.assertEqual(rej_event.details["restored_status"], "ACKNOWLEDGED")
        self.assertEqual(rej_event.details["original_cancellation_reason"], "Mistaken duplicate reorder")
        self.assertEqual(rej_event.details["original_cancellation_category"], CancellationReasonCategory.OPERATIONAL_MISTAKE.value)

        # 3. Order can now continue normal lifecycle to shipment dispatch
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-RECOVER-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T13:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, "GATEWAY")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

    # -------------------------------------------------------------------------
    # 15. Resolve Delivery Exception With Explicit Target Status
    # -------------------------------------------------------------------------

    def test_15_resolve_delivery_exception_with_target_status(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-TGT-EXC",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-TGT-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-15T14:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=10.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, "GATEWAY")
        grn = GoodsReceiptNote(
            grn_id="GRN-TGT-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-TGT-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-15T14:00:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-TGT-1",),
                ),
            ),
        )
        fo = self.service.receive_goods(fo.fo_id, grn, "RECEIVER_ARUN")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Record exception during dock handling
        fo = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Barcode scanner system down",
            timestamp="2024-11-15T14:30:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.DELIVERY_EXCEPTION)

        # Resolve explicitly to PARTIALLY_RECEIVED
        fo = self.service.resolve_delivery_exception(
            fo.fo_id,
            resolution_notes="Scanner recalibrated",
            target_status=FulfilmentOrderStatus.PARTIALLY_RECEIVED,
            timestamp="2024-11-15T15:00:00Z",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

    # -------------------------------------------------------------------------
    # 16. Supersession Atomic Rollback on Idempotency Conflict
    # -------------------------------------------------------------------------

    def test_16_supersession_atomic_rollback_on_idempotency_conflict(self):
        # Pre-seed an order that owns the key CREATE:FO-REPLACE-COLL:1
        pre_po = _make_approved_po(po_id="PO-PRE-IDEMP")
        self.service.create_order(
            pre_po,
            actor_id="AGENT",
            timestamp="2024-11-15T11:00:00Z",
            fo_id="FO-PRE-COLL",
            idempotency_key="CREATE:FO-REPLACE-COLL:1",
        )

        # Setup original order in SUPPLIER_CHANGE_PENDING_REVIEW
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CHG-COLL",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"suggested_price": 35.0},
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, actor_id="GATEWAY")
        initial_version = self.service.get_order_version(fo.fo_id)

        # Replacement order attempting to supersede with replacement_fo_id='FO-REPLACE-COLL'
        # which will attempt to use CREATE:FO-REPLACE-COLL:1, colliding with FO-PRE-COLL!
        new_approved_po = _make_approved_po(
            po_id="PO-REPLACE-COLL",
            final_order_qty=24.0,
            unit_procurement_cost=35.0,
            approval_timestamp="2024-11-15T12:00:00Z",
        )

        with self.assertRaises(IdempotencyConflictError):
            self.service.supersede_order(
                original_fo_id=fo.fo_id,
                replacement_po=new_approved_po,
                replacement_fo_id="FO-REPLACE-COLL",
                actor_id="BUYER_PRIYA",
                timestamp="2024-11-15T12:15:00Z",
                idempotency_key="SUP-COLL-KEY",
                reason="Terms renegotiated",
                supplier_cancellation_ref="SUPP-CNC-01",
            )

        # Verify rollback: original order was NOT cancelled in DB
        reloaded_old = self.service.get_order_or_raise(fo.fo_id)
        self.assertEqual(reloaded_old.status, FulfilmentOrderStatus.SUPPLIER_CHANGE_PENDING_REVIEW)
        self.assertEqual(self.service.get_order_version(fo.fo_id), initial_version)
        self.assertIsNone(self.service.get_order("FO-REPLACE-COLL"))

    # -------------------------------------------------------------------------
    # 17. Governance Defense Against Unapproved PO & Tampered Model
    # -------------------------------------------------------------------------

    def test_17_governance_defense_against_unapproved_po_and_tampered_model(self):
        # 1. Unapproved PO rejected by service.create_order
        unapproved_po = self.po.model_copy(update={"status": PurchaseOrderStatus.PENDING_APPROVAL})
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.create_order(unapproved_po)
        self.assertIn("approval evidence is strictly required", str(ctx.exception))

        # 2. Tampered FulfilmentOrder model rejected by repository.create
        valid_fo = self.service.create_order(self.po, fo_id="FO-VALID-GOV")
        # Attempting to mutate status directly via model_copy bypass
        tampered_fo = valid_fo.model_copy(update={"fo_id": "FO-TAMPERED-GOV", "status": FulfilmentOrderStatus.RECEIVED})

        with self.assertRaises(FulfilmentGovernanceError):
            self.repo.create(tampered_fo)

        self.assertIsNone(self.service.get_order("FO-TAMPERED-GOV"))

    # -------------------------------------------------------------------------
    # 18. Durable Idempotency Across Later Lifecycle Transitions
    # -------------------------------------------------------------------------

    def test_18_durable_idempotency_across_later_lifecycle_transitions(self):
        # 1. Setup order to IN_TRANSIT
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-DURABLE",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-DURABLE-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-15T14:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
        )
        fo = self.service.dispatch_shipment(fo.fo_id, asn, "GATEWAY")

        # 2. Record exception with explicit command ID
        fo = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Engine cooling failure",
            actor_id="DRIVER_MANOJ",
            timestamp="2024-11-15T13:00:00Z",
            idempotency_key="CMD-EXC-99",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.DELIVERY_EXCEPTION)

        # 3. Resolve exception with explicit command ID -> back to IN_TRANSIT
        fo = self.service.resolve_delivery_exception(
            fo.fo_id,
            resolution_notes="Coolant added, temperature normalized",
            actor_id="SUPERVISOR_PRIYA",
            timestamp="2024-11-15T13:30:00Z",
            idempotency_key="CMD-RES-99",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        # 4. Progress order through dock receipt to terminal RECEIVED state
        grn = GoodsReceiptNote(
            grn_id="GRN-DURABLE-1",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            shipment_id="ASN-DURABLE-1",
            store_id=fo.store_id,
            supplier_id=fo.supplier_id,
            receipt_at="2024-11-15T14:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,
                    batch_references=("BATCH-DURABLE-1",),
                ),
            ),
        )
        fo = self.service.receive_goods(fo.fo_id, grn, "RECEIVER_ARUN")
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)
        v_terminal = self.service.get_order_version(fo.fo_id)
        event_count = len(fo.event_history)

        # 5. RETRY original exception command with CMD-EXC-99 while order is in RECEIVED status
        fo_retry_exc = self.service.record_delivery_exception(
            fo.fo_id,
            exception_reason="Engine cooling failure",
            actor_id="DRIVER_MANOJ",
            timestamp="2024-11-15T13:00:00Z",
            idempotency_key="CMD-EXC-99",
        )
        # Durable idempotency guarantees exact no-op: status remains RECEIVED, version unchanged
        self.assertEqual(fo_retry_exc.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), v_terminal)
        self.assertEqual(len(fo_retry_exc.event_history), event_count)

        # 6. RETRY original resolution command with CMD-RES-99 while order is in RECEIVED status
        fo_retry_res = self.service.resolve_delivery_exception(
            fo.fo_id,
            resolution_notes="Coolant added, temperature normalized",
            actor_id="SUPERVISOR_PRIYA",
            timestamp="2024-11-15T13:30:00Z",
            idempotency_key="CMD-RES-99",
        )
        self.assertEqual(fo_retry_res.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), v_terminal)
        self.assertEqual(len(fo_retry_res.event_history), event_count)

        # 7. Persistence integrity verification
        reloaded = self.service.get_order_or_raise(fo.fo_id)
        self.assertEqual(reloaded, fo)
        db_events = self.service.get_order_events(fo.fo_id)
        self.assertEqual(len(db_events), event_count)
        self.assertEqual([e.sequence_number for e in db_events], list(range(1, event_count + 1)))

    # -------------------------------------------------------------------------
    # 19. Cancellation Cycle Recovery and Original Request Retry
    # -------------------------------------------------------------------------

    def test_19_cancellation_cycle_recovery_and_original_request_retry(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-CAN-CYCLE",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")

        # Cycle 1: Request cancellation
        fo = self.service.request_cancellation(
            fo.fo_id,
            reason="Mistake in demand forecasting model",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            actor_id="PLANNER_SITA",
            timestamp="2024-11-15T11:45:00Z",
            idempotency_key="CMD-REQ-CNC-1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED)

        # Cycle 1: Reject cancellation request -> restored to ACKNOWLEDGED
        fo = self.service.reject_cancellation(
            fo.fo_id,
            rejection_reason="Supplier production already scheduled",
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-15T12:00:00Z",
            idempotency_key="CMD-REJ-CNC-1",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)

        # Cycle 2: New cancellation request with distinct command ID
        fo = self.service.request_cancellation(
            fo.fo_id,
            reason="Cold chain compressor broke at dark-store",
            category=CancellationReasonCategory.STORE_DEMAND_CHANGED,
            actor_id="STORE_MANAGER",
            timestamp="2024-11-15T12:30:00Z",
            idempotency_key="CMD-REQ-CNC-2",
        )
        self.assertEqual(fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED)
        v_current = self.service.get_order_version(fo.fo_id)
        event_count = len(fo.event_history)

        # RETRY original request from Cycle 1 (CMD-REQ-CNC-1)
        fo_retry_cnc1 = self.service.request_cancellation(
            fo.fo_id,
            reason="Mistake in demand forecasting model",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            actor_id="PLANNER_SITA",
            timestamp="2024-11-15T11:45:00Z",
            idempotency_key="CMD-REQ-CNC-1",
        )
        # Must be exact no-op: does not append duplicate events, does not change version
        self.assertEqual(fo_retry_cnc1.status, FulfilmentOrderStatus.CANCEL_REQUESTED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), v_current)
        self.assertEqual(len(fo_retry_cnc1.event_history), event_count)

        # RETRY original rejection from Cycle 1 (CMD-REJ-CNC-1)
        fo_retry_rej1 = self.service.reject_cancellation(
            fo.fo_id,
            rejection_reason="Supplier production already scheduled",
            actor_id="OPERATIONS_SUPERVISOR",
            timestamp="2024-11-15T12:00:00Z",
            idempotency_key="CMD-REJ-CNC-1",
        )
        self.assertEqual(fo_retry_rej1.status, FulfilmentOrderStatus.CANCEL_REQUESTED)
        self.assertEqual(self.service.get_order_version(fo.fo_id), v_current)
        self.assertEqual(len(fo_retry_rej1.event_history), event_count)

    # -------------------------------------------------------------------------
    # 20. Same Command ID With Altered Payload Rejection Across Lifecycle
    # -------------------------------------------------------------------------

    def test_20_command_id_altered_payload_rejection_across_lifecycle(self):
        fo = self.service.create_order(self.po, actor_id="AGENT", timestamp="2024-11-15T11:00:00Z")
        fo = self.service.transmit_order(fo.fo_id, "2024-11-15T11:15:00Z", "AGENT")
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-ALTERED",
            fo_id=fo.fo_id,
            po_id=fo.po_id,
            supplier_id=fo.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T08:00:00Z",
        )
        fo = self.service.acknowledge_order(fo.fo_id, ack, "GATEWAY")

        # Record cancellation request with CMD-ALTER-1
        fo = self.service.request_cancellation(
            fo.fo_id,
            reason="Initial cancellation reason",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            actor_id="AGENT",
            timestamp="2024-11-15T11:45:00Z",
            idempotency_key="CMD-ALTER-1",
        )

        # Attempt replay with altered reason -> rejected with Idempotency Conflict
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.request_cancellation(
                fo.fo_id,
                reason="Altered reason completely",
                category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
                actor_id="AGENT",
                timestamp="2024-11-15T11:45:00Z",
                idempotency_key="CMD-ALTER-1",
            )
        self.assertIn("Idempotency Conflict", str(ctx.exception))

        # Attempt to reuse CMD-ALTER-1 on a different transition -> rejected with Idempotency Conflict
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.confirm_cancellation(
                fo.fo_id,
                actor_id="OPERATIONS_SUPERVISOR",
                timestamp="2024-11-15T12:00:00Z",
                supplier_cancellation_ref="SUPP-CNC-AGREED",
                idempotency_key="CMD-ALTER-1",  # Cross-operation key collision!
            )
        self.assertIn("Idempotency Conflict", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
