"""
Unit & Integration Tests: Durable Shipment Tracking Persistence & Fulfilment Integration
========================================================================================
Phase 5.5.2 Verification Test Suite:
1. Registration & Foreign Key Enforcement (linked to FO and shipment ID).
2. Rejection of invalid FO or unregistered shipment IDs.
3. Carrier validation against dispatch ASN carrier.
4. Multiple independently tracked shipments per fulfilment order.
5. Ingestion of provider events and atomic projection updates.
6. Provider and service recreation (restart recovery across simulator/service restarts).
7. Exact duplicate provider event deduplication (idempotent no-op, no version increment).
8. Conflicting reuse of provider_event_id rejection.
9. Out-of-order event arrival handling without status regression.
10. Out-of-order event after terminal DELIVERED status does not un-deliver in SQLite.
11. Optimistic locking and stale update rejection (StaleUpdateError).
12. Transaction rollback on simulated database failures.
13. Critical business boundary: DELIVERED tracking does NOT mutate GRNs, accepted qty, or pipeline qty.
14. Complete physical dock receiving workflow following tracking delivery.
"""

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from src.agents.fulfilment import (
    AdvanceShipmentNotice,
    FulfilmentGovernanceError,
    FulfilmentOrder,
    FulfilmentOrderNotFoundError,
    FulfilmentOrderStatus,
    FulfilmentPersistenceError,
    FulfilmentRepository,
    FulfilmentService,
    GoodsReceiptNote,
    IdempotencyConflictError,
    InvalidTrackingEventError,
    InvalidTrackingTransitionError,
    ShipmentLineItem,
    ShipmentNotFoundError,
    ShipmentStatus,
    ShipmentTrackingEvent,
    ShipmentTrackingRecord,
    ShipmentTrackingResult,
    ShipmentTrackingService,
    ShipmentTrackingStatus,
    SimulatedSupplierAdapter,
    SimulatedTrackingProvider,
    StaleUpdateError,
    SupplierAcknowledgment,
    SupplierAcknowledgmentLine,
    SupplierAcknowledgmentStatus,
    SupplierSimulationScenario,
    TrackingIdempotencyConflictError,
    TrackingLocation,
    TrackingProvider,
    TrackingScenario,
    assert_tracking_does_not_mutate_fulfilment_order,
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
    po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
    store_id: str = "DS_BLR_KORAMANGALA",
    supplier_id: str = "SIM_SUPP_DAIRY_PRIMARY",
    sku_id: str = "SKU_MILK_1L",
    category: str = "Dairy",
    final_order_qty: float = 24.0,
    unit_procurement_cost: float = 30.0,
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
        supplier_id=supplier_id,
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
        approval_timestamp="2024-11-15T10:30:00Z",
        approval_reason="Approved for fulfilment.",
    )


class TestShipmentTrackingPersistence(unittest.TestCase):
    """Integration tests for durable SQLite shipment tracking persistence and FulfilmentService."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_fulfilment.db"
        self.repo = FulfilmentRepository(db_path=self.db_path)
        self.gateway = SimulatedSupplierAdapter(db_path=self.db_path)
        self.tracking_provider = SimulatedTrackingProvider()
        self.service = FulfilmentService(
            repository=self.repo,
            gateway=self.gateway,
            tracking_provider=self.tracking_provider,
        )

        # Create approved PO, transmit, acknowledge, and dispatch shipment
        self.po = _make_approved_po()
        self.fo = self.service.create_order(self.po)
        self.fo, _ = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-PO-01",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        self.shipment_id = "SHP-FO-001-01"
        self.carrier_id = "BLUEDART"

        asn = AdvanceShipmentNotice(
            shipment_id=self.shipment_id,
            fo_id=self.fo.fo_id,
            po_id=self.po.po_id,
            supplier_id=self.po.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-16T12:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
            carrier=self.carrier_id,
            tracking_reference="WB-100293",
        )
        self.fo = self.service.apply_shipment_dispatch(
            fo_id=self.fo.fo_id,
            asn=asn,
            idempotency_key="CMD-DSP-01",
        )

    def tearDown(self):
        self.service.close()
        self.gateway.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Registration & Foreign Key Enforcement
    # -------------------------------------------------------------------------

    def test_01_register_shipment_persists_in_sqlite(self):
        """Registering a dispatched shipment creates projection and initial event in SQLite."""
        record = self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T12:00:00Z",
        )

        self.assertEqual(record.shipment_id, self.shipment_id)
        self.assertEqual(record.current_status, ShipmentTrackingStatus.LABEL_CREATED)
        self.assertEqual(record.carrier_id, self.carrier_id)
        self.assertEqual(record.event_count, 1)

        # Inspect raw database tables
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM shipment_tracking_records WHERE shipment_id = ?", (self.shipment_id,))
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["fo_id"], self.fo.fo_id)
        self.assertEqual(row["carrier_id"], self.carrier_id)
        self.assertEqual(row["current_status"], ShipmentTrackingStatus.LABEL_CREATED.value)
        self.assertEqual(row["version"], 1)

        cur_evts = conn.execute("SELECT * FROM shipment_tracking_events WHERE shipment_id = ?", (self.shipment_id,))
        evt_rows = cur_evts.fetchall()
        self.assertEqual(len(evt_rows), 1)
        self.assertEqual(evt_rows[0]["status"], ShipmentTrackingStatus.LABEL_CREATED.value)
        conn.close()

    def test_02_register_shipment_invalid_fo_rejected(self):
        """Attempting to register tracking for non-existent FO raises FulfilmentOrderNotFoundError."""
        with self.assertRaises(FulfilmentOrderNotFoundError):
            self.service.register_shipment(
                fo_id="FO-NON-EXISTENT-999",
                shipment_id=self.shipment_id,
                carrier_id=self.carrier_id,
            )

    def test_03_register_shipment_unregistered_shipment_id_rejected(self):
        """Attempting to register a shipment ID not associated with the FO raises FulfilmentGovernanceError."""
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.register_shipment(
                fo_id=self.fo.fo_id,
                shipment_id="SHP-ROGUE-999",
                carrier_id=self.carrier_id,
            )
        self.assertIn("not associated with FulfilmentOrder", str(ctx.exception))

    def test_04_carrier_mismatch_validation(self):
        """If dispatch ASN specified 'BLUEDART', registering with 'DELHIVERY' raises FulfilmentGovernanceError."""
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.register_shipment(
                fo_id=self.fo.fo_id,
                shipment_id=self.shipment_id,
                carrier_id="DELHIVERY",  # Mismatched carrier!
            )
        self.assertIn("Carrier mismatch", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Multiple Independently Tracked Shipments
    # -------------------------------------------------------------------------

    def test_05_multiple_shipments_per_fulfilment_order(self):
        """Multiple shipments for same FO maintain separate tracking records and progress independently."""
        po_multi = _make_approved_po(
            po_id="PO-TEST-MULTI-01",
            final_order_qty=48.0,
        )
        fo_created = self.service.create_order(po_multi, idempotency_key="CMD-CREATE-MULTI-01")
        fo_multi, _ = self.service.transmit_purchase_order(
            fo_created.fo_id,
            po_multi,
            idempotency_key="CMD-TXN-MULTI-01",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        shipment_1 = "SHP-MULTI-01"
        shipment_2 = "SHP-MULTI-02"

        asn1 = AdvanceShipmentNotice(
            shipment_id=shipment_1,
            fo_id=fo_multi.fo_id,
            po_id=po_multi.po_id,
            supplier_id=po_multi.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-16T12:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
            carrier=self.carrier_id,
        )
        self.service.dispatch_shipment(
            fo_id=fo_multi.fo_id,
            asn=asn1,
            idempotency_key="CMD-DSP-MULTI-01",
        )

        asn2 = AdvanceShipmentNotice(
            shipment_id=shipment_2,
            fo_id=fo_multi.fo_id,
            po_id=po_multi.po_id,
            supplier_id=po_multi.supplier_id,
            dispatch_at="2024-11-15T13:00:00Z",
            expected_arrival_at="2024-11-16T15:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
            carrier="DELHIVERY",
        )
        self.service.dispatch_shipment(
            fo_id=fo_multi.fo_id,
            asn=asn2,
            idempotency_key="CMD-DSP-MULTI-02",
        )

        rec1 = self.service.register_shipment(
            fo_id=fo_multi.fo_id,
            shipment_id=shipment_1,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
        )
        rec2 = self.service.register_shipment(
            fo_id=fo_multi.fo_id,
            shipment_id=shipment_2,
            carrier_id="DELHIVERY",
            scenario=TrackingScenario.RETURN_TO_ORIGIN,
        )

        # Advance SHP 1 to DELIVERED
        self.service.advance_to_tracking_status(shipment_1, ShipmentTrackingStatus.DELIVERED)
        track1 = self.service.get_shipment_tracking(shipment_1)
        self.assertEqual(track1.status, ShipmentTrackingStatus.DELIVERED)

        # SHP 2 remains at initial status
        track2 = self.service.get_shipment_tracking(shipment_2)
        self.assertEqual(track2.status, ShipmentTrackingStatus.LABEL_CREATED)

        # Retrieve both shipments for order
        all_shipments = self.service.get_shipments_for_order(fo_multi.fo_id)
        self.assertEqual(len(all_shipments), 2)
        statuses = {s.shipment_id: s.current_status for s in all_shipments}
        self.assertEqual(statuses[shipment_1], ShipmentTrackingStatus.DELIVERED)
        self.assertEqual(statuses[shipment_2], ShipmentTrackingStatus.LABEL_CREATED)

    # -------------------------------------------------------------------------
    # 3. Event Ingestion & Atomic State Progression
    # -------------------------------------------------------------------------

    def test_06_ingest_tracking_event_atomically_updates_projection(self):
        """Ingesting an event atomically records the event and increments record version in SQLite."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        initial_ver = self.repo.get_tracking_version(self.shipment_id)
        self.assertEqual(initial_ver, 1)

        event_pickup = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-BD-PICKUP-01",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:05:00Z",
            location=TrackingLocation(hub_id="HUB-VND", facility_name="Vendor Dispatch Hub"),
        )
        updated = self.service.ingest_tracking_event(event_pickup)
        self.assertEqual(updated.current_status, ShipmentTrackingStatus.PICKED_UP)
        self.assertEqual(updated.event_count, 2)

        new_ver = self.repo.get_tracking_version(self.shipment_id)
        self.assertEqual(new_ver, 2)

        # Verify event history query
        history = self.service.get_tracking_history(self.shipment_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1].provider_event_id, "EVT-BD-PICKUP-01")

    # -------------------------------------------------------------------------
    # 4. Service & Provider Recreation (Restart Recovery)
    # -------------------------------------------------------------------------

    def test_07_service_and_provider_recreation_restart_recovery(self):
        """Tracking state and advancement resume seamlessly across application/service restarts."""
        # 1. Register and advance to IN_TRANSIT with initial service
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T10:00:00Z",
        )
        self.service.advance_shipment_tracking(self.shipment_id, steps=2)
        rec_before = self.service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(rec_before.status, ShipmentTrackingStatus.IN_TRANSIT)
        self.assertEqual(rec_before.record.event_count, 3)

        # 2. Close initial service and drop references
        self.service.close()
        self.repo.close()

        # 3. Create fresh service and provider instances pointing to same SQLite database
        new_repo = FulfilmentRepository(db_path=self.db_path)
        new_tracking_provider = SimulatedTrackingProvider()
        new_service = FulfilmentService(
            repository=new_repo,
            tracking_provider=new_tracking_provider,
        )

        # 4. Query reconstituted tracking projection from SQLite
        rec_restarted = new_service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(rec_restarted.status, ShipmentTrackingStatus.IN_TRANSIT)
        self.assertEqual(rec_restarted.record.event_count, 3)

        # 5. Seamlessly advance simulation from where it left off
        new_service.advance_shipment_tracking(self.shipment_id, steps=1)
        rec_after = new_service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(rec_after.status, ShipmentTrackingStatus.AT_HUB)
        self.assertEqual(rec_after.record.event_count, 4)

        # 6. Advance to final delivery
        new_service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.DELIVERED)
        final_rec = new_service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(final_rec.status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(final_rec.record.is_delivered)

        new_service.close()
        new_repo.close()

    # -------------------------------------------------------------------------
    # 5. Deduplication & Idempotent Retries
    # -------------------------------------------------------------------------

    def test_08_exact_duplicate_event_is_no_op(self):
        """Ingesting the exact same event twice is an idempotent no-op without version increments."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        evt = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-DUP-01",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:05:00Z",
        )
        # First ingestion
        rec1 = self.service.ingest_tracking_event(evt)
        ver1 = self.repo.get_tracking_version(self.shipment_id)
        self.assertEqual(rec1.event_count, 2)
        self.assertEqual(ver1, 2)

        # Second exact duplicate ingestion
        rec2 = self.service.ingest_tracking_event(evt)
        ver2 = self.repo.get_tracking_version(self.shipment_id)
        self.assertEqual(rec2.event_count, 2)
        self.assertEqual(ver2, 2)  # Version did NOT increment!

    # -------------------------------------------------------------------------
    # 6. Conflicting Reuse Rejection
    # -------------------------------------------------------------------------

    def test_09_conflicting_event_id_reuse_rejected(self):
        """Reusing provider_event_id with differing payload raises TrackingIdempotencyConflictError."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        evt = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-REUSE-ID",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:05:00Z",
        )
        self.service.ingest_tracking_event(evt)

        # Same event ID but conflicting status (IN_TRANSIT)
        evt_conflicting = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-REUSE-ID",
            status=ShipmentTrackingStatus.IN_TRANSIT,
            event_timestamp="2024-11-15T14:00:00Z",
            received_timestamp="2024-11-15T14:05:00Z",
        )
        with self.assertRaises(TrackingIdempotencyConflictError):
            self.service.ingest_tracking_event(evt_conflicting)

    # -------------------------------------------------------------------------
    # 7. Out-of-Order Handling Without Status Regression
    # -------------------------------------------------------------------------

    def test_10_out_of_order_event_arrival_persisted_without_regression(self):
        """Late-arriving historical event persists in audit trail without regressing current status in SQLite."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T10:00:00Z",
        )
        e1 = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="E1",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T11:00:00Z",
            received_timestamp="2024-11-15T11:05:00Z",
        )
        e3 = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="E3",
            status=ShipmentTrackingStatus.OUT_FOR_DELIVERY,
            event_timestamp="2024-11-15T16:00:00Z",
            received_timestamp="2024-11-15T16:05:00Z",
        )
        self.service.ingest_tracking_event(e1)
        self.service.ingest_tracking_event(e3)

        track_mid = self.service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(track_mid.status, ShipmentTrackingStatus.OUT_FOR_DELIVERY)

        # Late-arriving IN_TRANSIT event from 13:00 (arrives at 16:30)
        e2_late = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="E2-LATE",
            status=ShipmentTrackingStatus.IN_TRANSIT,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T16:30:00Z",
        )
        self.service.ingest_tracking_event(e2_late)

        # Status in SQLite remains OUT_FOR_DELIVERY
        track_after = self.service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(track_after.status, ShipmentTrackingStatus.OUT_FOR_DELIVERY)

        # Events list has 4 events in chronological sequence
        history = self.service.get_tracking_history(self.shipment_id)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].status, ShipmentTrackingStatus.LABEL_CREATED)
        self.assertEqual(history[1].status, ShipmentTrackingStatus.PICKED_UP)
        self.assertEqual(history[2].status, ShipmentTrackingStatus.IN_TRANSIT)  # Correctly placed at 13:00
        self.assertEqual(history[3].status, ShipmentTrackingStatus.OUT_FOR_DELIVERY)

    # -------------------------------------------------------------------------
    # 8. Out-of-Order After Terminal DELIVERED Status
    # -------------------------------------------------------------------------

    def test_11_out_of_order_event_after_delivered_does_not_un_deliver(self):
        """Late-arriving packet after DELIVERED status persists without un-delivering the shipment."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T10:00:00Z",
        )
        # Advance to DELIVERED
        self.service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.DELIVERED)
        track_del = self.service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(track_del.status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(track_del.record.is_delivered)

        # Delayed AT_HUB event from 14:00 arrives late at 18:00
        e_delayed = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="E-DELAYED-PACKET",
            status=ShipmentTrackingStatus.AT_HUB,
            event_timestamp="2024-11-15T14:00:00Z",
            received_timestamp="2024-11-15T18:00:00Z",
        )
        self.service.ingest_tracking_event(e_delayed)

        # DELIVERED status is strictly preserved in SQLite
        track_del_after = self.service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(track_del_after.status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(track_del_after.record.is_delivered)

    # -------------------------------------------------------------------------
    # 9. Optimistic Locking & Stale Update Rejection
    # -------------------------------------------------------------------------

    def test_12_optimistic_locking_stale_version_rejected(self):
        """Ingesting an event with a stale expected_version raises StaleUpdateError."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T10:00:00Z",
        )
        evt = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="E-STALE-CHECK",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T11:00:00Z",
            received_timestamp="2024-11-15T11:05:00Z",
        )
        with self.assertRaises(StaleUpdateError):
            self.service.ingest_tracking_event(evt, expected_version=999)

    # -------------------------------------------------------------------------
    # 10. Transaction Rollback & Integrity
    # -------------------------------------------------------------------------

    def test_13_transaction_rollback_on_database_failure(self):
        """Simulated DB error during update rolls back transaction, preserving uncorrupted prior state."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T10:00:00Z",
        )
        orig_insert = self.repo._insert_tracking_event_internal

        def crashing_insert(conn, event):
            raise sqlite3.OperationalError("Simulated disk I/O failure during tracking event insert")

        self.repo._insert_tracking_event_internal = crashing_insert

        evt = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="E-CRASH-TEST",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T11:00:00Z",
            received_timestamp="2024-11-15T11:05:00Z",
        )
        with self.assertRaises(sqlite3.OperationalError):
            self.service.ingest_tracking_event(evt)

        # Restore normal insert
        self.repo._insert_tracking_event_internal = orig_insert

        # Verify state was rolled back to initial LABEL_CREATED
        rec = self.service.get_shipment_tracking(self.shipment_id)
        self.assertEqual(rec.status, ShipmentTrackingStatus.LABEL_CREATED)
        self.assertEqual(rec.record.event_count, 1)

    # -------------------------------------------------------------------------
    # 11. Critical Business Boundary: DELIVERED Tracking vs GRN
    # -------------------------------------------------------------------------

    def test_14_delivered_tracking_does_not_mutate_goods_receipt(self):
        """
        CRITICAL ARCHITECTURAL BOUNDARY:
        DELIVERED tracking status in SQLite does NOT create a GRN,
        does NOT increase accepted or delivered quantities,
        does NOT alter confirmed inbound pipeline quantity,
        and does NOT modify the approved PurchaseOrder.
        """
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T10:00:00Z",
        )
        # Advance tracking to DELIVERED
        self.service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.DELIVERED)

        # Retrieve FO from SQLite
        fo_persisted = self.repo.get_or_raise(self.fo.fo_id)

        # Enforce all Phase 5.5.1 boundary assertions
        assert_tracking_does_not_mutate_fulfilment_order(
            self.fo,
            fo_persisted,
            po_before=self.po,
            po_after=self.po,
        )

        self.assertEqual(fo_persisted.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(fo_persisted.total_confirmed_pipeline_qty, 24.0)
        self.assertEqual(fo_persisted.lines[0].delivered_quantity, 0.0)
        self.assertEqual(fo_persisted.lines[0].accepted_quantity, 0.0)
        self.assertEqual(fo_persisted.total_received_accepted_qty, 0.0)
        self.assertEqual(len(fo_persisted.grn_ids), 0)

    # -------------------------------------------------------------------------
    # 12. Complete Physical Dock Receipt Workflow Following Tracking
    # -------------------------------------------------------------------------

    def test_15_dock_receiving_workflow_following_tracking_delivery(self):
        """After tracking reaches DELIVERED, dark-store receiver applies formal goods receipt."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T10:00:00Z",
        )
        self.service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.DELIVERED)

        # Now receiver conducts physical dock inspection and creates GRN
        from src.agents.fulfilment.schemas import DockInspectionLine, GoodsReceiptNote
        inspection = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            damaged_quantity=0.0,
            unit_cost=30.0,
            batch_references=("BATCH-BLR-MILK-20241116-01",),
            remaining_shelf_life_days=5,
        )
        grn = GoodsReceiptNote(
            grn_id="GRN-001",
            fo_id=self.fo.fo_id,
            po_id=self.po.po_id,
            shipment_id=self.shipment_id,
            store_id=self.fo.store_id,
            supplier_id=self.fo.supplier_id,
            receipt_at="2024-11-16T14:30:00Z",
            receiver_id="RECEIVER_MOHAN",
            inspection_lines=(inspection,),
        )
        fo_received = self.service.receive_goods(
            fo_id=self.fo.fo_id,
            grn=grn,
            actor_id="RECEIVER_MOHAN",
        )

        # NOW AND ONLY NOW has status transitioned to RECEIVED
        self.assertEqual(fo_received.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(fo_received.total_received_accepted_qty, 24.0)
        self.assertEqual(fo_received.lines[0].accepted_quantity, 24.0)
        self.assertEqual(fo_received.lines[0].delivered_quantity, 24.0)
        self.assertEqual(fo_received.total_confirmed_pipeline_qty, 0.0)  # Pipeline fulfilled
        self.assertIn("GRN-001", fo_received.grn_ids)


if __name__ == "__main__":
    unittest.main()

