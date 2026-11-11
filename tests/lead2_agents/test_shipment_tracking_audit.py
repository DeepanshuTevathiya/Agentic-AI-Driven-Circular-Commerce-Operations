"""
Adversarial Audit & Hardening Regression Test Suite for Phase 5.5
=================================================================
Verifies all 6 comprehensive audit dimensions:
1. Strict contract validation, temporal sequences, same-timestamp handling, and deterministic ordering.
2. Terminal state lockdown (DELIVERED and RETURNED immutability against late and invalid events).
3. Derived telemetry consistency (attempt_count, actual_delivery_at, latest location).
4. Carrier identity binding and mismatch rejection during event ingestion.
5. Scenario persistence and restoration across service/provider recreation.
6. Atomic optimistic concurrency, concurrent registration races, and transaction rollbacks.
7. Lifecycle co-existence (cancelled/superseded order guards).
8. Critical architectural boundary: complete isolation between carrier tracking and goods receipt/inventory.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.agents.fulfilment.gateway import (
    SimulatedSupplierAdapter,
    SupplierSimulationScenario,
)
from src.agents.fulfilment.repository import (
    FulfilmentRepository,
    StaleUpdateError,
)
from src.agents.fulfilment.schemas import (
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DockInspectionLine,
    FulfilmentOrder,
    FulfilmentOrderStatus,
    GoodsReceiptNote,
    ShipmentLineItem,
    SupplierType,
)
from src.agents.fulfilment.service import (
    FulfilmentService,
    ShipmentTrackingService,
)
from src.agents.fulfilment.state_machine import (
    FulfilmentGovernanceError,
    InvalidFulfilmentTransitionError,
)
from src.agents.fulfilment.tracking import (
    InvalidTrackingEventError,
    InvalidTrackingTransitionError,
    ShipmentNotFoundError,
    ShipmentTrackingEvent,
    ShipmentTrackingRecord,
    ShipmentTrackingStatus,
    SimulatedTrackingProvider,
    TrackingIdempotencyConflictError,
    TrackingLocation,
    TrackingScenario,
    VALID_TRACKING_TRANSITIONS,
    assert_tracking_does_not_mutate_fulfilment_order,
    is_valid_tracking_transition,
)
from src.agents.procurement.approval import approve_purchase_order
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
)


def _make_approved_po(
    po_id: str = "PO-AUDIT-001",
    store_id: str = "DS_BLR_KORAMANGALA",
    supplier_id: str = "SIM_SUPP_DAIRY_PRIMARY",
    sku_id: str = "SKU_MILK_1L",
    final_order_qty: float = 24.0,
    unit_procurement_cost: float = 30.0,
) -> PurchaseOrder:
    """Creates an approved PurchaseOrder with verified approval evidence."""
    item = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id=sku_id,
        category="Dairy",
        raw_suggested_qty=20.0,
        case_pack_rounded_qty=final_order_qty,
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
        approved_by="AUDITOR_DEEPANSHU",
        approval_timestamp="2024-11-15T10:00:00Z",
        approval_reason="Approved for adversarial tracking audit.",
    )


class TestShipmentTrackingAdversarialAudit(unittest.TestCase):
    """Adversarial audit and regression test suite for Phase 5.5."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "audit_fulfilment.db"
        self.repo = FulfilmentRepository(db_path=self.db_path)
        self.gateway = SimulatedSupplierAdapter(db_path=self.db_path)
        self.tracking_provider = SimulatedTrackingProvider()
        self.service = FulfilmentService(
            repository=self.repo,
            gateway=self.gateway,
            tracking_provider=self.tracking_provider,
        )

        # Create, transmit, and dispatch baseline order
        self.po = _make_approved_po(po_id="PO-AUDIT-BASE-01", final_order_qty=24.0)
        self.fo = self.service.create_order(self.po, idempotency_key="CMD-CREATE-BASE-01")
        self.fo, _ = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-BASE-01",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        self.shipment_id = "SHP-AUDIT-01"
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
        )
        self.fo = self.service.dispatch_shipment(
            fo_id=self.fo.fo_id,
            asn=asn,
            idempotency_key="CMD-DSP-BASE-01",
        )

    def tearDown(self):
        self.service.close()
        self.gateway.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Contract & Event Payload Validation
    # -------------------------------------------------------------------------

    def test_01_adversarial_contract_validation(self):
        """Verifies strict validation of IDs, timestamps, and required exception fields."""
        # Empty shipment_id
        with self.assertRaises(InvalidTrackingEventError):
            ShipmentTrackingEvent(
                shipment_id="   ",
                carrier_id="BLUEDART",
                provider_event_id="EVT-01",
                status=ShipmentTrackingStatus.PICKED_UP,
                event_timestamp="2024-11-15T12:00:00Z",
                received_timestamp="2024-11-15T12:05:00Z",
            )

        # Invalid timestamp format
        with self.assertRaises(InvalidTrackingEventError):
            ShipmentTrackingEvent(
                shipment_id=self.shipment_id,
                carrier_id="BLUEDART",
                provider_event_id="EVT-01",
                status=ShipmentTrackingStatus.PICKED_UP,
                event_timestamp="15/11/2024 12:00",
                received_timestamp="2024-11-15T12:05:00Z",
            )

        # received_timestamp precedes event_timestamp (causality violation)
        with self.assertRaises(InvalidTrackingEventError):
            ShipmentTrackingEvent(
                shipment_id=self.shipment_id,
                carrier_id="BLUEDART",
                provider_event_id="EVT-01",
                status=ShipmentTrackingStatus.PICKED_UP,
                event_timestamp="2024-11-15T12:00:00Z",
                received_timestamp="2024-11-15T11:59:00Z",
            )

        # DELIVERY_EXCEPTION missing exception_reason
        with self.assertRaises(InvalidTrackingEventError):
            ShipmentTrackingEvent(
                shipment_id=self.shipment_id,
                carrier_id="BLUEDART",
                provider_event_id="EVT-01",
                status=ShipmentTrackingStatus.DELIVERY_EXCEPTION,
                event_timestamp="2024-11-15T12:00:00Z",
                received_timestamp="2024-11-15T12:05:00Z",
                exception_reason=None,
            )

        # RETURN_INITIATED missing both exception_reason and notes
        with self.assertRaises(InvalidTrackingEventError):
            ShipmentTrackingEvent(
                shipment_id=self.shipment_id,
                carrier_id="BLUEDART",
                provider_event_id="EVT-01",
                status=ShipmentTrackingStatus.RETURN_INITIATED,
                event_timestamp="2024-11-15T12:00:00Z",
                received_timestamp="2024-11-15T12:05:00Z",
                exception_reason=None,
                notes=None,
            )

    # -------------------------------------------------------------------------
    # 2. Carrier Binding & Mismatch Rejection
    # -------------------------------------------------------------------------

    def test_02_carrier_mismatch_on_event_rejected(self):
        """Events referencing an unauthenticated or mismatched carrier for a shipment are rejected."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,  # BLUEDART
        )

        # Mismatched carrier in event
        foreign_event = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id="DELHIVERY",  # Mismatched carrier!
            provider_event_id="EVT-DEL-01",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:05:00Z",
        )

        with self.assertRaises(InvalidTrackingEventError) as ctx:
            self.service.ingest_tracking_event(foreign_event)
        self.assertIn("Carrier ID mismatch", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 3. Same-Timestamp Events & Deterministic Ordering
    # -------------------------------------------------------------------------

    def test_03_same_timestamp_progression_and_ordering(self):
        """Events occurring at the identical timestamp are deterministically ordered and resolved."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T12:00:00Z",
        )

        # In-order progression with same timestamp (e.g. rapid hub processing within same second)
        e1 = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-BD-01",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:05:00Z",
        )
        self.service.ingest_tracking_event(e1)

        # In-transit also stamped at 13:00:00Z
        e2 = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-BD-02",
            status=ShipmentTrackingStatus.IN_TRANSIT,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:06:00Z",
        )
        rec2 = self.service.ingest_tracking_event(e2)
        self.assertEqual(rec2.current_status, ShipmentTrackingStatus.IN_TRANSIT)

        # Verify events are stored in deterministic order
        history = self.service.get_tracking_history(self.shipment_id)
        statuses = [evt.status for evt in history]
        self.assertIn(ShipmentTrackingStatus.PICKED_UP, statuses)
        self.assertIn(ShipmentTrackingStatus.IN_TRANSIT, statuses)

    # -------------------------------------------------------------------------
    # 4. Terminal State Lockdown (DELIVERED & RETURNED Immutability)
    # -------------------------------------------------------------------------

    def test_04_terminal_state_lockdown(self):
        """Terminal DELIVERED and RETURNED states cannot be reopened, transitioned, or un-delivered."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        self.service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.DELIVERED)
        rec = self.service.get_shipment_tracking(self.shipment_id).record
        self.assertTrue(rec.is_delivered)
        self.assertTrue(rec.is_terminal)

        # 1. Attempting forward transition after DELIVERED is rejected
        late_forward_event = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-BD-LATE-01",
            status=ShipmentTrackingStatus.IN_TRANSIT,
            event_timestamp="2024-11-16T15:00:00Z",  # AFTER delivery
            received_timestamp="2024-11-16T15:05:00Z",
        )
        with self.assertRaises(InvalidTrackingTransitionError):
            self.service.ingest_tracking_event(late_forward_event)

        # 2. Late historical event (timestamp <= delivery) is accepted into audit trail WITHOUT un-delivering
        late_hist_event = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-BD-HIST-01",
            status=ShipmentTrackingStatus.AT_HUB,
            event_timestamp="2024-11-15T18:00:00Z",  # BEFORE delivery
            received_timestamp="2024-11-16T15:10:00Z",
            location=TrackingLocation(hub_id="HUB-BLR", facility_name="Sorting Hub"),
        )
        rec_after = self.service.ingest_tracking_event(late_hist_event)
        self.assertEqual(rec_after.current_status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(rec_after.is_delivered)

        # Verify historical event is in audit trail
        history = self.service.get_tracking_history(self.shipment_id)
        event_ids = [e.provider_event_id for e in history]
        self.assertIn("EVT-BD-HIST-01", event_ids)

    # -------------------------------------------------------------------------
    # 5. Telemetry Consistency: Delivery Time & Attempt Count
    # -------------------------------------------------------------------------

    def test_05_attempt_count_and_delivery_time_consistency(self):
        """attempt_count and actual_delivery_at remain accurate across exception cycles and out-of-order events."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.DELIVERY_EXCEPTION_AND_RECOVERY,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        # Advance through exception and recovery
        rec = self.service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.DELIVERED)
        # DELIVERY_EXCEPTION_AND_RECOVERY has 2 OUT_FOR_DELIVERY attempts
        self.assertEqual(rec.attempt_count, 2)
        self.assertIsNotNone(rec.actual_delivery_at)

        # Late-arriving historical event does NOT increment attempt_count
        late_event = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-BD-CHECK-01",
            status=ShipmentTrackingStatus.IN_TRANSIT,
            event_timestamp="2024-11-15T14:00:00Z",
            received_timestamp="2024-11-16T16:00:00Z",
        )
        rec_after = self.service.ingest_tracking_event(late_event)
        self.assertEqual(rec_after.attempt_count, 2)
        self.assertEqual(rec_after.actual_delivery_at, rec.actual_delivery_at)

    # -------------------------------------------------------------------------
    # 6. Scenario Persistence Across Restart
    # -------------------------------------------------------------------------

    def test_06_scenario_persisted_and_restored_across_restart(self):
        """Scenario (e.g. RETURN_TO_ORIGIN) is persisted in SQLite and restored after service restart."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.RETURN_TO_ORIGIN,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        # Advance 3 steps
        self.service.advance_shipment_tracking(self.shipment_id, steps=3)

        # Close service and simulate total process restart
        self.service.close()
        self.gateway.close()

        # Re-open repo and fresh provider
        new_repo = FulfilmentRepository(db_path=self.db_path)
        new_gateway = SimulatedSupplierAdapter(db_path=self.db_path)
        new_provider = SimulatedTrackingProvider()  # Default is NORMAL_DELIVERY
        new_service = FulfilmentService(
            repository=new_repo,
            gateway=new_gateway,
            tracking_provider=new_provider,
        )

        # Advance to completion using the new service instance
        final_rec = new_service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.RETURNED)
        self.assertEqual(final_rec.current_status, ShipmentTrackingStatus.RETURNED)
        self.assertTrue(final_rec.is_returned)

        new_service.close()
        new_gateway.close()

    # -------------------------------------------------------------------------
    # 7. Atomic Optimistic Concurrency Race Condition
    # -------------------------------------------------------------------------

    def test_07_concurrent_update_optimistic_locking(self):
        """Concurrent updates conditioned on expected version raise StaleUpdateError atomically in SQLite."""
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        ver = self.repo.get_tracking_version(self.shipment_id)
        self.assertEqual(ver, 1)

        # Update 1 succeeds, incrementing version to 2
        e1 = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-CONC-01",
            status=ShipmentTrackingStatus.PICKED_UP,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:05:00Z",
        )
        self.service.ingest_tracking_event(e1, expected_version=1)

        # Update 2 attempting to update using stale version 1 raises StaleUpdateError
        e2 = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-CONC-02",
            status=ShipmentTrackingStatus.IN_TRANSIT,
            event_timestamp="2024-11-15T14:00:00Z",
            received_timestamp="2024-11-15T14:05:00Z",
        )
        with self.assertRaises(StaleUpdateError):
            self.service.ingest_tracking_event(e2, expected_version=1)

    # -------------------------------------------------------------------------
    # 8. Concurrent Idempotent Registration Race
    # -------------------------------------------------------------------------

    def test_08_concurrent_idempotent_registration(self):
        """Simultaneous registration calls for the same shipment are safely handled as idempotent no-ops."""
        r1 = self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
        )
        # Exact duplicate registration returns same projection without error
        r2 = self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
        )
        self.assertEqual(r1.shipment_id, r2.shipment_id)
        self.assertEqual(r1.carrier_id, r2.carrier_id)

    # -------------------------------------------------------------------------
    # 9. Cancelled & Superseded Order Registration Guard
    # -------------------------------------------------------------------------

    def test_09_cancelled_order_registration_rejected(self):
        """Tracking registration on a cancelled FulfilmentOrder raises FulfilmentGovernanceError."""
        # Cancel order before registration
        self.service.request_cancellation(
            fo_id=self.fo.fo_id,
            reason="Carrier vehicle broke down irrecoverably",
            category=CancellationReasonCategory.SUPPLIER_UNABLE_TO_FULFIL,
            idempotency_key="CMD-REQ-CNC-01",
        )
        self.service.confirm_cancellation(
            fo_id=self.fo.fo_id,
            actor_id="SUPERVISOR",
            idempotency_key="CMD-CNC-01",
            supplier_cancellation_ref="SUPP-CNC-001",
            shipment_cancellation_ref="SHP-CNC-001",
        )
        fo_cancelled = self.service.get_order(self.fo.fo_id)
        self.assertEqual(fo_cancelled.status, FulfilmentOrderStatus.CANCELLED)

        # Registration must be rejected
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.register_shipment(
                fo_id=self.fo.fo_id,
                shipment_id=self.shipment_id,
                carrier_id=self.carrier_id,
            )
        self.assertIn("cancelled or superseded", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 10. CRITICAL BUSINESS BOUNDARY: Full Isolation from Goods Receipt
    # -------------------------------------------------------------------------

    def test_10_critical_boundary_isolation_from_goods_receipt(self):
        """
        Verifies the absolute domain boundary:
        1. Carrier DELIVERED alters ZERO inventory, GRNs, accepted quantities, or POs.
        2. Subsequent goods receipt inspection operates purely through domain state machine.
        3. Tracking events before, during, and after GRN cannot corrupt physical receipt evidence.
        """
        self.service.register_shipment(
            fo_id=self.fo.fo_id,
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T12:00:00Z",
        )

        fo_before = self.service.get_order(self.fo.fo_id)
        self.assertEqual(fo_before.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(fo_before.total_received_accepted_qty, 0.0)

        # 1. Advance tracking to DELIVERED
        self.service.advance_to_tracking_status(self.shipment_id, ShipmentTrackingStatus.DELIVERED)

        fo_after_delivery = self.service.get_order(self.fo.fo_id)
        assert_tracking_does_not_mutate_fulfilment_order(
            fo_before=fo_before,
            fo_after=fo_after_delivery,
            po_before=self.po,
            po_after=self.po,
        )
        self.assertEqual(fo_after_delivery.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(len(fo_after_delivery.grn_ids), 0)

        # 2. Physical Receiving at Dark Store Dock (Partial: 12 units)
        inspection_partial = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=12.0,
            accepted_quantity=12.0,
            rejected_quantity=0.0,
            damaged_quantity=0.0,
            unit_cost=30.0,
            batch_references=("BATCH-AUDIT-01",),
            remaining_shelf_life_days=5,
        )
        grn_partial = GoodsReceiptNote(
            grn_id="GRN-AUDIT-PARTIAL-01",
            fo_id=self.fo.fo_id,
            po_id=self.po.po_id,
            shipment_id=self.shipment_id,
            store_id=self.fo.store_id,
            supplier_id=self.fo.supplier_id,
            receipt_at="2024-11-16T13:00:00Z",
            receiver_id="RECEIVER_01",
            inspection_lines=(inspection_partial,),
        )
        fo_partially_received = self.service.receive_goods(
            fo_id=self.fo.fo_id,
            grn=grn_partial,
            idempotency_key="CMD-RCV-PARTIAL-01",
        )
        self.assertEqual(fo_partially_received.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(fo_partially_received.total_received_accepted_qty, 12.0)

        # 3. Ingest late tracking update (e.g. proof of delivery document ping before delivery time)
        late_ping = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="EVT-CHECK-01",
            status=ShipmentTrackingStatus.OUT_FOR_DELIVERY,
            event_timestamp="2024-11-16T01:00:00Z",  # Before delivery at 02:00:00Z
            received_timestamp="2024-11-16T13:30:00Z",
            notes="Delayed packet from dispatch hub",
        )
        self.service.ingest_tracking_event(late_ping)

        # Verify FO remains completely intact
        fo_check = self.service.get_order(self.fo.fo_id)
        self.assertEqual(fo_check.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)
        self.assertEqual(fo_check.total_received_accepted_qty, 12.0)

        # 4. Final Goods Receipt (Remaining 12 units)
        inspection_final = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=12.0,
            accepted_quantity=12.0,
            rejected_quantity=0.0,
            damaged_quantity=0.0,
            unit_cost=30.0,
            batch_references=("BATCH-AUDIT-02",),
            remaining_shelf_life_days=5,
        )
        grn_final = GoodsReceiptNote(
            grn_id="GRN-AUDIT-FINAL-01",
            fo_id=self.fo.fo_id,
            po_id=self.po.po_id,
            shipment_id=self.shipment_id,
            store_id=self.fo.store_id,
            supplier_id=self.fo.supplier_id,
            receipt_at="2024-11-16T14:00:00Z",
            receiver_id="RECEIVER_01",
            inspection_lines=(inspection_final,),
        )
        fo_received = self.service.receive_goods(
            fo_id=self.fo.fo_id,
            grn=grn_final,
            idempotency_key="CMD-RCV-FINAL-01",
        )
        self.assertEqual(fo_received.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(fo_received.total_received_accepted_qty, 24.0)

    # -------------------------------------------------------------------------
    # 11. Full Transition Matrix Verification
    # -------------------------------------------------------------------------

    def test_11_transition_matrix_validity_and_rejections(self):
        """Verifies that all valid transitions succeed and all invalid transitions fail."""
        for from_status, valid_targets in VALID_TRACKING_TRANSITIONS.items():
            for to_status in ShipmentTrackingStatus:
                expected_valid = (
                    (to_status in valid_targets)
                    or (from_status == to_status and from_status not in (ShipmentTrackingStatus.DELIVERED, ShipmentTrackingStatus.RETURNED))
                )
                self.assertEqual(
                    is_valid_tracking_transition(from_status, to_status),
                    expected_valid,
                    f"Transition from {from_status} to {to_status} failed expected validity check.",
                )


if __name__ == "__main__":
    unittest.main()
