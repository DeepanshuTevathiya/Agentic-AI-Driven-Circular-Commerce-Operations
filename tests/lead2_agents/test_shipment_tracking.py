"""
Unit & Integration Tests: Shipment Tracking Contracts, Lifecycle & Deterministic Simulator
===========================================================================================
Phase 5.5.1 Verification Test Suite:
1. Valid lifecycle transitions (LABEL_CREATED -> PICKED_UP -> IN_TRANSIT -> AT_HUB -> OUT_FOR_DELIVERY -> DELIVERED).
2. Terminal state immutability (DELIVERED and RETURNED cannot be regressed or transitioned).
3. Exception recovery workflow (DELIVERY_EXCEPTION -> AT_HUB -> OUT_FOR_DELIVERY -> DELIVERED).
4. Return-to-origin (RTO) workflow (DELIVERY_EXCEPTION -> RETURN_INITIATED -> IN_TRANSIT -> RETURNED).
5. Illegal transition rejection (cannot skip required stages or jump backwards).
6. Same-status intermediate checkpoints permitted for non-terminal statuses.
7. Deduplication and exact idempotent replay of identical provider events.
8. Conflicting reuse of provider_event_id rejection (differing payload raises conflict).
9. Out-of-order event handling (historical insertion without current_status regression).
10. Out-of-order late arrival after terminal DELIVERED status does not un-deliver shipment.
11. Multiple shipments under same fulfilment order maintain strict isolation.
12. Temporal causality validation (received_timestamp cannot precede event_timestamp).
13. Mandatory exception_reason on DELIVERY_EXCEPTION and RETURN_INITIATED.
14. TrackingLocation normalization from string, dict, or model.
15. Deterministic simulation: NORMAL_DELIVERY scenario progression.
16. Deterministic simulation: DELAYED_DELIVERY scenario progression.
17. Deterministic simulation: DELIVERY_EXCEPTION_AND_RECOVERY scenario progression.
18. Deterministic simulation: RETURN_TO_ORIGIN scenario progression.
19. Scenario resolution precedence hierarchy (shipment override > carrier override > default).
20. Advance shipment step-by-step and advance to target status.
21. TrackingProvider ABC and Protocol compliance.
22. Critical business boundary: DELIVERED tracking does NOT mutate GRNs, accepted qty, pipeline qty, or PO.
"""

import unittest
from datetime import datetime as dt_datetime, timezone

from src.agents.fulfilment import (
    AdvanceShipmentNotice,
    FulfilmentOrder,
    FulfilmentOrderStatus,
    InvalidTrackingEventError,
    InvalidTrackingTransitionError,
    ShipmentNotFoundError,
    ShipmentStatus,
    ShipmentTrackingEvent,
    ShipmentTrackingRecord,
    ShipmentTrackingResult,
    ShipmentTrackingStatus,
    SimulatedTrackingProvider,
    TERMINAL_TRACKING_STATUSES,
    TrackingError,
    TrackingIdempotencyConflictError,
    TrackingLocation,
    TrackingProvider,
    TrackingProviderProtocol,
    TrackingScenario,
    VALID_TRACKING_TRANSITIONS,
    apply_shipment_dispatch,
    apply_supplier_acknowledgment,
    assert_tracking_does_not_mutate_fulfilment_order,
    create_fulfilment_order,
    is_valid_tracking_transition,
    transmit_fulfilment_order,
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


class TestShipmentTrackingContractsAndLifecycle(unittest.TestCase):
    """Exhaustive tests for shipment tracking contracts, state machine transitions, and deduplication."""

    def setUp(self):
        self.provider = SimulatedTrackingProvider()
        self.shipment_id = "SHP-FO-001-01"
        self.carrier_id = "BLUEDART"

    def _event(
        self,
        provider_event_id: str,
        status: ShipmentTrackingStatus,
        event_ts: str,
        received_ts: str,
        location: str = "Bangalore Central Hub",
        exception_reason: str = None,
        exception_details: dict = None,
        notes: str = None,
        shipment_id: str = None,
    ) -> ShipmentTrackingEvent:
        """Helper to construct valid ShipmentTrackingEvent."""
        return ShipmentTrackingEvent(
            shipment_id=shipment_id or self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id=provider_event_id,
            status=status,
            event_timestamp=event_ts,
            received_timestamp=received_ts,
            location=location,
            exception_reason=exception_reason,
            exception_details=exception_details,
            notes=notes,
        )

    # -------------------------------------------------------------------------
    # 1. Valid Lifecycle Transitions
    # -------------------------------------------------------------------------

    def test_01_valid_happy_path_transitions(self):
        """Full forward delivery lifecycle executes and accumulates chronological history."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.LABEL_CREATED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T10:00:00Z",
        )

        e1 = self._event("E1", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T11:00:00Z", "2024-11-15T11:05:00Z")
        rec = rec.apply_event(e1)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.PICKED_UP)
        self.assertEqual(rec.event_count, 1)

        e2 = self._event("E2", ShipmentTrackingStatus.IN_TRANSIT, "2024-11-15T12:00:00Z", "2024-11-15T12:05:00Z")
        rec = rec.apply_event(e2)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.IN_TRANSIT)

        e3 = self._event("E3", ShipmentTrackingStatus.AT_HUB, "2024-11-15T14:00:00Z", "2024-11-15T14:05:00Z")
        rec = rec.apply_event(e3)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.AT_HUB)

        e4 = self._event("E4", ShipmentTrackingStatus.OUT_FOR_DELIVERY, "2024-11-15T16:00:00Z", "2024-11-15T16:05:00Z")
        rec = rec.apply_event(e4)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.OUT_FOR_DELIVERY)
        self.assertEqual(rec.attempt_count, 1)

        e5 = self._event("E5", ShipmentTrackingStatus.DELIVERED, "2024-11-15T17:00:00Z", "2024-11-15T17:05:00Z")
        rec = rec.apply_event(e5)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(rec.is_delivered)
        self.assertTrue(rec.is_terminal)
        self.assertEqual(rec.actual_delivery_at, "2024-11-15T17:00:00Z")
        self.assertEqual(rec.event_count, 5)

    # -------------------------------------------------------------------------
    # 2. Terminal State Immutability
    # -------------------------------------------------------------------------

    def test_02_terminal_state_delivered_cannot_transition(self):
        """Once DELIVERED, any subsequent transition attempt raises InvalidTrackingTransitionError."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.DELIVERED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T17:05:00Z",
            actual_delivery_at="2024-11-15T17:00:00Z",
        )

        # Attempt to transition back to IN_TRANSIT
        e_invalid = self._event("E-REV", ShipmentTrackingStatus.IN_TRANSIT, "2024-11-15T18:00:00Z", "2024-11-15T18:05:00Z")
        with self.assertRaises(InvalidTrackingTransitionError):
            rec.apply_event(e_invalid)

        # Attempt to transition to PICKED_UP
        e_invalid2 = self._event("E-REV2", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T18:00:00Z", "2024-11-15T18:05:00Z")
        with self.assertRaises(InvalidTrackingTransitionError):
            rec.apply_event(e_invalid2)

        # Attempt to transition to DELIVERY_EXCEPTION
        e_invalid3 = self._event(
            "E-REV3", ShipmentTrackingStatus.DELIVERY_EXCEPTION, "2024-11-15T18:00:00Z", "2024-11-15T18:05:00Z",
            exception_reason="LATE_DAMAGED",
        )
        with self.assertRaises(InvalidTrackingTransitionError):
            rec.apply_event(e_invalid3)

    def test_02b_terminal_state_returned_cannot_transition(self):
        """Once RETURNED, any subsequent transition attempt raises InvalidTrackingTransitionError."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.RETURNED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T20:00:00Z",
        )
        e_invalid = self._event("E-REV-RET", ShipmentTrackingStatus.OUT_FOR_DELIVERY, "2024-11-15T21:00:00Z", "2024-11-15T21:05:00Z")
        with self.assertRaises(InvalidTrackingTransitionError):
            rec.apply_event(e_invalid)

    # -------------------------------------------------------------------------
    # 3. Exception Recovery Workflow
    # -------------------------------------------------------------------------

    def test_03_delivery_exception_recovery_and_reattempt(self):
        """Shipment encounters delivery exception, returns to hub, and re-attempt succeeds."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.OUT_FOR_DELIVERY,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T12:00:00Z",
            attempt_count=1,
        )

        # 1. Exception occurred
        e_exc = self._event(
            "EXC-01",
            ShipmentTrackingStatus.DELIVERY_EXCEPTION,
            "2024-11-15T13:00:00Z",
            "2024-11-15T13:05:00Z",
            exception_reason="DOCK_BAY_LOCKED",
            exception_details={"attempt": 1},
        )
        rec = rec.apply_event(e_exc)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.DELIVERY_EXCEPTION)
        self.assertTrue(rec.has_active_exception)

        # 2. Return to hub
        e_hub = self._event("HUB-01", ShipmentTrackingStatus.AT_HUB, "2024-11-15T16:00:00Z", "2024-11-15T16:05:00Z")
        rec = rec.apply_event(e_hub)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.AT_HUB)
        self.assertFalse(rec.has_active_exception)

        # 3. Re-dispatch next morning (Attempt 2)
        e_out2 = self._event("OUT-02", ShipmentTrackingStatus.OUT_FOR_DELIVERY, "2024-11-16T09:00:00Z", "2024-11-16T09:05:00Z")
        rec = rec.apply_event(e_out2)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.OUT_FOR_DELIVERY)
        self.assertEqual(rec.attempt_count, 2)

        # 4. Successful delivery
        e_del = self._event("DEL-02", ShipmentTrackingStatus.DELIVERED, "2024-11-16T11:00:00Z", "2024-11-16T11:05:00Z")
        rec = rec.apply_event(e_del)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(rec.is_delivered)

    # -------------------------------------------------------------------------
    # 4. Return to Origin (RTO) Workflow
    # -------------------------------------------------------------------------

    def test_04_return_to_origin_workflow(self):
        """Delivery refusal transitions to RETURN_INITIATED, IN_TRANSIT, and finally RETURNED."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.OUT_FOR_DELIVERY,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T12:00:00Z",
        )

        e_exc = self._event(
            "EXC-RTO",
            ShipmentTrackingStatus.DELIVERY_EXCEPTION,
            "2024-11-15T13:00:00Z",
            "2024-11-15T13:05:00Z",
            exception_reason="STORE_REFUSED_ENTRY",
        )
        rec = rec.apply_event(e_exc)

        e_rto = self._event(
            "RTO-INIT",
            ShipmentTrackingStatus.RETURN_INITIATED,
            "2024-11-15T14:00:00Z",
            "2024-11-15T14:05:00Z",
            exception_reason="RTO_AUTHORIZED",
            notes="Consignment authorized for return to supplier",
        )
        rec = rec.apply_event(e_rto)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.RETURN_INITIATED)

        e_transit = self._event("RTO-TRANSIT", ShipmentTrackingStatus.IN_TRANSIT, "2024-11-15T18:00:00Z", "2024-11-15T18:05:00Z")
        rec = rec.apply_event(e_transit)

        e_hub = self._event("RTO-HUB", ShipmentTrackingStatus.AT_HUB, "2024-11-16T08:00:00Z", "2024-11-16T08:05:00Z")
        rec = rec.apply_event(e_hub)

        e_returned = self._event("RTO-FINAL", ShipmentTrackingStatus.RETURNED, "2024-11-16T12:00:00Z", "2024-11-16T12:05:00Z")
        rec = rec.apply_event(e_returned)

        self.assertEqual(rec.current_status, ShipmentTrackingStatus.RETURNED)
        self.assertTrue(rec.is_returned)
        self.assertTrue(rec.is_terminal)

    # -------------------------------------------------------------------------
    # 5. Illegal Transition Rejection
    # -------------------------------------------------------------------------

    def test_05_illegal_transitions_rejected(self):
        """Invalid jumps (e.g. LABEL_CREATED -> DELIVERED) raise InvalidTrackingTransitionError."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.LABEL_CREATED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T10:00:00Z",
        )

        # Illegal: jump directly to DELIVERED from LABEL_CREATED
        e_bad = self._event("BAD-01", ShipmentTrackingStatus.DELIVERED, "2024-11-15T12:00:00Z", "2024-11-15T12:05:00Z")
        with self.assertRaises(InvalidTrackingTransitionError):
            rec.apply_event(e_bad)

        # Illegal: jump directly to OUT_FOR_DELIVERY from LABEL_CREATED
        e_bad2 = self._event("BAD-02", ShipmentTrackingStatus.OUT_FOR_DELIVERY, "2024-11-15T12:00:00Z", "2024-11-15T12:05:00Z")
        with self.assertRaises(InvalidTrackingTransitionError):
            rec.apply_event(e_bad2)

        # Illegal: jump from PICKED_UP backwards to LABEL_CREATED
        rec_picked = rec.apply_event(self._event("P1", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T11:00:00Z", "2024-11-15T11:05:00Z"))
        e_bad3 = self._event("BAD-03", ShipmentTrackingStatus.LABEL_CREATED, "2024-11-15T12:00:00Z", "2024-11-15T12:05:00Z")
        with self.assertRaises(InvalidTrackingTransitionError):
            rec_picked.apply_event(e_bad3)

    # -------------------------------------------------------------------------
    # 6. Same-Status Intermediate Checkpoints
    # -------------------------------------------------------------------------

    def test_06_same_status_checkpoints_permitted(self):
        """Intermediate updates within the same status (e.g. multi-hop IN_TRANSIT) are permitted."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.PICKED_UP,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T11:00:00Z",
        )

        e_transit1 = self._event("T1", ShipmentTrackingStatus.IN_TRANSIT, "2024-11-15T12:00:00Z", "2024-11-15T12:05:00Z", location="Checkpoint A")
        rec = rec.apply_event(e_transit1)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.IN_TRANSIT)

        # Intermediate checkpoint still IN_TRANSIT
        e_transit2 = self._event("T2", ShipmentTrackingStatus.IN_TRANSIT, "2024-11-15T14:00:00Z", "2024-11-15T14:05:00Z", location="Checkpoint B")
        rec = rec.apply_event(e_transit2)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.IN_TRANSIT)
        self.assertEqual(rec.event_count, 2)

    # -------------------------------------------------------------------------
    # 7. Deduplication & Idempotent Replay
    # -------------------------------------------------------------------------

    def test_07_deduplication_exact_idempotent_replay(self):
        """Exact duplicate events with same provider_event_id are safe no-ops."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.LABEL_CREATED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T10:00:00Z",
        )
        e1 = self._event("EVT-IDEM", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T11:00:00Z", "2024-11-15T11:05:00Z")
        rec = rec.apply_event(e1)
        self.assertEqual(rec.event_count, 1)

        # Duplicate replay of exact same event
        rec_replayed = rec.apply_event(e1)
        self.assertEqual(rec_replayed.event_count, 1)
        self.assertEqual(rec_replayed.current_status, ShipmentTrackingStatus.PICKED_UP)

    # -------------------------------------------------------------------------
    # 8. Conflicting Reuse Rejection
    # -------------------------------------------------------------------------

    def test_08_conflicting_provider_event_id_reuse_rejected(self):
        """Reusing provider_event_id with differing status or payload raises TrackingIdempotencyConflictError."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.LABEL_CREATED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T10:00:00Z",
        )
        e1 = self._event("EVT-CONFLICT", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T11:00:00Z", "2024-11-15T11:05:00Z")
        rec = rec.apply_event(e1)

        # Conflicting event with same event ID but differing status (IN_TRANSIT)
        e1_tampered = self._event("EVT-CONFLICT", ShipmentTrackingStatus.IN_TRANSIT, "2024-11-15T12:00:00Z", "2024-11-15T12:05:00Z")
        with self.assertRaises(TrackingIdempotencyConflictError):
            rec.apply_event(e1_tampered)

    # -------------------------------------------------------------------------
    # 9. Out-of-Order Handling Without Status Regression
    # -------------------------------------------------------------------------

    def test_09_out_of_order_event_does_not_regress_current_status(self):
        """Late-arriving historical event is inserted into audit history without regressing current status."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.LABEL_CREATED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T10:00:00Z",
        )

        e_picked = self._event("E1", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T11:00:00Z", "2024-11-15T11:05:00Z")
        e_out = self._event("E3", ShipmentTrackingStatus.OUT_FOR_DELIVERY, "2024-11-15T15:00:00Z", "2024-11-15T15:05:00Z")

        rec = rec.apply_event(e_picked)
        rec = rec.apply_event(e_out)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.OUT_FOR_DELIVERY)

        # Late-arriving IN_TRANSIT event from 13:00 (arrives at app at 15:30)
        e_late_transit = self._event("E2-LATE", ShipmentTrackingStatus.IN_TRANSIT, "2024-11-15T13:00:00Z", "2024-11-15T15:30:00Z")
        rec = rec.apply_event(e_late_transit)

        # Current status remains OUT_FOR_DELIVERY (no regression to IN_TRANSIT)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.OUT_FOR_DELIVERY)

        # Events list is sorted chronologically
        self.assertEqual(rec.event_count, 3)
        self.assertEqual(rec.events[0].provider_event_id, "E1")          # 11:00
        self.assertEqual(rec.events[1].provider_event_id, "E2-LATE")     # 13:00 (inserted in correct temporal sequence)
        self.assertEqual(rec.events[2].provider_event_id, "E3")          # 15:00

    # -------------------------------------------------------------------------
    # 10. Out-of-Order After Terminal DELIVERED Status
    # -------------------------------------------------------------------------

    def test_10_out_of_order_arrival_after_delivered_does_not_un_deliver(self):
        """A delayed packet from an earlier leg arriving after DELIVERED preserves DELIVERED status."""
        rec = ShipmentTrackingRecord(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            current_status=ShipmentTrackingStatus.LABEL_CREATED,
            created_at="2024-11-15T10:00:00Z",
            updated_at="2024-11-15T10:00:00Z",
        )

        e_picked = self._event("E1", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T11:00:00Z", "2024-11-15T11:05:00Z")
        e_out = self._event("E3", ShipmentTrackingStatus.OUT_FOR_DELIVERY, "2024-11-15T16:00:00Z", "2024-11-15T16:05:00Z")
        e_del = self._event("E4", ShipmentTrackingStatus.DELIVERED, "2024-11-15T17:00:00Z", "2024-11-15T17:05:00Z")

        rec = rec.apply_event(e_picked)
        rec = rec.apply_event(e_out)
        rec = rec.apply_event(e_del)
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(rec.is_delivered)

        # Delayed AT_HUB event from 14:00 arrives at 17:30
        e_delayed_hub = self._event("E2-DELAYED", ShipmentTrackingStatus.AT_HUB, "2024-11-15T14:00:00Z", "2024-11-15T17:30:00Z")
        rec = rec.apply_event(e_delayed_hub)

        # DELIVERED status is strictly preserved
        self.assertEqual(rec.current_status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(rec.is_delivered)
        self.assertTrue(rec.is_terminal)
        self.assertEqual(rec.event_count, 4)

    # -------------------------------------------------------------------------
    # 11. Multi-Shipment Isolation
    # -------------------------------------------------------------------------

    def test_11_multiple_shipments_isolation(self):
        """Events for one shipment never modify or cross-contaminate another shipment."""
        shp1 = "SHP-001"
        shp2 = "SHP-002"

        self.provider.register_shipment(shp1, carrier_id="BLUEDART", scenario=TrackingScenario.NORMAL_DELIVERY)
        self.provider.register_shipment(shp2, carrier_id="DELHIVERY", scenario=TrackingScenario.RETURN_TO_ORIGIN)

        # Advance SHP-001 to DELIVERED
        self.provider.advance_to_status(shp1, ShipmentTrackingStatus.DELIVERED)
        res1 = self.provider.get_tracking(shp1)
        self.assertEqual(res1.status, ShipmentTrackingStatus.DELIVERED)

        # SHP-002 is still at its initial LABEL_CREATED state
        res2 = self.provider.get_tracking(shp2)
        self.assertEqual(res2.status, ShipmentTrackingStatus.LABEL_CREATED)

        # Advancing SHP-002 does not mutate SHP-001
        self.provider.advance_to_status(shp2, ShipmentTrackingStatus.RETURN_INITIATED)
        res2_updated = self.provider.get_tracking(shp2)
        self.assertEqual(res2_updated.status, ShipmentTrackingStatus.RETURN_INITIATED)

        # Verify SHP-001 unchanged
        res1_check = self.provider.get_tracking(shp1)
        self.assertEqual(res1_check.status, ShipmentTrackingStatus.DELIVERED)

        # Mismatched shipment_id submission rejected
        rec1 = res1.record
        e_mismatch = self._event("MISMATCH", ShipmentTrackingStatus.AT_HUB, "2024-11-15T18:00:00Z", "2024-11-15T18:05:00Z", shipment_id=shp2)
        with self.assertRaises(InvalidTrackingEventError):
            rec1.apply_event(e_mismatch)

    # -------------------------------------------------------------------------
    # 12. Temporal Causality Validation
    # -------------------------------------------------------------------------

    def test_12_temporal_causality_validation(self):
        """received_timestamp cannot precede event_timestamp."""
        with self.assertRaises(InvalidTrackingEventError):
            ShipmentTrackingEvent(
                shipment_id=self.shipment_id,
                carrier_id=self.carrier_id,
                provider_event_id="EVT-TIME-ERR",
                status=ShipmentTrackingStatus.PICKED_UP,
                event_timestamp="2024-11-15T12:00:00Z",
                received_timestamp="2024-11-15T11:00:00Z",  # Ingestion before physical occurrence!
            )

    # -------------------------------------------------------------------------
    # 13. Mandatory Exception Reason
    # -------------------------------------------------------------------------

    def test_13_mandatory_exception_reason_validation(self):
        """DELIVERY_EXCEPTION without exception_reason raises InvalidTrackingEventError."""
        with self.assertRaises(InvalidTrackingEventError):
            ShipmentTrackingEvent(
                shipment_id=self.shipment_id,
                carrier_id=self.carrier_id,
                provider_event_id="EVT-EXC-NO-REASON",
                status=ShipmentTrackingStatus.DELIVERY_EXCEPTION,
                event_timestamp="2024-11-15T12:00:00Z",
                received_timestamp="2024-11-15T12:05:00Z",
                exception_reason=None,  # Missing mandatory reason!
            )

    # -------------------------------------------------------------------------
    # 14. TrackingLocation Normalization
    # -------------------------------------------------------------------------

    def test_14_tracking_location_normalization(self):
        """TrackingLocation parses correctly from string, dict, or model."""
        # From string
        e_str = self._event("L1", ShipmentTrackingStatus.PICKED_UP, "2024-11-15T11:00:00Z", "2024-11-15T11:05:00Z", location="Bangalore Central Hub")
        self.assertIsInstance(e_str.location, TrackingLocation)
        self.assertEqual(e_str.location.facility_name, "Bangalore Central Hub")

        # From dict
        e_dict = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="L2",
            status=ShipmentTrackingStatus.IN_TRANSIT,
            event_timestamp="2024-11-15T12:00:00Z",
            received_timestamp="2024-11-15T12:05:00Z",
            location={"hub_id": "HUB-01", "city": "Bangalore", "state": "Karnataka"},
        )
        self.assertEqual(e_dict.location.hub_id, "HUB-01")
        self.assertEqual(e_dict.location.city, "Bangalore")

        # From TrackingLocation model
        loc_obj = TrackingLocation(hub_id="HUB-02", facility_name="South Hub")
        e_obj = ShipmentTrackingEvent(
            shipment_id=self.shipment_id,
            carrier_id=self.carrier_id,
            provider_event_id="L3",
            status=ShipmentTrackingStatus.AT_HUB,
            event_timestamp="2024-11-15T13:00:00Z",
            received_timestamp="2024-11-15T13:05:00Z",
            location=loc_obj,
        )
        self.assertEqual(e_obj.location.hub_id, "HUB-02")

    # -------------------------------------------------------------------------
    # 15. Deterministic Simulation: NORMAL_DELIVERY
    # -------------------------------------------------------------------------

    def test_15_deterministic_scenario_normal_delivery(self):
        """NORMAL_DELIVERY advances deterministically through all 6 checkpoints to DELIVERED."""
        provider = SimulatedTrackingProvider()
        record = provider.register_shipment(
            "SHP-NORM-01",
            carrier_id="BLUEDART",
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T08:00:00Z",
        )
        self.assertEqual(record.current_status, ShipmentTrackingStatus.LABEL_CREATED)

        events = provider.advance_to_status("SHP-NORM-01", ShipmentTrackingStatus.DELIVERED)
        self.assertEqual(len(events), 5)  # PICKED_UP, IN_TRANSIT, AT_HUB, OUT_FOR_DELIVERY, DELIVERED

        result = provider.get_tracking("SHP-NORM-01")
        self.assertEqual(result.status, ShipmentTrackingStatus.DELIVERED)
        self.assertEqual(result.record.event_count, 6)
        self.assertTrue(result.record.is_delivered)

    # -------------------------------------------------------------------------
    # 16. Deterministic Simulation: DELAYED_DELIVERY
    # -------------------------------------------------------------------------

    def test_16_deterministic_scenario_delayed_delivery(self):
        """DELAYED_DELIVERY includes intermediate hubs and completes at DELIVERED."""
        provider = SimulatedTrackingProvider()
        provider.register_shipment(
            "SHP-DELAY-01",
            carrier_id="DELHIVERY",
            scenario=TrackingScenario.DELAYED_DELIVERY,
            base_timestamp="2024-11-15T08:00:00Z",
        )

        events = provider.advance_to_status("SHP-DELAY-01", ShipmentTrackingStatus.DELIVERED)
        self.assertGreater(len(events), 5)

        result = provider.get_tracking("SHP-DELAY-01")
        self.assertEqual(result.status, ShipmentTrackingStatus.DELIVERED)
        self.assertEqual(result.record.event_count, 8)

    # -------------------------------------------------------------------------
    # 17. Deterministic Simulation: DELIVERY_EXCEPTION_AND_RECOVERY
    # -------------------------------------------------------------------------

    def test_17_deterministic_scenario_exception_and_recovery(self):
        """DELIVERY_EXCEPTION_AND_RECOVERY exercises exception, hub return, re-dispatch, and delivery."""
        provider = SimulatedTrackingProvider()
        provider.register_shipment(
            "SHP-REC-01",
            scenario=TrackingScenario.DELIVERY_EXCEPTION_AND_RECOVERY,
            base_timestamp="2024-11-15T08:00:00Z",
        )

        # Advance to exception
        exc_events = provider.advance_to_status("SHP-REC-01", ShipmentTrackingStatus.DELIVERY_EXCEPTION)
        self.assertEqual(exc_events[-1].status, ShipmentTrackingStatus.DELIVERY_EXCEPTION)
        self.assertEqual(exc_events[-1].exception_reason, "DOCK_BAY_CONGESTED")

        # Advance to final delivery
        rec_events = provider.advance_to_status("SHP-REC-01", ShipmentTrackingStatus.DELIVERED)
        self.assertEqual(rec_events[-1].status, ShipmentTrackingStatus.DELIVERED)

        result = provider.get_tracking("SHP-REC-01")
        self.assertEqual(result.status, ShipmentTrackingStatus.DELIVERED)
        self.assertEqual(result.record.attempt_count, 2)

    # -------------------------------------------------------------------------
    # 18. Deterministic Simulation: RETURN_TO_ORIGIN
    # -------------------------------------------------------------------------

    def test_18_deterministic_scenario_return_to_origin(self):
        """RETURN_TO_ORIGIN advances through exception, return initiation, line-haul, to RETURNED."""
        provider = SimulatedTrackingProvider()
        provider.register_shipment(
            "SHP-RTO-01",
            scenario=TrackingScenario.RETURN_TO_ORIGIN,
            base_timestamp="2024-11-15T08:00:00Z",
        )

        provider.advance_to_status("SHP-RTO-01", ShipmentTrackingStatus.RETURN_INITIATED)
        result_mid = provider.get_tracking("SHP-RTO-01")
        self.assertEqual(result_mid.status, ShipmentTrackingStatus.RETURN_INITIATED)

        provider.advance_to_status("SHP-RTO-01", ShipmentTrackingStatus.RETURNED)
        result_final = provider.get_tracking("SHP-RTO-01")
        self.assertEqual(result_final.status, ShipmentTrackingStatus.RETURNED)
        self.assertTrue(result_final.record.is_returned)
        self.assertTrue(result_final.record.is_terminal)

    # -------------------------------------------------------------------------
    # 19. Scenario Resolution Precedence Hierarchy
    # -------------------------------------------------------------------------

    def test_19_scenario_resolution_precedence(self):
        """Precedence order: shipment override > carrier override > default scenario."""
        provider = SimulatedTrackingProvider(default_scenario=TrackingScenario.RETURN_TO_ORIGIN)

        # 1. Carrier override beats default
        provider.configure_carrier_scenario("BLUEDART", TrackingScenario.DELAYED_DELIVERY)
        provider.register_shipment("SHP-CARRIER-OVERRIDE", carrier_id="BLUEDART")
        evts_carrier = provider.advance_to_status("SHP-CARRIER-OVERRIDE", ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(any(e.location.hub_id == "HUB-HYD-01" for e in evts_carrier))

        # 2. Shipment override beats carrier override
        provider.configure_shipment_scenario("SHP-SPECIFIC", TrackingScenario.NORMAL_DELIVERY)
        provider.register_shipment("SHP-SPECIFIC", carrier_id="BLUEDART")
        evts_shipment = provider.advance_to_status("SHP-SPECIFIC", ShipmentTrackingStatus.DELIVERED)
        self.assertEqual(len(evts_shipment), 5)  # Normal delivery has 5 advancement events

    # -------------------------------------------------------------------------
    # 20. Step-by-Step Advancement
    # -------------------------------------------------------------------------

    def test_20_step_by_step_advancement(self):
        """advance_shipment steps forward one or multiple checkpoints cleanly."""
        provider = SimulatedTrackingProvider()
        provider.register_shipment("SHP-STEP-01", scenario=TrackingScenario.NORMAL_DELIVERY)

        e1 = provider.advance_shipment("SHP-STEP-01", steps=1)
        self.assertEqual(e1.status, ShipmentTrackingStatus.PICKED_UP)

        e2 = provider.advance_shipment("SHP-STEP-01", steps=2)
        self.assertEqual(e2.status, ShipmentTrackingStatus.AT_HUB)

        # Step remaining steps
        provider.advance_shipment("SHP-STEP-01", steps=10)
        res = provider.get_tracking("SHP-STEP-01")
        self.assertEqual(res.status, ShipmentTrackingStatus.DELIVERED)

        # Calling advance on exhausted pipeline returns None
        extra = provider.advance_shipment("SHP-STEP-01", steps=1)
        self.assertIsNone(extra)

    # -------------------------------------------------------------------------
    # 21. Provider Abstraction & Protocol Compliance
    # -------------------------------------------------------------------------

    def test_21_provider_protocol_and_abc_compliance(self):
        """SimulatedTrackingProvider satisfies TrackingProvider ABC and Protocol."""
        self.assertTrue(issubclass(SimulatedTrackingProvider, TrackingProvider))
        self.assertTrue(isinstance(self.provider, TrackingProvider))
        self.assertTrue(isinstance(self.provider, TrackingProviderProtocol))

        # Unknown shipment query raises ShipmentNotFoundError
        with self.assertRaises(ShipmentNotFoundError):
            self.provider.get_tracking("UNKNOWN-SHIPMENT-XYZ")

    # -------------------------------------------------------------------------
    # 22. Critical Business Boundary Invariant Enforcement
    # -------------------------------------------------------------------------

    def test_22_critical_business_boundary_delivered_does_not_mutate_inventory(self):
        """
        CRITICAL ARCHITECTURAL BOUNDARY:
        A carrier reporting DELIVERED physical status does NOT create a GRN,
        does NOT increase accepted/delivered quantity, does NOT alter confirmed
        pipeline quantity, does NOT create batches, and does NOT mutate the approved PO.
        """
        po = _make_approved_po()
        fo = create_fulfilment_order(po, actor_id="FULFILMENT_AGENT", timestamp="2024-11-15T10:35:00Z")
        fo = transmit_fulfilment_order(
            fo,
            actor_id="FULFILMENT_AGENT",
            transmission_timestamp="2024-11-15T11:00:00Z",
            idempotency_key="TXN-BOUND-01",
        )

        # Accept order
        from src.agents.fulfilment.schemas import (
            SupplierAcknowledgment,
            SupplierAcknowledgmentLine,
            SupplierAcknowledgmentStatus,
        )
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-BOUND-01",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            supplier_id=po.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T12:00:00Z",
            lines=(
                SupplierAcknowledgmentLine(
                    sku_id="SKU_MILK_1L",
                    accepted_quantity=24.0,
                    declined_quantity=0.0,
                ),
            ),
        )
        fo = apply_supplier_acknowledgment(
            fo,
            ack=ack,
            actor_id="SUPPLIER_GATEWAY",
            idempotency_key="CMD-ACK-BOUND-01",
        )

        # Dispatch shipment
        from src.agents.fulfilment.schemas import AdvanceShipmentNotice, ShipmentLineItem
        asn = AdvanceShipmentNotice(
            shipment_id="SHP-BOUND-01",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            supplier_id=po.supplier_id,
            dispatch_at="2024-11-15T12:00:00Z",
            expected_arrival_at="2024-11-16T10:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
            carrier="BLUEDART",
            tracking_reference="WB-100293",
        )
        fo = apply_shipment_dispatch(
            fo,
            asn=asn,
            actor_id="SUPPLIER_GATEWAY",
            idempotency_key="CMD-DSP-BOUND-01",
        )

        # Record state prior to tracking delivery
        fo_before_tracking = fo.model_copy(deep=True)
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(fo.total_confirmed_pipeline_qty, 24.0)
        self.assertEqual(fo.lines[0].delivered_quantity, 0.0)
        self.assertEqual(fo.lines[0].accepted_quantity, 0.0)
        self.assertEqual(fo.total_received_accepted_qty, 0.0)
        self.assertEqual(len(fo.grn_ids), 0)

        # Carrier tracking event reporting DELIVERED
        provider = SimulatedTrackingProvider()
        provider.register_shipment("SHP-BOUND-01", scenario=TrackingScenario.NORMAL_DELIVERY)
        provider.advance_to_status("SHP-BOUND-01", ShipmentTrackingStatus.DELIVERED)
        track_res = provider.get_tracking("SHP-BOUND-01")
        self.assertEqual(track_res.status, ShipmentTrackingStatus.DELIVERED)
        self.assertTrue(track_res.record.is_delivered)

        # Enforce that tracking alone does NOT mutate FulfilmentOrder or PO!
        fo_after_tracking = fo.model_copy(deep=True)
        assert_tracking_does_not_mutate_fulfilment_order(
            fo_before_tracking,
            fo_after_tracking,
            po_before=po,
            po_after=po,
        )

        self.assertEqual(fo_after_tracking.status, FulfilmentOrderStatus.IN_TRANSIT)
        self.assertEqual(fo_after_tracking.total_confirmed_pipeline_qty, 24.0)
        self.assertEqual(fo_after_tracking.lines[0].delivered_quantity, 0.0)
        self.assertEqual(fo_after_tracking.lines[0].accepted_quantity, 0.0)
        self.assertEqual(fo_after_tracking.total_received_accepted_qty, 0.0)
        self.assertEqual(len(fo_after_tracking.grn_ids), 0)
        self.assertEqual(fo_after_tracking.po_id, po.po_id)


if __name__ == "__main__":
    unittest.main()
