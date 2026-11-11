"""
Unit & Integration Tests: Supplier Gateway Integration with FulfilmentService & SQLite
======================================================================================
Phase 5.4.2 Verification Tests:
1. Confirmed acceptance transmission flow (READY_TO_TRANSMIT -> ACKNOWLEDGED).
2. Confirmed rejection transmission flow (READY_TO_TRANSMIT -> REJECTED_BY_SUPPLIER).
3. Definite pre-acceptance failure flow (order remains READY_TO_TRANSMIT; subsequent retry succeeds).
4. Ambiguous outcome flow (blocks fresh submissions; requires reconciliation).
5. Adapter recreation with durable submission recovery across simulator restarts.
6. Application crash recovery (supplier accepted, app crashed before saving response).
7. Idempotent duplicate retry (no duplicate events, no unnecessary version increments).
8. Conflicting idempotency key reuse rejection.
9. Confirmed order cannot be retransmitted with a new key.
10. Commercial mismatch validation (PO vs FO store, supplier, SKUs, quantities, unit costs).
11. Phase 4 human approval governance and PO immutability enforcement.
12. Optimistic concurrency control (stale update rejection).
13. Two-phase commit: transmission intent stored in SQLite prior to gateway invocation.
"""

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from src.agents.fulfilment import (
    CancellationReasonCategory,
    FulfilmentGovernanceError,
    FulfilmentOrder,
    FulfilmentOrderStatus,
    FulfilmentRepository,
    FulfilmentService,
    IdempotencyConflictError,
    InvalidFulfilmentTransitionError,
    InvalidPurchaseOrderError,
    SimulatedSupplierAdapter,
    StaleUpdateError,
    SupplierGateway,
    SupplierGatewayIdempotencyConflictError,
    SupplierSimulationScenario,
    SupplierTransmissionResult,
    SupplierTransmissionStatus,
)
from src.agents.procurement.approval import approve_purchase_order, reject_purchase_order
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
    urgency: PurchaseOrderUrgency = PurchaseOrderUrgency.STANDARD,
    supplier_type: SupplierType = SupplierType.PRIMARY,
    approval_timestamp: str = "2024-11-15T10:30:00Z",
    approved_by: str = "OPERATOR_DEEPANSHU",
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
        supplier_type=supplier_type,
        items=[item],
        total_order_qty=final_order_qty,
        total_order_cost=round(final_order_qty * unit_procurement_cost, 2),
        urgency=urgency,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        requires_escalation=False,
        escalation_reasons=[],
        audit_trail=(),
    )
    return approve_purchase_order(
        po=pending_po,
        approved_by=approved_by,
        approval_timestamp=approval_timestamp,
        approval_reason="Routine replenishment approved.",
    )


class TestGatewayServiceIntegration(unittest.TestCase):
    """Integration test suite for Supplier Gateway, FulfilmentService, and SQLite persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_gateway_service.db"
        self.repo = FulfilmentRepository(self.db_path)
        self.gateway = SimulatedSupplierAdapter(db_path=self.db_path)
        self.service = FulfilmentService(repository=self.repo, gateway=self.gateway)
        self.po = _make_approved_po()
        self.fo = self.service.create_order(self.po, actor_id="PROCUREMENT_AGENT", timestamp="2024-11-15T10:45:00Z")

    def tearDown(self):
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Confirmed Acceptance Flow
    # -------------------------------------------------------------------------

    def test_01_confirmed_acceptance_flow(self):
        """Gateway ACCEPT outcome transitions order to ACKNOWLEDGED and persists atomically."""
        self.assertEqual(self.fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)
        v_before = self.service.get_order_version(self.fo.fo_id)

        fo, result = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-01",
            timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertGreater(self.service.get_order_version(fo.fo_id), v_before)

        # Lines updated with confirmed supplier accepted quantity
        self.assertEqual(fo.lines[0].supplier_accepted_quantity, 24.0)
        self.assertEqual(fo.lines[0].supplier_declined_quantity, 0.0)

        # Event history contains TRANSMIT_TO_SUPPLIER and SUPPLIER_ACKNOWLEDGE
        event_types = [e.event_type.value for e in fo.event_history]
        self.assertIn("TRANSMIT_TO_SUPPLIER", event_types)
        self.assertIn("SUPPLIER_ACKNOWLEDGE", event_types)

        # Transmission record in SQLite
        tx = self.service.get_transmission_record("CMD-TXN-01")
        self.assertIsNotNone(tx)
        self.assertEqual(tx["transmission_status"], "CONFIRMED_ACCEPTANCE")
        self.assertEqual(tx["fo_id"], fo.fo_id)
        self.assertEqual(tx["po_id"], self.po.po_id)

    # -------------------------------------------------------------------------
    # 2. Confirmed Rejection Flow
    # -------------------------------------------------------------------------

    def test_02_confirmed_rejection_flow(self):
        """Gateway REJECT outcome transitions order to REJECTED_BY_SUPPLIER."""
        fo, result = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-REJ",
            timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.REJECT,
            rejection_reason="Supplier dairy plant shutdown",
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.CONFIRMED_REJECTION)
        self.assertEqual(fo.status, FulfilmentOrderStatus.REJECTED_BY_SUPPLIER)
        self.assertEqual(fo.lines[0].supplier_accepted_quantity, 0.0)
        self.assertEqual(fo.lines[0].supplier_declined_quantity, 24.0)

        event_types = [e.event_type.value for e in fo.event_history]
        self.assertIn("SUPPLIER_REJECT", event_types)

        tx = self.service.get_transmission_record("CMD-TXN-REJ")
        self.assertIsNotNone(tx)
        self.assertEqual(tx["transmission_status"], "CONFIRMED_REJECTION")

    # -------------------------------------------------------------------------
    # 3. Definite Pre-Acceptance Failure & Subsequent Retry
    # -------------------------------------------------------------------------

    def test_03_definite_failure_and_subsequent_retry(self):
        """Transmission failure does not falsely transition order; subsequent retry succeeds."""
        fo, result = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-FAIL-1",
            timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.TRANSMISSION_FAILURE,
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.TRANSMISSION_FAILURE)
        # Invariant: order must NOT be marked ISSUED or ACKNOWLEDGED!
        self.assertEqual(fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

        tx = self.service.get_transmission_record("CMD-TXN-FAIL-1")
        self.assertEqual(tx["transmission_status"], "TRANSMISSION_FAILURE")

        # Explicit subsequent retry with new key after network fixed
        fo_retried, result_retry = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-RETRY-SUCCESS",
            timestamp="2024-11-15T11:15:00Z",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        self.assertEqual(result_retry.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(fo_retried.status, FulfilmentOrderStatus.ACKNOWLEDGED)

    # -------------------------------------------------------------------------
    # 4. Ambiguous Outcome & Reconciliation Flow
    # -------------------------------------------------------------------------

    def test_04_ambiguous_outcome_and_reconciliation(self):
        """Ambiguous outcome prevents fresh submissions; requires reconciliation with original key."""
        fo, result = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-AMBIG",
            timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.AMBIGUOUS_OUTCOME,
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.AMBIGUOUS_OUTCOME)
        self.assertIsNone(result.acknowledgment)
        self.assertEqual(fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

        tx = self.service.get_transmission_record("CMD-TXN-AMBIG")
        self.assertEqual(tx["transmission_status"], "AMBIGUOUS_OUTCOME")

        # Invariant: attempting a fresh transmission with a NEW key while ambiguous attempt is open is blocked!
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.transmit_purchase_order(
                self.fo.fo_id,
                self.po,
                idempotency_key="CMD-TXN-FRESH-KEY",
                timestamp="2024-11-15T11:30:00Z",
            )
        self.assertIn("must be reconciled", str(ctx.exception))

        # Reconcile using original key
        fo_rec, res_rec = self.service.reconcile_transmission(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-AMBIG",
            reconciliation_timestamp="2024-11-15T11:30:00Z",
        )

        self.assertEqual(res_rec.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(fo_rec.status, FulfilmentOrderStatus.ACKNOWLEDGED)

        tx_after = self.service.get_transmission_record("CMD-TXN-AMBIG")
        self.assertEqual(tx_after["transmission_status"], "CONFIRMED_ACCEPTANCE")

    # -------------------------------------------------------------------------
    # 5. Adapter Recreation: Durable Simulation Recovery
    # -------------------------------------------------------------------------

    def test_05_adapter_recreation_durable_reconciliation(self):
        """Simulated supplier's submission record is durable across adapter recreation."""
        # 1. Transmit with adapter 1 under AMBIGUOUS_OUTCOME
        adapter1 = SimulatedSupplierAdapter(db_path=self.db_path)
        fo, res1 = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-DURABLE",
            override_scenario=SupplierSimulationScenario.AMBIGUOUS_OUTCOME,
            gateway=adapter1,
        )
        self.assertEqual(res1.status, SupplierTransmissionStatus.AMBIGUOUS_OUTCOME)

        # 2. Recreate adapter (simulating process restart)
        del adapter1
        adapter2 = SimulatedSupplierAdapter(db_path=self.db_path)
        self.assertTrue(adapter2.has_supplier_received(self.po.po_id))

        # 3. Reconcile with adapter 2
        fo_rec, res_rec = self.service.reconcile_transmission(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-DURABLE",
            gateway=adapter2,
        )
        self.assertEqual(res_rec.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(fo_rec.status, FulfilmentOrderStatus.ACKNOWLEDGED)

    # -------------------------------------------------------------------------
    # 6. Application Crash Recovery
    # -------------------------------------------------------------------------

    def test_06_application_crash_recovery_after_supplier_acceptance(self):
        """Application crash between gateway response and DB commit recovers cleanly on restart."""
        key = "CMD-TXN-CRASH"
        clean_ts = "2024-11-15T11:00:00Z"

        # Simulate crash: record intent in DB, supplier receives and accepts order, but app crashed before DB commit
        self.repo.record_transmission_intent(
            fo_id=self.fo.fo_id,
            po_id=self.po.po_id,
            idempotency_key=key,
            payload_fingerprint="fp-dummy",
            correlation_id=f"CORR-{self.po.po_id}",
            timestamp=clean_ts,
        )
        # Supplier physically recorded acceptance
        self.gateway._record_supplier_submission(self.po.po_id, clean_ts, "ACCEPTED", scenario="ACCEPT")

        # Simulate restart: new service instance on the same DB
        with FulfilmentService(db_path=self.db_path) as restarted_service:
            # Caller retries the transmission command after crash
            fo_recovered, result = restarted_service.reconcile_transmission(
                self.fo.fo_id,
                self.po,
                idempotency_key=key,
                reconciliation_timestamp="2024-11-15T11:05:00Z",
            )

            self.assertEqual(result.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
            self.assertEqual(fo_recovered.status, FulfilmentOrderStatus.ACKNOWLEDGED)

            tx = restarted_service.get_transmission_record(key)
            self.assertEqual(tx["transmission_status"], "CONFIRMED_ACCEPTANCE")

    # -------------------------------------------------------------------------
    # 7. Idempotency & Version Invariant
    # -------------------------------------------------------------------------

    def test_07_idempotent_duplicate_retry_no_version_increment(self):
        """Retrying an identical transmission command does not increment version or duplicate events."""
        key = "CMD-TXN-IDEMP"
        fo1, res1 = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key=key,
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        self.assertEqual(fo1.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        v1 = self.service.get_order_version(self.fo.fo_id)
        evt_count1 = len(fo1.event_history)
        sub_count1 = self.gateway.get_submission_count(self.po.po_id)

        # Retry identical transmission
        fo2, res2 = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key=key,
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        self.assertEqual(res1.transmission_id, res2.transmission_id)
        self.assertEqual(self.service.get_order_version(self.fo.fo_id), v1)
        self.assertEqual(len(fo2.event_history), evt_count1)
        self.assertEqual(self.gateway.get_submission_count(self.po.po_id), sub_count1)

    # -------------------------------------------------------------------------
    # 8. Conflicting Idempotency Key Reuse
    # -------------------------------------------------------------------------

    def test_08_conflicting_idempotency_key_reuse_rejected(self):
        """Reusing an idempotency key with differing PO or payload raises conflict."""
        key = "CMD-SHARED-KEY"
        self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key=key,
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        # Create second order
        po2 = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-02")
        fo2 = self.service.create_order(po2, actor_id="AGENT", timestamp="2024-11-15T10:50:00Z")

        with self.assertRaises(SupplierGatewayIdempotencyConflictError):
            self.service.transmit_purchase_order(
                fo2.fo_id,
                po2,
                idempotency_key=key,
            )

    # -------------------------------------------------------------------------
    # 9. Confirmed Order Retransmission Blocked
    # -------------------------------------------------------------------------

    def test_09_confirmed_order_cannot_be_retransmitted_with_new_key(self):
        """Once acknowledged, transmitting with a NEW key is blocked."""
        self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TXN-K1",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        # Attempt to transmit with a different key
        with self.assertRaises(InvalidFulfilmentTransitionError):
            self.service.transmit_purchase_order(
                self.fo.fo_id,
                self.po,
                idempotency_key="CMD-TXN-K2",
            )

    # -------------------------------------------------------------------------
    # 10. Commercial Detail Verification (PO vs FO)
    # -------------------------------------------------------------------------

    def test_10_commercial_detail_mismatches_rejected(self):
        """Mismatches in store, supplier, SKUs, or quantities are strictly rejected."""
        # 1. Store mismatch
        po_wrong_store = _make_approved_po(
            po_id=self.po.po_id,
            store_id="DS_BLR_HSR_LAYOUT",  # FO is KORAMANGALA
        )
        with self.assertRaises(FulfilmentGovernanceError):
            self.service.transmit_purchase_order(self.fo.fo_id, po_wrong_store, "KEY-M1")

        # 2. Quantity mismatch
        po_wrong_qty = _make_approved_po(
            po_id=self.po.po_id,
            final_order_qty=48.0,  # FO is 24.0
        )
        with self.assertRaises(FulfilmentGovernanceError):
            self.service.transmit_purchase_order(self.fo.fo_id, po_wrong_qty, "KEY-M2")

        # 3. Unit cost mismatch
        po_wrong_cost = _make_approved_po(
            po_id=self.po.po_id,
            unit_procurement_cost=35.0,  # FO is 30.0
        )
        with self.assertRaises(FulfilmentGovernanceError):
            self.service.transmit_purchase_order(self.fo.fo_id, po_wrong_cost, "KEY-M3")

    # -------------------------------------------------------------------------
    # 11. Phase 4 Approval Governance & Immutability
    # -------------------------------------------------------------------------

    def test_11_phase_4_governance_and_po_immutability(self):
        """Unapproved POs rejected; approved PO is never mutated."""
        # Unapproved PO
        item = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
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
        unapproved_po = PurchaseOrder(
            po_id=self.po.po_id,
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            supplier_type=SupplierType.PRIMARY,
            items=[item],
            total_order_qty=24.0,
            total_order_cost=720.0,
            urgency=PurchaseOrderUrgency.STANDARD,
            status=PurchaseOrderStatus.PENDING_APPROVAL,
            requires_escalation=False,
            escalation_reasons=[],
            audit_trail=(),
        )

        with self.assertRaises(InvalidPurchaseOrderError):
            self.service.transmit_purchase_order(self.fo.fo_id, unapproved_po, "KEY-UNAPP")

        # PO immutability check
        orig_dict = self.po.model_dump()
        self.service.transmit_purchase_order(self.fo.fo_id, self.po, "KEY-IMMUTABLE")
        self.assertEqual(self.po.model_dump(), orig_dict)

    # -------------------------------------------------------------------------
    # 12. Optimistic Concurrency Control
    # -------------------------------------------------------------------------

    def test_12_optimistic_concurrency_stale_update_rejected(self):
        """Stale expected_version is rejected before committing update."""
        with self.assertRaises(StaleUpdateError):
            self.service.transmit_purchase_order(
                self.fo.fo_id,
                self.po,
                idempotency_key="KEY-STALE",
                expected_version=999,
            )

    # -------------------------------------------------------------------------
    # 13. Two-Phase Commit Intent Verification
    # -------------------------------------------------------------------------

    def test_13_two_phase_intent_recorded_before_gateway_call(self):
        """Gateway verifies that transmission intent was stored in SQLite prior to execution."""
        intent_verified = {"status": False}

        class VerifyingGateway(SupplierGateway):
            def __init__(self, repo: FulfilmentRepository, inner: SupplierGateway):
                self.repo = repo
                self.inner = inner

            def transmit_purchase_order(self, po, idempotency_key, **kwargs):
                # Inspect repository to confirm intent was stored before invocation
                tx = self.repo.get_transmission(idempotency_key)
                if tx is not None and tx["transmission_status"] == "INITIATED":
                    intent_verified["status"] = True
                return self.inner.transmit_purchase_order(po, idempotency_key, **kwargs)

            def reconcile_transmission(self, idempotency_key, po, **kwargs):
                return self.inner.reconcile_transmission(idempotency_key, po, **kwargs)

        custom_gateway = VerifyingGateway(self.repo, self.gateway)
        fo, result = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-INTENT-CHECK",
            gateway=custom_gateway,
        )

        self.assertTrue(intent_verified["status"])
        self.assertEqual(result.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)

    # -------------------------------------------------------------------------
    # 14. Multi-Attempt Failure Then Success with Historical Replay
    # -------------------------------------------------------------------------

    def test_14_multi_attempt_failure_then_success_historical_replay(self):
        """First attempt fails pre-acceptance; second attempt succeeds; both attempts replayable."""
        # 1. First attempt fails
        fo1, res1 = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-MULTI-1",
            override_scenario=SupplierSimulationScenario.TRANSMISSION_FAILURE,
        )
        self.assertEqual(res1.status, SupplierTransmissionStatus.TRANSMISSION_FAILURE)
        self.assertEqual(fo1.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

        # 2. Second attempt with distinct key succeeds
        fo2, res2 = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-MULTI-2",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        self.assertEqual(res2.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(fo2.status, FulfilmentOrderStatus.ACKNOWLEDGED)

        # 3. Replay first key: returns original failure
        fo_rep1, res_rep1 = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-MULTI-1",
            override_scenario=SupplierSimulationScenario.TRANSMISSION_FAILURE,
        )
        self.assertEqual(res_rep1.status, SupplierTransmissionStatus.TRANSMISSION_FAILURE)

        # 4. Replay second key: returns original acceptance
        fo_rep2, res_rep2 = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-MULTI-2",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        self.assertEqual(res_rep2.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)

        # 5. Check both are stored in SQLite
        history = self.repo.get_transmissions_for_order(self.fo.fo_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["idempotency_key"], "CMD-MULTI-1")
        self.assertEqual(history[0]["transmission_status"], "TRANSMISSION_FAILURE")
        self.assertEqual(history[1]["idempotency_key"], "CMD-MULTI-2")
        self.assertEqual(history[1]["transmission_status"], "CONFIRMED_ACCEPTANCE")

    # -------------------------------------------------------------------------
    # 15. Conflicting Lifecycle Operations Blocked While Ambiguous
    # -------------------------------------------------------------------------

    def test_15_conflicting_lifecycle_operations_blocked_while_transmission_is_ambiguous(self):
        """Order in AMBIGUOUS_OUTCOME blocks cancellation, supersession, and closure until reconciled."""
        # Ambiguous transmission
        fo, res = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-AMB-LOCK",
            override_scenario=SupplierSimulationScenario.AMBIGUOUS_OUTCOME,
        )
        self.assertEqual(res.status, SupplierTransmissionStatus.AMBIGUOUS_OUTCOME)
        self.assertEqual(fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)

        # Attempt cancellation request -> blocked
        with self.assertRaises(FulfilmentGovernanceError) as ctx:
            self.service.request_cancellation(
                self.fo.fo_id,
                reason="Supplier unresponsive, aborting order",
                category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
            )
        self.assertIn("unresolved transmission", str(ctx.exception))

        # Attempt short closure -> blocked
        with self.assertRaises(FulfilmentGovernanceError) as ctx2:
            self.service.close_short_order(
                self.fo.fo_id,
                closure_reason="Closing out order prematurely",
            )
        self.assertIn("unresolved transmission", str(ctx2.exception))

        # Attempt supersession -> blocked
        rep_po = _make_approved_po(po_id="PO-REPL-FOR-AMB-01")
        with self.assertRaises(FulfilmentGovernanceError) as ctx3:
            self.service.supersede_order(
                original_fo_id=self.fo.fo_id,
                replacement_po=rep_po,
                replacement_fo_id="FO-REPL-FOR-AMB-01",
                actor_id="OPERATOR_DEEPANSHU",
                timestamp="2024-11-15T15:00:00Z",
                idempotency_key="CMD-SUP-AMB",
                reason="Superseding ambiguous order",
            )
        self.assertIn("unresolved transmission", str(ctx3.exception))

        # Reconcile transmission -> resolves to ACKNOWLEDGED
        rec_fo, rec_res = self.service.reconcile_transmission(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-AMB-LOCK",
        )
        self.assertEqual(rec_res.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(rec_fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)

        # Now cancellation can be requested (as permitted from ACKNOWLEDGED status)
        can_fo = self.service.request_cancellation(
            self.fo.fo_id,
            reason="Operational cancellation after reconciliation",
            category=CancellationReasonCategory.OPERATIONAL_MISTAKE,
        )
        self.assertEqual(can_fo.status, FulfilmentOrderStatus.CANCEL_REQUESTED)

    # -------------------------------------------------------------------------
    # 16. Gateway Invocation Crash (Boundary 3) Recovery
    # -------------------------------------------------------------------------

    def test_16_gateway_invocation_crash_boundary_3_recovery(self):
        """Boundary 3 crash: unexpected error during gateway call records AMBIGUOUS and recovers via reconciliation."""
        class CrashingGateway(SupplierGateway):
            def __init__(self, inner: SupplierGateway):
                self.inner = inner

            def transmit_purchase_order(self, po, idempotency_key, **kwargs):
                raise RuntimeError("Simulated low-level socket crash during transmission")

            def reconcile_transmission(self, idempotency_key, po, **kwargs):
                return self.inner.reconcile_transmission(idempotency_key, po, **kwargs)

        crashing_gw = CrashingGateway(self.gateway)

        # Transmit raises RuntimeError
        with self.assertRaises(RuntimeError):
            self.service.transmit_purchase_order(
                self.fo.fo_id,
                self.po,
                idempotency_key="CMD-CRASH-3",
                gateway=crashing_gw,
            )

        # Verify SQLite recorded AMBIGUOUS_OUTCOME with exception details
        tx = self.repo.get_transmission("CMD-CRASH-3")
        self.assertIsNotNone(tx)
        self.assertEqual(tx["transmission_status"], SupplierTransmissionStatus.AMBIGUOUS_OUTCOME.value)
        self.assertEqual(tx["error_code"], "GATEWAY_INVOCATION_EXCEPTION")

        # Verify fresh transmission is blocked while ambiguous
        with self.assertRaises(FulfilmentGovernanceError):
            self.service.transmit_purchase_order(
                self.fo.fo_id,
                self.po,
                idempotency_key="CMD-NEW-AFTER-CRASH",
            )

        # Reconcile using original key confirms vendor never received it (ORDER_NOT_FOUND_AT_SUPPLIER)
        rec_fo, rec_res = self.service.reconcile_transmission(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-CRASH-3",
        )
        self.assertEqual(rec_fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)
        self.assertEqual(rec_res.status, SupplierTransmissionStatus.TRANSMISSION_FAILURE)
        self.assertEqual(rec_res.error_code, "ORDER_NOT_FOUND_AT_SUPPLIER")

        # Now that ambiguous state is reconciled to TRANSMISSION_FAILURE, fresh submission succeeds
        fo_succ, res_succ = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-NEW-AFTER-CRASH",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        self.assertEqual(res_succ.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(fo_succ.status, FulfilmentOrderStatus.ACKNOWLEDGED)

    # -------------------------------------------------------------------------
    # 17. Boundary 5 DB Commit Failure Recovery
    # -------------------------------------------------------------------------

    def test_17_boundary_5_db_commit_failure_recovery(self):
        """Boundary 5 recovery: transmission succeeds at gateway but fails DB commit; retry recovers via reconciliation."""
        orig_record_outcome = self.repo.record_transmission_outcome
        failure_injected = {"injected": False}

        def flaky_record_outcome(key, status, result, updated_fo=None, expected_version=None):
            if not failure_injected["injected"]:
                failure_injected["injected"] = True
                raise sqlite3.OperationalError("Simulated disk I/O lock during outcome commit")
            return orig_record_outcome(key, status, result, updated_fo=updated_fo, expected_version=expected_version)

        self.repo.record_transmission_outcome = flaky_record_outcome

        # Transmit fails at Boundary 5
        with self.assertRaises(sqlite3.OperationalError):
            self.service.transmit_purchase_order(
                self.fo.fo_id,
                self.po,
                idempotency_key="CMD-FAIL-B5",
            )

        # Restore normal record_transmission_outcome
        self.repo.record_transmission_outcome = orig_record_outcome

        # Intent was recorded in DB
        tx = self.repo.get_transmission("CMD-FAIL-B5")
        self.assertIsNotNone(tx)
        self.assertEqual(tx["transmission_status"], "INITIATED")

        # Retry with the same key routes through reconcile_transmission
        rec_fo, rec_res = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-FAIL-B5",
        )
        self.assertEqual(rec_res.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertEqual(rec_fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)

    # -------------------------------------------------------------------------
    # 18. Concurrent Transmission Intent Idempotency
    # -------------------------------------------------------------------------

    def test_18_concurrent_transmission_intent_idempotency(self):
        """Concurrent or repeated calls to record_transmission_intent are idempotent for matching payload."""
        # First call succeeds
        self.repo.record_transmission_intent(
            fo_id=self.fo.fo_id,
            po_id=self.po.po_id,
            idempotency_key="CMD-CONC-KEY",
            payload_fingerprint="FP_EXACT_MATCH_123",
            correlation_id="CORR-123",
            timestamp="2024-11-15T11:00:00Z",
        )

        # Second call with same fingerprint succeeds cleanly (no UniqueConstraint crash)
        self.repo.record_transmission_intent(
            fo_id=self.fo.fo_id,
            po_id=self.po.po_id,
            idempotency_key="CMD-CONC-KEY",
            payload_fingerprint="FP_EXACT_MATCH_123",
            correlation_id="CORR-123",
            timestamp="2024-11-15T11:00:00Z",
        )

        # Third call with differing fingerprint raises IdempotencyConflictError
        with self.assertRaises(IdempotencyConflictError):
            self.repo.record_transmission_intent(
                fo_id=self.fo.fo_id,
                po_id=self.po.po_id,
                idempotency_key="CMD-CONC-KEY",
                payload_fingerprint="FP_DIFFERING_TAMPERED",
                correlation_id="CORR-123",
                timestamp="2024-11-15T11:00:00Z",
            )

    # -------------------------------------------------------------------------
    # 19. Scenario Precedence Hierarchy
    # -------------------------------------------------------------------------

    def test_19_scenario_precedence_hierarchy(self):
        """SimulatedSupplierAdapter adheres to scenario hierarchy: call override > PO > Supplier > Default."""
        # 1. Default scenario is REJECT
        adapter = SimulatedSupplierAdapter(default_scenario=SupplierSimulationScenario.REJECT)
        res_def = adapter.transmit_purchase_order(self.po, "KEY-PREC-DEF")
        self.assertEqual(res_def.status, SupplierTransmissionStatus.CONFIRMED_REJECTION)

        # 2. Supplier scenario overrides default
        adapter.set_supplier_scenario(self.po.supplier_id, SupplierSimulationScenario.ACCEPT)
        res_sup = adapter.transmit_purchase_order(self.po, "KEY-PREC-SUP")
        self.assertEqual(res_sup.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)

        # 3. PO scenario overrides supplier scenario
        adapter.set_po_scenario(self.po.po_id, SupplierSimulationScenario.TRANSMISSION_FAILURE)
        res_po = adapter.transmit_purchase_order(self.po, "KEY-PREC-PO")
        self.assertEqual(res_po.status, SupplierTransmissionStatus.TRANSMISSION_FAILURE)

        # 4. Explicit call override beats PO scenario
        res_call = adapter.transmit_purchase_order(
            self.po, "KEY-PREC-CALL", override_scenario=SupplierSimulationScenario.ACCEPT
        )
        self.assertEqual(res_call.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)

    # -------------------------------------------------------------------------
    # 20. Commercial Terms Tampering Detection
    # -------------------------------------------------------------------------

    def test_20_commercial_terms_tampering_fingerprint_mismatch(self):
        """Reusing key with modified urgency, supplier_type, or created_date triggers idempotency conflict."""
        # Transmit initial PO
        self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-TAMPER-KEY",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        # Tampered PO with urgency changed to EMERGENCY
        tampered_urgency_po = _make_approved_po(
            po_id=self.po.po_id,
            store_id=self.po.store_id,
            supplier_id=self.po.supplier_id,
            urgency=PurchaseOrderUrgency.EMERGENCY,
        )

        with self.assertRaises(SupplierGatewayIdempotencyConflictError):
            self.service.transmit_purchase_order(
                self.fo.fo_id,
                tampered_urgency_po,
                idempotency_key="CMD-TAMPER-KEY",
            )

    # -------------------------------------------------------------------------
    # 21. Pipeline Inventory Isolation Across Outcomes
    # -------------------------------------------------------------------------

    def test_21_pipeline_inventory_isolation_across_outcomes(self):
        """Active inbound pipeline quantity strictly isolates outcomes: only CONFIRMED_ACCEPTANCE contributes."""
        # 1. READY_TO_TRANSMIT contributes 0
        self.assertEqual(self.fo.total_confirmed_pipeline_qty, 0.0)

        # 2. Definite failure contributes 0
        fo_fail, _ = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-PIPE-FAIL",
            override_scenario=SupplierSimulationScenario.TRANSMISSION_FAILURE,
        )
        self.assertEqual(fo_fail.total_confirmed_pipeline_qty, 0.0)

        # 3. Ambiguous outcome contributes 0
        fo_amb, _ = self.service.transmit_purchase_order(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-PIPE-AMB",
            override_scenario=SupplierSimulationScenario.AMBIGUOUS_OUTCOME,
        )
        self.assertEqual(fo_amb.total_confirmed_pipeline_qty, 0.0)

        # 4. Confirmed rejection contributes 0
        po_rej = _make_approved_po(po_id="PO-REJ-PIPE-01")
        fo_rej_init = self.service.create_order(po_rej)
        fo_rej, _ = self.service.transmit_purchase_order(
            fo_rej_init.fo_id,
            po_rej,
            idempotency_key="CMD-PIPE-REJ",
            override_scenario=SupplierSimulationScenario.REJECT,
        )
        self.assertEqual(fo_rej.status, FulfilmentOrderStatus.REJECTED_BY_SUPPLIER)
        self.assertEqual(fo_rej.total_confirmed_pipeline_qty, 0.0)

        # 5. Confirmed acceptance contributes exactly final_order_qty (24.0)
        rec_fo, _ = self.service.reconcile_transmission(
            self.fo.fo_id,
            self.po,
            idempotency_key="CMD-PIPE-AMB",
        )
        self.assertEqual(rec_fo.status, FulfilmentOrderStatus.ACKNOWLEDGED)
        self.assertEqual(rec_fo.total_confirmed_pipeline_qty, 24.0)


if __name__ == "__main__":
    unittest.main()
