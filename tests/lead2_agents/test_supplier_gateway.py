"""
Unit & Integration Tests: Supplier Gateway Abstraction & Deterministic Simulation Adapter
========================================================================================
Phase 5.4.1 Verification Tests:
1. Interface adherence (ABC and Protocol conformance).
2. Deterministic simulation scenarios:
   - Supplier accepts order (CONFIRMED_ACCEPTANCE).
   - Supplier rejects order (CONFIRMED_REJECTION).
   - Transmission fails before acceptance (TRANSMISSION_FAILURE).
   - Supplier accepts but response unavailable (AMBIGUOUS_OUTCOME).
3. Strict idempotency enforcement:
   - Repeating identical transmission key is exact no-op without duplicate submission.
   - Reusing transmission key with differing PO or payload raises conflict.
4. Determinism guarantees (zero randomness, identical outputs across instances).
5. Phase 4 approval governance enforcement (unapproved, rejected, or tampered POs rejected).
6. PurchaseOrder immutability preservation.
7. Scenario routing (default, per-PO, per-supplier, and per-call overrides).
8. Result contract validation (invalid status/acknowledgment combinations rejected).
"""

import unittest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.agents.fulfilment.gateway import (
    InvalidPurchaseOrderError,
    SimulatedSupplierAdapter,
    SupplierGateway,
    SupplierGatewayIdempotencyConflictError,
    SupplierGatewayProtocol,
    SupplierSimulationScenario,
    SupplierTransmissionResult,
    SupplierTransmissionStatus,
)
from src.agents.fulfilment.schemas import (
    SupplierAcknowledgment,
    SupplierAcknowledgmentLine,
    SupplierAcknowledgmentStatus,
)
from src.agents.procurement.approval import approve_purchase_order, reject_purchase_order
from src.agents.procurement.schemas import (
    ApprovalAction,
    PurchaseOrder,
    PurchaseOrderAuditRecord,
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
    approval_timestamp: str = "2024-11-15T10:30:00Z",
    approved_by: str = "OPERATOR_DEEPANSHU",
) -> PurchaseOrder:
    """Constructs a valid APPROVED PurchaseOrder with verified human approval evidence."""
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
        approved_by=approved_by,
        approval_timestamp=approval_timestamp,
        approval_reason="Routine replenishment approved.",
    )


class TestSupplierGateway(unittest.TestCase):
    """Test suite for SupplierGateway abstraction and SimulatedSupplierAdapter."""

    def setUp(self):
        self.adapter = SimulatedSupplierAdapter()
        self.approved_po = _make_approved_po()

    # -------------------------------------------------------------------------
    # 1. Interface Adherence
    # -------------------------------------------------------------------------

    def test_01_interface_adherence(self):
        """Adapter conforms to SupplierGateway ABC and SupplierGatewayProtocol."""
        self.assertIsInstance(self.adapter, SupplierGateway)
        self.assertIsInstance(self.adapter, SupplierGatewayProtocol)

        # Incomplete subclass cannot be instantiated
        class IncompleteGateway(SupplierGateway):
            pass

        with self.assertRaises(TypeError):
            IncompleteGateway()

    # -------------------------------------------------------------------------
    # 2. Simulation Scenarios
    # -------------------------------------------------------------------------

    def test_02_scenario_confirmed_acceptance(self):
        """Scenario ACCEPT produces CONFIRMED_ACCEPTANCE with valid acknowledgment."""
        key = "KEY-ACCEPT-01"
        result = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)
        self.assertTrue(result.is_success)
        self.assertFalse(result.is_rejected)
        self.assertFalse(result.is_failure)
        self.assertFalse(result.is_ambiguous)

        self.assertEqual(result.po_id, self.approved_po.po_id)
        self.assertEqual(result.supplier_id, self.approved_po.supplier_id)
        self.assertEqual(result.transmitted_at, "2024-11-15T11:00:00Z")
        self.assertTrue(result.correlation_id.startswith("CORR-"))
        self.assertTrue(result.transmission_id.startswith("TXN-"))
        self.assertEqual(result.supplier_reference, f"SUP-REF-{self.approved_po.po_id}")
        self.assertIsNone(result.error_code)
        self.assertIsNone(result.error_message)

        # Validate acknowledgment contract
        self.assertIsNotNone(result.acknowledgment)
        ack = result.acknowledgment
        self.assertEqual(ack.status, SupplierAcknowledgmentStatus.ACCEPTED)
        self.assertEqual(ack.po_id, self.approved_po.po_id)
        self.assertEqual(ack.supplier_id, self.approved_po.supplier_id)
        self.assertEqual(ack.acknowledged_at, "2024-11-15T11:00:00Z")
        self.assertIsNotNone(ack.expected_delivery_at)
        self.assertGreater(ack.expected_delivery_at, ack.acknowledged_at)

        # Lines match PO items
        self.assertEqual(len(ack.lines), len(self.approved_po.items))
        self.assertEqual(ack.lines[0].sku_id, "SKU_MILK_1L")
        self.assertEqual(ack.lines[0].accepted_quantity, 24.0)
        self.assertEqual(ack.lines[0].declined_quantity, 0.0)

        # Internal supplier submission recorded
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 1)
        self.assertTrue(self.adapter.has_supplier_received(self.approved_po.po_id))

    def test_03_scenario_confirmed_rejection(self):
        """Scenario REJECT produces CONFIRMED_REJECTION with mandatory rejection reason."""
        key = "KEY-REJECT-01"
        result = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.REJECT,
            rejection_reason="Factory cold-storage outage",
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.CONFIRMED_REJECTION)
        self.assertFalse(result.is_success)
        self.assertTrue(result.is_rejected)
        self.assertFalse(result.is_failure)
        self.assertFalse(result.is_ambiguous)

        self.assertIsNotNone(result.acknowledgment)
        ack = result.acknowledgment
        self.assertEqual(ack.status, SupplierAcknowledgmentStatus.REJECTED)
        self.assertEqual(ack.rejection_reason, "Factory cold-storage outage")
        self.assertEqual(ack.lines[0].accepted_quantity, 0.0)
        self.assertEqual(ack.lines[0].declined_quantity, 24.0)

        # Internal supplier submission recorded
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 1)
        self.assertTrue(self.adapter.has_supplier_received(self.approved_po.po_id))

    def test_04_scenario_transmission_failure(self):
        """Scenario TRANSMISSION_FAILURE produces failure before acceptance, no supplier submission."""
        key = "KEY-FAIL-01"
        result = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.TRANSMISSION_FAILURE,
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.TRANSMISSION_FAILURE)
        self.assertFalse(result.is_success)
        self.assertFalse(result.is_rejected)
        self.assertTrue(result.is_failure)
        self.assertFalse(result.is_ambiguous)

        self.assertIsNone(result.acknowledgment)
        self.assertIsNone(result.supplier_reference)
        self.assertEqual(result.error_code, "TRANSMISSION_FAILED_PRE_ACCEPTANCE")
        self.assertIn("failed before acceptance", result.error_message)

        # Crucial check: supplier was NEVER reached, submission count is 0
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 0)
        self.assertFalse(self.adapter.has_supplier_received(self.approved_po.po_id))

    def test_05_scenario_ambiguous_outcome(self):
        """Scenario AMBIGUOUS_OUTCOME: supplier accepted order, but response dropped/unavailable."""
        key = "KEY-AMBIG-01"
        result = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.AMBIGUOUS_OUTCOME,
        )

        self.assertEqual(result.status, SupplierTransmissionStatus.AMBIGUOUS_OUTCOME)
        self.assertFalse(result.is_success)
        self.assertFalse(result.is_rejected)
        self.assertFalse(result.is_failure)
        self.assertTrue(result.is_ambiguous)

        # Caller received NO acknowledgment
        self.assertIsNone(result.acknowledgment)
        self.assertEqual(result.error_code, "RESPONSE_TIMEOUT_AMBIGUOUS")
        self.assertIn("ambiguous outcome", result.error_message)

        # Crucial check: under the hood, simulated supplier DID receive and accept order
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 1)
        self.assertTrue(self.adapter.has_supplier_received(self.approved_po.po_id))
        self.assertEqual(result.details.get("internal_supplier_status"), "ACCEPTED")

    # -------------------------------------------------------------------------
    # 3. Strict Idempotency Enforcement
    # -------------------------------------------------------------------------

    def test_06_idempotent_retry_creates_no_second_submission(self):
        """Repeating the same transmission key returns the cached result without second submission."""
        key = "KEY-IDEMP-01"

        # First transmission
        result1 = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 1)

        # Exact retry with same key and same PO
        result2 = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp="2024-11-15T11:00:00Z",
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        # Results must be identical
        self.assertEqual(result1.transmission_id, result2.transmission_id)
        self.assertEqual(result1.correlation_id, result2.correlation_id)
        self.assertEqual(result1.transmitted_at, result2.transmitted_at)
        self.assertEqual(result1.status, result2.status)
        self.assertEqual(result1.acknowledgment.model_dump(), result2.acknowledgment.model_dump())

        # Crucial invariant: submission count STILL exactly 1 (no duplicate submission)
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 1)

    def test_07_idempotency_across_failure_and_ambiguous_scenarios(self):
        """Idempotent retry works consistently for TRANSMISSION_FAILURE and AMBIGUOUS_OUTCOME."""
        # Failure scenario retry
        fail_key = "KEY-FAIL-RETRY"
        res_fail_1 = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=fail_key,
            override_scenario=SupplierSimulationScenario.TRANSMISSION_FAILURE,
        )
        res_fail_2 = self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=fail_key,
            override_scenario=SupplierSimulationScenario.TRANSMISSION_FAILURE,
        )
        self.assertEqual(res_fail_1.transmission_id, res_fail_2.transmission_id)
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 0)

        # Ambiguous scenario retry
        po2 = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-02")
        ambig_key = "KEY-AMBIG-RETRY"
        res_amb_1 = self.adapter.transmit_purchase_order(
            po2,
            idempotency_key=ambig_key,
            override_scenario=SupplierSimulationScenario.AMBIGUOUS_OUTCOME,
        )
        self.assertEqual(self.adapter.get_submission_count(po2.po_id), 1)

        res_amb_2 = self.adapter.transmit_purchase_order(
            po2,
            idempotency_key=ambig_key,
            override_scenario=SupplierSimulationScenario.AMBIGUOUS_OUTCOME,
        )
        self.assertEqual(res_amb_1.transmission_id, res_amb_2.transmission_id)
        self.assertEqual(self.adapter.get_submission_count(po2.po_id), 1)

    def test_08_reusing_key_with_different_po_raises_conflict(self):
        """Reusing an idempotency key with a different Purchase Order raises conflict error."""
        shared_key = "KEY-SHARED-01"
        po_a = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01")
        po_b = _make_approved_po(po_id="PO-20241115-DS_BLR_HSR_LAYOUT-DAIRY-02", store_id="DS_BLR_HSR_LAYOUT")

        self.adapter.transmit_purchase_order(po_a, idempotency_key=shared_key)

        with self.assertRaises(SupplierGatewayIdempotencyConflictError) as ctx:
            self.adapter.transmit_purchase_order(po_b, idempotency_key=shared_key)

        self.assertIn("Idempotency Conflict", str(ctx.exception))
        self.assertIn(shared_key, str(ctx.exception))

    def test_09_reusing_key_with_altered_payload_or_scenario_raises_conflict(self):
        """Reusing an idempotency key with altered line item quantities or scenario raises conflict."""
        key = "KEY-PAYLOAD-01"
        self.adapter.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        # Attempt to retry same key but with REJECT scenario
        with self.assertRaises(SupplierGatewayIdempotencyConflictError):
            self.adapter.transmit_purchase_order(
                self.approved_po,
                idempotency_key=key,
                override_scenario=SupplierSimulationScenario.REJECT,
            )

    # -------------------------------------------------------------------------
    # 4. Determinism Guarantees
    # -------------------------------------------------------------------------

    def test_10_deterministic_behavior_across_adapter_instances(self):
        """Separate adapter instances given identical input and scenario produce bit-for-bit identical results."""
        adapter1 = SimulatedSupplierAdapter()
        adapter2 = SimulatedSupplierAdapter()

        key = "KEY-DETERMINISM-01"
        ts = "2024-11-15T12:00:00Z"

        res1 = adapter1.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp=ts,
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )
        res2 = adapter2.transmit_purchase_order(
            self.approved_po,
            idempotency_key=key,
            transmission_timestamp=ts,
            override_scenario=SupplierSimulationScenario.ACCEPT,
        )

        self.assertEqual(res1.model_dump(), res2.model_dump())

    # -------------------------------------------------------------------------
    # 5. Phase 4 Approval Governance Enforcement
    # -------------------------------------------------------------------------

    def test_11_unapproved_pending_po_rejected(self):
        """A Purchase Order in PENDING_APPROVAL status cannot be transmitted."""
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
        pending_po = PurchaseOrder(
            po_id="PO-PENDING-01",
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

        with self.assertRaises(InvalidPurchaseOrderError) as ctx:
            self.adapter.transmit_purchase_order(pending_po, idempotency_key="KEY-UNAPPROVED")

        self.assertIn("Only orders in APPROVED status can be transmitted", str(ctx.exception))

    def test_12_rejected_po_cannot_be_transmitted(self):
        """A rejected Purchase Order cannot be transmitted."""
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
        pending_po = PurchaseOrder(
            po_id="PO-REJECT-01",
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
        rej_po = reject_purchase_order(
            pending_po,
            rejected_by="MANAGER_PRIYA",
            rejection_timestamp="2024-11-15T10:00:00Z",
            rejection_reason="Excess inventory on hand.",
        )

        with self.assertRaises(InvalidPurchaseOrderError):
            self.adapter.transmit_purchase_order(rej_po, idempotency_key="KEY-REJ-PO")

    def test_13_tampered_approval_evidence_rejected(self):
        """A PO with status APPROVED but missing/tampered approval evidence is rejected."""
        # Construct raw dict claiming APPROVED without audit trail record
        raw_data = self.approved_po.model_dump()
        raw_data["audit_trail"] = ()  # Strip audit record

        # Pydantic model_validate itself catches this or can_be_dispatched catches it
        with self.assertRaises((ValueError, InvalidPurchaseOrderError)):
            tampered_po = PurchaseOrder.model_validate(raw_data)
            self.adapter.transmit_purchase_order(tampered_po, idempotency_key="KEY-TAMPER")

    def test_14_non_purchase_order_input_rejected(self):
        """Passing a non-PurchaseOrder object raises InvalidPurchaseOrderError."""
        with self.assertRaises(InvalidPurchaseOrderError):
            self.adapter.transmit_purchase_order("NOT_A_PO", idempotency_key="KEY-INVALID")

    def test_15_empty_idempotency_key_rejected(self):
        """Empty or whitespace-only idempotency key raises ValueError."""
        with self.assertRaises(ValueError):
            self.adapter.transmit_purchase_order(self.approved_po, idempotency_key="   ")

    # -------------------------------------------------------------------------
    # 6. PurchaseOrder Immutability
    # -------------------------------------------------------------------------

    def test_16_purchase_order_immutability(self):
        """PurchaseOrder is completely unmodified by gateway transmission."""
        orig_dict = self.approved_po.model_dump()

        self.adapter.transmit_purchase_order(self.approved_po, idempotency_key="KEY-IMMUTABLE")

        # PO state must remain 100% unchanged
        self.assertEqual(self.approved_po.model_dump(), orig_dict)
        self.assertEqual(self.approved_po.status, PurchaseOrderStatus.APPROVED)

    # -------------------------------------------------------------------------
    # 7. Scenario Routing (Default, Per-PO, Per-Supplier)
    # -------------------------------------------------------------------------

    def test_17_scenario_routing_per_po_and_per_supplier(self):
        """Adapter correctly routes scenarios based on per-PO and per-supplier configurations."""
        po_accept = _make_approved_po(po_id="PO-ROUTE-ACCEPT")
        po_reject = _make_approved_po(po_id="PO-ROUTE-REJECT")
        po_fail = _make_approved_po(po_id="PO-ROUTE-FAIL", supplier_id="SIM_SUPP_FAIL_VENDOR")

        self.adapter.set_default_scenario(SupplierSimulationScenario.ACCEPT)
        self.adapter.set_po_scenario(po_reject.po_id, SupplierSimulationScenario.REJECT, "Item out of stock")
        self.adapter.set_supplier_scenario("SIM_SUPP_FAIL_VENDOR", SupplierSimulationScenario.TRANSMISSION_FAILURE)

        res_acc = self.adapter.transmit_purchase_order(po_accept, idempotency_key="KEY-R1")
        self.assertEqual(res_acc.status, SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE)

        res_rej = self.adapter.transmit_purchase_order(po_reject, idempotency_key="KEY-R2")
        self.assertEqual(res_rej.status, SupplierTransmissionStatus.CONFIRMED_REJECTION)
        self.assertEqual(res_rej.acknowledgment.rejection_reason, "Item out of stock")

        res_fail = self.adapter.transmit_purchase_order(po_fail, idempotency_key="KEY-R3")
        self.assertEqual(res_fail.status, SupplierTransmissionStatus.TRANSMISSION_FAILURE)

    def test_18_adapter_reset_and_record_lookup(self):
        """Adapter get_transmission_record and reset methods work properly."""
        key = "KEY-LOOKUP-01"
        res = self.adapter.transmit_purchase_order(self.approved_po, idempotency_key=key)

        lookup = self.adapter.get_transmission_record(key)
        self.assertIsNotNone(lookup)
        self.assertEqual(lookup.transmission_id, res.transmission_id)

        self.adapter.reset()
        self.assertIsNone(self.adapter.get_transmission_record(key))
        self.assertEqual(self.adapter.get_submission_count(self.approved_po.po_id), 0)

    # -------------------------------------------------------------------------
    # 8. Result Contract Invariant Enforcement
    # -------------------------------------------------------------------------

    def test_19_result_contract_validation_rules(self):
        """SupplierTransmissionResult enforces consistency between status, acknowledgment, and error fields."""
        # ACCEPTANCE requires acknowledgment
        with self.assertRaises(ValidationError):
            SupplierTransmissionResult(
                transmission_id="TXN-01",
                idempotency_key="KEY-01",
                po_id=self.approved_po.po_id,
                supplier_id=self.approved_po.supplier_id,
                status=SupplierTransmissionStatus.CONFIRMED_ACCEPTANCE,
                transmitted_at="2024-11-15T11:00:00Z",
                correlation_id="CORR-01",
                acknowledgment=None,
            )

        # TRANSMISSION_FAILURE requires error_code and error_message
        with self.assertRaises(ValidationError):
            SupplierTransmissionResult(
                transmission_id="TXN-02",
                idempotency_key="KEY-02",
                po_id=self.approved_po.po_id,
                supplier_id=self.approved_po.supplier_id,
                status=SupplierTransmissionStatus.TRANSMISSION_FAILURE,
                transmitted_at="2024-11-15T11:00:00Z",
                correlation_id="CORR-02",
                error_code=None,
                error_message=None,
            )

        # AMBIGUOUS_OUTCOME forbids acknowledgment
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-01",
            fo_id="FO-01",
            po_id=self.approved_po.po_id,
            supplier_id=self.approved_po.supplier_id,
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:00:00Z",
            expected_delivery_at="2024-11-16T11:00:00Z",
        )
        with self.assertRaises(ValidationError):
            SupplierTransmissionResult(
                transmission_id="TXN-03",
                idempotency_key="KEY-03",
                po_id=self.approved_po.po_id,
                supplier_id=self.approved_po.supplier_id,
                status=SupplierTransmissionStatus.AMBIGUOUS_OUTCOME,
                transmitted_at="2024-11-15T11:00:00Z",
                correlation_id="CORR-03",
                acknowledgment=ack,
                error_code="RESPONSE_TIMEOUT_AMBIGUOUS",
                error_message="Ambiguous outcome",
            )


if __name__ == "__main__":
    unittest.main()
