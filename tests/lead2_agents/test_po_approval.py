"""
Unit & Integration Tests: Purchase Order Approval Workflow & Escalation Policies
================================================================================
Phase 4.4 Verification Tests:
1. Approval without an approver (raises ApprovalGovernanceError).
2. Approval without a timestamp (raises ApprovalGovernanceError).
3. Successful human approval (status -> APPROVED, audit trail updated, approved_by and timestamp recorded).
4. Successful human rejection with reason (status -> REJECTED, audit trail updated, reason recorded).
5. Rejection without reviewer or reason (raises ApprovalGovernanceError).
6. Invalid status transitions (APPROVED -> APPROVED, APPROVED -> REJECTED, REJECTED -> APPROVED, etc.).
7. Critical Pydantic validation safeguard: model_copy bypass detection and rejection at workflow boundary.
8. Escalation policies: line cost > ₹5k, store PO > ₹25k, MOQ excess, assumed-zero pipeline with LOW_STOCK_REORDER.
9. Every PO requiring approval, including low-value POs (no automatic approval).
10. Approved POs not treated as issued or dispatched (strict decoupling of approval and issuance).
"""

import unittest
from pydantic import ValidationError

from src.agents.procurement.schemas import (
    ApprovalAction,
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
)
from src.agents.procurement.approval import (
    ApprovalGovernanceError,
    InvalidStatusTransitionError,
    PurchaseOrderApprovalWorkflow,
    approve_purchase_order,
    can_be_dispatched,
    evaluate_escalation_policies,
    is_issued_or_dispatched,
    reject_purchase_order,
    validate_inbound_po,
    verify_approval_evidence,
)


def _make_test_line_item(
    store_id: str = "DS_BLR_KORAMANGALA",
    sku_id: str = "SKU_MILK_1L",
    category: str = "Dairy",
    final_order_qty: float = 24.0,
    case_pack_size: int = 12,
    moq: float = 10.0,
    unit_retail_price: float = 50.0,
    unit_procurement_cost: float = 30.0,
    total_line_cost: float = 720.0,
    requires_escalation: bool = False,
    escalation_reasons: list = None,
) -> PurchaseOrderLineItem:
    return PurchaseOrderLineItem(
        store_id=store_id,
        sku_id=sku_id,
        category=category,
        raw_suggested_qty=20.0,
        case_pack_rounded_qty=24.0,
        final_order_qty=final_order_qty,
        case_pack_size=case_pack_size,
        moq=moq,
        unit_retail_price=unit_retail_price,
        unit_procurement_cost=unit_procurement_cost,
        total_line_cost=total_line_cost,
        requires_escalation=requires_escalation,
        escalation_reasons=escalation_reasons or [],
    )


def _make_test_po(
    po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
    store_id: str = "DS_BLR_KORAMANGALA",
    supplier_id: str = "SIM_SUPP_DAIRY_PRIMARY",
    status: PurchaseOrderStatus = PurchaseOrderStatus.PENDING_APPROVAL,
    items: list = None,
    total_order_qty: float = 24.0,
    total_order_cost: float = 720.0,
    requires_escalation: bool = False,
    escalation_reasons: list = None,
    approved_by: str = None,
    approval_timestamp: str = None,
    rejected_by: str = None,
    rejection_timestamp: str = None,
    rejection_reason: str = None,
) -> PurchaseOrder:
    if items is None:
        items = [_make_test_line_item(store_id=store_id)]
    return PurchaseOrder(
        po_id=po_id,
        created_date="2024-11-15",
        store_id=store_id,
        supplier_id=supplier_id,
        supplier_type=SupplierType.PRIMARY,
        items=items,
        total_order_qty=total_order_qty,
        total_order_cost=total_order_cost,
        urgency=PurchaseOrderUrgency.STANDARD,
        status=status,
        requires_escalation=requires_escalation,
        escalation_reasons=escalation_reasons or [],
        approved_by=approved_by,
        approval_timestamp=approval_timestamp,
        rejected_by=rejected_by,
        rejection_timestamp=rejection_timestamp,
        rejection_reason=rejection_reason,
    )


class TestPurchaseOrderApprovalWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = PurchaseOrderApprovalWorkflow()
        self.pending_po = _make_test_po()

    def test_approval_without_approver_rejected(self):
        """Approval must strictly require a non-empty approved_by identifier."""
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=self.pending_po,
                approved_by="",  # Empty approver
                approval_timestamp="2024-11-15T10:30:00Z",
            )
        self.assertIn("approved_by", str(ctx.exception))

        with self.assertRaises(ApprovalGovernanceError):
            self.workflow.approve(
                po=self.pending_po,
                approved_by="   ",  # Whitespace-only approver
                approval_timestamp="2024-11-15T10:30:00Z",
            )

    def test_approval_without_timestamp_rejected(self):
        """Approval must strictly require a non-empty approval_timestamp."""
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=self.pending_po,
                approved_by="Operator_Lead_01",
                approval_timestamp="",  # Empty timestamp
            )
        self.assertIn("approval_timestamp", str(ctx.exception))

        with self.assertRaises(ApprovalGovernanceError):
            self.workflow.approve(
                po=self.pending_po,
                approved_by="Operator_Lead_01",
                approval_timestamp="   ",  # Whitespace timestamp
            )

    def test_successful_human_approval(self):
        """Valid human approval transitions PO to APPROVED and logs audit record."""
        approved_po = self.workflow.approve(
            po=self.pending_po,
            approved_by="Operator_Lead_BLR_01",
            approval_timestamp="2024-11-15T10:30:00Z",
            approval_reason="Routine cycle replenishment reviewed and confirmed.",
        )

        # 1. State changes
        self.assertEqual(approved_po.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(approved_po.approved_by, "Operator_Lead_BLR_01")
        self.assertEqual(approved_po.approval_timestamp, "2024-11-15T10:30:00Z")
        self.assertEqual(approved_po.approval_reason, "Routine cycle replenishment reviewed and confirmed.")

        # 2. Audit trail verification
        self.assertEqual(len(approved_po.audit_trail), 1)
        audit = approved_po.audit_trail[0]
        self.assertEqual(audit.po_id, self.pending_po.po_id)
        self.assertEqual(audit.previous_status, PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertEqual(audit.new_status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(audit.action, ApprovalAction.APPROVE)
        self.assertEqual(audit.reviewer_id, "Operator_Lead_BLR_01")
        self.assertEqual(audit.timestamp, "2024-11-15T10:30:00Z")
        self.assertIn("Routine cycle replenishment", audit.reason)

        # 3. Immutability check: original pending_po remains unchanged
        self.assertEqual(self.pending_po.status, PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertIsNone(self.pending_po.approved_by)

    def test_successful_human_rejection_with_reason(self):
        """Valid human rejection transitions PO to REJECTED and logs audit record with mandatory reason."""
        rejected_po = self.workflow.reject(
            po=self.pending_po,
            rejected_by="Ops_Manager_BLR_02",
            rejection_timestamp="2024-11-15T11:00:00Z",
            rejection_reason="Duplicate replenishment detected; manual adjustment required.",
        )

        # 1. State changes
        self.assertEqual(rejected_po.status, PurchaseOrderStatus.REJECTED)
        self.assertEqual(rejected_po.rejected_by, "Ops_Manager_BLR_02")
        self.assertEqual(rejected_po.rejection_timestamp, "2024-11-15T11:00:00Z")
        self.assertEqual(rejected_po.rejection_reason, "Duplicate replenishment detected; manual adjustment required.")

        # 2. Audit trail verification
        self.assertEqual(len(rejected_po.audit_trail), 1)
        audit = rejected_po.audit_trail[0]
        self.assertEqual(audit.action, ApprovalAction.REJECT)
        self.assertEqual(audit.previous_status, PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertEqual(audit.new_status, PurchaseOrderStatus.REJECTED)
        self.assertEqual(audit.reviewer_id, "Ops_Manager_BLR_02")
        self.assertEqual(audit.reason, "Duplicate replenishment detected; manual adjustment required.")

    def test_rejection_without_reason_rejected(self):
        """Rejection must fail if rejection_reason is empty or missing."""
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.reject(
                po=self.pending_po,
                rejected_by="Ops_Manager_BLR_02",
                rejection_timestamp="2024-11-15T11:00:00Z",
                rejection_reason="",  # Empty reason forbidden
            )
        self.assertIn("rejection_reason", str(ctx.exception))

        with self.assertRaises(ApprovalGovernanceError):
            self.workflow.reject(
                po=self.pending_po,
                rejected_by="Ops_Manager_BLR_02",
                rejection_timestamp="2024-11-15T11:00:00Z",
                rejection_reason="    ",  # Whitespace forbidden
            )

    def test_rejection_without_reviewer_rejected(self):
        """Rejection must fail if rejected_by is empty."""
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.reject(
                po=self.pending_po,
                rejected_by="",
                rejection_timestamp="2024-11-15T11:00:00Z",
                rejection_reason="Valid reason",
            )
        self.assertIn("rejected_by", str(ctx.exception))

    def test_invalid_status_transitions(self):
        """Transitions out of non-PENDING_APPROVAL states must be strictly rejected."""
        # 1. Approve a PO
        approved_po = self.workflow.approve(
            po=self.pending_po,
            approved_by="Lead_01",
            approval_timestamp="2024-11-15T10:00:00Z",
        )

        # Trying to approve an already APPROVED PO must fail
        with self.assertRaises(InvalidStatusTransitionError) as ctx:
            self.workflow.approve(
                po=approved_po,
                approved_by="Lead_02",
                approval_timestamp="2024-11-15T10:05:00Z",
            )
        self.assertIn("Cannot approve Purchase Order", str(ctx.exception))
        self.assertIn("APPROVED", str(ctx.exception))

        # Trying to reject an already APPROVED PO must fail
        with self.assertRaises(InvalidStatusTransitionError) as ctx:
            self.workflow.reject(
                po=approved_po,
                rejected_by="Lead_02",
                rejection_timestamp="2024-11-15T10:05:00Z",
                rejection_reason="Cannot reject approved order",
            )
        self.assertIn("Cannot reject Purchase Order", str(ctx.exception))

        # 2. Reject a PO
        rejected_po = self.workflow.reject(
            po=self.pending_po,
            rejected_by="Lead_01",
            rejection_timestamp="2024-11-15T10:00:00Z",
            rejection_reason="Budget constraint",
        )

        # Trying to approve an already REJECTED PO must fail
        with self.assertRaises(InvalidStatusTransitionError):
            self.workflow.approve(
                po=rejected_po,
                approved_by="Lead_02",
                approval_timestamp="2024-11-15T10:05:00Z",
            )

        # Trying to reject an already REJECTED PO must fail
        with self.assertRaises(InvalidStatusTransitionError):
            self.workflow.reject(
                po=rejected_po,
                rejected_by="Lead_02",
                rejection_timestamp="2024-11-15T10:05:00Z",
                rejection_reason="Duplicate rejection attempt",
            )


class TestAuditTrailImmutability(unittest.TestCase):
    def setUp(self):
        self.workflow = PurchaseOrderApprovalWorkflow()
        self.pending_po = _make_test_po()

    def test_audit_trail_and_nested_collections_immutable(self):
        """
        AUDIT IMMUTABILITY VERIFICATION:
        Ensures audit_trail, items, and escalation_reasons are immutable tuples
        and cannot be mutated in-place outside the approved workflow.
        """
        approved_po = self.workflow.approve(
            po=self.pending_po,
            approved_by="Auditor_Lead",
            approval_timestamp="2024-11-15T14:00:00Z",
        )

        # 1. Verify audit_trail is an immutable tuple
        self.assertIsInstance(approved_po.audit_trail, tuple)
        self.assertEqual(len(approved_po.audit_trail), 1)

        # Attempting in-place mutation of audit_trail must fail
        with self.assertRaises(AttributeError):
            approved_po.audit_trail.append("fake_record")  # Tuple has no append

        with self.assertRaises(AttributeError):
            approved_po.audit_trail.clear()  # Tuple has no clear

        with self.assertRaises(TypeError):
            approved_po.audit_trail[0] = "tampered_record"  # Tuple does not support assignment

        # 2. Verify items is an immutable tuple
        self.assertIsInstance(approved_po.items, tuple)
        with self.assertRaises(AttributeError):
            approved_po.items.append("fake_item")

        with self.assertRaises(TypeError):
            approved_po.items[0] = "tampered_item"

        # 3. Verify escalation_reasons is an immutable tuple
        self.assertIsInstance(approved_po.escalation_reasons, tuple)
        with self.assertRaises(AttributeError):
            approved_po.escalation_reasons.append("fake_reason")

    def test_po_serialization_and_reconstruction_with_audit_trail(self):
        """Final PO with audit trail can be serialized and reconstructed with full validation."""
        approved_po = self.workflow.approve(
            po=self.pending_po,
            approved_by="Auditor_Lead",
            approval_timestamp="2024-11-15T14:00:00Z",
            approval_reason="Verified against master inventory levels.",
        )
        # Serialization to JSON
        json_str = approved_po.model_dump_json()

        # Deserialization and full Pydantic validation
        reconstructed = PurchaseOrder.model_validate_json(json_str)
        self.assertEqual(reconstructed.po_id, approved_po.po_id)
        self.assertEqual(reconstructed.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(reconstructed.approved_by, "Auditor_Lead")
        self.assertIsInstance(reconstructed.audit_trail, tuple)
        self.assertEqual(len(reconstructed.audit_trail), 1)
        self.assertEqual(reconstructed.audit_trail[0].reviewer_id, "Auditor_Lead")


class TestPydanticModelCopySafeguard(unittest.TestCase):
    def setUp(self):
        self.workflow = PurchaseOrderApprovalWorkflow()
        self.pending_po = _make_test_po()

    def test_model_copy_bypass_demonstration_and_boundary_rejection(self):
        """
        CRITICAL PYDANTIC SAFEGUARD TEST:
        Demonstrates that Pydantic's model_copy(update=...) bypasses model validators,
        and verifies that the workflow service boundary strictly detects and rejects it.
        """
        # 1. Demonstrate the vulnerability of model_copy:
        # Pydantic's model_copy creates an invalid instance without executing validators!
        unvalidated_bypassed_po = self.pending_po.model_copy(
            update={"status": PurchaseOrderStatus.APPROVED, "approved_by": None}
        )
        # unvalidated_bypassed_po in memory now has status == APPROVED, but approved_by is None!
        self.assertEqual(unvalidated_bypassed_po.status, PurchaseOrderStatus.APPROVED)
        self.assertIsNone(unvalidated_bypassed_po.approved_by)

        # 2. Verify that boundary validation detects the unvalidated object and raises ApprovalGovernanceError
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            validate_inbound_po(unvalidated_bypassed_po)
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))
        self.assertIn("Mandatory Governance Violation", str(ctx.exception))

        # 3. Verify that the workflow service rejects the bypassed PO on approve() call
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=unvalidated_bypassed_po,
                approved_by="Hacker_Attempt",
                approval_timestamp="2024-11-15T12:00:00Z",
            )
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))

        # 4. Verify that the workflow service rejects the bypassed PO on reject() call
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.reject(
                po=unvalidated_bypassed_po,
                rejected_by="Hacker_Attempt",
                rejection_timestamp="2024-11-15T12:00:00Z",
                rejection_reason="Boundary check",
            )
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))


class TestApprovalAndIssuanceSeparation(unittest.TestCase):
    def setUp(self):
        self.workflow = PurchaseOrderApprovalWorkflow()
        self.pending_po = _make_test_po()

    def test_approved_po_not_treated_as_issued_or_dispatched(self):
        """An APPROVED PO must strictly NOT be considered issued or dispatched."""
        approved_po = self.workflow.approve(
            po=self.pending_po,
            approved_by="Operator_Lead_01",
            approval_timestamp="2024-11-15T10:00:00Z",
        )

        # 1. Status is APPROVED, never ISSUED
        self.assertEqual(approved_po.status, PurchaseOrderStatus.APPROVED)
        self.assertNotEqual(approved_po.status, PurchaseOrderStatus.ISSUED)

        # 2. Workflow helper checks
        self.assertTrue(self.workflow.is_approved(approved_po))
        self.assertFalse(self.workflow.is_issued_or_dispatched(approved_po))

        # 3. Verified approval evidence allows future dispatch
        self.assertTrue(self.workflow.can_be_dispatched(approved_po))

        # 4. Pending PO cannot be dispatched
        self.assertFalse(self.workflow.can_be_dispatched(self.pending_po))
        self.assertFalse(self.workflow.is_issued_or_dispatched(self.pending_po))

    def test_every_po_requires_approval_including_low_value(self):
        """Even a ₹720 low-value order strictly requires human approval (no auto-approval)."""
        low_val_item = _make_test_line_item(total_line_cost=720.0, final_order_qty=24.0)
        low_val_po = _make_test_po(
            items=[low_val_item],
            total_order_cost=720.0,
            total_order_qty=24.0,
            status=PurchaseOrderStatus.PENDING_APPROVAL,
        )

        # Pending order cannot be dispatched without human approval
        self.assertFalse(can_be_dispatched(low_val_po))
        self.assertFalse(verify_approval_evidence(low_val_po))

        # Must go through human approval
        approved = self.workflow.approve(
            po=low_val_po,
            approved_by="Junior_Operator_01",
            approval_timestamp="2024-11-15T09:30:00Z",
        )
        self.assertTrue(can_be_dispatched(approved))


class TestEscalationPolicies(unittest.TestCase):
    def test_escalation_line_threshold_above_5000(self):
        """Line cost > ₹5,000 must trigger escalation."""
        high_line_item = _make_test_line_item(
            sku_id="SKU_CHICKEN_500G",
            category="Meat",
            final_order_qty=60.0,
            unit_retail_price=160.0,
            unit_procurement_cost=96.0,
            total_line_cost=5760.0,  # > ₹5,000
            requires_escalation=True,
            escalation_reasons=["High Line Value: Line cost exceeds ₹5,000"],
        )
        po = _make_test_po(
            items=[high_line_item],
            total_order_cost=5760.0,
            total_order_qty=60.0,
            requires_escalation=True,
            escalation_reasons=["High Line Value: Line cost exceeds ₹5,000"],
        )
        esc, reasons = evaluate_escalation_policies(po)
        self.assertTrue(esc)
        self.assertTrue(any("High Line Value" in r for r in reasons))

    def test_escalation_line_threshold_boundary_exactly_5000(self):
        """Line cost of exactly ₹5,000.00 must NOT trigger escalation (strictly > ₹5,000)."""
        line_item_5000 = _make_test_line_item(
            sku_id="SKU_POTATO_1KG",
            category="Produce",
            case_pack_size=10,
            moq=10.0,
            final_order_qty=200.0,
            unit_retail_price=50.0,
            unit_procurement_cost=25.0,
            total_line_cost=5000.0,
            requires_escalation=False,
            escalation_reasons=[],
        )
        po = _make_test_po(
            supplier_id="SIM_SUPP_PRODUCE_PRIMARY",
            items=[line_item_5000],
            total_order_cost=5000.0,
            total_order_qty=200.0,
            requires_escalation=False,
            escalation_reasons=[],
        )
        esc, reasons = evaluate_escalation_policies(po)
        self.assertFalse(esc)
        self.assertEqual(len(reasons), 0)

        # Boundary check: line cost > ₹5,000 (e.g. ₹5,001.00) MUST trigger escalation
        line_item_5001 = _make_test_line_item(
            sku_id="SKU_POTATO_1KG",
            category="Produce",
            case_pack_size=10,
            moq=10.0,
            final_order_qty=200.0,
            unit_retail_price=50.0,
            unit_procurement_cost=25.005,
            total_line_cost=5001.0,
            requires_escalation=True,
            escalation_reasons=["High Line Value: Line cost exceeds ₹5,000"],
        )
        po_above = _make_test_po(
            supplier_id="SIM_SUPP_PRODUCE_PRIMARY",
            items=[line_item_5001],
            total_order_cost=5001.0,
            total_order_qty=200.0,
            requires_escalation=True,
            escalation_reasons=["High Line Value: Line cost exceeds ₹5,000"],
        )
        esc_above, reasons_above = evaluate_escalation_policies(po_above)
        self.assertTrue(esc_above)
        self.assertTrue(any("High Line Value" in r for r in reasons_above))

    def test_escalation_store_total_boundary_exactly_25000(self):
        """Store PO total of exactly ₹25,000.00 must NOT trigger store-level escalation (strictly > ₹25,000)."""
        produce_skus = ["SKU_BANANA_DZ", "SKU_ONION_1KG", "SKU_POTATO_1KG", "SKU_SPINACH_500G", "SKU_TOMATO_1KG"]
        items_5000 = [
            _make_test_line_item(
                sku_id=sku,
                category="Produce",
                case_pack_size=10,
                moq=10.0,
                final_order_qty=200.0,
                unit_retail_price=50.0,
                unit_procurement_cost=25.0,
                total_line_cost=5000.0,
                requires_escalation=False,
                escalation_reasons=[],
            )
            for sku in produce_skus
        ]
        po_25k = _make_test_po(
            supplier_id="SIM_SUPP_PRODUCE_PRIMARY",
            items=items_5000,
            total_order_cost=25000.0,
            total_order_qty=1000.0,
            requires_escalation=False,
            escalation_reasons=[],
        )
        esc, reasons = evaluate_escalation_policies(po_25k)
        self.assertFalse(esc)
        self.assertEqual(len(reasons), 0)

        # Boundary check: store total > ₹25,000 (e.g. ₹25,001.00) MUST trigger escalation
        items_25001 = items_5000[:4] + [
            _make_test_line_item(
                sku_id="SKU_TOMATO_1KG",
                category="Produce",
                case_pack_size=10,
                moq=10.0,
                final_order_qty=200.0,
                unit_retail_price=50.0,
                unit_procurement_cost=25.005,
                total_line_cost=5001.0,
                requires_escalation=True,
                escalation_reasons=["High Line Value: Line cost exceeds ₹5,000"],
            )
        ]
        po_25001 = _make_test_po(
            supplier_id="SIM_SUPP_PRODUCE_PRIMARY",
            items=items_25001,
            total_order_cost=25001.0,
            total_order_qty=1000.0,
            requires_escalation=True,
            escalation_reasons=["Store order total exceeds ₹25,000"],
        )
        esc_above, reasons_above = evaluate_escalation_policies(po_25001)
        self.assertTrue(esc_above)
        self.assertTrue(any("High Order Value" in r for r in reasons_above))

    def test_escalation_store_total_above_25000(self):
        """Store PO total > ₹25,000 must trigger escalation."""
        item = _make_test_line_item(
            sku_id="SKU_CHICKEN_500G",
            category="Meat",
            final_order_qty=300.0,
            unit_retail_price=160.0,
            unit_procurement_cost=96.0,
            total_line_cost=28800.0,
            requires_escalation=True,
            escalation_reasons=["High Line Value"],
        )
        po = _make_test_po(
            items=[item],
            total_order_cost=28800.0,  # > ₹25,000
            total_order_qty=300.0,
            requires_escalation=True,
            escalation_reasons=["Store order total exceeds ₹25,000"],
        )
        esc, reasons = evaluate_escalation_policies(po)
        self.assertTrue(esc)
        self.assertTrue(any("High Order Value" in r or "Store order total exceeds" in r for r in reasons))

    def test_escalation_non_critical_moq_excess(self):
        """Non-critical MOQ adjustment exceeding target stock must preserve manual review flag."""
        item = _make_test_line_item(
            sku_id="SKU_BREAD_WHITE",
            category="Bakery",
            requires_escalation=True,
            escalation_reasons=["MOQ Adjustment Excess: Non-critical stock position raised to MOQ"],
        )
        po = _make_test_po(
            items=[item],
            requires_escalation=True,
            escalation_reasons=["MOQ Adjustment Excess: Non-critical stock position raised to MOQ"],
        )
        esc, reasons = evaluate_escalation_policies(po)
        self.assertTrue(esc)
        self.assertTrue(any("MOQ Adjustment Excess" in r for r in reasons))

    def test_escalation_assumed_zero_pipeline_with_low_stock_reorder(self):
        """Assumed-zero pipeline with LOW_STOCK_REORDER must flag for review."""
        item = _make_test_line_item(
            sku_id="SKU_MILK_1L",
            requires_escalation=True,
            escalation_reasons=["Assumed-Zero Pipeline with LOW_STOCK_REORDER: Physical verification required"],
        )
        po = _make_test_po(
            items=[item],
            requires_escalation=True,
            escalation_reasons=["Assumed-Zero Pipeline with LOW_STOCK_REORDER: Physical verification required"],
        )
        esc, reasons = evaluate_escalation_policies(po)
        self.assertTrue(esc)
        self.assertTrue(any("Assumed-Zero Pipeline" in r for r in reasons))

    def test_escalated_po_can_still_be_approved_by_human(self):
        """Escalation is a review requirement, NOT an approval exemption: an operator CAN approve it."""
        escalated_item = _make_test_line_item(
            sku_id="SKU_CHICKEN_500G",
            category="Meat",
            final_order_qty=60.0,
            case_pack_size=6,
            moq=5.0,
            unit_retail_price=160.0,
            unit_procurement_cost=96.0,
            total_line_cost=5760.0,
            requires_escalation=True,
            escalation_reasons=["High Line Value"],
        )
        escalated_po = _make_test_po(
            items=[escalated_item],
            total_order_cost=5760.0,
            total_order_qty=60.0,
            requires_escalation=True,
            escalation_reasons=["High Line Value"],
        )
        workflow = PurchaseOrderApprovalWorkflow()

        # Human operator reviews escalation reasons and signs off
        approved = workflow.approve(
            po=escalated_po,
            approved_by="Senior_Director_Ops",
            approval_timestamp="2024-11-15T15:00:00Z",
            approval_reason="High value verified against regional festival demand; approved.",
        )
        self.assertEqual(approved.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(approved.approved_by, "Senior_Director_Ops")
        self.assertTrue(approved.requires_escalation)
        self.assertEqual(approved.audit_trail[0].requires_escalation, True)


if __name__ == "__main__":
    unittest.main()
