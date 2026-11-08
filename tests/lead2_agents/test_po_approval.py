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
    ISO8601_STRICT_PATTERN,
    PurchaseOrder,
    PurchaseOrderAuditRecord,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
    validate_and_normalize_iso_timestamp,
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
    audit_trail: list = None,
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
        audit_trail=tuple(audit_trail) if audit_trail is not None else (),
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


class TestStrictTimestampValidation(unittest.TestCase):
    """
    Verification of Fix 2: Strict ISO 8601 timestamp validation throughout the
    Purchase Order approval, rejection, and audit workflows.
    """
    def setUp(self):
        self.workflow = PurchaseOrderApprovalWorkflow()
        self.pending_po = _make_test_po()

    def test_case_01_valid_utc_timestamp_ending_in_z(self):
        """Case 1: Valid UTC timestamp ending in 'Z' is accepted and preserved."""
        approved = self.workflow.approve(
            po=self.pending_po,
            approved_by="Lead_Approver_01",
            approval_timestamp="2026-11-08T12:30:00Z",
        )
        self.assertEqual(approved.approval_timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(approved.audit_trail[0].timestamp, "2026-11-08T12:30:00Z")

    def test_case_02_valid_utc_timestamp_with_zero_offset(self):
        """Case 2: Valid UTC timestamp with explicit '+00:00' is normalized to 'Z'."""
        approved = self.workflow.approve(
            po=self.pending_po,
            approved_by="Lead_Approver_01",
            approval_timestamp="2026-11-08T12:30:00+00:00",
        )
        self.assertEqual(approved.approval_timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(approved.audit_trail[0].timestamp, "2026-11-08T12:30:00Z")

    def test_case_03_valid_non_utc_offset_converted_to_utc(self):
        """Case 3: Valid non-UTC offset (+05:30) is correctly converted to UTC ending in 'Z'."""
        # 18:00:00 in IST (+05:30) is 12:30:00 UTC
        approved = self.workflow.approve(
            po=self.pending_po,
            approved_by="Lead_Approver_01",
            approval_timestamp="2026-11-08T18:00:00+05:30",
        )
        self.assertEqual(approved.approval_timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(approved.audit_trail[0].timestamp, "2026-11-08T12:30:00Z")

        # Also test negative offset: 07:30:00 in EST (-05:00) is 12:30:00 UTC
        rejected = self.workflow.reject(
            po=self.pending_po,
            rejected_by="Ops_Reviewer_01",
            rejection_timestamp="2026-11-08T07:30:00-05:00",
            rejection_reason="Stock re-evaluation required.",
        )
        self.assertEqual(rejected.rejection_timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(rejected.audit_trail[0].timestamp, "2026-11-08T12:30:00Z")

    def test_case_04_naive_timestamp_rejected(self):
        """Case 4: Timezone-naive timestamp without offset/Z is rejected."""
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=self.pending_po,
                approved_by="Lead_Approver_01",
                approval_timestamp="2026-11-08T12:30:00",
            )
        self.assertIn("timezone", str(ctx.exception).lower())

    def test_case_05_missing_timezone_and_space_separator_rejected(self):
        """Case 5: Timestamp with space separator and no timezone is rejected."""
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=self.pending_po,
                approved_by="Lead_Approver_01",
                approval_timestamp="2026-11-08 12:30:00",
            )
        self.assertIn("timezone", str(ctx.exception).lower())

    def test_case_06_malformed_timestamp_strings_rejected(self):
        """Case 6: Malformed date/time strings are strictly rejected."""
        malformed_inputs = [
            "not-a-timestamp",
            "08-11-2026",
            "2026/11/08",
            "2026-11-08",
            "12:30:00Z",
            "2026.11.08T12:30:00Z",
        ]
        for bad_ts in malformed_inputs:
            with self.subTest(bad_ts=bad_ts):
                with self.assertRaises(ApprovalGovernanceError):
                    self.workflow.approve(
                        po=self.pending_po,
                        approved_by="Lead_Approver_01",
                        approval_timestamp=bad_ts,
                    )

    def test_case_07_invalid_calendar_dates_rejected(self):
        """Case 7: Non-existent calendar dates (e.g. Feb 30, April 31) are rejected."""
        invalid_dates = [
            "2026-02-30T12:00:00Z",
            "2026-04-31T12:00:00Z",
            "2026-02-29T12:00:00Z",  # 2026 is not a leap year
            "2026-13-01T12:00:00Z",  # Month 13
            "2026-00-10T12:00:00Z",  # Month 00
            "2026-11-00T12:00:00Z",  # Day 00
            "2026-11-32T12:00:00Z",  # Day 32
        ]
        for bad_date in invalid_dates:
            with self.subTest(bad_date=bad_date):
                with self.assertRaises(ApprovalGovernanceError):
                    self.workflow.approve(
                        po=self.pending_po,
                        approved_by="Lead_Approver_01",
                        approval_timestamp=bad_date,
                    )

    def test_case_08_invalid_time_or_utc_offset_rejected(self):
        """Case 8: Invalid hour, minute, second, or timezone offsets are rejected."""
        invalid_times_or_offsets = [
            "2026-11-08T24:00:00Z",       # Hour 24
            "2026-11-08T25:00:00Z",       # Hour 25
            "2026-11-08T12:60:00Z",       # Minute 60
            "2026-11-08T12:00:60Z",       # Second 60
            "2026-11-08T12:00:00+25:00",   # Offset +25:00
            "2026-11-08T12:00:00+05:60",   # Offset minute 60
            "2026-11-08T12:00:00+14:01",   # Offset beyond +14:00
            "2026-11-08T12:00:00-15:00",   # Offset beyond -14:00
        ]
        for bad_val in invalid_times_or_offsets:
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(ApprovalGovernanceError):
                    self.workflow.approve(
                        po=self.pending_po,
                        approved_by="Lead_Approver_01",
                        approval_timestamp=bad_val,
                    )

    def test_case_09_empty_and_whitespace_only_timestamps_rejected(self):
        """Case 9: Empty and whitespace-only timestamps are strictly rejected."""
        blank_inputs = ["", " ", "   ", "\t", "\n"]
        for blank in blank_inputs:
            with self.subTest(blank=blank):
                with self.assertRaises(ApprovalGovernanceError) as ctx:
                    self.workflow.approve(
                        po=self.pending_po,
                        approved_by="Lead_Approver_01",
                        approval_timestamp=blank,
                    )
                self.assertIn("approval_timestamp", str(ctx.exception))

    def test_case_10_direct_approved_po_with_invalid_timestamp_rejected(self):
        """Case 10: Direct instantiation of APPROVED PO with invalid timestamp raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-DIRECT-TEST-01",
                created_date="2026-11-08",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                supplier_type=SupplierType.PRIMARY,
                items=[_make_test_line_item()],
                total_order_qty=24.0,
                total_order_cost=720.0,
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Direct_Approver",
                approval_timestamp="not-a-valid-timestamp",
            )
        self.assertIn("approval_timestamp", str(ctx.exception))

        # Also test timezone-naive string on direct instantiation
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-DIRECT-TEST-02",
                created_date="2026-11-08",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                supplier_type=SupplierType.PRIMARY,
                items=[_make_test_line_item()],
                total_order_qty=24.0,
                total_order_cost=720.0,
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Direct_Approver",
                approval_timestamp="2026-11-08 12:30:00",
            )
        self.assertIn("approval_timestamp", str(ctx.exception))

    def test_case_11_direct_rejected_po_with_invalid_timestamp_rejected(self):
        """Case 11: Direct instantiation of REJECTED PO with invalid timestamp raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-DIRECT-REJ-01",
                created_date="2026-11-08",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                supplier_type=SupplierType.PRIMARY,
                items=[_make_test_line_item()],
                total_order_qty=24.0,
                total_order_cost=720.0,
                status=PurchaseOrderStatus.REJECTED,
                rejected_by="Direct_Reviewer",
                rejection_timestamp="2026-11-08T12:30:00",  # Naive
                rejection_reason="Stock level adequate.",
            )
        self.assertIn("rejection_timestamp", str(ctx.exception))

    def test_case_12_invalid_audit_record_timestamp_rejected(self):
        """Case 12: PurchaseOrderAuditRecord rejects invalid/naive timestamps with ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderAuditRecord(
                po_id="PO-AUDIT-TEST",
                previous_status=PurchaseOrderStatus.PENDING_APPROVAL,
                new_status=PurchaseOrderStatus.APPROVED,
                reviewer_id="Reviewer_01",
                timestamp="invalid-timestamp",
                action=ApprovalAction.APPROVE,
                reason="Routine approval.",
                requires_escalation=False,
            )
        self.assertIn("timestamp", str(ctx.exception))

        # Test naive timestamp on audit record
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderAuditRecord(
                po_id="PO-AUDIT-TEST-2",
                previous_status=PurchaseOrderStatus.PENDING_APPROVAL,
                new_status=PurchaseOrderStatus.APPROVED,
                reviewer_id="Reviewer_01",
                timestamp="2026-11-08T12:30:00",
                action=ApprovalAction.APPROVE,
                reason="Routine approval.",
                requires_escalation=False,
            )
        self.assertIn("timestamp", str(ctx.exception))

    def test_case_13_approval_workflow_rejects_invalid_timestamp(self):
        """Case 13: Approval workflow rejects invalid timestamps raising ApprovalGovernanceError."""
        invalid_ts_list = [
            "2026-11-08T12:30:00",
            "2026-11-08 12:30:00",
            "not-an-iso-string",
            "2026-02-30T10:00:00Z",
        ]
        for bad_ts in invalid_ts_list:
            with self.subTest(bad_ts=bad_ts):
                with self.assertRaises(ApprovalGovernanceError) as ctx:
                    self.workflow.approve(
                        po=self.pending_po,
                        approved_by="Lead_01",
                        approval_timestamp=bad_ts,
                    )
                self.assertIn("approval_timestamp", str(ctx.exception))

    def test_case_14_rejection_workflow_rejects_invalid_timestamp(self):
        """Case 14: Rejection workflow rejects invalid timestamps raising ApprovalGovernanceError."""
        invalid_ts_list = [
            "2026-11-08T12:30:00",
            "2026-11-08 12:30:00",
            "not-an-iso-string",
            "2026-02-30T10:00:00Z",
            "",
            "   ",
        ]
        for bad_ts in invalid_ts_list:
            with self.subTest(bad_ts=bad_ts):
                with self.assertRaises(ApprovalGovernanceError) as ctx:
                    self.workflow.reject(
                        po=self.pending_po,
                        rejected_by="Reviewer_01",
                        rejection_timestamp=bad_ts,
                        rejection_reason="Rejection reason",
                    )
                self.assertIn("rejection_timestamp", str(ctx.exception))

    def test_case_15_identical_normalized_timestamps_in_po_and_audit(self):
        """Case 15: Valid approval and rejection store identical normalized UTC timestamps in PO and audit."""
        # Non-UTC input normalized to UTC
        approved = self.workflow.approve(
            po=self.pending_po,
            approved_by="Approver_01",
            approval_timestamp="2026-11-08T18:00:00+05:30",
        )
        expected_utc = "2026-11-08T12:30:00Z"
        self.assertEqual(approved.approval_timestamp, expected_utc)
        self.assertEqual(approved.audit_trail[-1].timestamp, expected_utc)
        self.assertEqual(approved.approval_timestamp, approved.audit_trail[-1].timestamp)

        # Rejection with +00:00 normalized to Z
        pending2 = _make_test_po(po_id="PO-REJ-IDENTICAL-01")
        rejected = self.workflow.reject(
            po=pending2,
            rejected_by="Reviewer_01",
            rejection_timestamp="2026-11-08T12:30:00+00:00",
            rejection_reason="Duplicate order.",
        )
        self.assertEqual(rejected.rejection_timestamp, expected_utc)
        self.assertEqual(rejected.audit_trail[-1].timestamp, expected_utc)
        self.assertEqual(rejected.rejection_timestamp, rejected.audit_trail[-1].timestamp)

    def test_case_16_model_copy_cannot_bypass_timestamp_validation(self):
        """Case 16: model_copy cannot bypass timestamp validation at the workflow boundary."""
        # Maliciously attempt to set an invalid naive timestamp via model_copy
        tampered_po = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Bypass_Attacker",
                "approval_timestamp": "2026-11-08T12:30:00",  # Naive!
            }
        )
        # 1. Boundary validation must catch and reject this
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            validate_inbound_po(tampered_po)
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))

        # 2. Workflow approve/reject must also reject it at the boundary
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=tampered_po,
                approved_by="Legal_Lead",
                approval_timestamp="2026-11-08T14:00:00Z",
            )
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))

    def test_verify_approval_evidence_validates_timestamp(self):
        """verify_approval_evidence returns False if approval_timestamp is invalid or naive."""
        # Unvalidated construct with invalid timestamp
        tampered_po = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Attacker",
                "approval_timestamp": "invalid-timestamp",
            }
        )
        self.assertFalse(verify_approval_evidence(tampered_po))
        self.assertFalse(can_be_dispatched(tampered_po))


class TestPOAuditEvidenceIntegrity(unittest.TestCase):
    """
    Verification of Additional Governance Fix: Enforce PO Audit Evidence Integrity.

    Invariants Enforced:
    1. Every APPROVED or ISSUED PO must have an APPROVE audit record:
       - po_id matching PO
       - previous_status == PENDING_APPROVAL
       - new_status == APPROVED
       - reviewer_id matching approved_by
       - timestamp matching normalized approval_timestamp
    2. Every REJECTED PO must have a REJECT audit record:
       - po_id matching PO
       - previous_status == PENDING_APPROVAL
       - new_status == REJECTED
       - reviewer_id matching rejected_by
       - timestamp matching normalized rejection_timestamp
       - reason matching rejection_reason
    3. Reject missing, duplicate, or inconsistent terminal decision records.
    4. Workflow boundary protection against model_copy bypass.
    5. verify_approval_evidence() and can_be_dispatched() fail closed.
    6. Roundtrip JSON serialization and reconstruction.
    """
    def setUp(self):
        self.workflow = PurchaseOrderApprovalWorkflow()
        self.pending_po = _make_test_po()

    def _make_approve_audit(
        self,
        po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
        reviewer_id: str = "Operator_Lead_01",
        timestamp: str = "2026-11-08T12:30:00Z",
        previous_status: PurchaseOrderStatus = PurchaseOrderStatus.PENDING_APPROVAL,
        new_status: PurchaseOrderStatus = PurchaseOrderStatus.APPROVED,
        action: ApprovalAction = ApprovalAction.APPROVE,
        reason: str = "Approved by authorized human operator.",
    ) -> PurchaseOrderAuditRecord:
        return PurchaseOrderAuditRecord(
            po_id=po_id,
            previous_status=previous_status,
            new_status=new_status,
            reviewer_id=reviewer_id,
            timestamp=timestamp,
            action=action,
            reason=reason,
            requires_escalation=False,
            escalation_reasons=(),
        )

    def _make_reject_audit(
        self,
        po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
        reviewer_id: str = "Ops_Manager_01",
        timestamp: str = "2026-11-08T12:30:00Z",
        previous_status: PurchaseOrderStatus = PurchaseOrderStatus.PENDING_APPROVAL,
        new_status: PurchaseOrderStatus = PurchaseOrderStatus.REJECTED,
        action: ApprovalAction = ApprovalAction.REJECT,
        reason: str = "Duplicate replenishment detected.",
    ) -> PurchaseOrderAuditRecord:
        return PurchaseOrderAuditRecord(
            po_id=po_id,
            previous_status=previous_status,
            new_status=new_status,
            reviewer_id=reviewer_id,
            timestamp=timestamp,
            action=action,
            reason=reason,
            requires_escalation=False,
            escalation_reasons=(),
        )

    def test_01_direct_approved_po_with_no_audit_record_rejected(self):
        """Test 1: Direct instantiation of APPROVED or ISSUED PO with empty audit_trail is rejected."""
        # APPROVED with empty audit_trail
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[],
            )
        self.assertIn("must contain a matching APPROVE audit record in 'audit_trail'", str(ctx.exception))

        # ISSUED with empty audit_trail
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.ISSUED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[],
            )
        self.assertIn("must contain a matching APPROVE audit record in 'audit_trail'", str(ctx.exception))

    def test_02_direct_rejected_po_with_no_audit_record_rejected(self):
        """Test 2: Direct instantiation of REJECTED PO with empty audit_trail is rejected."""
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.REJECTED,
                rejected_by="Ops_Manager_01",
                rejection_timestamp="2026-11-08T12:30:00Z",
                rejection_reason="Budget constraint",
                audit_trail=[],
            )
        self.assertIn("must contain a matching REJECT audit record in 'audit_trail'", str(ctx.exception))

    def test_03_mismatched_reviewer_id_rejected(self):
        """Test 3: Mismatched reviewer_id between PO and audit record is strictly rejected."""
        # APPROVED: PO approved_by != audit reviewer_id
        audit = self._make_approve_audit(reviewer_id="Reviewer_Actual")
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Reviewer_Spoofed",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[audit],
            )
        self.assertIn("reviewer_id 'Reviewer_Actual' does not match Purchase Order approved_by 'Reviewer_Spoofed'", str(ctx.exception))

        # REJECTED: PO rejected_by != audit reviewer_id
        rej_audit = self._make_reject_audit(reviewer_id="Reviewer_Actual")
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.REJECTED,
                rejected_by="Reviewer_Spoofed",
                rejection_timestamp="2026-11-08T12:30:00Z",
                rejection_reason="Duplicate replenishment detected.",
                audit_trail=[rej_audit],
            )
        self.assertIn("reviewer_id 'Reviewer_Actual' does not match Purchase Order rejected_by 'Reviewer_Spoofed'", str(ctx.exception))

    def test_04_mismatched_timestamp_rejected(self):
        """Test 4: Mismatched timestamp between PO and audit record is strictly rejected."""
        # APPROVED: PO approval_timestamp != audit timestamp
        audit = self._make_approve_audit(timestamp="2026-11-08T10:00:00Z")
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[audit],
            )
        self.assertIn("timestamp '2026-11-08T10:00:00Z' does not match Purchase Order approval_timestamp", str(ctx.exception))

        # REJECTED: PO rejection_timestamp != audit timestamp
        rej_audit = self._make_reject_audit(timestamp="2026-11-08T10:00:00Z")
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.REJECTED,
                rejected_by="Ops_Manager_01",
                rejection_timestamp="2026-11-08T12:30:00Z",
                rejection_reason="Duplicate replenishment detected.",
                audit_trail=[rej_audit],
            )
        self.assertIn("timestamp '2026-11-08T10:00:00Z' does not match Purchase Order rejection_timestamp", str(ctx.exception))

    def test_05_mismatched_po_id_rejected(self):
        """Test 5: Mismatched po_id in audit record is strictly rejected."""
        audit = self._make_approve_audit(po_id="PO-OTHER-STORE-001")
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[audit],
            )
        self.assertIn("does not match Purchase Order po_id", str(ctx.exception))

    def test_06_incorrect_previous_or_new_status_in_audit_record_rejected(self):
        """Test 6: Incorrect previous_status or new_status in audit record is rejected."""
        # 1. previous_status must be PENDING_APPROVAL
        audit_bad_prev = self._make_approve_audit(previous_status=PurchaseOrderStatus.APPROVED)
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[audit_bad_prev],
            )
        self.assertIn("previous_status 'APPROVED' must be PENDING_APPROVAL", str(ctx.exception))

        # 2. new_status must be APPROVED for approved PO
        audit_bad_new = self._make_approve_audit(new_status=PurchaseOrderStatus.PENDING_APPROVAL)
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[audit_bad_new],
            )
        self.assertIn("new_status 'PENDING_APPROVAL' must be APPROVED", str(ctx.exception))

        # 3. For REJECTED PO: previous_status must be PENDING_APPROVAL and new_status must be REJECTED
        rej_bad_prev = self._make_reject_audit(previous_status=PurchaseOrderStatus.REJECTED)
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.REJECTED,
                rejected_by="Ops_Manager_01",
                rejection_timestamp="2026-11-08T12:30:00Z",
                rejection_reason="Duplicate replenishment detected.",
                audit_trail=[rej_bad_prev],
            )
        self.assertIn("previous_status 'REJECTED' must be PENDING_APPROVAL", str(ctx.exception))

        # 4. For REJECTED PO: reason mismatch
        rej_bad_reason = self._make_reject_audit(reason="Reason in audit")
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.REJECTED,
                rejected_by="Ops_Manager_01",
                rejection_timestamp="2026-11-08T12:30:00Z",
                rejection_reason="Different reason on PO",
                audit_trail=[rej_bad_reason],
            )
        self.assertIn("does not match Purchase Order rejection_reason", str(ctx.exception))

    def test_07_duplicate_terminal_decision_records_rejected(self):
        """Test 7: Duplicate or conflicting terminal decision audit records are rejected."""
        audit1 = self._make_approve_audit(reviewer_id="Operator_Lead_01")
        audit2 = self._make_approve_audit(reviewer_id="Operator_Lead_01")
        # Duplicate APPROVE records
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[audit1, audit2],
            )
        self.assertIn("contains duplicate or conflicting terminal decision audit records", str(ctx.exception))

        # Conflicting APPROVE + REJECT records
        rej_audit = self._make_reject_audit(reviewer_id="Operator_Lead_01")
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_01",
                approval_timestamp="2026-11-08T12:30:00Z",
                audit_trail=[audit1, rej_audit],
            )
        self.assertIn("contains duplicate or conflicting terminal decision audit records", str(ctx.exception))

        # Terminal decision record on PENDING_APPROVAL order
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.PENDING_APPROVAL,
                audit_trail=[audit1],
            )
        self.assertIn("cannot contain terminal decision audit records", str(ctx.exception))

    def test_08_valid_workflow_generated_approval(self):
        """Test 8: Normal approval via workflow produces valid matching audit record and evidence."""
        approved = self.workflow.approve(
            po=self.pending_po,
            approved_by="Lead_Approver_BLR",
            approval_timestamp="2026-11-08T18:00:00+05:30",  # Normalizes to 12:30:00Z
            approval_reason="Verified against master inventory balance.",
        )
        # 1. State changes
        self.assertEqual(approved.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(approved.approved_by, "Lead_Approver_BLR")
        self.assertEqual(approved.approval_timestamp, "2026-11-08T12:30:00Z")

        # 2. Audit record validation
        self.assertEqual(len(approved.audit_trail), 1)
        rec = approved.audit_trail[0]
        self.assertEqual(rec.po_id, approved.po_id)
        self.assertEqual(rec.previous_status, PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertEqual(rec.new_status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(rec.reviewer_id, "Lead_Approver_BLR")
        self.assertEqual(rec.timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(rec.action, ApprovalAction.APPROVE)

        # 3. Evidence verification
        self.assertTrue(verify_approval_evidence(approved))
        self.assertTrue(can_be_dispatched(approved))
        self.assertFalse(is_issued_or_dispatched(approved))

    def test_09_valid_workflow_generated_rejection(self):
        """Test 9: Normal rejection via workflow produces valid matching audit record."""
        rejected = self.workflow.reject(
            po=self.pending_po,
            rejected_by="Ops_Reviewer_BLR",
            rejection_timestamp="2026-11-08T12:30:00Z",
            rejection_reason="Duplicate replenishment run.",
        )
        # 1. State changes
        self.assertEqual(rejected.status, PurchaseOrderStatus.REJECTED)
        self.assertEqual(rejected.rejected_by, "Ops_Reviewer_BLR")
        self.assertEqual(rejected.rejection_timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(rejected.rejection_reason, "Duplicate replenishment run.")

        # 2. Audit record validation
        self.assertEqual(len(rejected.audit_trail), 1)
        rec = rejected.audit_trail[0]
        self.assertEqual(rec.po_id, rejected.po_id)
        self.assertEqual(rec.previous_status, PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertEqual(rec.new_status, PurchaseOrderStatus.REJECTED)
        self.assertEqual(rec.reviewer_id, "Ops_Reviewer_BLR")
        self.assertEqual(rec.timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(rec.reason, "Duplicate replenishment run.")
        self.assertEqual(rec.action, ApprovalAction.REJECT)

        # 3. Evidence verification fails closed
        self.assertFalse(verify_approval_evidence(rejected))
        self.assertFalse(can_be_dispatched(rejected))

    def test_10_invalid_model_copy_approval_bypass(self):
        """Test 10: model_copy bypass attempt is rejected at workflow boundary."""
        # Attacker tries to bypass approval using model_copy without audit record
        tampered_po = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Attacker_Lead",
                "approval_timestamp": "2026-11-08T12:30:00Z",
            }
        )
        # 1. Boundary check directly rejects it
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            validate_inbound_po(tampered_po)
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))
        self.assertIn("must contain a matching APPROVE audit record", str(ctx.exception))

        # 2. Workflow approve rejects it at boundary
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=tampered_po,
                approved_by="Auditor",
                approval_timestamp="2026-11-08T13:00:00Z",
            )
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))

        # 3. Workflow reject rejects it at boundary
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.reject(
                po=tampered_po,
                rejected_by="Auditor",
                rejection_timestamp="2026-11-08T13:00:00Z",
                rejection_reason="Boundary check",
            )
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))

    def test_11_dispatch_eligibility_for_invalid_approval_evidence(self):
        """Test 11: verify_approval_evidence and can_be_dispatched fail closed on any defect."""
        # 1. Unvalidated object with empty audit_trail
        no_audit = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Operator_01",
                "approval_timestamp": "2026-11-08T12:30:00Z",
                "audit_trail": (),
            }
        )
        self.assertFalse(verify_approval_evidence(no_audit))
        self.assertFalse(can_be_dispatched(no_audit))

        # 2. Unvalidated object with mismatched reviewer_id in audit
        bad_reviewer = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Operator_01",
                "approval_timestamp": "2026-11-08T12:30:00Z",
                "audit_trail": (self._make_approve_audit(reviewer_id="Different_Reviewer"),),
            }
        )
        self.assertFalse(verify_approval_evidence(bad_reviewer))
        self.assertFalse(can_be_dispatched(bad_reviewer))

        # 3. Unvalidated object with mismatched timestamp in audit
        bad_ts = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Operator_01",
                "approval_timestamp": "2026-11-08T12:30:00Z",
                "audit_trail": (self._make_approve_audit(timestamp="2026-11-08T10:00:00Z"),),
            }
        )
        self.assertFalse(verify_approval_evidence(bad_ts))
        self.assertFalse(can_be_dispatched(bad_ts))

        # 4. Unvalidated object with mismatched po_id in audit
        bad_poid = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Operator_01",
                "approval_timestamp": "2026-11-08T12:30:00Z",
                "audit_trail": (self._make_approve_audit(po_id="PO-OTHER-ID"),),
            }
        )
        self.assertFalse(verify_approval_evidence(bad_poid))
        self.assertFalse(can_be_dispatched(bad_poid))

        # 5. Unvalidated object with REJECT action on APPROVED PO
        rej_action = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Operator_01",
                "approval_timestamp": "2026-11-08T12:30:00Z",
                "audit_trail": (self._make_reject_audit(reviewer_id="Operator_01"),),
            }
        )
        self.assertFalse(verify_approval_evidence(rej_action))
        self.assertFalse(can_be_dispatched(rej_action))

        # 6. Unvalidated object with duplicate terminal records
        audit1 = self._make_approve_audit()
        audit2 = self._make_approve_audit()
        dup_audit = self.pending_po.model_copy(
            update={
                "status": PurchaseOrderStatus.APPROVED,
                "approved_by": "Operator_Lead_01",
                "approval_timestamp": "2026-11-08T12:30:00Z",
                "audit_trail": (audit1, audit2),
            }
        )
        self.assertFalse(verify_approval_evidence(dup_audit))
        self.assertFalse(can_be_dispatched(dup_audit))

    def test_12_json_serialization_and_reconstruction_of_valid_pos(self):
        """Test 12: JSON serialization and reconstruction of valid approved and rejected POs."""
        # 1. Roundtrip for approved PO
        approved = self.workflow.approve(
            po=self.pending_po,
            approved_by="Auditor_Director_01",
            approval_timestamp="2026-11-08T12:30:00Z",
            approval_reason="Full audit verified.",
        )
        json_data = approved.model_dump_json()
        reconstructed_appr = PurchaseOrder.model_validate_json(json_data)
        self.assertEqual(reconstructed_appr.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(reconstructed_appr.approved_by, "Auditor_Director_01")
        self.assertEqual(reconstructed_appr.approval_timestamp, "2026-11-08T12:30:00Z")
        self.assertEqual(len(reconstructed_appr.audit_trail), 1)
        self.assertEqual(reconstructed_appr.audit_trail[0].reviewer_id, "Auditor_Director_01")
        self.assertTrue(verify_approval_evidence(reconstructed_appr))
        self.assertTrue(can_be_dispatched(reconstructed_appr))

        # 2. Roundtrip for rejected PO
        rejected = self.workflow.reject(
            po=self.pending_po,
            rejected_by="Reviewer_Director_02",
            rejection_timestamp="2026-11-08T12:30:00Z",
            rejection_reason="Threshold violation.",
        )
        json_rej = rejected.model_dump_json()
        reconstructed_rej = PurchaseOrder.model_validate_json(json_rej)
        self.assertEqual(reconstructed_rej.status, PurchaseOrderStatus.REJECTED)
        self.assertEqual(reconstructed_rej.rejected_by, "Reviewer_Director_02")
        self.assertEqual(reconstructed_rej.rejection_reason, "Threshold violation.")
        self.assertEqual(len(reconstructed_rej.audit_trail), 1)
        self.assertEqual(reconstructed_rej.audit_trail[0].reviewer_id, "Reviewer_Director_02")
        self.assertFalse(verify_approval_evidence(reconstructed_rej))
        self.assertFalse(can_be_dispatched(reconstructed_rej))


class TestCancelledStatusGovernance(unittest.TestCase):
    """
    Verification of Final Governance Check: CANCELLED Purchase Order State.

    Required Behaviors:
    1. Direct construction of a PurchaseOrder with status CANCELLED is strictly rejected.
    2. Any attempt to transition a PO to CANCELLED through the existing approval workflow is rejected.
    3. model_copy(update={"status": "CANCELLED"}) cannot bypass validation at workflow boundaries.
    4. Keep the enum value available for a future explicit cancellation workflow.
    5. Existing APPROVED, REJECTED, DRAFT, and PENDING_APPROVAL behavior remains unchanged.
    """
    def setUp(self):
        self.workflow = PurchaseOrderApprovalWorkflow()
        self.pending_po = _make_test_po()

    def test_direct_cancelled_po_construction_rejected(self):
        """Direct construction of PurchaseOrder with status CANCELLED raises ValidationError."""
        # Enum value
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(status=PurchaseOrderStatus.CANCELLED)
        self.assertIn("cannot have status 'CANCELLED'", str(ctx.exception))
        self.assertIn("Order cancellation is not supported in Phase 4 governance", str(ctx.exception))

        # String value
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(status="CANCELLED")
        self.assertIn("cannot have status 'CANCELLED'", str(ctx.exception))

    def test_cancelled_po_construction_with_empty_audit_history_rejected(self):
        """Direct construction of CANCELLED PO with empty audit_trail is strictly rejected."""
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.CANCELLED,
                audit_trail=[],
            )
        self.assertIn("cannot have status 'CANCELLED'", str(ctx.exception))

    def test_cancelled_po_construction_with_approval_audit_record_rejected(self):
        """Direct construction of CANCELLED PO even with an attached audit record is rejected."""
        audit = PurchaseOrderAuditRecord(
            po_id=self.pending_po.po_id,
            previous_status=PurchaseOrderStatus.PENDING_APPROVAL,
            new_status=PurchaseOrderStatus.APPROVED,
            reviewer_id="Operator_01",
            timestamp="2026-11-08T12:00:00Z",
            action=ApprovalAction.APPROVE,
            reason="Approved.",
            requires_escalation=False,
            escalation_reasons=(),
        )
        with self.assertRaises(ValidationError) as ctx:
            _make_test_po(
                status=PurchaseOrderStatus.CANCELLED,
                approved_by="Operator_01",
                approval_timestamp="2026-11-08T12:00:00Z",
                audit_trail=[audit],
            )
        self.assertIn("cannot have status 'CANCELLED'", str(ctx.exception))

    def test_model_copy_cancellation_bypass_rejected_at_workflow_boundary(self):
        """model_copy cancellation bypass attempts are detected and rejected at workflow boundaries."""
        tampered_cancelled_po = self.pending_po.model_copy(
            update={"status": PurchaseOrderStatus.CANCELLED}
        )
        self.assertEqual(tampered_cancelled_po.status, PurchaseOrderStatus.CANCELLED)

        # 1. validate_inbound_po boundary check rejects it
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            validate_inbound_po(tampered_cancelled_po)
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))
        self.assertIn("cannot have status 'CANCELLED'", str(ctx.exception))

        # 2. Workflow approve rejects it at boundary
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.approve(
                po=tampered_cancelled_po,
                approved_by="Lead_01",
                approval_timestamp="2026-11-08T12:00:00Z",
            )
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))

        # 3. Workflow reject rejects it at boundary
        with self.assertRaises(ApprovalGovernanceError) as ctx:
            self.workflow.reject(
                po=tampered_cancelled_po,
                rejected_by="Lead_01",
                rejection_timestamp="2026-11-08T12:00:00Z",
                rejection_reason="Boundary check",
            )
        self.assertIn("Workflow Boundary Rejection", str(ctx.exception))

    def test_cancelled_po_evidence_and_dispatch_fail_closed(self):
        """verify_approval_evidence and can_be_dispatched fail closed on CANCELLED PO."""
        tampered_cancelled_po = self.pending_po.model_copy(
            update={"status": PurchaseOrderStatus.CANCELLED}
        )
        self.assertFalse(verify_approval_evidence(tampered_cancelled_po))
        self.assertFalse(can_be_dispatched(tampered_cancelled_po))
        self.assertFalse(is_issued_or_dispatched(tampered_cancelled_po))

    def test_existing_approved_rejected_draft_pending_behavior_unchanged(self):
        """Preserves normal behavior for APPROVED, REJECTED, DRAFT, and PENDING_APPROVAL."""
        # 1. DRAFT is valid without terminal audit records
        draft_po = _make_test_po(status=PurchaseOrderStatus.DRAFT)
        self.assertEqual(draft_po.status, PurchaseOrderStatus.DRAFT)

        # 2. PENDING_APPROVAL is valid without terminal audit records
        pending = _make_test_po(status=PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertEqual(pending.status, PurchaseOrderStatus.PENDING_APPROVAL)

        # 3. Normal workflow approval produces valid APPROVED PO
        approved = self.workflow.approve(
            po=pending,
            approved_by="Auditor_Lead",
            approval_timestamp="2026-11-08T12:00:00Z",
        )
        self.assertEqual(approved.status, PurchaseOrderStatus.APPROVED)
        self.assertTrue(verify_approval_evidence(approved))

        # 4. Normal workflow rejection produces valid REJECTED PO
        pending2 = _make_test_po(po_id="PO-REJ-UNCHANGED-01")
        rejected = self.workflow.reject(
            po=pending2,
            rejected_by="Auditor_Lead",
            rejection_timestamp="2026-11-08T12:00:00Z",
            rejection_reason="Operational adjustment.",
        )
        self.assertEqual(rejected.status, PurchaseOrderStatus.REJECTED)


if __name__ == "__main__":
    unittest.main()
