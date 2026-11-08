"""
Lead 2 Purchase Order Approval Workflow & Escalation Policies
============================================================
Implements strict human governance, audit trail persistence, and validated state
transitions for quick-commerce Purchase Orders.

CORE GOVERNANCE POLICIES & CONSTRAINTS:
1. MANDATORY HUMAN APPROVAL:
   Every Purchase Order requires explicit human approval before it can transition
   to APPROVED or be considered for future dispatch, regardless of:
   - PO monetary value (even low-value ₹720 orders)
   - Order urgency (STANDARD, EXPEDITE, EMERGENCY)
   - Supplier tier (PRIMARY or BACKUP)
   - Inventory risk level (CRITICAL_LOW, LOW_STOCK_REORDER)
   - Escalation status (escalated or non-escalated)
   Automated approval or unattended dispatch is strictly forbidden.

2. ALLOWED TRANSITIONS IN THIS PHASE:
   - PENDING_APPROVAL -> APPROVED (requires non-empty approved_by & approval_timestamp)
   - PENDING_APPROVAL -> REJECTED (requires non-empty rejected_by, rejection_timestamp, & rejection_reason)
   All other transitions (e.g. approving already approved/rejected/issued orders) are rejected.

3. PYDANTIC VALIDATION SAFEGUARD:
   Pydantic's model_copy(update=...) does NOT run model or field validators.
   Therefore, this workflow service:
   - Reconstructs and validates incoming PO data at workflow boundaries.
   - Executes transitions exclusively through model_validate() / full reconstruction.
   - Strictly forbids using model_copy(update=...) to alter approval states.

4. SEPARATION OF APPROVAL AND ISSUANCE:
   An APPROVED order is NOT issued or dispatched. PO issuance and supplier transmission
   are decoupled and strictly require verified human approval evidence.

5. ESCALATION POLICIES (Review Triggers, NOT Approval Exemptions):
   - Line cost > ₹5,000 -> Flag line item for escalation.
   - Store PO total > ₹25,000 -> Flag order for escalation.
   - Non-critical MOQ adjustment exceeds target stock -> Preserve manual-review flag.
   - Assumed-zero pipeline with LOW_STOCK_REORDER -> Flag for additional review.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import ValidationError

from ..common.config import Lead2Config, get_default_config
from .schemas import (
    ApprovalAction,
    PurchaseOrder,
    PurchaseOrderAuditRecord,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    validate_and_normalize_iso_timestamp,
)


class InvalidStatusTransitionError(ValueError):
    """Raised when an unsupported, illegal, or out-of-order PO status transition is attempted."""
    pass


class ApprovalGovernanceError(ValueError):
    """Raised when mandatory human approval, evidence, or validation governance rules are violated."""
    pass


def _validate_non_empty_str(v: Any, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ApprovalGovernanceError(
            f"Governance Violation: Field '{field_name}' must be a non-empty, non-whitespace string."
        )
    return v.strip()


def validate_inbound_po(po: PurchaseOrder) -> PurchaseOrder:
    """
    Validates incoming PurchaseOrder at workflow boundaries.

    CRITICAL SAFEGUARD:
    Protects against unvalidated model_copy(update=...) mutations. Reconstructs and
    executes all Pydantic validators to ensure incoming object integrity before
    processing approval or transition decisions.

    Args:
        po: Inbound PurchaseOrder instance.

    Returns:
        Validated PurchaseOrder.

    Raises:
        ApprovalGovernanceError: If incoming object fails Pydantic schema or governance rules.
    """
    if not isinstance(po, PurchaseOrder):
        raise ApprovalGovernanceError(f"Expected PurchaseOrder instance, got {type(po)}.")

    try:
        # Reconstruct and validate using official Pydantic validation method
        return PurchaseOrder.model_validate(po.model_dump())
    except ValidationError as err:
        raise ApprovalGovernanceError(
            f"Workflow Boundary Rejection: Incoming Purchase Order '{po.po_id}' failed model validation: {err}"
        ) from err


def evaluate_escalation_policies(
    po: PurchaseOrder,
    config: Optional[Lead2Config] = None,
) -> Tuple[bool, List[str]]:
    """
    Evaluates standard escalation policies on a Purchase Order.

    ESCALATION RULES:
    1. Line cost > ₹5,000
    2. Store PO total > ₹25,000
    3. Non-critical MOQ adjustment exceeding target stock
    4. Assumed-zero pipeline with LOW_STOCK_REORDER

    Returns:
        Tuple of (requires_escalation: bool, consolidated_reasons: List[str]).
    """
    cfg = config or get_default_config()
    max_line = cfg.procurement_thresholds.max_line_item_cost_inr
    max_order = cfg.procurement_thresholds.max_total_po_cost_inr

    reasons: List[str] = []

    # 1. Total order threshold check (> ₹25,000)
    if po.total_order_cost > max_order:
        reasons.append(
            f"High Order Value: Total store order cost ₹{po.total_order_cost:,.2f} exceeds "
            f"store-level review threshold ₹{max_order:,.2f}."
        )

    # 2. Line item threshold and rule checks
    for item in po.items:
        # Line cost threshold (> ₹5,000)
        if item.total_line_cost > max_line:
            line_reason = (
                f"[{item.sku_id}] High Line Value: Line cost ₹{item.total_line_cost:,.2f} exceeds "
                f"single-line review limit ₹{max_line:,.2f}."
            )
            if line_reason not in reasons:
                reasons.append(line_reason)

        # Inherit any item-level escalation triggers (e.g. MOQ excess, pipeline verification)
        if item.requires_escalation:
            for r in item.escalation_reasons:
                annotated = f"[{item.sku_id}] {r}"
                if annotated not in reasons:
                    reasons.append(annotated)

    # 3. Preserve existing PO-level escalation reasons if present
    for r in po.escalation_reasons:
        if r not in reasons:
            reasons.append(r)

    requires_escalation = len(reasons) > 0
    return requires_escalation, reasons


def approve_purchase_order(
    po: PurchaseOrder,
    approved_by: str,
    approval_timestamp: str,
    approval_reason: Optional[str] = None,
) -> PurchaseOrder:
    """
    Executes a validated human approval transition on a pending Purchase Order:
        PENDING_APPROVAL -> APPROVED

    CRITICAL SAFEGUARDS:
    1. Validates incoming PO data at the workflow boundary (protects against model_copy bypass).
    2. Enforces that only PENDING_APPROVAL orders can be transitioned.
    3. Mandates non-empty human approver identifier and ISO timestamp.
    4. Appends an immutable PurchaseOrderAuditRecord to the audit trail.
    5. Reconstructs and validates updated state using PurchaseOrder.model_validate().
    6. Strictly does NOT use model_copy(update=...).
    7. Approved state is decoupled from issuance (status remains APPROVED, NOT ISSUED).

    Args:
        po: Inbound PurchaseOrder in PENDING_APPROVAL status.
        approved_by: Non-empty identifier of authorized human operator.
        approval_timestamp: Non-empty ISO timestamp of approval.
        approval_reason: Optional recorded rationale for approval.

    Returns:
        New immutable PurchaseOrder in APPROVED status.

    Raises:
        ApprovalGovernanceError: If approver evidence is missing or boundary validation fails.
        InvalidStatusTransitionError: If order is not in PENDING_APPROVAL status.
    """
    # 1. Inbound validation at workflow boundary
    validated_inbound = validate_inbound_po(po)

    # 2. Strict state transition check
    if validated_inbound.status != PurchaseOrderStatus.PENDING_APPROVAL:
        raise InvalidStatusTransitionError(
            f"Invalid status transition: Cannot approve Purchase Order '{validated_inbound.po_id}' "
            f"in status '{validated_inbound.status.value}'. Only PENDING_APPROVAL orders can be approved."
        )

    # 3. Mandatory human approval evidence validation
    clean_approver = _validate_non_empty_str(approved_by, "approved_by")
    clean_timestamp = validate_and_normalize_iso_timestamp(
        approval_timestamp,
        field_name="approval_timestamp",
        error_cls=ApprovalGovernanceError,
    )
    clean_reason = (
        approval_reason.strip()
        if approval_reason and approval_reason.strip()
        else "Approved by authorized human operator."
    )

    # 4. Construct immutable audit record
    audit_entry = PurchaseOrderAuditRecord(
        po_id=validated_inbound.po_id,
        previous_status=validated_inbound.status,
        new_status=PurchaseOrderStatus.APPROVED,
        reviewer_id=clean_approver,
        timestamp=clean_timestamp,
        action=ApprovalAction.APPROVE,
        reason=clean_reason,
        requires_escalation=validated_inbound.requires_escalation,
        escalation_reasons=list(validated_inbound.escalation_reasons),
    )

    # 5. Full state reconstruction through Pydantic validation (NO model_copy bypass)
    data = validated_inbound.model_dump()
    data["status"] = PurchaseOrderStatus.APPROVED.value
    data["approved_by"] = clean_approver
    data["approval_timestamp"] = clean_timestamp
    data["approval_reason"] = clean_reason
    data["audit_trail"] = [rec.model_dump() for rec in validated_inbound.audit_trail] + [audit_entry.model_dump()]

    try:
        return PurchaseOrder.model_validate(data)
    except ValidationError as err:
        raise ApprovalGovernanceError(
            f"Approval Transition Failed: Reconstructed Purchase Order '{validated_inbound.po_id}' "
            f"failed model validation: {err}"
        ) from err


def reject_purchase_order(
    po: PurchaseOrder,
    rejected_by: str,
    rejection_timestamp: str,
    rejection_reason: str,
) -> PurchaseOrder:
    """
    Executes a validated human rejection transition on a pending Purchase Order:
        PENDING_APPROVAL -> REJECTED

    CRITICAL SAFEGUARDS:
    1. Validates incoming PO data at the workflow boundary.
    2. Enforces that only PENDING_APPROVAL orders can be rejected.
    3. Mandates non-empty human reviewer identifier, ISO timestamp, and non-empty rejection reason.
    4. Appends an immutable PurchaseOrderAuditRecord to the audit trail.
    5. Reconstructs and validates updated state using PurchaseOrder.model_validate().

    Args:
        po: Inbound PurchaseOrder in PENDING_APPROVAL status.
        rejected_by: Non-empty identifier of human reviewer rejecting the order.
        rejection_timestamp: Non-empty ISO timestamp of rejection.
        rejection_reason: Non-empty mandatory explanation for rejection.

    Returns:
        New immutable PurchaseOrder in REJECTED status.

    Raises:
        ApprovalGovernanceError: If reviewer evidence or reason is missing.
        InvalidStatusTransitionError: If order is not in PENDING_APPROVAL status.
    """
    # 1. Inbound validation at workflow boundary
    validated_inbound = validate_inbound_po(po)

    # 2. Strict state transition check
    if validated_inbound.status != PurchaseOrderStatus.PENDING_APPROVAL:
        raise InvalidStatusTransitionError(
            f"Invalid status transition: Cannot reject Purchase Order '{validated_inbound.po_id}' "
            f"in status '{validated_inbound.status.value}'. Only PENDING_APPROVAL orders can be rejected."
        )

    # 3. Mandatory human rejection evidence validation
    clean_reviewer = _validate_non_empty_str(rejected_by, "rejected_by")
    clean_timestamp = validate_and_normalize_iso_timestamp(
        rejection_timestamp,
        field_name="rejection_timestamp",
        error_cls=ApprovalGovernanceError,
    )
    clean_reason = _validate_non_empty_str(rejection_reason, "rejection_reason")

    # 4. Construct immutable audit record
    audit_entry = PurchaseOrderAuditRecord(
        po_id=validated_inbound.po_id,
        previous_status=validated_inbound.status,
        new_status=PurchaseOrderStatus.REJECTED,
        reviewer_id=clean_reviewer,
        timestamp=clean_timestamp,
        action=ApprovalAction.REJECT,
        reason=clean_reason,
        requires_escalation=validated_inbound.requires_escalation,
        escalation_reasons=list(validated_inbound.escalation_reasons),
    )

    # 5. Full state reconstruction through Pydantic validation (NO model_copy bypass)
    data = validated_inbound.model_dump()
    data["status"] = PurchaseOrderStatus.REJECTED.value
    data["rejected_by"] = clean_reviewer
    data["rejection_timestamp"] = clean_timestamp
    data["rejection_reason"] = clean_reason
    data["audit_trail"] = [rec.model_dump() for rec in validated_inbound.audit_trail] + [audit_entry.model_dump()]

    try:
        return PurchaseOrder.model_validate(data)
    except ValidationError as err:
        raise ApprovalGovernanceError(
            f"Rejection Transition Failed: Reconstructed Purchase Order '{validated_inbound.po_id}' "
            f"failed model validation: {err}"
        ) from err


def verify_approval_evidence(po: PurchaseOrder) -> bool:
    """
    Verifies that a Purchase Order contains valid, verifiable human approval evidence.
    Required before any future dispatch or issuance operation.

    Fails closed (returns False) if:
    - Status is not APPROVED
    - approved_by is missing or empty
    - approval_timestamp is missing, empty, or not a valid ISO 8601 timestamp
    - audit_trail lacks an exact matching APPROVE audit record
    - audit_trail contains duplicate or conflicting terminal decision records
    - Any audit record attribute (po_id, previous_status, new_status, reviewer_id, timestamp) mismatches

    NOTE: Reviewer ID integrity is verified against audit trail evidence; human identity
    authentication remains an application-layer responsibility.

    Returns:
        True if order is APPROVED with verified matching audit evidence; False otherwise.
    """
    if po.status != PurchaseOrderStatus.APPROVED:
        return False
    if not po.approved_by or not po.approved_by.strip():
        return False
    if not po.approval_timestamp or not po.approval_timestamp.strip():
        return False
    try:
        clean_ts = validate_and_normalize_iso_timestamp(po.approval_timestamp, "approval_timestamp")
    except Exception:
        return False

    terminal_records = [
        rec for rec in po.audit_trail
        if rec.action in [ApprovalAction.APPROVE, ApprovalAction.REJECT]
    ]
    if len(terminal_records) != 1:
        return False

    audit = terminal_records[0]
    if audit.action != ApprovalAction.APPROVE:
        return False
    if audit.po_id != po.po_id:
        return False
    if audit.previous_status != PurchaseOrderStatus.PENDING_APPROVAL:
        return False
    if audit.new_status != PurchaseOrderStatus.APPROVED:
        return False
    if audit.reviewer_id != po.approved_by:
        return False
    if audit.timestamp != clean_ts or audit.timestamp != po.approval_timestamp:
        return False

    return True


def is_issued_or_dispatched(po: PurchaseOrder) -> bool:
    """
    Checks whether a Purchase Order has been issued or dispatched.
    Confirms the strict decoupling between approval and issuance:
    An APPROVED order is NOT issued or dispatched.
    """
    return po.status == PurchaseOrderStatus.ISSUED


def can_be_dispatched(po: PurchaseOrder) -> bool:
    """
    Pre-condition check for any future dispatch or issuance operation:
    Guarantees that no order may be dispatched without valid human approval evidence.
    """
    return verify_approval_evidence(po) and (po.status == PurchaseOrderStatus.APPROVED)


class PurchaseOrderApprovalWorkflow:
    """
    Domain service orchestrating human review, audit trail tracking, and validated
    state transitions for Purchase Orders.
    """

    def __init__(self, config: Optional[Lead2Config] = None):
        self.config = config or get_default_config()

    def validate_inbound_po(self, po: PurchaseOrder) -> PurchaseOrder:
        """Validates incoming PO at workflow boundary."""
        return validate_inbound_po(po)

    def evaluate_escalation(self, po: PurchaseOrder) -> Tuple[bool, List[str]]:
        """Evaluates escalation review policies."""
        return evaluate_escalation_policies(po, self.config)

    def approve(
        self,
        po: PurchaseOrder,
        approved_by: str,
        approval_timestamp: str,
        approval_reason: Optional[str] = None,
    ) -> PurchaseOrder:
        """Executes human approval transition (PENDING_APPROVAL -> APPROVED)."""
        return approve_purchase_order(
            po=po,
            approved_by=approved_by,
            approval_timestamp=approval_timestamp,
            approval_reason=approval_reason,
        )

    def reject(
        self,
        po: PurchaseOrder,
        rejected_by: str,
        rejection_timestamp: str,
        rejection_reason: str,
    ) -> PurchaseOrder:
        """Executes human rejection transition (PENDING_APPROVAL -> REJECTED)."""
        return reject_purchase_order(
            po=po,
            rejected_by=rejected_by,
            rejection_timestamp=rejection_timestamp,
            rejection_reason=rejection_reason,
        )

    def is_approved(self, po: PurchaseOrder) -> bool:
        """Checks if PO is validly approved."""
        return verify_approval_evidence(po)

    def is_issued_or_dispatched(self, po: PurchaseOrder) -> bool:
        """Checks if PO has been issued or dispatched (strictly False for approved orders)."""
        return is_issued_or_dispatched(po)

    def can_be_dispatched(self, po: PurchaseOrder) -> bool:
        """Pre-condition check ensuring valid approval evidence exists prior to dispatch."""
        return can_be_dispatched(po)
