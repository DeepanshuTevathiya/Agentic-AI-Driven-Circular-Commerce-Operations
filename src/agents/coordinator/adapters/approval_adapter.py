"""
Lead 2 Coordinator Adapter: Purchase Order Approval Workflow
============================================================
Enforces strict Phase 4 human approval governance at CoordinatorStage.HUMAN_APPROVAL_GATE.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import CoordinatorStage, TaskContext
from ..registry import CoordinatorAgent, TaskDelegationError
from ...procurement.schemas import PurchaseOrder, PurchaseOrderStatus
from ...procurement.approval import (
    ApprovalGovernanceError,
    InvalidStatusTransitionError,
    approve_purchase_order,
    evaluate_escalation_policies,
    reject_purchase_order,
)
from ...common.config import Lead2Config, get_default_config


class ApprovalWorkflowAdapter:
    """
    Adapter enforcing mandatory human approval governance for CoordinatorStage.HUMAN_APPROVAL_GATE.

    CRITICAL GOVERNANCE INVARIANTS:
    1. The coordinator CANNOT approve PurchaseOrders automatically.
    2. In the absence of explicit human operator credentials ('approved_by' and 'approval_timestamp'),
       the adapter records the task as SUSPENDED awaiting human signoff.
    3. When authorized human review credentials are supplied, it delegates exclusively
       to approve_purchase_order() or reject_purchase_order().
    """

    def __init__(self, config: Optional[Lead2Config] = None) -> None:
        self.config = config or get_default_config()

    @property
    def agent_id(self) -> str:
        return "approval_workflow_adapter"

    @property
    def agent_name(self) -> str:
        return "Purchase Order Approval Workflow Adapter"

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        return (CoordinatorStage.HUMAN_APPROVAL_GATE,)

    def can_handle(self, task: TaskContext) -> bool:
        return task.stage == CoordinatorStage.HUMAN_APPROVAL_GATE

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates that a purchase_order is provided in input payload."""
        payload = task.input_payload
        if "purchase_order" not in payload and "purchase_orders" not in payload:
            raise ValueError(
                f"Task '{task.task_id}' missing 'purchase_order' in input payload."
            )
        return True

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Executes approval review.
        - If human approval credentials are provided: approves the PO.
        - If rejection credentials are provided: rejects the PO.
        - If no decision credentials provided: evaluates escalation policies and suspends.
        """
        payload = task.input_payload

        # Single PO or primary PO handling
        raw_po = payload.get("purchase_order")
        if raw_po is None and "purchase_orders" in payload and payload["purchase_orders"]:
            raw_po = payload["purchase_orders"][0]

        po = PurchaseOrder.model_validate(raw_po) if isinstance(raw_po, dict) else raw_po

        approved_by = payload.get("approved_by")
        approval_timestamp = payload.get("approval_timestamp")
        approval_reason = payload.get("approval_reason")

        rejected_by = payload.get("rejected_by")
        rejection_timestamp = payload.get("rejection_timestamp")
        rejection_reason = payload.get("rejection_reason")

        try:
            # 1. Approval Decision Path
            if approved_by and approval_timestamp:
                approved_po = approve_purchase_order(
                    po=po,
                    approved_by=approved_by,
                    approval_timestamp=approval_timestamp,
                    approval_reason=approval_reason,
                )
                return {
                    "action": "APPROVED",
                    "status": "APPROVED",
                    "po_id": approved_po.po_id,
                    "approved_by": approved_po.approved_by,
                    "approval_timestamp": approved_po.approval_timestamp,
                    "purchase_order": approved_po.model_dump(),
                }

            # 2. Rejection Decision Path
            if rejected_by and rejection_timestamp and rejection_reason:
                rejected_po = reject_purchase_order(
                    po=po,
                    rejected_by=rejected_by,
                    rejection_timestamp=rejection_timestamp,
                    rejection_reason=rejection_reason,
                )
                return {
                    "action": "REJECTED",
                    "status": "REJECTED",
                    "po_id": rejected_po.po_id,
                    "rejected_by": rejected_po.rejected_by,
                    "rejection_timestamp": rejected_po.rejection_timestamp,
                    "rejection_reason": rejected_po.rejection_reason,
                    "purchase_order": rejected_po.model_dump(),
                }

            # 3. Automated Check / Suspension Path (Mandatory Human Signoff Required)
            requires_esc, esc_reasons = evaluate_escalation_policies(po, self.config)
            return {
                "action": "SUSPENDED",
                "status": "SUSPENDED",
                "action_required": "HUMAN_APPROVAL_REQUIRED",
                "po_id": po.po_id,
                "current_status": po.status.value,
                "requires_escalation": requires_esc,
                "escalation_reasons": esc_reasons,
                "total_order_cost": po.total_order_cost,
                "total_order_qty": po.total_order_qty,
                "purchase_order": po.model_dump(),
            }

        except (ApprovalGovernanceError, InvalidStatusTransitionError) as gov_err:
            raise TaskDelegationError(
                f"Approval governance failure for task '{task.task_id}': {gov_err}"
            ) from gov_err
        except Exception as exc:
            raise TaskDelegationError(
                f"Unexpected approval workflow error for task '{task.task_id}': {exc}"
            ) from exc

