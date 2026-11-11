"""
Lead 2 Coordinator Adapter: Procurement Engine
==============================================
Wraps the existing ProcurementEngine to provide standardized PROCUREMENT_PLANNING stage execution.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import CoordinatorStage, TaskContext
from ..registry import CoordinatorAgent, TaskDelegationError
from ...procurement.engine import ProcurementEngine
from ...common.schemas import InventoryCalculationResult, InventoryRiskAssessment
from ...common.config import Lead2Config, get_default_config


def _serialize_recommendation(rec: Any) -> Dict[str, Any]:
    """Helper to reliably serialize ProcurementRecommendation dataclass instances."""
    if is_dataclass(rec):
        d = asdict(rec)
        if "selected_supplier" in d:
            supp = d["selected_supplier"]
            if hasattr(supp, "model_dump"):
                d["selected_supplier"] = supp.model_dump()
            elif isinstance(supp, dict) and "supplier_type" in supp and hasattr(supp["supplier_type"], "value"):
                supp["supplier_type"] = supp["supplier_type"].value
        if "line_item" in d and d["line_item"] is not None:
            li = d["line_item"]
            if hasattr(li, "model_dump"):
                d["line_item"] = li.model_dump()
        return d
    if hasattr(rec, "model_dump"):
        return rec.model_dump()
    return dict(rec)


class ProcurementEngineAdapter:
    """
    Adapter wrapping ProcurementEngine for CoordinatorStage.PROCUREMENT_PLANNING.
    Applies MOQ, case-pack rounding, supplier selection, financial valuation,
    and clusters lines into draft Purchase Orders in status PENDING_APPROVAL.
    """

    def __init__(self, engine: Optional[ProcurementEngine] = None, config: Optional[Lead2Config] = None) -> None:
        self.config = config or get_default_config()
        self.engine = engine or ProcurementEngine(self.config)

    @property
    def agent_id(self) -> str:
        return "procurement_engine_adapter"

    @property
    def agent_name(self) -> str:
        return "Procurement Engine Adapter"

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        return (CoordinatorStage.PROCUREMENT_PLANNING,)

    def can_handle(self, task: TaskContext) -> bool:
        return task.stage == CoordinatorStage.PROCUREMENT_PLANNING

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates that calculation_results are provided in input payload."""
        payload = task.input_payload
        if "calculation_results" not in payload or not isinstance(payload["calculation_results"], list):
            raise ValueError(
                f"Task '{task.task_id}' missing 'calculation_results' list in input payload."
            )
        return True

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Executes procurement planning across a set of inventory calculation results.
        Aggregates items into draft Purchase Orders grouped by (store_id, supplier_id).
        """
        payload = task.input_payload
        raw_calcs = payload["calculation_results"]
        op_date = payload.get("operating_date")

        calc_results = [
            InventoryCalculationResult.model_validate(c) if isinstance(c, dict) else c
            for c in raw_calcs
        ]

        # Optional risk assessments
        risk_map = {}
        if "risk_assessments" in payload and payload["risk_assessments"]:
            raw_risks = payload["risk_assessments"]
            for r in raw_risks:
                risk_obj = InventoryRiskAssessment.model_validate(r) if isinstance(r, dict) else r
                key = (risk_obj.store_id, risk_obj.sku_id, risk_obj.date)
                risk_map[key] = risk_obj

        # Optional custom pricing catalog
        price_catalog = payload.get("price_catalog")

        try:
            purchase_orders, recommendations, audit_summary = self.engine.generate_purchase_orders(
                calculation_results=calc_results,
                price_catalog=price_catalog,
                risk_assessments=risk_map,
                date=op_date,
            )

            draft_po_ids = [po.po_id for po in purchase_orders]
            po_dicts = [po.model_dump() for po in purchase_orders]
            return {
                "store_id": task.node_key,
                "operating_date": op_date,
                "purchase_orders": po_dicts,
                "draft_pos": po_dicts,
                "recommendations": [_serialize_recommendation(rec) for rec in recommendations],
                "audit_summary": audit_summary,
                "draft_po_count": len(purchase_orders),
                "draft_po_ids": draft_po_ids,
                "evaluated_items_count": len(calc_results),
            }
        except Exception as exc:
            raise TaskDelegationError(
                f"Procurement planning failed for task '{task.task_id}': {exc}"
            ) from exc
