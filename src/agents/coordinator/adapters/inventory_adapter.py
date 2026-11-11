"""
Lead 2 Coordinator Adapter: Inventory Engine
============================================
Wraps the existing InventoryEngine to provide standardized INVENTORY_EVALUATION stage execution.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import CoordinatorStage, TaskContext
from ..registry import CoordinatorAgent, TaskDelegationError
from ...inventory.engine import InventoryEngine
from ...common.schemas import DailyAgentInput, InventoryBatch, InventoryPosition
from ...common.config import Lead2Config, get_default_config


class InventoryEngineAdapter:
    """
    Adapter wrapping InventoryEngine for CoordinatorStage.INVENTORY_EVALUATION.
    Calculates dynamic safety stock, reorder point, target stock, and evaluates multi-factor risk.
    """

    def __init__(self, engine: Optional[InventoryEngine] = None, config: Optional[Lead2Config] = None) -> None:
        self.config = config or get_default_config()
        self.engine = engine or InventoryEngine(self.config)

    @property
    def agent_id(self) -> str:
        return "inventory_engine_adapter"

    @property
    def agent_name(self) -> str:
        return "Inventory Engine Adapter"

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        return (CoordinatorStage.INVENTORY_EVALUATION,)

    def can_handle(self, task: TaskContext) -> bool:
        return task.stage == CoordinatorStage.INVENTORY_EVALUATION

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates that daily_agent_input or necessary node parameters are present."""
        payload = task.input_payload
        if "daily_agent_input" not in payload and "forecast" not in payload:
            raise ValueError(
                f"Task '{task.task_id}' missing 'daily_agent_input' in input payload."
            )
        return True

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Executes deterministic inventory calculation and risk assessment for a single node.
        """
        payload = task.input_payload

        # Reconstruct DailyAgentInput
        if "daily_agent_input" in payload:
            raw_inp = payload["daily_agent_input"]
            daily_input = DailyAgentInput.model_validate(raw_inp) if isinstance(raw_inp, dict) else raw_inp
        else:
            daily_input = DailyAgentInput.model_validate(payload)

        # Optional InventoryPosition override
        inv_pos = None
        if "inventory_position" in payload and payload["inventory_position"]:
            raw_pos = payload["inventory_position"]
            inv_pos = InventoryPosition.model_validate(raw_pos) if isinstance(raw_pos, dict) else raw_pos

        # Optional active batches
        batches = None
        if "batches" in payload and payload["batches"]:
            raw_batches = payload["batches"]
            batches = [
                InventoryBatch.model_validate(b) if isinstance(b, dict) else b
                for b in raw_batches
            ]

        z_score = payload.get("z_score")

        try:
            calc_result, risk_assessment = self.engine.calculate_and_assess_risk(
                daily_input=daily_input,
                inventory_position=inv_pos,
                batches=batches,
                z_score=z_score,
            )
            return {
                "store_id": daily_input.store_id,
                "sku_id": daily_input.sku_id,
                "date": daily_input.date,
                "calculation_result": calc_result.model_dump(),
                "risk_assessment": risk_assessment.model_dump(),
                "reorder_triggered": calc_result.reorder_triggered,
                "suggested_replenishment_qty": calc_result.suggested_replenishment_qty,
            }
        except Exception as exc:
            raise TaskDelegationError(
                f"InventoryEngine calculation failed for task '{task.task_id}': {exc}"
            ) from exc

