"""
Lead 2 Coordinator Adapter: Circularity Prototype Adapter
=========================================================
Wraps the existing Step 4 prototype circularity logic to provide CIRCULARITY_DISPOSITION stage execution.

ARCHITECTURAL DISCLOSURE:
This adapter wraps the existing prototype implementation from Step 4 (MockRajatCircularityAgent /
overstock risk clearance formula). It is NOT an independent production-grade agent.
Production-grade circularity orchestration remains scheduled for future development.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import CoordinatorStage, TaskContext
from ..registry import CoordinatorAgent, TaskDelegationError


class CircularityAdapter:
    """
    Adapter wrapping Step 4 prototype circularity logic for CoordinatorStage.CIRCULARITY_DISPOSITION.
    
    IMPLEMENTATION NOTE:
    Clearly identified as a prototype wrapper around Step 4 math:
    - Overstock Risk Units: Omega_t = max(0.0, Closing_Stock - D_avg * Remaining_SL)
    - Dynamic Markdown Thresholds: 20% or 40% discount
    - Food Bank Rescue Eligibility
    """

    IMPLEMENTATION_STATUS: str = "PROTOTYPE_WRAPPER_STEP4"
    IS_PRODUCTION_GRADE: bool = False

    @property
    def agent_id(self) -> str:
        return "circularity_adapter"

    @property
    def agent_name(self) -> str:
        return "Circularity Prototype Adapter (Step 4 Wrapper)"

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        return (CoordinatorStage.CIRCULARITY_DISPOSITION,)

    def can_handle(self, task: TaskContext) -> bool:
        return task.stage == CoordinatorStage.CIRCULARITY_DISPOSITION

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates that necessary inventory and demand velocity fields exist."""
        payload = task.input_payload
        required = ["store_id", "sku_id"]
        for f in required:
            if f not in payload:
                raise ValueError(f"Task '{task.task_id}' missing required field '{f}' in input payload.")

        has_stock = "closing_stock" in payload or "on_hand_stock" in payload
        has_demand = "p50_demand" in payload or "average_daily_demand" in payload
        if not has_stock or not has_demand:
            raise ValueError(
                f"Task '{task.task_id}' requires stock and demand velocity fields for circularity evaluation."
            )
        return True

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Executes prototype overstock risk calculation and dynamic markdown routing.
        """
        payload = task.input_payload
        store_id = payload["store_id"]
        sku_id = payload["sku_id"]
        date = payload.get("operating_date") or payload.get("date")

        # Resolve stock and demand velocity
        closing_stock = float(payload.get("closing_stock", payload.get("on_hand_stock", 0.0)))
        d_avg = float(payload.get("p50_demand", payload.get("average_daily_demand", 1.0)))
        remaining_sl = int(payload.get("remaining_shelf_life_days", 3))

        d_avg_safe = max(0.01, d_avg)
        remaining_sl_safe = max(0, remaining_sl)

        # Mathematical overstock risk from Step 4 Dossier:
        # Omega_t = max(0, Closing_Stock - D_avg * max(1, Remaining_SL))
        expected_sl_clearance = d_avg_safe * max(1, remaining_sl_safe)
        overstock_units = max(0.0, closing_stock - expected_sl_clearance)

        # Prototype Decision Rules from Step 4 / Step 5:
        if overstock_units > (d_avg_safe * 3.0) and remaining_sl_safe <= 1:
            discount = 50.0
            action = "DIVERT_TO_FOOD_BANK"
            rescue = True
        elif overstock_units > 0 and remaining_sl_safe <= 1:
            discount = 40.0
            action = "DYNAMIC_MARKDOWN_40"
            rescue = True
        elif overstock_units > 0 and remaining_sl_safe <= 2:
            discount = 20.0
            action = "DYNAMIC_MARKDOWN_20"
            rescue = False
        else:
            discount = 0.0
            action = "STANDARD_PRICING"
            rescue = False

        return {
            "store_id": store_id,
            "sku_id": sku_id,
            "date": date,
            "closing_stock": round(closing_stock, 2),
            "remaining_shelf_life_days": remaining_sl_safe,
            "overstock_risk_units": round(overstock_units, 2),
            "markdown_discount_pct": discount,
            "food_rescue_eligible": rescue,
            "circularity_action": action,
            "implementation_status": self.IMPLEMENTATION_STATUS,
            "is_production_grade": self.IS_PRODUCTION_GRADE,
        }

