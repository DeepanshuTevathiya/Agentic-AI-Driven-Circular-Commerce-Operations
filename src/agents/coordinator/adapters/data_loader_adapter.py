"""
Lead 2 Coordinator Adapter: DataLoader
======================================
Wraps the existing DataLoader to provide standardized INGESTION stage execution.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from ..schemas import CoordinatorStage, TaskContext
from ..registry import CoordinatorAgent, TaskDelegationError
from ...common.data_loader import DataLoader
from ...common.schemas import DailyAgentInput
from ...common.config import Lead2Config, get_default_config


class DataLoaderAdapter:
    """
    Adapter wrapping DataLoader for CoordinatorStage.INGESTION.
    Loads and normalizes forecasts, beginning-of-day opening inventory, and diagnostic preorders.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None, config: Optional[Lead2Config] = None) -> None:
        self.config = config or get_default_config()
        self.data_loader = data_loader or DataLoader(self.config)

    @property
    def agent_id(self) -> str:
        return "data_loader_adapter"

    @property
    def agent_name(self) -> str:
        return "DataLoader Ingestion Adapter"

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        return (CoordinatorStage.INGESTION,)

    def can_handle(self, task: TaskContext) -> bool:
        return task.stage == CoordinatorStage.INGESTION

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates that operating_date and store_id are present in task input payload."""
        payload = task.input_payload
        if "operating_date" not in payload or not payload["operating_date"]:
            raise ValueError(f"Task '{task.task_id}' missing 'operating_date' in input payload.")
        if "store_id" not in payload and "target_stores" not in payload:
            raise ValueError(f"Task '{task.task_id}' missing 'store_id' or 'target_stores' in input payload.")
        return True

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Executes ingestion for the requested store(s) and operating date.
        If pre-computed daily_agent_inputs are supplied in input_payload, validates and returns them.
        Otherwise loads harmonized records from parquet datasets using DataLoader.
        """
        payload = task.input_payload
        op_date = payload["operating_date"]
        target_store = payload.get("store_id")
        target_skus = payload.get("target_skus")

        # Support direct input passing for isolated testing and downstream DAG chaining
        if "daily_agent_inputs" in payload and payload["daily_agent_inputs"]:
            raw_inputs = payload["daily_agent_inputs"]
            validated_inputs = []
            for item in raw_inputs:
                inp = DailyAgentInput.model_validate(item) if isinstance(item, dict) else item
                if target_store and inp.store_id != target_store:
                    continue
                if inp.date != op_date:
                    continue
                if target_skus and inp.sku_id not in target_skus:
                    continue
                validated_inputs.append(inp)

            return {
                "store_id": target_store,
                "operating_date": op_date,
                "daily_agent_inputs": [inp.model_dump() for inp in validated_inputs],
                "count": len(validated_inputs),
                "source": "DIRECT_PAYLOAD",
            }

        # Otherwise load from configured parquet files
        try:
            inputs, eval_recs, report = self.data_loader.load_daily_agent_inputs()
            filtered = [
                inp for inp in inputs
                if (not target_store or inp.store_id == target_store)
                and inp.date == op_date
                and (not target_skus or inp.sku_id in target_skus)
            ]
            return {
                "store_id": target_store,
                "operating_date": op_date,
                "daily_agent_inputs": [inp.model_dump() for inp in filtered],
                "count": len(filtered),
                "source": "PARQUET_INGESTION",
                "validation_passed": report.validation_passed,
            }
        except Exception as exc:
            raise TaskDelegationError(f"DataLoader failed to ingest records for task '{task.task_id}': {exc}") from exc

