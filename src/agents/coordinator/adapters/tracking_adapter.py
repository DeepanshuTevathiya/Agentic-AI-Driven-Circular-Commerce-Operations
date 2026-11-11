"""
Lead 2 Coordinator Adapter: Shipment Tracking Service
=====================================================
Wraps the existing ShipmentTrackingService to provide standardized SHIPMENT_TRACKING stage execution.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import CoordinatorStage, TaskContext
from ..registry import CoordinatorAgent, TaskDelegationError
from ...fulfilment.service import ShipmentTrackingService
from ...fulfilment.tracking import (
    ShipmentTrackingStatus,
    TrackingScenario,
)


class TrackingServiceAdapter:
    """
    Adapter wrapping ShipmentTrackingService for CoordinatorStage.SHIPMENT_TRACKING.
    Manages carrier registration, event ingestion, and simulated tracking advancement.
    """

    def __init__(self, tracking_service: Optional[ShipmentTrackingService] = None) -> None:
        self.tracking_service = tracking_service

    @property
    def agent_id(self) -> str:
        return "tracking_service_adapter"

    @property
    def agent_name(self) -> str:
        return "Shipment Tracking Service Adapter"

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        return (CoordinatorStage.SHIPMENT_TRACKING,)

    def can_handle(self, task: TaskContext) -> bool:
        return task.stage == CoordinatorStage.SHIPMENT_TRACKING

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates that shipment_id is provided in input payload."""
        payload = task.input_payload
        if "shipment_id" not in payload:
            raise ValueError(f"Task '{task.task_id}' missing 'shipment_id' in input payload.")
        return True

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Executes shipment tracking registration and/or status advancement.
        """
        if self.tracking_service is None:
            raise TaskDelegationError(
                f"ShipmentTrackingService is not initialized on adapter for task '{task.task_id}'."
            )

        payload = task.input_payload
        shipment_id = payload["shipment_id"]
        fo_id = payload.get("fo_id")
        carrier_id = payload.get("carrier_id", "BLR_COLD_EXPRESS")
        raw_scenario = payload.get("scenario")
        target_status_raw = payload.get("target_status")
        steps = payload.get("steps")

        scenario = TrackingScenario(raw_scenario) if isinstance(raw_scenario, str) else raw_scenario

        try:
            # Register if not already registered and fo_id is available
            existing = self.tracking_service.repository.get_tracking_record(shipment_id)
            if not existing and fo_id:
                record = self.tracking_service.register_shipment(
                    fo_id=fo_id,
                    shipment_id=shipment_id,
                    carrier_id=carrier_id,
                    scenario=scenario,
                )
            else:
                record = existing or self.tracking_service.repository.get_tracking_record_or_raise(shipment_id)

            # Advance tracking if requested
            if target_status_raw:
                target_status = (
                    ShipmentTrackingStatus(target_status_raw)
                    if isinstance(target_status_raw, str)
                    else target_status_raw
                )
                record = self.tracking_service.advance_to_tracking_status(
                    shipment_id=shipment_id,
                    target_status=target_status,
                )
            elif steps is not None and steps > 0:
                advanced = self.tracking_service.advance_shipment_tracking(
                    shipment_id=shipment_id,
                    steps=steps,
                )
                if advanced:
                    record = advanced

            return {
                "shipment_id": record.shipment_id,
                "carrier_id": record.carrier_id,
                "current_status": record.current_status.value,
                "event_count": record.event_count,
                "scenario": record.scenario,
                "updated_at": record.updated_at,
            }
        except Exception as exc:
            raise TaskDelegationError(
                f"ShipmentTrackingService failed for task '{task.task_id}': {exc}"
            ) from exc

