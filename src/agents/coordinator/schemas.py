"""
Lead 2 Multi-Agent Coordinator Data Contracts & Pydantic V2 Schemas
===================================================================
Defines validated, immutable data schemas for the centralized Multi-Agent Coordinator:
- CoordinatorStage (standard operational lifecycle stages)
- TaskStatus (state machine for individual task lifecycle)
- CoordinatorExecutionMode (execution control: dry-run, approval-gate, full e2e, step-by-step)
- CoordinatorRequest (external operational trigger)
- TaskContext (unit of execution in the Coordinator DAG, with 1-to-N and N-to-1 provenance)
- CoordinatorAuditRecord (immutable event audit trail entry)
- CoordinatorResult (authoritative end-to-end outcome)

GOVERNANCE & ARCHITECTURAL INVARIANTS:
1. Strict Pydantic V2 validation with extra='forbid' and allow_inf_nan=False.
2. Mandatory deterministic task identifiers and scoped idempotency keys.
3. Provenance preservation for 1-to-N outputs and N-to-1 aggregation boundaries.
4. Reliable JSON round-trip serialization and deserialization.
"""

from datetime import date as dt_date, datetime as dt_datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..common.config import (
    CANONICAL_CATEGORIES,
    CANONICAL_SKUS,
    CANONICAL_STORES,
)


class CoordinatorStage(str, Enum):
    """
    Standard operational stages in the circular commerce pipeline.
    Ordered sequentially from upstream demand through circular disposition.
    """
    INGESTION = "INGESTION"                              # Load forecasts, opening inventory, alerts
    INVENTORY_EVALUATION = "INVENTORY_EVALUATION"        # Core inventory policy (SS, ROP, target stock) & risk
    PROCUREMENT_PLANNING = "PROCUREMENT_PLANNING"        # MOQ, case packs, supplier selection, draft POs
    HUMAN_APPROVAL_GATE = "HUMAN_APPROVAL_GATE"          # Mandatory human sign-off on draft POs
    FULFILMENT_TRANSMISSION = "FULFILMENT_TRANSMISSION"  # Gateway 2PC transmission to vendor
    SHIPMENT_TRACKING = "SHIPMENT_TRACKING"              # Carrier tracking projection & event ingestion
    DOCK_RECEIVING = "DOCK_RECEIVING"                    # Physical dock intake & GRN inspection
    INVENTORY_POSTING = "INVENTORY_POSTING"              # Accepted sellable stock posted to ledger
    CIRCULARITY_DISPOSITION = "CIRCULARITY_DISPOSITION"  # Dynamic markdown & waste diversion
    COMPLETED = "COMPLETED"                              # Terminal pipeline milestone


class TaskStatus(str, Enum):
    """Lifecycle state machine for individual tasks in the execution DAG."""
    PENDING = "PENDING"        # Prerequisite dependencies not yet satisfied or task not started
    RUNNING = "RUNNING"        # Currently executing
    SUSPENDED = "SUSPENDED"    # Paused awaiting external action (e.g. human approval signoff)
    COMPLETED = "COMPLETED"    # Successfully executed
    FAILED = "FAILED"          # Execution failed with an unhandled exception or error
    SKIPPED = "SKIPPED"        # Skipped because prerequisites or business conditions omitted it
    CANCELLED = "CANCELLED"    # Cancelled before or during execution


class CoordinatorExecutionMode(str, Enum):
    """Execution control mode for the coordinator."""
    DRY_RUN = "DRY_RUN"                              # Evaluate up through procurement without dispatch
    AUTOMATED_TO_APPROVAL = "AUTOMATED_TO_APPROVAL"  # Execute until human approval gate, then suspend
    FULL_END_TO_END = "FULL_END_TO_END"              # Full lifecycle (requires approval evidence provided)
    STEP_BY_STEP = "STEP_BY_STEP"                    # Pause after each stage for inspection


# Strict ISO 8601 regex pattern
ISO8601_STRICT_PATTERN = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"[Tt](?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?"
    r"(?:[Zz]|[+-](?:(?:0\d|1[0-3]):[0-5]\d|14:00))$"
)


def _validate_non_empty_str(v: str, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


def _normalize_date_str(v: Union[str, dt_date, dt_datetime]) -> str:
    """Helper to canonicalize dates to YYYY-MM-DD string format."""
    if isinstance(v, (dt_datetime, dt_date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, str):
        cleaned = v.strip()
        try:
            parsed = dt_datetime.strptime(cleaned[:10], "%Y-%m-%d").date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError as err:
            raise ValueError(f"Invalid date string '{v}'. Expected YYYY-MM-DD format.") from err
    raise ValueError(f"Unsupported date type: {type(v)}")


def validate_and_normalize_iso_timestamp(
    v: Any,
    field_name: str = "timestamp",
    error_cls: type = ValueError,
) -> str:
    """Validates timezone-aware ISO 8601 timestamp and normalizes to UTC ending with 'Z'."""
    if v is None:
        raise error_cls(f"Field '{field_name}' cannot be None.")

    if isinstance(v, dt_datetime):
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise error_cls(f"Field '{field_name}' must be timezone-aware. Naive datetime is prohibited.")
        dt_utc = v.astimezone(timezone.utc)
        if dt_utc.microsecond == 0:
            return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    if not isinstance(v, str):
        raise error_cls(f"Field '{field_name}' must be a string or datetime, got {type(v).__name__}.")

    v_clean = v.strip()
    if not v_clean:
        raise error_cls(f"Field '{field_name}' cannot be empty or whitespace-only.")

    if not ISO8601_STRICT_PATTERN.match(v_clean):
        raise error_cls(
            f"Field '{field_name}' must be a valid timezone-aware ISO 8601 timestamp "
            f"(e.g. 'YYYY-MM-DDTHH:MM:SSZ' or '+HH:MM'). Got '{v}'."
        )

    try:
        dt = dt_datetime.fromisoformat(v_clean)
    except Exception as err:
        raise error_cls(f"Field '{field_name}' has invalid date or time values: '{v}'.") from err

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise error_cls(f"Field '{field_name}' must have an explicit timezone offset or UTC indicator: '{v}'.")

    dt_utc = dt.astimezone(timezone.utc)
    if dt_utc.microsecond == 0:
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_payload_fingerprint(payload: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 fingerprint for a canonicalized payload dictionary."""
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


class InvalidTaskStatusTransitionError(ValueError):
    """Raised when an illegal task status transition is attempted."""
    pass


class CoordinatorRequest(BaseModel):
    """
    Validated external business operational trigger received by the Multi-Agent Coordinator.
    Specifies operating date, optional store/SKU filters, and execution parameters.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    request_id: str = Field(..., description="Unique request identifier")
    operating_date: str = Field(..., description="Operating calculation date (YYYY-MM-DD)")
    target_stores: Optional[Tuple[str, ...]] = Field(default=None, description="Optional target dark store filter")
    target_skus: Optional[Tuple[str, ...]] = Field(default=None, description="Optional product SKU filter")
    execution_mode: CoordinatorExecutionMode = Field(
        default=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
        description="Execution boundary control mode"
    )
    initiator_id: str = Field(default="OPERATIONS_CONTROLLER", description="Actor or service initiating request")
    correlation_id: str = Field(..., description="Global correlation ID for end-to-end telemetry")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary execution metadata")

    @field_validator("request_id", "initiator_id", "correlation_id")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("operating_date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)

    @field_validator("target_stores")
    @classmethod
    def validate_stores(cls, v: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
        if v is None:
            return None
        cleaned = []
        for s in v:
            c = _validate_non_empty_str(s, "target_stores item")
            if c not in CANONICAL_STORES:
                raise ValueError(f"target_store '{c}' is not one of canonical stores: {CANONICAL_STORES}")
            cleaned.append(c)
        return tuple(cleaned)

    @field_validator("target_skus")
    @classmethod
    def validate_skus(cls, v: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
        if v is None:
            return None
        cleaned = []
        for s in v:
            c = _validate_non_empty_str(s, "target_skus item")
            if c not in CANONICAL_SKUS:
                raise ValueError(f"target_sku '{c}' is not one of canonical SKUs: {CANONICAL_SKUS}")
            cleaned.append(c)
        return tuple(cleaned)


class TaskContext(BaseModel):
    """
    Stateful execution context for an individual task node in the Coordinator DAG.
    Tracks execution status, prerequisite dependencies, inputs, outputs, error state,
    and 1-to-N / N-to-1 aggregation provenance.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    task_id: str = Field(..., description="Unique deterministic task identifier (e.g. TASK:REQ-1:INGESTION:DS_HSR)")
    execution_id: str = Field(..., description="Coordinator execution run identifier")
    stage: CoordinatorStage = Field(..., description="Lifecycle stage of this task")
    node_key: str = Field(..., description="Subject node identifier (e.g. 'DS_BLR_HSR:SKU_MILK' or 'PO-123')")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current execution status")

    dependencies: Tuple[str, ...] = Field(default_factory=tuple, description="Tuple of prerequisite task_ids")
    input_payload: Dict[str, Any] = Field(default_factory=dict, description="Input data payload provided to agent")
    output_payload: Optional[Dict[str, Any]] = Field(default=None, description="Output payload produced upon completion")
    error_message: Optional[str] = Field(default=None, description="Error detail if status == FAILED")
    retry_count: int = Field(default=0, ge=0, description="Number of execution attempts")
    idempotency_key: str = Field(..., description="Deterministic idempotency token for this task")

    started_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of task start")
    completed_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of task finish")

    # 1-to-N outputs and N-to-1 aggregation provenance tracking
    parent_task_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Direct upstream producer tasks")
    child_task_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Direct downstream derived tasks")
    source_node_keys: Tuple[str, ...] = Field(default_factory=tuple, description="Source entity keys aggregated into this task")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Detailed lineage metadata")

    @field_validator("task_id", "execution_id", "node_key", "idempotency_key")
    @classmethod
    def validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("started_at", "completed_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v: Any, info) -> Optional[str]:
        if v is None:
            return None
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

    def can_transition_to(self, new_status: TaskStatus) -> bool:
        """
        Validates whether a transition from current status to new_status is legal.
        """
        if self.status == new_status:
            return True  # Idempotent re-affirmation

        allowed_map: Dict[TaskStatus, Set[TaskStatus]] = {
            TaskStatus.PENDING: {TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.CANCELLED},
            TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SUSPENDED, TaskStatus.SKIPPED, TaskStatus.CANCELLED},
            TaskStatus.SUSPENDED: {TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.PENDING},
            TaskStatus.FAILED: {TaskStatus.RUNNING, TaskStatus.SKIPPED, TaskStatus.CANCELLED},  # Retry path
            TaskStatus.COMPLETED: {TaskStatus.RUNNING},  # Explicit restart/recompute path
            TaskStatus.SKIPPED: {TaskStatus.RUNNING, TaskStatus.PENDING},
            TaskStatus.CANCELLED: set(),  # Terminal state
        }
        return new_status in allowed_map.get(self.status, set())

    def with_transition(
        self,
        new_status: TaskStatus,
        output_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        timestamp: Optional[str] = None,
        child_task_ids: Optional[Sequence[str]] = None,
    ) -> "TaskContext":
        """
        Creates an updated immutable TaskContext reflecting a validated state transition.
        """
        if not self.can_transition_to(new_status):
            raise InvalidTaskStatusTransitionError(
                f"Illegal task transition for task '{self.task_id}': cannot move from "
                f"'{self.status.value}' to '{new_status.value}'."
            )

        now_utc = validate_and_normalize_iso_timestamp(
            timestamp or dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        started = self.started_at
        if new_status == TaskStatus.RUNNING and started is None:
            started = now_utc

        completed = self.completed_at
        if new_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED):
            completed = now_utc

        retries = self.retry_count
        if new_status == TaskStatus.RUNNING and self.status == TaskStatus.FAILED:
            retries += 1

        effective_output = output_payload if output_payload is not None else self.output_payload
        effective_error = error_message if error_message is not None else (None if new_status == TaskStatus.COMPLETED else self.error_message)

        updated_children = tuple(child_task_ids) if child_task_ids is not None else self.child_task_ids

        data = self.model_dump()
        data.update({
            "status": new_status,
            "started_at": started,
            "completed_at": completed,
            "retry_count": retries,
            "output_payload": effective_output,
            "error_message": effective_error,
            "child_task_ids": updated_children,
        })
        return TaskContext.model_validate(data)


class CoordinatorAuditRecord(BaseModel):
    """
    Immutable audit record documenting coordinator orchestration actions and lifecycle events.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    execution_id: str = Field(..., description="Unique coordinator execution run identifier")
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    stage: CoordinatorStage = Field(..., description="Pipeline stage associated with event")
    action: str = Field(..., description="Action name (e.g. TASK_STARTED, APPROVAL_SUSPENDED)")
    actor_id: str = Field(..., description="Entity or agent responsible for the action")
    task_id: Optional[str] = Field(default=None, description="Associated task identifier if applicable")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured event payload")
    payload_fingerprint: str = Field(..., description="SHA-256 fingerprint of event details")

    @field_validator("execution_id", "event_id", "action", "actor_id", "payload_fingerprint")
    @classmethod
    def validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_audit_timestamp(cls, v: Any) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name="timestamp", error_cls=ValueError)


class CoordinatorResult(BaseModel):
    """
    Authoritative end-to-end outcome delivered by the Multi-Agent Coordinator.
    Consolidates task execution statuses, generated business artifacts, and full audit history.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    execution_id: str = Field(..., description="Unique execution run identifier")
    request_id: str = Field(..., description="Originating request identifier")
    operating_date: str = Field(..., description="Operating calculation date (YYYY-MM-DD)")
    overall_status: TaskStatus = Field(..., description="Overall execution status")
    current_stage: CoordinatorStage = Field(..., description="Stage reached at completion or suspension")

    # Task metrics
    total_tasks: int = Field(..., ge=0, description="Total planned tasks in execution DAG")
    completed_tasks: int = Field(..., ge=0, description="Number of completed tasks")
    failed_tasks: int = Field(..., ge=0, description="Number of failed tasks")
    suspended_tasks: int = Field(..., ge=0, description="Number of suspended tasks")
    skipped_tasks: int = Field(..., ge=0, description="Number of skipped tasks")
    cancelled_tasks: int = Field(default=0, ge=0, description="Number of cancelled tasks")

    # Commercial and operational entity references
    evaluated_sku_count: int = Field(default=0, ge=0, description="Distinct SKUs evaluated")
    draft_po_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Draft PurchaseOrder IDs created")
    approved_po_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Approved PurchaseOrder IDs")
    transmitted_fo_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Transmitted FulfilmentOrder IDs")
    grn_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Goods Receipt Note IDs created")
    posted_inventory_entry_count: int = Field(default=0, ge=0, description="Ledger entries posted")

    # Performance & Diagnostics
    execution_duration_ms: float = Field(default=0.0, ge=0.0, description="Total execution wall-clock time in ms")
    audit_trail: Tuple[CoordinatorAuditRecord, ...] = Field(default_factory=tuple, description="Chronological audit records")
    errors: Tuple[Dict[str, Any], ...] = Field(default_factory=tuple, description="Detailed error records if any")

    @field_validator("execution_id", "request_id")
    @classmethod
    def validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("operating_date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)
