"""
Lead 2 Multi-Agent Coordinator Package
======================================
Core orchestration foundation for Lead 2 circular commerce operations:
- Schemas & Data Contracts (Pydantic V2)
- Agent Registry & Delegation Protocol
- Execution DAG & Planning Engine
- Standard Adapters wrapping existing Phase 1-5 services
"""

from .schemas import (
    CoordinatorStage,
    TaskStatus,
    CoordinatorExecutionMode,
    CoordinatorRequest,
    TaskContext,
    CoordinatorAuditRecord,
    CoordinatorResult,
    InvalidTaskStatusTransitionError,
    validate_and_normalize_iso_timestamp,
    compute_payload_fingerprint,
)

from .registry import (
    CoordinatorAgent,
    AgentRegistry,
    CoordinatorRegistryError,
    DuplicateAgentRegistrationError,
    AgentNotFoundError,
    AmbiguousAgentCapabilityError,
    IncompatibleAgentCapabilityError,
    TaskDelegationError,
)

from .dag import (
    STAGE_EXECUTION_ORDER,
    ExecutionDAG,
    DAGError,
    DAGCycleError,
    MissingDependencyError,
    DuplicateTaskError,
    TaskNotFoundError,
    make_task_id,
    make_idempotency_key,
    build_replenishment_dag,
)

from .adapters import (
    DataLoaderAdapter,
    InventoryEngineAdapter,
    ProcurementEngineAdapter,
    ApprovalWorkflowAdapter,
    FulfilmentServiceAdapter,
    TrackingServiceAdapter,
    CircularityAdapter,
)

from .repository import (
    CoordinatorRepository,
    CoordinatorPersistenceError,
    ExecutionNotFoundError,
    TaskNotFoundError as RepoTaskNotFoundError,
    StaleExecutionError,
    IdempotencyConflictError,
)

from .service import (
    MultiAgentCoordinator,
)

__all__ = [
    # Schemas
    "CoordinatorStage",
    "TaskStatus",
    "CoordinatorExecutionMode",
    "CoordinatorRequest",
    "TaskContext",
    "CoordinatorAuditRecord",
    "CoordinatorResult",
    "InvalidTaskStatusTransitionError",
    "validate_and_normalize_iso_timestamp",
    "compute_payload_fingerprint",
    # Registry
    "CoordinatorAgent",
    "AgentRegistry",
    "CoordinatorRegistryError",
    "DuplicateAgentRegistrationError",
    "AgentNotFoundError",
    "AmbiguousAgentCapabilityError",
    "IncompatibleAgentCapabilityError",
    "TaskDelegationError",
    # DAG
    "STAGE_EXECUTION_ORDER",
    "ExecutionDAG",
    "DAGError",
    "DAGCycleError",
    "MissingDependencyError",
    "DuplicateTaskError",
    "TaskNotFoundError",
    "make_task_id",
    "make_idempotency_key",
    "build_replenishment_dag",
    # Adapters
    "DataLoaderAdapter",
    "InventoryEngineAdapter",
    "ProcurementEngineAdapter",
    "ApprovalWorkflowAdapter",
    "FulfilmentServiceAdapter",
    "TrackingServiceAdapter",
    "CircularityAdapter",
    # Persistence & Service
    "CoordinatorRepository",
    "CoordinatorPersistenceError",
    "ExecutionNotFoundError",
    "StaleExecutionError",
    "IdempotencyConflictError",
    "MultiAgentCoordinator",
]


