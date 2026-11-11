"""
Lead 2 Multi-Agent Coordinator DAG & Execution Planning
======================================================
Implements dependency-aware task graph construction, cycle detection, topological sorting,
ready-task discovery, and stage-by-stage planning for circular commerce operations.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .schemas import (
    CoordinatorExecutionMode,
    CoordinatorRequest,
    CoordinatorStage,
    TaskContext,
    TaskStatus,
)
from ..common.config import CANONICAL_SKUS, CANONICAL_STORES, Lead2Config, get_default_config


STAGE_EXECUTION_ORDER: Tuple[CoordinatorStage, ...] = (
    CoordinatorStage.INGESTION,
    CoordinatorStage.INVENTORY_EVALUATION,
    CoordinatorStage.PROCUREMENT_PLANNING,
    CoordinatorStage.HUMAN_APPROVAL_GATE,
    CoordinatorStage.FULFILMENT_TRANSMISSION,
    CoordinatorStage.SHIPMENT_TRACKING,
    CoordinatorStage.DOCK_RECEIVING,
    CoordinatorStage.INVENTORY_POSTING,
    CoordinatorStage.CIRCULARITY_DISPOSITION,
    CoordinatorStage.COMPLETED,
)


def make_task_id(execution_id: str, stage: Union[CoordinatorStage, str], node_key: str) -> str:
    """
    Generates a deterministic task identifier.
    Format: 'TASK:{execution_id}:{stage}:{node_key}'
    """
    stage_val = stage.value if isinstance(stage, CoordinatorStage) else str(stage).strip()
    clean_exec = execution_id.strip()
    clean_node = node_key.strip().replace(" ", "_")
    return f"TASK:{clean_exec}:{stage_val}:{clean_node}"


def make_idempotency_key(execution_id: str, stage: Union[CoordinatorStage, str], node_key: str) -> str:
    """
    Generates a deterministic idempotency token for an operational task.
    Format: 'COORD:{execution_id}:{stage}:{node_key}'
    """
    stage_val = stage.value if isinstance(stage, CoordinatorStage) else str(stage).strip()
    clean_exec = execution_id.strip()
    clean_node = node_key.strip().replace(" ", "_")
    return f"COORD:{clean_exec}:{stage_val}:{clean_node}"


class DAGError(Exception):
    """Base exception for DAG construction and validation errors."""
    pass


class DAGCycleError(DAGError):
    """Raised when a cyclical dependency is detected in the execution DAG."""
    pass


class MissingDependencyError(DAGError):
    """Raised when a task references a prerequisite dependency that does not exist in the DAG."""
    pass


class DuplicateTaskError(DAGError):
    """Raised when adding a task whose task_id already exists in the DAG."""
    pass


class TaskNotFoundError(DAGError):
    """Raised when querying a task_id that does not exist in the DAG."""
    pass


class ExecutionDAG:
    """
    Directed Acyclic Graph (DAG) managing tasks, prerequisites, execution statuses,
    and stage-based topological ordering.
    """

    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id.strip()
        self._tasks: Dict[str, TaskContext] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # task_id -> set of task_ids it depends on
        self._dependents: Dict[str, Set[str]] = defaultdict(set)    # task_id -> set of task_ids depending on it

    @property
    def task_count(self) -> int:
        """Returns the total number of tasks in the DAG."""
        return len(self._tasks)

    def add_task(self, task: TaskContext) -> None:
        """
        Adds a TaskContext to the DAG.
        
        Raises:
            DuplicateTaskError: If task.task_id already exists.
        """
        tid = task.task_id
        if tid in self._tasks:
            raise DuplicateTaskError(f"Task with ID '{tid}' already exists in DAG.")

        self._tasks[tid] = task
        for dep in task.dependencies:
            self._dependencies[tid].add(dep)
            self._dependents[dep].add(tid)

    def add_dependency(self, task_id: str, depends_on_task_id: str) -> None:
        """
        Adds a dependency edge from depends_on_task_id to task_id:
        task_id cannot execute until depends_on_task_id completes.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task '{task_id}' not found in DAG.")
        if depends_on_task_id not in self._tasks:
            raise TaskNotFoundError(f"Dependency task '{depends_on_task_id}' not found in DAG.")

        self._dependencies[task_id].add(depends_on_task_id)
        self._dependents[depends_on_task_id].add(task_id)

        # Update immutable TaskContext dependency tuple
        current_task = self._tasks[task_id]
        if depends_on_task_id not in current_task.dependencies:
            new_deps = current_task.dependencies + (depends_on_task_id,)
            self._tasks[task_id] = current_task.model_copy(update={"dependencies": new_deps})

    def has_task(self, task_id: str) -> bool:
        """Returns True if task_id exists in the DAG."""
        return task_id in self._tasks

    def get_task(self, task_id: str) -> TaskContext:
        """
        Retrieves a TaskContext by task_id.
        
        Raises:
            TaskNotFoundError: If task_id does not exist.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task '{task_id}' not found in DAG.")
        return self._tasks[task_id]

    def get_all_tasks(self) -> List[TaskContext]:
        """Returns all TaskContext instances in the DAG."""
        return list(self._tasks.values())

    def validate(self) -> None:
        """
        Validates the DAG integrity:
        1. Verifies that all referenced dependencies exist in the graph.
        2. Detects cycles using Kahn's algorithm.

        Raises:
            MissingDependencyError: If a task depends on a missing task ID.
            DAGCycleError: If a cyclical dependency is detected.
        """
        # 1. Missing dependency check
        for tid, deps in self._dependencies.items():
            for dep in deps:
                if dep not in self._tasks:
                    raise MissingDependencyError(
                        f"Task '{tid}' depends on '{dep}', which is not registered in the DAG."
                    )

        # 2. Cycle detection via Kahn's algorithm
        in_degree = {tid: len(self._dependencies[tid]) for tid in self._tasks}
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for dependent in self._dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if visited_count != len(self._tasks):
            unresolved = [tid for tid, deg in in_degree.items() if deg > 0]
            raise DAGCycleError(
                f"Cyclical dependency detected in DAG. Unresolvable tasks: {unresolved}"
            )

    def topological_sort(self) -> List[str]:
        """
        Returns a list of task IDs in a topologically valid execution order.
        Validates the DAG before sorting.
        """
        self.validate()

        in_degree = {tid: len(self._dependencies[tid]) for tid in self._tasks}
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for dependent in self._dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return order

    def get_ready_tasks(self, stage: Optional[CoordinatorStage] = None) -> List[TaskContext]:
        """
        Returns all tasks currently in status PENDING whose prerequisite dependencies
        are fully satisfied (all dependencies have status COMPLETED or SKIPPED).
        """
        ready: List[TaskContext] = []
        for tid, task in self._tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            if stage is not None and task.stage != stage:
                continue

            deps = self._dependencies[tid]
            deps_satisfied = True
            for dep_id in deps:
                dep_task = self._tasks.get(dep_id)
                if not dep_task or dep_task.status not in (TaskStatus.COMPLETED, TaskStatus.SKIPPED):
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append(task)

        return ready

    def get_parallel_ready_tasks(self, stage: Optional[CoordinatorStage] = None) -> List[TaskContext]:
        """
        Alias to get_ready_tasks representing tasks that can be executed concurrently
        in parallel without ordering conflicts.
        """
        return self.get_ready_tasks(stage=stage)

    def get_tasks_by_stage(self, stage: CoordinatorStage) -> List[TaskContext]:
        """Returns all tasks belonging to a specified stage."""
        return [t for t in self._tasks.values() if t.stage == stage]

    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskContext]:
        """Returns all tasks in a specified status."""
        return [t for t in self._tasks.values() if t.status == status]

    def transition_task(
        self,
        task_id: str,
        new_status: TaskStatus,
        output_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        timestamp: Optional[str] = None,
        child_task_ids: Optional[Sequence[str]] = None,
    ) -> TaskContext:
        """
        Applies a validated status transition to a task in the DAG.
        Updates internal task record and returns the updated immutable TaskContext.
        """
        current_task = self.get_task(task_id)
        updated_task = current_task.with_transition(
            new_status=new_status,
            output_payload=output_payload,
            error_message=error_message,
            timestamp=timestamp,
            child_task_ids=child_task_ids,
        )
        self._tasks[task_id] = updated_task
        return updated_task

    def skip_task_and_descendants(self, task_id: str, reason: str = "Prerequisites skipped") -> List[str]:
        """
        Recursively marks a task and all downstream dependent tasks as SKIPPED.
        Returns list of task IDs that were skipped.
        """
        skipped_ids: List[str] = []
        queue = deque([task_id])

        while queue:
            curr_id = queue.popleft()
            if curr_id not in self._tasks:
                continue
            curr_task = self._tasks[curr_id]
            if curr_task.status not in (TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.CANCELLED):
                self.transition_task(
                    curr_id,
                    TaskStatus.SKIPPED,
                    error_message=reason,
                )
                skipped_ids.append(curr_id)

            for child_id in self._dependents[curr_id]:
                queue.append(child_id)

        return skipped_ids

    def is_completed(self) -> bool:
        """
        Returns True if all non-skipped/non-cancelled tasks in the DAG are COMPLETED.
        """
        if not self._tasks:
            return False
        for task in self._tasks.values():
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.CANCELLED):
                return False
        return True

    def has_failures(self) -> bool:
        """Returns True if any task is in FAILED status."""
        return any(t.status == TaskStatus.FAILED for t in self._tasks.values())

    def has_suspended(self) -> bool:
        """Returns True if any task is in SUSPENDED status (e.g. human approval gate)."""
        return any(t.status == TaskStatus.SUSPENDED for t in self._tasks.values())


def build_replenishment_dag(
    request: CoordinatorRequest,
    config: Optional[Lead2Config] = None,
) -> ExecutionDAG:
    """
    Builds a structured replenishment ExecutionDAG from an operational CoordinatorRequest.
    
    Generates:
    1. INGESTION tasks: 1 per target store.
    2. INVENTORY_EVALUATION tasks: 1 per (store, SKU), depending on the store's ingestion task.
    3. PROCUREMENT_PLANNING task: 1 per store, aggregating all SKU inventory tasks for that store (N-to-1 provenance).
    4. HUMAN_APPROVAL_GATE task placeholder: 1 per store, depending on the procurement planning task.

    Downstream fulfilment, tracking, dock receiving, inventory posting, and circularity tasks
    can be dynamically instantiated or linked once procurement produces concrete PurchaseOrders.
    """
    cfg = config or get_default_config()
    dag = ExecutionDAG(execution_id=request.request_id)

    stores = list(request.target_stores) if request.target_stores else list(CANONICAL_STORES)
    skus = list(request.target_skus) if request.target_skus else list(CANONICAL_SKUS)

    for store_id in stores:
        # Stage 1: Ingestion task per store
        ingest_task_id = make_task_id(request.request_id, CoordinatorStage.INGESTION, store_id)
        ingest_task = TaskContext(
            task_id=ingest_task_id,
            execution_id=request.request_id,
            stage=CoordinatorStage.INGESTION,
            node_key=store_id,
            status=TaskStatus.PENDING,
            idempotency_key=make_idempotency_key(request.request_id, CoordinatorStage.INGESTION, store_id),
            input_payload={
                "store_id": store_id,
                "operating_date": request.operating_date,
                "target_skus": skus,
            },
        )
        dag.add_task(ingest_task)

        # Stage 2: Inventory evaluation tasks per (store, SKU)
        sku_eval_task_ids: List[str] = []
        for sku_id in skus:
            node_key = f"{store_id}:{sku_id}"
            eval_task_id = make_task_id(request.request_id, CoordinatorStage.INVENTORY_EVALUATION, node_key)
            eval_task = TaskContext(
                task_id=eval_task_id,
                execution_id=request.request_id,
                stage=CoordinatorStage.INVENTORY_EVALUATION,
                node_key=node_key,
                status=TaskStatus.PENDING,
                dependencies=(ingest_task_id,),
                parent_task_ids=(ingest_task_id,),
                idempotency_key=make_idempotency_key(request.request_id, CoordinatorStage.INVENTORY_EVALUATION, node_key),
                input_payload={
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "operating_date": request.operating_date,
                },
            )
            dag.add_task(eval_task)
            sku_eval_task_ids.append(eval_task_id)

        # Stage 3: Procurement planning task per store (aggregating all SKU evaluations)
        proc_task_id = make_task_id(request.request_id, CoordinatorStage.PROCUREMENT_PLANNING, store_id)
        proc_task = TaskContext(
            task_id=proc_task_id,
            execution_id=request.request_id,
            stage=CoordinatorStage.PROCUREMENT_PLANNING,
            node_key=store_id,
            status=TaskStatus.PENDING,
            dependencies=tuple(sku_eval_task_ids),
            parent_task_ids=tuple(sku_eval_task_ids),
            source_node_keys=tuple(f"{store_id}:{sku}" for sku in skus),
            idempotency_key=make_idempotency_key(request.request_id, CoordinatorStage.PROCUREMENT_PLANNING, store_id),
            input_payload={
                "store_id": store_id,
                "operating_date": request.operating_date,
            },
            provenance={
                "aggregation_type": "N_TO_1_STORE_PROCUREMENT",
                "evaluated_sku_count": len(skus),
            },
        )
        dag.add_task(proc_task)

        # Stage 4: Human Approval Gate placeholder task per store
        approval_task_id = make_task_id(request.request_id, CoordinatorStage.HUMAN_APPROVAL_GATE, store_id)
        approval_task = TaskContext(
            task_id=approval_task_id,
            execution_id=request.request_id,
            stage=CoordinatorStage.HUMAN_APPROVAL_GATE,
            node_key=store_id,
            status=TaskStatus.PENDING,
            dependencies=(proc_task_id,),
            parent_task_ids=(proc_task_id,),
            idempotency_key=make_idempotency_key(request.request_id, CoordinatorStage.HUMAN_APPROVAL_GATE, store_id),
            input_payload={
                "store_id": store_id,
                "operating_date": request.operating_date,
            },
        )
        dag.add_task(approval_task)

    dag.validate()
    return dag

