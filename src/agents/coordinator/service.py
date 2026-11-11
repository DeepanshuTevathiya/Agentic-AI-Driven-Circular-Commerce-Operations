"""
Lead 2 Multi-Agent Coordinator Service
======================================
The central orchestration layer for circular commerce operations.
Coordinates existing agents through the DAG, manages durable human approval pauses,
delegates tasks through registered adapters, and safely resumes or recovers workflows.

GOVERNANCE & ARCHITECTURAL INVARIANTS:
1. NON-DUPLICATION: Delegates business logic to existing Lead 1 and Lead 2 agents/services.
2. HUMAN APPROVAL GOVERNANCE: The coordinator NEVER auto-approves PurchaseOrders.
   In the absence of explicit authorized human credentials, it safely pauses at HUMAN_APPROVAL_GATE.
3. PERSISTENCE & CONCURRENCY: Uses durable SQLite repository with optimistic concurrency control.
4. REPLAY & CONFLICT PROTECTION: Identical request_id + identical payload returns existing result;
   different payload raises IdempotencyConflictError.
5. FAILURE ISOLATION: A failure on one SKU or node does not crash independent DAG branches.
"""

from datetime import datetime as dt_datetime, timezone
import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .schemas import (
    CoordinatorAuditRecord,
    CoordinatorExecutionMode,
    CoordinatorRequest,
    CoordinatorResult,
    CoordinatorStage,
    TaskContext,
    TaskStatus,
    compute_payload_fingerprint,
    validate_and_normalize_iso_timestamp,
)
from .registry import (
    AgentRegistry,
    CoordinatorAgent,
    TaskDelegationError,
)
from .dag import (
    ExecutionDAG,
    build_replenishment_dag,
    make_idempotency_key,
    make_task_id,
)
from .repository import (
    CoordinatorPersistenceError,
    CoordinatorRepository,
    ExecutionNotFoundError,
    IdempotencyConflictError,
    StaleExecutionError,
    TaskNotFoundError,
)
from .adapters.data_loader_adapter import DataLoaderAdapter
from .adapters.inventory_adapter import InventoryEngineAdapter
from .adapters.procurement_adapter import ProcurementEngineAdapter
from .adapters.approval_adapter import ApprovalWorkflowAdapter
from .adapters.fulfilment_adapter import FulfilmentServiceAdapter
from .adapters.tracking_adapter import TrackingServiceAdapter
from .adapters.circularity_adapter import CircularityAdapter

from ..common.config import Lead2Config, get_default_config
from ..fulfilment.service import FulfilmentService, ShipmentTrackingService
from ..fulfilment.repository import FulfilmentRepository


class MultiAgentCoordinator:
    """
    Central Multi-Agent Coordinator service for circular commerce operations.
    """

    def __init__(
        self,
        config: Optional[Lead2Config] = None,
        registry: Optional[AgentRegistry] = None,
        repository: Optional[CoordinatorRepository] = None,
        max_workers: int = 4,
    ) -> None:
        self.config = config or get_default_config()
        self.max_workers = max(1, max_workers)
        self._owned_services: List[Any] = []
        self._owned_repository: bool = False

        if repository is not None:
            self.repository = repository
        else:
            self.repository = CoordinatorRepository(db_path=self.config.coordinator_db_path)
            self._owned_repository = True

        if registry is not None:
            self.registry = registry
        else:
            self.registry = self._create_default_registry()

        self._execution_operational_inputs: Dict[str, Dict[str, Any]] = {}

    def _create_default_registry(self) -> AgentRegistry:
        """Constructs and registers the standard production and prototype adapters."""
        registry = AgentRegistry()
        registry.register(DataLoaderAdapter(config=self.config))
        registry.register(InventoryEngineAdapter(config=self.config))
        registry.register(ProcurementEngineAdapter(config=self.config))
        registry.register(ApprovalWorkflowAdapter(config=self.config))

        fulfilment_svc = FulfilmentService(
            db_path=self.config.fulfilment_db_path,
        )
        self._owned_services.append(fulfilment_svc)
        registry.register(FulfilmentServiceAdapter(service=fulfilment_svc))

        tracking_svc = ShipmentTrackingService(
            repository=fulfilment_svc.repository,
        )
        registry.register(TrackingServiceAdapter(tracking_service=tracking_svc))
        registry.register(CircularityAdapter())
        return registry

    def close(self) -> None:
        """Closes all owned connections, services, and repositories."""
        for svc in self._owned_services:
            try:
                if hasattr(svc, "close"):
                    svc.close()
            except Exception:
                pass
        self._owned_services.clear()

        if self.repository is not None:
            try:
                self.repository.close()
            except Exception:
                pass

    def __enter__(self) -> "MultiAgentCoordinator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Audit Logging Helper
    # -------------------------------------------------------------------------

    def _log_audit(
        self,
        execution_id: str,
        stage: CoordinatorStage,
        action: str,
        actor_id: str = "COORDINATOR",
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> CoordinatorAuditRecord:
        """Constructs, persists, and returns an immutable audit record."""
        det = details or {}
        fp = compute_payload_fingerprint(det)
        now = dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        now_ns = time.time_ns()
        task_suffix = f":{task_id}" if task_id else ""
        event_id = f"EVT:{execution_id}:{action}{task_suffix}:{now_ns}"
        record = CoordinatorAuditRecord(
            execution_id=execution_id,
            event_id=event_id,
            timestamp=now,
            stage=stage,
            action=action,
            actor_id=actor_id,
            task_id=task_id,
            details=det,
            payload_fingerprint=fp,
        )
        self.repository.save_audit_record(record)
        return record

    # -------------------------------------------------------------------------
    # Lifecycle & Execution Operations
    # -------------------------------------------------------------------------

    def start_execution(
        self,
        request: CoordinatorRequest,
        operational_inputs: Optional[Dict[str, Any]] = None,
    ) -> CoordinatorResult:
        """
        Initiates a coordinator execution run from an operational request.

        If request_id already exists:
        - If payload matches exactly: returns existing execution result (idempotent replay).
        - If payload differs: raises IdempotencyConflictError.
        """
        start_mono = time.monotonic()
        exec_id = request.request_id

        # 1. Store operational inputs if provided
        op_inputs = dict(operational_inputs or {})
        if "operational_inputs" in request.metadata and isinstance(request.metadata["operational_inputs"], dict):
            op_inputs.update(request.metadata["operational_inputs"])
        self._execution_operational_inputs[exec_id] = op_inputs

        # 2. Idempotent check or register in repository
        existing_exec = self.repository.get_execution_by_request_id(request.request_id)
        if existing_exec is not None:
            payload_to_fingerprint = {
                "request": request.model_dump(),
                "operational_inputs": op_inputs,
            }
            incoming_fp = compute_payload_fingerprint(payload_to_fingerprint)
            if existing_exec["payload_fingerprint"] != incoming_fp:
                raise IdempotencyConflictError(
                    f"Execution with request_id '{request.request_id}' already exists with different payload."
                )
            # Replay existing execution result
            replayed_res = self.get_execution(existing_exec["execution_id"])
            if replayed_res is not None:
                return replayed_res

        # Register initial execution record
        self.repository.create_execution(
            request=request,
            execution_id=exec_id,
            initial_status=TaskStatus.RUNNING,
            initial_stage=CoordinatorStage.INGESTION,
            operational_inputs=op_inputs,
        )

        # 3. Build initial replenishment DAG
        dag = build_replenishment_dag(request, config=self.config)

        # Inject direct daily_agent_inputs if provided in operational_inputs
        if "daily_agent_inputs" in op_inputs:
            for task in dag.get_tasks_by_stage(CoordinatorStage.INGESTION):
                updated_input = dict(task.input_payload)
                updated_input["daily_agent_inputs"] = op_inputs["daily_agent_inputs"]
                updated_task = task.model_copy(update={"input_payload": updated_input})
                dag._tasks[task.task_id] = updated_task

        # 4. Save planned tasks
        self.repository.save_tasks(dag.get_all_tasks())

        # 5. Log audit event
        self._log_audit(
            execution_id=exec_id,
            stage=CoordinatorStage.INGESTION,
            action="EXECUTION_STARTED",
            details={
                "request_id": request.request_id,
                "operating_date": request.operating_date,
                "execution_mode": request.execution_mode.value,
                "initial_tasks_count": dag.task_count,
            },
        )

        # 6. Run DAG execution loop
        result = self._run_dag_loop(
            dag=dag,
            execution_id=exec_id,
            request=request,
            mode=request.execution_mode,
            start_mono=start_mono,
        )

        return result

    def get_execution(self, execution_id: str) -> Optional[CoordinatorResult]:
        """Retrieves and consolidates execution outcome from persistence."""
        exec_row = self.repository.get_execution(execution_id)
        if exec_row is None:
            return None

        if exec_row["result"] is not None:
            return exec_row["result"]

        # Reconstruct result if not yet finalized
        tasks = self.repository.get_tasks_for_execution(execution_id)
        audit_trail = self.repository.get_audit_records_for_execution(execution_id)
        return self._reconstruct_result(
            exec_row=exec_row,
            tasks=tasks,
            audit_trail=audit_trail,
            duration_ms=0.0,
        )

    def get_execution_tasks(self, execution_id: str) -> List[TaskContext]:
        """Returns all persisted tasks for the execution."""
        return self.repository.get_tasks_for_execution(execution_id)

    def resume_execution(
        self,
        execution_id: str,
        human_decisions: Optional[List[Dict[str, Any]]] = None,
        operational_inputs: Optional[Dict[str, Any]] = None,
    ) -> CoordinatorResult:
        """
        Resumes a paused (SUSPENDED) or interrupted execution.
        Applies verified human approval decisions to approval gate tasks.
        """
        start_mono = time.monotonic()
        exec_row = self.repository.get_execution(execution_id)
        if exec_row is None:
            raise ExecutionNotFoundError(f"Execution '{execution_id}' not found.")

        current_status = exec_row["status"]
        if current_status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            raise ValueError(f"Cannot resume execution '{execution_id}' in terminal state '{current_status.value}'.")

        # Update stored operational inputs
        if execution_id not in self._execution_operational_inputs:
            self._execution_operational_inputs[execution_id] = {}
        if operational_inputs:
            self._execution_operational_inputs[execution_id].update(operational_inputs)

        # Reconstruct DAG from stored tasks
        persisted_tasks = self.repository.get_tasks_for_execution(execution_id)
        dag = ExecutionDAG(execution_id=execution_id)
        for t in persisted_tasks:
            dag.add_task(t)

        # Apply human decisions if provided
        if human_decisions:
            self._apply_human_decisions(dag, execution_id, human_decisions)

        # Transition execution back to RUNNING
        self.repository.update_execution(
            execution_id=execution_id,
            status=TaskStatus.RUNNING,
            current_stage=exec_row["current_stage"],
            expected_version=exec_row["version"],
        )

        self._log_audit(
            execution_id=execution_id,
            stage=exec_row["current_stage"],
            action="EXECUTION_RESUMED",
            details={
                "has_human_decisions": bool(human_decisions),
                "decision_count": len(human_decisions) if human_decisions else 0,
            },
        )

        result = self._run_dag_loop(
            dag=dag,
            execution_id=execution_id,
            request=exec_row["request"],
            mode=exec_row["execution_mode"],
            start_mono=start_mono,
        )

        return result

    def retry_failed_tasks(
        self,
        execution_id: str,
        task_ids: Optional[Sequence[str]] = None,
        max_retries: int = 3,
    ) -> CoordinatorResult:
        """
        Retries tasks in FAILED status within retry limits and resumes DAG execution.
        """
        start_mono = time.monotonic()
        exec_row = self.repository.get_execution(execution_id)
        if exec_row is None:
            raise ExecutionNotFoundError(f"Execution '{execution_id}' not found.")

        persisted_tasks = self.repository.get_tasks_for_execution(execution_id)
        dag = ExecutionDAG(execution_id=execution_id)
        for t in persisted_tasks:
            dag.add_task(t)

        target_tasks = [
            t for t in dag.get_tasks_by_status(TaskStatus.FAILED)
            if (task_ids is None or t.task_id in task_ids)
        ]

        retried_count = 0
        now_utc = dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for task in target_tasks:
            if task.retry_count < max_retries:
                new_prov = dict(task.provenance)
                history = list(new_prov.get("attempt_history", []))
                history.append({
                    "attempt": task.retry_count + 1,
                    "failed_at": task.completed_at or now_utc,
                    "error_message": task.error_message,
                })
                new_prov["attempt_history"] = history

                updated = task.model_copy(
                    update={
                        "status": TaskStatus.PENDING,
                        "error_message": None,
                        "retry_count": task.retry_count + 1,
                        "provenance": new_prov,
                    }
                )
                dag._tasks[task.task_id] = updated
                self.repository.save_task(updated)
                retried_count += 1
                self._log_audit(
                    execution_id=execution_id,
                    stage=task.stage,
                    action="TASK_RETRY_SCHEDULED",
                    task_id=task.task_id,
                    details={
                        "retry_attempt": task.retry_count + 1,
                        "max_retries": max_retries,
                    },
                )

        if retried_count > 0:
            self.repository.update_execution(
                execution_id=execution_id,
                status=TaskStatus.RUNNING,
                current_stage=exec_row["current_stage"],
                expected_version=exec_row["version"],
            )

        result = self._run_dag_loop(
            dag=dag,
            execution_id=execution_id,
            request=exec_row["request"],
            mode=exec_row["execution_mode"],
            start_mono=start_mono,
        )

        return result

    def cancel_execution(
        self,
        execution_id: str,
        reason: str = "Cancelled by operator",
    ) -> CoordinatorResult:
        """
        Cancels an active or suspended execution and marks non-terminal tasks as CANCELLED.
        """
        exec_row = self.repository.get_execution(execution_id)
        if exec_row is None:
            raise ExecutionNotFoundError(f"Execution '{execution_id}' not found.")

        persisted_tasks = self.repository.get_tasks_for_execution(execution_id)
        dag = ExecutionDAG(execution_id=execution_id)
        for t in persisted_tasks:
            dag.add_task(t)

        for task in persisted_tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.SUSPENDED):
                updated = dag.transition_task(
                    task.task_id,
                    TaskStatus.CANCELLED,
                    error_message=reason,
                )
                self.repository.save_task(updated)

        self._log_audit(
            execution_id=execution_id,
            stage=exec_row["current_stage"],
            action="EXECUTION_CANCELLED",
            details={"reason": reason},
        )

        tasks = self.repository.get_tasks_for_execution(execution_id)
        audit_trail = self.repository.get_audit_records_for_execution(execution_id)

        result = self._reconstruct_result(
            exec_row=exec_row,
            tasks=tasks,
            audit_trail=audit_trail,
            duration_ms=0.0,
            override_status=TaskStatus.CANCELLED,
        )

        self.repository.update_execution(
            execution_id=execution_id,
            status=TaskStatus.CANCELLED,
            current_stage=exec_row["current_stage"],
            result=result,
            expected_version=exec_row["version"],
        )

        return result

    def recover_interrupted_executions(self) -> List[str]:
        """
        Discovers executions left in RUNNING status after an abrupt shutdown or crash,
        marks stuck tasks as FAILED, and updates execution state.
        """
        return self.repository.recover_interrupted_executions()

    # -------------------------------------------------------------------------
    # Internal DAG Execution Loop
    # -------------------------------------------------------------------------

    def _run_dag_loop(
        self,
        dag: ExecutionDAG,
        execution_id: str,
        request: CoordinatorRequest,
        mode: CoordinatorExecutionMode,
        start_mono: float,
    ) -> CoordinatorResult:
        """
        Executes ready tasks through topological levels until completion, suspension,
        or failure isolation barrier.
        """
        current_stage = CoordinatorStage.INGESTION

        while True:
            ready_tasks = dag.get_ready_tasks()

            if not ready_tasks:
                # No tasks ready. Determine termination condition.
                if dag.has_suspended():
                    overall_status = TaskStatus.SUSPENDED
                    suspended_tasks = dag.get_tasks_by_status(TaskStatus.SUSPENDED)
                    current_stage = suspended_tasks[0].stage if suspended_tasks else CoordinatorStage.HUMAN_APPROVAL_GATE
                elif dag.has_failures():
                    overall_status = TaskStatus.FAILED
                    failed_tasks = dag.get_tasks_by_status(TaskStatus.FAILED)
                    current_stage = failed_tasks[0].stage if failed_tasks else CoordinatorStage.COMPLETED
                elif dag.is_completed():
                    overall_status = TaskStatus.COMPLETED
                    current_stage = CoordinatorStage.COMPLETED
                else:
                    # All remaining non-completed tasks are blocked or skipped
                    overall_status = TaskStatus.COMPLETED
                    current_stage = CoordinatorStage.COMPLETED
                break

            # Filter ready tasks based on execution mode
            tasks_to_execute: List[TaskContext] = []
            for task in ready_tasks:
                if mode == CoordinatorExecutionMode.DRY_RUN:
                    if task.stage in (
                        CoordinatorStage.HUMAN_APPROVAL_GATE,
                        CoordinatorStage.FULFILMENT_TRANSMISSION,
                        CoordinatorStage.SHIPMENT_TRACKING,
                        CoordinatorStage.DOCK_RECEIVING,
                        CoordinatorStage.INVENTORY_POSTING,
                        CoordinatorStage.CIRCULARITY_DISPOSITION,
                    ):
                        dag.skip_task_and_descendants(task.task_id, reason="Omitted in DRY_RUN mode")
                        self.repository.save_task(dag.get_task(task.task_id))
                        continue

                tasks_to_execute.append(task)

            if not tasks_to_execute:
                # Re-check ready tasks after mode-based skips
                continue

            # Execute batch of ready tasks (with bounded concurrency when multiple tasks are ready)
            max_parallel = min(self.max_workers, len(tasks_to_execute))
            if max_parallel <= 1:
                # Sequential execution path (single task or max_workers=1)
                for task in tasks_to_execute:
                    current_stage = task.stage
                    running_task = dag.transition_task(task.task_id, TaskStatus.RUNNING)
                    self.repository.save_task(running_task)
                    self._log_audit(
                        execution_id=execution_id,
                        stage=running_task.stage,
                        action="TASK_STARTED",
                        task_id=running_task.task_id,
                    )

                    try:
                        output = self.registry.delegate_task(running_task)
                        if output.get("action") == "SUSPENDED" or output.get("status") == "SUSPENDED":
                            suspended_task = dag.transition_task(
                                running_task.task_id,
                                TaskStatus.SUSPENDED,
                                output_payload=output,
                            )
                            self.repository.save_task(suspended_task)
                            self._log_audit(
                                execution_id=execution_id,
                                stage=suspended_task.stage,
                                action="TASK_SUSPENDED",
                                task_id=suspended_task.task_id,
                                details=output,
                            )
                        else:
                            completed_task = dag.transition_task(
                                running_task.task_id,
                                TaskStatus.COMPLETED,
                                output_payload=output,
                            )
                            self.repository.save_task(completed_task)
                            self._log_audit(
                                execution_id=execution_id,
                                stage=completed_task.stage,
                                action="TASK_COMPLETED",
                                task_id=completed_task.task_id,
                                details={"output_keys": list(output.keys())},
                            )
                            self._handle_downstream_task_generation(dag, completed_task, execution_id)
                    except Exception as exc:
                        curr = dag.get_task(running_task.task_id)
                        if curr.status == TaskStatus.RUNNING:
                            failed_task = dag.transition_task(
                                running_task.task_id,
                                TaskStatus.FAILED,
                                error_message=str(exc),
                            )
                            self.repository.save_task(failed_task)
                        self._log_audit(
                            execution_id=execution_id,
                            stage=running_task.stage,
                            action="TASK_FAILED",
                            task_id=running_task.task_id,
                            details={"error": str(exc)},
                        )
            else:
                # Concurrent execution path across independent ready tasks
                running_tasks: List[TaskContext] = []
                for task in tasks_to_execute:
                    current_stage = task.stage
                    rt = dag.transition_task(task.task_id, TaskStatus.RUNNING)
                    self.repository.save_task(rt)
                    self._log_audit(
                        execution_id=execution_id,
                        stage=rt.stage,
                        action="TASK_STARTED",
                        task_id=rt.task_id,
                    )
                    running_tasks.append(rt)

                def _delegate_worker(t: TaskContext) -> Tuple[str, Optional[Dict[str, Any]], Optional[Exception]]:
                    try:
                        out = self.registry.delegate_task(t)
                        return (t.task_id, out, None)
                    except Exception as exc:
                        return (t.task_id, None, exc)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                    futures = [executor.submit(_delegate_worker, t) for t in running_tasks]
                    for future in concurrent.futures.as_completed(futures):
                        t_id, output, err = future.result()
                        task_item = dag.get_task(t_id)
                        if err is not None:
                            curr = dag.get_task(t_id)
                            if curr.status == TaskStatus.RUNNING:
                                failed_task = dag.transition_task(
                                    t_id,
                                    TaskStatus.FAILED,
                                    error_message=str(err),
                                )
                                self.repository.save_task(failed_task)
                            self._log_audit(
                                execution_id=execution_id,
                                stage=task_item.stage,
                                action="TASK_FAILED",
                                task_id=t_id,
                                details={"error": str(err)},
                            )
                        elif output is not None and (output.get("action") == "SUSPENDED" or output.get("status") == "SUSPENDED"):
                            suspended_task = dag.transition_task(
                                t_id,
                                TaskStatus.SUSPENDED,
                                output_payload=output,
                            )
                            self.repository.save_task(suspended_task)
                            self._log_audit(
                                execution_id=execution_id,
                                stage=suspended_task.stage,
                                action="TASK_SUSPENDED",
                                task_id=t_id,
                                details=output,
                            )
                        else:
                            completed_task = dag.transition_task(
                                t_id,
                                TaskStatus.COMPLETED,
                                output_payload=output or {},
                            )
                            self.repository.save_task(completed_task)
                            self._log_audit(
                                execution_id=execution_id,
                                stage=completed_task.stage,
                                action="TASK_COMPLETED",
                                task_id=t_id,
                                details={"output_keys": list((output or {}).keys())},
                            )
                            self._handle_downstream_task_generation(dag, completed_task, execution_id)

            # In STEP_BY_STEP mode: pause after processing current stage
            if mode == CoordinatorExecutionMode.STEP_BY_STEP:
                overall_status = TaskStatus.SUSPENDED
                break

        # Calculate final metrics and record result
        duration_ms = (time.monotonic() - start_mono) * 1000.0
        all_tasks = self.repository.get_tasks_for_execution(execution_id)
        audit_trail = self.repository.get_audit_records_for_execution(execution_id)

        latest_exec = self.repository.get_execution(execution_id)
        current_version = latest_exec["version"] if latest_exec else None

        result = self._reconstruct_result(
            exec_row=latest_exec,
            tasks=all_tasks,
            audit_trail=audit_trail,
            duration_ms=duration_ms,
            override_status=overall_status,
            override_stage=current_stage,
        )

        self.repository.update_execution(
            execution_id=execution_id,
            status=overall_status,
            current_stage=current_stage,
            result=result,
            expected_version=current_version,
        )

        return result

    # -------------------------------------------------------------------------
    # Dynamic Task Generation & Downstream Linking
    # -------------------------------------------------------------------------

    def _handle_downstream_task_generation(
        self,
        dag: ExecutionDAG,
        completed_task: TaskContext,
        execution_id: str,
    ) -> None:
        """
        Dynamically generates or links downstream DAG tasks upon task completion.
        Maintains parent-child provenance tuples.
        """
        output = completed_task.output_payload or {}
        op_inputs = self._execution_operational_inputs.get(execution_id, {})

        # 0. Post-Ingestion -> Inject DailyAgentInputs to SKU Inventory Evaluations
        if completed_task.stage == CoordinatorStage.INGESTION:
            store_id = completed_task.node_key
            daily_inputs = output.get("daily_agent_inputs", [])
            for inp_dict in daily_inputs:
                sku_id = inp_dict.get("sku_id") if isinstance(inp_dict, dict) else getattr(inp_dict, "sku_id", None)
                if not sku_id:
                    continue
                node_key = f"{store_id}:{sku_id}"
                eval_task_id = make_task_id(execution_id, CoordinatorStage.INVENTORY_EVALUATION, node_key)
                if dag.has_task(eval_task_id):
                    eval_task = dag.get_task(eval_task_id)
                    updated_input = dict(eval_task.input_payload)
                    updated_input["daily_agent_input"] = inp_dict
                    updated_task = eval_task.model_copy(update={"input_payload": updated_input})
                    dag._tasks[eval_task_id] = updated_task
                    self.repository.save_task(updated_task)

        # 0b. Post-Inventory Evaluation -> Accumulate Calculation Results in Store Procurement Planning
        elif completed_task.stage == CoordinatorStage.INVENTORY_EVALUATION:
            store_id = completed_task.input_payload.get("store_id")
            calc_res = output.get("calculation_result")
            if store_id and calc_res:
                proc_task_id = make_task_id(execution_id, CoordinatorStage.PROCUREMENT_PLANNING, store_id)
                if dag.has_task(proc_task_id):
                    proc_task = dag.get_task(proc_task_id)
                    updated_input = dict(proc_task.input_payload)
                    calcs = list(updated_input.get("calculation_results", []))
                    calcs.append(calc_res)
                    updated_input["calculation_results"] = calcs
                    updated_task = proc_task.model_copy(update={"input_payload": updated_input})
                    dag._tasks[proc_task_id] = updated_task
                    self.repository.save_task(updated_task)

        # 1. Post-Procurement -> Link Draft Purchase Orders to Approval Gate
        elif completed_task.stage == CoordinatorStage.PROCUREMENT_PLANNING:
            store_id = completed_task.node_key
            draft_pos = output.get("draft_pos") or output.get("purchase_orders", [])
            approval_task_id = make_task_id(execution_id, CoordinatorStage.HUMAN_APPROVAL_GATE, store_id)

            if not dag.has_task(approval_task_id):
                return

            approval_task = dag.get_task(approval_task_id)

            if not draft_pos:
                # No POs generated for this store: skip approval gate
                updated_app = dag.transition_task(
                    approval_task_id,
                    TaskStatus.SKIPPED,
                    error_message="No draft purchase orders generated for store.",
                )
                self.repository.save_task(updated_app)
            else:
                # Attach draft POs into approval task payload
                updated_input = dict(approval_task.input_payload)
                updated_input["purchase_orders"] = draft_pos
                updated_input["purchase_order"] = draft_pos[0]

                # Check if approval decisions were pre-supplied in operational_inputs
                pre_decisions = op_inputs.get("human_decisions", [])
                for d in pre_decisions:
                    if d.get("po_id") == draft_pos[0].get("po_id") or d.get("store_id") == store_id:
                        updated_input.update({
                            "approved_by": d.get("approved_by"),
                            "approval_timestamp": d.get("approval_timestamp"),
                            "approval_reason": d.get("approval_reason"),
                            "rejected_by": d.get("rejected_by"),
                            "rejection_timestamp": d.get("rejection_timestamp"),
                            "rejection_reason": d.get("rejection_reason"),
                        })
                        break

                updated_app = approval_task.model_copy(update={"input_payload": updated_input})
                dag._tasks[approval_task_id] = updated_app
                self.repository.save_task(updated_app)

        # 2. Post-Approval -> Generate Fulfilment Transmission for Approved POs
        elif completed_task.stage == CoordinatorStage.HUMAN_APPROVAL_GATE:
            if output.get("action") == "APPROVED":
                raw_po = output.get("purchase_order", {})
                po_id = output.get("po_id") or raw_po.get("po_id")
                if not po_id:
                    return

                trans_task_id = make_task_id(execution_id, CoordinatorStage.FULFILMENT_TRANSMISSION, po_id)
                if not dag.has_task(trans_task_id):
                    trans_input = {
                        "purchase_order": raw_po,
                        "fo_id": f"FO-{po_id.replace('PO-', '')}",
                        "actor_id": "COORDINATOR_TRANSMISSION",
                    }
                    if "override_scenario" in op_inputs:
                        trans_input["override_scenario"] = op_inputs["override_scenario"]
                    if "asn" in op_inputs:
                        trans_input["asn"] = op_inputs["asn"]
                    elif "grn" in op_inputs:
                        raw_grn = op_inputs["grn"]
                        shp_id = raw_grn.get("shipment_id") if isinstance(raw_grn, dict) else getattr(raw_grn, "shipment_id", None)
                        if shp_id:
                            supp_id = raw_po.get("supplier_id") if isinstance(raw_po, dict) else getattr(raw_po, "supplier_id", "SUPPLIER")
                            c_date = raw_po.get("created_date") if isinstance(raw_po, dict) else getattr(raw_po, "created_date", "2024-11-15")
                            lines_raw = raw_grn.get("inspection_lines", []) if isinstance(raw_grn, dict) else getattr(raw_grn, "inspection_lines", [])
                            trans_input["asn"] = {
                                "shipment_id": shp_id,
                                "fo_id": trans_input["fo_id"],
                                "po_id": po_id,
                                "supplier_id": supp_id,
                                "dispatch_at": f"{c_date}T11:00:00Z",
                                "expected_arrival_at": f"{c_date}T14:00:00Z",
                                "lines": [
                                    {
                                        "sku_id": line.get("sku_id") if isinstance(line, dict) else getattr(line, "sku_id"),
                                        "shipped_quantity": float(line.get("shipped_quantity", line.get("ordered_quantity", 1.0)) if isinstance(line, dict) else getattr(line, "shipped_quantity", getattr(line, "ordered_quantity", 1.0))),
                                    }
                                    for line in lines_raw
                                ],
                                "carrier": "BLR_COLD_EXPRESS",
                            }

                    trans_task = TaskContext(
                        task_id=trans_task_id,
                        execution_id=execution_id,
                        stage=CoordinatorStage.FULFILMENT_TRANSMISSION,
                        node_key=po_id,
                        status=TaskStatus.PENDING,
                        dependencies=(completed_task.task_id,),
                        parent_task_ids=(completed_task.task_id,),
                        idempotency_key=make_idempotency_key(execution_id, CoordinatorStage.FULFILMENT_TRANSMISSION, po_id),
                        input_payload=trans_input,
                    )
                    dag.add_task(trans_task)
                    self.repository.save_task(trans_task)

                    # Update completed task's child list
                    updated_completed = completed_task.model_copy(
                        update={"child_task_ids": completed_task.child_task_ids + (trans_task_id,)}
                    )
                    dag._tasks[completed_task.task_id] = updated_completed
                    self.repository.save_task(updated_completed)

        # 3. Post-Transmission -> Generate Shipment Tracking
        elif completed_task.stage == CoordinatorStage.FULFILMENT_TRANSMISSION:
            fo_id = output.get("fo_id")
            shipment_ids = output.get("shipment_ids", [])
            for shipment_id in shipment_ids:
                track_task_id = make_task_id(execution_id, CoordinatorStage.SHIPMENT_TRACKING, shipment_id)
                if not dag.has_task(track_task_id):
                    track_input = {
                        "shipment_id": shipment_id,
                        "fo_id": fo_id,
                        "carrier_id": op_inputs.get("carrier_id", "BLR_COLD_EXPRESS"),
                    }
                    if "tracking_scenario" in op_inputs:
                        track_input["scenario"] = op_inputs["tracking_scenario"]
                    if "target_tracking_status" in op_inputs:
                        track_input["target_status"] = op_inputs["target_tracking_status"]

                    track_task = TaskContext(
                        task_id=track_task_id,
                        execution_id=execution_id,
                        stage=CoordinatorStage.SHIPMENT_TRACKING,
                        node_key=shipment_id,
                        status=TaskStatus.PENDING,
                        dependencies=(completed_task.task_id,),
                        parent_task_ids=(completed_task.task_id,),
                        idempotency_key=make_idempotency_key(execution_id, CoordinatorStage.SHIPMENT_TRACKING, shipment_id),
                        input_payload=track_input,
                    )
                    dag.add_task(track_task)
                    self.repository.save_task(track_task)

        # 4. Post-Shipment Tracking -> Generate Dock Intake if GRN provided
        elif completed_task.stage == CoordinatorStage.SHIPMENT_TRACKING:
            shipment_id = completed_task.node_key
            grn_payload = op_inputs.get("grns", {}).get(shipment_id) or op_inputs.get("grn")
            fo_id = completed_task.input_payload.get("fo_id")

            if grn_payload and fo_id:
                grn_id = grn_payload.get("grn_id") if isinstance(grn_payload, dict) else getattr(grn_payload, "grn_id", "GRN-DEFAULT")
                dock_task_id = make_task_id(execution_id, CoordinatorStage.DOCK_RECEIVING, grn_id)

                if not dag.has_task(dock_task_id):
                    dock_task = TaskContext(
                        task_id=dock_task_id,
                        execution_id=execution_id,
                        stage=CoordinatorStage.DOCK_RECEIVING,
                        node_key=grn_id,
                        status=TaskStatus.PENDING,
                        dependencies=(completed_task.task_id,),
                        parent_task_ids=(completed_task.task_id,),
                        idempotency_key=make_idempotency_key(execution_id, CoordinatorStage.DOCK_RECEIVING, grn_id),
                        input_payload={
                            "fo_id": fo_id,
                            "grn": grn_payload,
                            "actor_id": op_inputs.get("dock_receiver_id", "STORE_RECEIVER"),
                            "auto_post_inventory": False,
                        },
                    )
                    dag.add_task(dock_task)
                    self.repository.save_task(dock_task)

        # 5. Post-Dock Intake -> Generate Inventory Ledger Posting for Accepted Stock
        elif completed_task.stage == CoordinatorStage.DOCK_RECEIVING:
            fo_id = output.get("fo_id")
            grn_id = output.get("grn_id")
            accepted_qty = float(output.get("total_accepted_quantity", 0.0))

            if fo_id and grn_id and accepted_qty > 0:
                post_task_id = make_task_id(execution_id, CoordinatorStage.INVENTORY_POSTING, grn_id)
                if not dag.has_task(post_task_id):
                    post_task = TaskContext(
                        task_id=post_task_id,
                        execution_id=execution_id,
                        stage=CoordinatorStage.INVENTORY_POSTING,
                        node_key=grn_id,
                        status=TaskStatus.PENDING,
                        dependencies=(completed_task.task_id,),
                        parent_task_ids=(completed_task.task_id,),
                        idempotency_key=make_idempotency_key(execution_id, CoordinatorStage.INVENTORY_POSTING, grn_id),
                        input_payload={
                            "fo_id": fo_id,
                            "grn_id": grn_id,
                            "actor_id": op_inputs.get("inventory_poster_id", "STORE_RECEIVER"),
                        },
                    )
                    dag.add_task(post_task)
                    self.repository.save_task(post_task)

        # 6. Post-Inventory Evaluation / Circularity Inputs -> Dynamic Circularity Disposition
        if op_inputs.get("evaluate_circularity"):
            for circ_item in op_inputs.get("circularity_items", []):
                store = circ_item.get("store_id")
                sku = circ_item.get("sku_id")
                node = f"{store}:{sku}"
                circ_task_id = make_task_id(execution_id, CoordinatorStage.CIRCULARITY_DISPOSITION, node)
                if not dag.has_task(circ_task_id):
                    circ_task = TaskContext(
                        task_id=circ_task_id,
                        execution_id=execution_id,
                        stage=CoordinatorStage.CIRCULARITY_DISPOSITION,
                        node_key=node,
                        status=TaskStatus.PENDING,
                        idempotency_key=make_idempotency_key(execution_id, CoordinatorStage.CIRCULARITY_DISPOSITION, node),
                        input_payload=circ_item,
                    )
                    dag.add_task(circ_task)
                    self.repository.save_task(circ_task)

    # -------------------------------------------------------------------------
    # Human Decisions Processor
    # -------------------------------------------------------------------------

    def _apply_human_decisions(
        self,
        dag: ExecutionDAG,
        execution_id: str,
        human_decisions: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Applies verified human review decisions to approval gate tasks in the DAG.
        Transitions suspended approval tasks back to PENDING.
        """
        suspended_approval_tasks = [
            t for t in dag.get_tasks_by_stage(CoordinatorStage.HUMAN_APPROVAL_GATE)
            if t.status in (TaskStatus.SUSPENDED, TaskStatus.PENDING)
        ]

        for decision in human_decisions:
            target_po_id = decision.get("po_id")
            target_store_id = decision.get("store_id")

            for app_task in suspended_approval_tasks:
                payload = app_task.input_payload
                po_match = False

                if target_po_id:
                    po_val = payload.get("purchase_order")
                    pos_val = payload.get("purchase_orders", [])
                    if isinstance(po_val, dict) and po_val.get("po_id") == target_po_id:
                        po_match = True
                    elif any(isinstance(p, dict) and p.get("po_id") == target_po_id for p in pos_val):
                        po_match = True

                store_match = bool(target_store_id and app_task.node_key == target_store_id)

                if po_match or store_match:
                    updated_input = dict(payload)
                    action = decision.get("action", "APPROVED")

                    if action == "APPROVED":
                        updated_input.update({
                            "approved_by": decision.get("approved_by"),
                            "approval_timestamp": decision.get("approval_timestamp"),
                            "approval_reason": decision.get("approval_reason"),
                        })
                    else:
                        updated_input.update({
                            "rejected_by": decision.get("rejected_by"),
                            "rejection_timestamp": decision.get("rejection_timestamp"),
                            "rejection_reason": decision.get("rejection_reason"),
                        })

                    # Transition task back to PENDING
                    updated_task = app_task.model_copy(
                        update={
                            "input_payload": updated_input,
                            "status": TaskStatus.PENDING,
                        }
                    )
                    dag._tasks[app_task.task_id] = updated_task
                    self.repository.save_task(updated_task)

                    self._log_audit(
                        execution_id=execution_id,
                        stage=CoordinatorStage.HUMAN_APPROVAL_GATE,
                        action="HUMAN_APPROVAL_DECISION_RECORDED",
                        task_id=app_task.task_id,
                        details={
                            "decision_action": action,
                            "po_id": target_po_id,
                            "actor": decision.get("approved_by") or decision.get("rejected_by"),
                        },
                    )

    # -------------------------------------------------------------------------
    # Result Consolidation
    # -------------------------------------------------------------------------

    def _reconstruct_result(
        self,
        exec_row: Optional[Dict[str, Any]],
        tasks: Sequence[TaskContext],
        audit_trail: Sequence[CoordinatorAuditRecord],
        duration_ms: float = 0.0,
        override_status: Optional[TaskStatus] = None,
        override_stage: Optional[CoordinatorStage] = None,
    ) -> CoordinatorResult:
        """Consolidates tasks and audit records into an authoritative CoordinatorResult."""
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        suspended_tasks = sum(1 for t in tasks if t.status == TaskStatus.SUSPENDED)
        skipped_tasks = sum(1 for t in tasks if t.status == TaskStatus.SKIPPED)
        cancelled_tasks = sum(1 for t in tasks if t.status == TaskStatus.CANCELLED)

        # Distinct SKUs evaluated
        evaluated_skus: Set[str] = set()
        for t in tasks:
            if t.stage == CoordinatorStage.INVENTORY_EVALUATION and t.status == TaskStatus.COMPLETED:
                sku = t.input_payload.get("sku_id")
                if sku:
                    evaluated_skus.add(sku)

        # Draft PO IDs
        draft_po_ids: List[str] = []
        for t in tasks:
            if t.stage == CoordinatorStage.PROCUREMENT_PLANNING and t.output_payload:
                for poid in t.output_payload.get("draft_po_ids", []):
                    if poid and poid not in draft_po_ids:
                        draft_po_ids.append(poid)
                for po in (t.output_payload.get("draft_pos") or t.output_payload.get("purchase_orders", [])):
                    poid = po.get("po_id") if isinstance(po, dict) else getattr(po, "po_id", None)
                    if poid and poid not in draft_po_ids:
                        draft_po_ids.append(poid)

        # Approved PO IDs
        approved_po_ids: List[str] = []
        for t in tasks:
            if t.stage == CoordinatorStage.HUMAN_APPROVAL_GATE and t.output_payload:
                if t.output_payload.get("action") == "APPROVED":
                    poid = t.output_payload.get("po_id")
                    if poid and poid not in approved_po_ids:
                        approved_po_ids.append(poid)

        # Transmitted FO IDs
        transmitted_fo_ids: List[str] = []
        for t in tasks:
            if t.stage == CoordinatorStage.FULFILMENT_TRANSMISSION and t.output_payload:
                foid = t.output_payload.get("fo_id")
                if foid and foid not in transmitted_fo_ids:
                    transmitted_fo_ids.append(foid)

        # GRN IDs
        grn_ids: List[str] = []
        for t in tasks:
            if t.stage == CoordinatorStage.DOCK_RECEIVING and t.output_payload:
                grnid = t.output_payload.get("grn_id")
                if grnid and grnid not in grn_ids:
                    grn_ids.append(grnid)

        # Posted inventory count
        posted_inventory_entry_count = 0
        for t in tasks:
            if t.stage == CoordinatorStage.INVENTORY_POSTING and t.output_payload:
                posted_inventory_entry_count += t.output_payload.get("posted_entries_count", 0)

        # Errors list
        errors: List[Dict[str, Any]] = []
        for t in tasks:
            if t.status == TaskStatus.FAILED and t.error_message:
                errors.append({
                    "task_id": t.task_id,
                    "stage": t.stage.value,
                    "node_key": t.node_key,
                    "error_message": t.error_message,
                })

        overall_status = override_status or (exec_row["status"] if exec_row else TaskStatus.COMPLETED)
        current_stage = override_stage or (exec_row["current_stage"] if exec_row else CoordinatorStage.COMPLETED)

        req_id = exec_row["request_id"] if exec_row else "REQ-UNKNOWN"
        exec_id = exec_row["execution_id"] if exec_row else "EXEC-UNKNOWN"
        op_date = exec_row["operating_date"] if exec_row else "2026-11-11"

        return CoordinatorResult(
            execution_id=exec_id,
            request_id=req_id,
            operating_date=op_date,
            overall_status=overall_status,
            current_stage=current_stage,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            suspended_tasks=suspended_tasks,
            skipped_tasks=skipped_tasks,
            cancelled_tasks=cancelled_tasks,
            evaluated_sku_count=len(evaluated_skus),
            draft_po_ids=tuple(draft_po_ids),
            approved_po_ids=tuple(approved_po_ids),
            transmitted_fo_ids=tuple(transmitted_fo_ids),
            grn_ids=tuple(grn_ids),
            posted_inventory_entry_count=posted_inventory_entry_count,
            execution_duration_ms=max(0.0, duration_ms),
            audit_trail=tuple(audit_trail),
            errors=tuple(errors),
        )
