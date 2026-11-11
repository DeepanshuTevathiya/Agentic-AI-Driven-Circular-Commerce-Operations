"""
Unit & Integration Tests: Multi-Agent Coordinator Core (Phase 6.2)
==================================================================
Comprehensive test suite verifying:
1. Pydantic V2 Coordinator contracts, validation, state transitions, and JSON round trips.
2. AgentRegistry capability discovery, duplicate protection, delegation, and error handling.
3. ExecutionDAG dependency validation, cycle detection, topological sorting, ready-task discovery,
   and deterministic task/idempotency key generation.
4. Multi-store/SKU task generation and N-to-1 procurement aggregation provenance.
5. Standard thin adapters:
   - DataLoaderAdapter
   - InventoryEngineAdapter
   - ProcurementEngineAdapter
   - ApprovalWorkflowAdapter (human approval governance)
   - FulfilmentServiceAdapter
   - TrackingServiceAdapter
   - CircularityAdapter (prototype wrapper disclosure)
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import unittest

from src.agents import (
    ApprovalWorkflowAdapter,
    CircularityAdapter,
    CoordinatorAuditRecord,
    CoordinatorExecutionMode,
    CoordinatorRequest,
    CoordinatorResult,
    CoordinatorStage,
    DataLoaderAdapter,
    ExecutionDAG,
    FulfilmentServiceAdapter,
    InventoryEngineAdapter,
    ProcurementEngineAdapter,
    TaskContext,
    TaskStatus,
    TrackingServiceAdapter,
    build_replenishment_dag,
    make_idempotency_key,
    make_task_id,
)
from src.agents.common.schemas import (
    DailyAgentInput,
    DemandForecast,
    InventoryPosition,
    InventorySnapshot,
    InventorySnapshotBasis,
)
from src.agents.coordinator.dag import (
    DAGCycleError,
    DuplicateTaskError,
    MissingDependencyError,
    TaskNotFoundError,
)
from src.agents.coordinator.registry import (
    AgentNotFoundError,
    AgentRegistry,
    AmbiguousAgentCapabilityError,
    CoordinatorAgent,
    DuplicateAgentRegistrationError,
    IncompatibleAgentCapabilityError,
    TaskDelegationError,
)
from src.agents.coordinator.schemas import (
    InvalidTaskStatusTransitionError,
    compute_payload_fingerprint,
    validate_and_normalize_iso_timestamp,
)
from src.agents.fulfilment.schemas import (
    AdvanceShipmentNotice,
    DockInspectionLine,
    FulfilmentBatch,
    GoodsReceiptNote,
    ShipmentLineItem,
)
from src.agents.fulfilment.service import FulfilmentService
from src.agents.fulfilment.tracking import ShipmentTrackingStatus, TrackingScenario
from src.agents.procurement.approval import approve_purchase_order
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
)


def _make_test_daily_input(
    store_id: str = "DS_BLR_HSR_LAYOUT",
    sku_id: str = "SKU_MILK_1L",
    date: str = "2024-11-15",
    p50: float = 30.0,
    on_hand: float = 15.0,
) -> DailyAgentInput:
    """Helper to construct valid DailyAgentInput."""
    return DailyAgentInput(
        store_id=store_id,
        sku_id=sku_id,
        category="Dairy",
        date=date,
        forecast=DemandForecast(
            store_id=store_id,
            sku_id=sku_id,
            category="Dairy",
            date=date,
            p10_demand=round(p50 * 0.7, 2),
            p50_demand=p50,
            p90_demand=round(p50 * 1.3, 2),
            sigma_demand=5.0,
        ),
        inventory=InventorySnapshot(
            store_id=store_id,
            sku_id=sku_id,
            date=date,
            on_hand_stock=on_hand,
            snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
        ),
    )


def _make_test_draft_po(
    po_id: str = "PO-20241115-DS_BLR_HSR_LAYOUT-SIM_SUPP_DAIRY_PRIMARY",
    store_id: str = "DS_BLR_HSR_LAYOUT",
    sku_id: str = "SKU_MILK_1L",
    qty: float = 24.0,
    unit_cost: float = 30.0,
) -> PurchaseOrder:
    """Helper to construct valid draft PurchaseOrder in PENDING_APPROVAL status."""
    item = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id=sku_id,
        category="Dairy",
        raw_suggested_qty=qty,
        case_pack_rounded_qty=qty,
        final_order_qty=qty,
        case_pack_size=12,
        moq=10.0,
        unit_retail_price=50.0,
        unit_procurement_cost=unit_cost,
        total_line_cost=round(qty * unit_cost, 2),
        requires_escalation=False,
        escalation_reasons=[],
    )
    return PurchaseOrder(
        po_id=po_id,
        created_date="2024-11-15",
        store_id=store_id,
        supplier_id="SIM_SUPP_DAIRY_PRIMARY",
        supplier_type=SupplierType.PRIMARY,
        items=[item],
        total_order_qty=qty,
        total_order_cost=round(qty * unit_cost, 2),
        urgency=PurchaseOrderUrgency.STANDARD,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        requires_escalation=False,
        escalation_reasons=[],
        approved_by=None,
        approval_timestamp=None,
    )


class TestCoordinatorContracts(unittest.TestCase):
    """Verifies Pydantic V2 Coordinator contracts, validation, and JSON round trips."""

    def test_coordinator_request_validation_and_round_trip(self):
        req = CoordinatorRequest(
            request_id="REQ-TEST-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT", "DS_BLR_KORAMANGALA"),
            target_skus=("SKU_MILK_1L", "SKU_CURD_400G"),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-REQ-001",
            metadata={"source": "unit_test"},
        )
        self.assertEqual(req.request_id, "REQ-TEST-001")
        self.assertEqual(req.operating_date, "2024-11-15")
        self.assertEqual(len(req.target_stores), 2)

        # JSON Round Trip
        json_str = req.model_dump_json()
        reconstructed = CoordinatorRequest.model_validate_json(json_str)
        self.assertEqual(req, reconstructed)

    def test_coordinator_request_invalid_inputs(self):
        # Invalid store
        with self.assertRaises(ValueError):
            CoordinatorRequest(
                request_id="REQ-BAD-STORE",
                operating_date="2024-11-15",
                target_stores=("INVALID_STORE",),
                correlation_id="CORR-1",
            )
        # Invalid date
        with self.assertRaises(ValueError):
            CoordinatorRequest(
                request_id="REQ-BAD-DATE",
                operating_date="2024-13-45",
                correlation_id="CORR-1",
            )

    def test_task_context_lifecycle_and_transitions(self):
        task = TaskContext(
            task_id="TASK-TEST-1",
            execution_id="EXEC-1",
            stage=CoordinatorStage.INGESTION,
            node_key="DS_BLR_HSR_LAYOUT",
            status=TaskStatus.PENDING,
            idempotency_key="COORD:EXEC-1:INGESTION:DS_BLR_HSR_LAYOUT",
            input_payload={"store_id": "DS_BLR_HSR_LAYOUT"},
        )
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.started_at)
        self.assertIsNone(task.completed_at)

        # Legal transition: PENDING -> RUNNING
        t_running = task.with_transition(TaskStatus.RUNNING, timestamp="2024-11-15T10:00:00Z")
        self.assertEqual(t_running.status, TaskStatus.RUNNING)
        self.assertEqual(t_running.started_at, "2024-11-15T10:00:00Z")
        self.assertIsNone(t_running.completed_at)

        # Legal transition: RUNNING -> COMPLETED
        t_completed = t_running.with_transition(
            TaskStatus.COMPLETED,
            output_payload={"count": 15},
            timestamp="2024-11-15T10:01:00Z",
        )
        self.assertEqual(t_completed.status, TaskStatus.COMPLETED)
        self.assertEqual(t_completed.completed_at, "2024-11-15T10:01:00Z")
        self.assertEqual(t_completed.output_payload, {"count": 15})

        # Illegal transition: COMPLETED -> FAILED
        with self.assertRaises(InvalidTaskStatusTransitionError):
            t_completed.with_transition(TaskStatus.FAILED)

        # Legal suspension: RUNNING -> SUSPENDED -> RUNNING
        t_run2 = task.with_transition(TaskStatus.RUNNING, timestamp="2024-11-15T10:00:00Z")
        t_suspended = t_run2.with_transition(TaskStatus.SUSPENDED)
        self.assertEqual(t_suspended.status, TaskStatus.SUSPENDED)
        t_resumed = t_suspended.with_transition(TaskStatus.RUNNING)
        self.assertEqual(t_resumed.status, TaskStatus.RUNNING)

        # Retry transition: RUNNING -> FAILED -> RUNNING increments retry_count
        t_failed = t_run2.with_transition(TaskStatus.FAILED, error_message="Network glitch")
        self.assertEqual(t_failed.retry_count, 0)
        self.assertEqual(t_failed.error_message, "Network glitch")
        t_retry = t_failed.with_transition(TaskStatus.RUNNING)
        self.assertEqual(t_retry.retry_count, 1)

    def test_task_context_provenance_and_round_trip(self):
        task = TaskContext(
            task_id="TASK-AGG-1",
            execution_id="EXEC-1",
            stage=CoordinatorStage.PROCUREMENT_PLANNING,
            node_key="DS_BLR_HSR_LAYOUT",
            status=TaskStatus.PENDING,
            dependencies=("DEP-1", "DEP-2"),
            parent_task_ids=("DEP-1", "DEP-2"),
            source_node_keys=("DS_BLR_HSR:SKU_A", "DS_BLR_HSR:SKU_B"),
            idempotency_key="COORD:EXEC-1:PROC:DS_HSR",
            provenance={"aggregation_ratio": "2_to_1"},
        )
        self.assertEqual(len(task.parent_task_ids), 2)
        self.assertEqual(len(task.source_node_keys), 2)

        # JSON round trip
        json_str = task.model_dump_json()
        reconstructed = TaskContext.model_validate_json(json_str)
        self.assertEqual(task, reconstructed)

    def test_coordinator_audit_record_and_result(self):
        audit = CoordinatorAuditRecord(
            execution_id="EXEC-1",
            event_id="EVT-1",
            timestamp="2024-11-15T10:00:00Z",
            stage=CoordinatorStage.INGESTION,
            action="INGESTION_COMPLETED",
            actor_id="COORDINATOR_TEST",
            details={"rows_loaded": 15},
            payload_fingerprint=compute_payload_fingerprint({"rows_loaded": 15}),
        )
        self.assertEqual(audit.action, "INGESTION_COMPLETED")

        res = CoordinatorResult(
            execution_id="EXEC-1",
            request_id="REQ-1",
            operating_date="2024-11-15",
            overall_status=TaskStatus.COMPLETED,
            current_stage=CoordinatorStage.COMPLETED,
            total_tasks=10,
            completed_tasks=10,
            failed_tasks=0,
            suspended_tasks=0,
            skipped_tasks=0,
            evaluated_sku_count=15,
            draft_po_ids=("PO-1", "PO-2"),
            audit_trail=(audit,),
        )
        self.assertEqual(res.overall_status, TaskStatus.COMPLETED)
        self.assertEqual(len(res.draft_po_ids), 2)

        # JSON round trip
        json_str = res.model_dump_json()
        rec_res = CoordinatorResult.model_validate_json(json_str)
        self.assertEqual(res, rec_res)


class TestAgentRegistry(unittest.TestCase):
    """Verifies AgentRegistry discovery, registration, duplicate checks, and delegation."""

    def setUp(self):
        self.registry = AgentRegistry()

    def test_agent_registration_and_discovery(self):
        adapter = InventoryEngineAdapter()
        self.registry.register(adapter)
        self.assertTrue(self.registry.has_agent(adapter.agent_id))

        retrieved = self.registry.get_agent(adapter.agent_id)
        self.assertEqual(retrieved.agent_id, adapter.agent_id)

        agents_for_stage = self.registry.get_agents_for_stage(CoordinatorStage.INVENTORY_EVALUATION)
        self.assertEqual(len(agents_for_stage), 1)
        self.assertEqual(agents_for_stage[0].agent_id, adapter.agent_id)

    def test_duplicate_registration_protection(self):
        adapter = InventoryEngineAdapter()
        self.registry.register(adapter)

        # Re-registering without override raises DuplicateAgentRegistrationError
        with self.assertRaises(DuplicateAgentRegistrationError):
            self.registry.register(adapter, override=False)

        # Re-registering with override succeeds
        self.registry.register(adapter, override=True)
        self.assertEqual(len(self.registry.list_agents()), 1)

    def test_incompatible_agent_protection(self):
        class NotAnAgent:
            pass

        with self.assertRaises(IncompatibleAgentCapabilityError):
            self.registry.register(NotAnAgent())  # type: ignore

    def test_find_agent_and_missing_agent_error(self):
        task = TaskContext(
            task_id="TASK-1",
            execution_id="EXEC-1",
            stage=CoordinatorStage.INGESTION,
            node_key="DS_HSR",
            idempotency_key="KEY-1",
        )
        with self.assertRaises(AgentNotFoundError):
            self.registry.find_agent_for_task(task)

        # Register DataLoaderAdapter
        loader_adapter = DataLoaderAdapter()
        self.registry.register(loader_adapter)
        found = self.registry.find_agent_for_task(task)
        self.assertEqual(found.agent_id, loader_adapter.agent_id)

    def test_ambiguous_agent_capability_error(self):
        class MockInventoryAgent1:
            agent_id = "mock_inv_1"
            agent_name = "Mock Inv 1"
            handled_stages = (CoordinatorStage.INVENTORY_EVALUATION,)
            def can_handle(self, task: TaskContext) -> bool: return True
            def validate_task_input(self, task: TaskContext) -> bool: return True
            def execute_task(self, task: TaskContext) -> Dict[str, Any]: return {}

        class MockInventoryAgent2:
            agent_id = "mock_inv_2"
            agent_name = "Mock Inv 2"
            handled_stages = (CoordinatorStage.INVENTORY_EVALUATION,)
            def can_handle(self, task: TaskContext) -> bool: return True
            def validate_task_input(self, task: TaskContext) -> bool: return True
            def execute_task(self, task: TaskContext) -> Dict[str, Any]: return {}

        self.registry.register(MockInventoryAgent1())
        self.registry.register(MockInventoryAgent2())

        task = TaskContext(
            task_id="TASK-INV",
            execution_id="EXEC-1",
            stage=CoordinatorStage.INVENTORY_EVALUATION,
            node_key="DS_HSR:SKU_MILK",
            idempotency_key="KEY-INV",
        )
        with self.assertRaises(AmbiguousAgentCapabilityError):
            self.registry.find_agent_for_task(task)

    def test_task_delegation_execution(self):
        inv_adapter = InventoryEngineAdapter()
        self.registry.register(inv_adapter)

        inp = _make_test_daily_input()
        task = TaskContext(
            task_id="TASK-INV-EXEC",
            execution_id="EXEC-1",
            stage=CoordinatorStage.INVENTORY_EVALUATION,
            node_key="DS_BLR_HSR_LAYOUT:SKU_MILK_1L",
            idempotency_key="KEY-EXEC",
            input_payload={"daily_agent_input": inp.model_dump()},
        )

        output = self.registry.delegate_task(task)
        self.assertIn("calculation_result", output)
        self.assertIn("risk_assessment", output)
        self.assertEqual(output["store_id"], "DS_BLR_HSR_LAYOUT")
        self.assertEqual(output["sku_id"], "SKU_MILK_1L")


class TestExecutionDAG(unittest.TestCase):
    """Verifies ExecutionDAG construction, cycle detection, topological sorting, and planning."""

    def test_dag_cycle_detection(self):
        dag = ExecutionDAG("EXEC-CYCLE")
        t1 = TaskContext(task_id="T1", execution_id="EXEC-CYCLE", stage=CoordinatorStage.INGESTION, node_key="N1", idempotency_key="K1", dependencies=("T2",))
        t2 = TaskContext(task_id="T2", execution_id="EXEC-CYCLE", stage=CoordinatorStage.INVENTORY_EVALUATION, node_key="N2", idempotency_key="K2", dependencies=("T1",))

        dag.add_task(t1)
        dag.add_task(t2)

        with self.assertRaises(DAGCycleError):
            dag.validate()

    def test_missing_dependency_detection(self):
        dag = ExecutionDAG("EXEC-MISSING")
        t1 = TaskContext(task_id="T1", execution_id="EXEC-MISSING", stage=CoordinatorStage.INGESTION, node_key="N1", idempotency_key="K1", dependencies=("NON_EXISTENT",))
        dag.add_task(t1)

        with self.assertRaises(MissingDependencyError):
            dag.validate()

    def test_topological_sort_and_ready_tasks(self):
        dag = ExecutionDAG("EXEC-SORT")
        t_ingest = TaskContext(task_id="T_INGEST", execution_id="EXEC-SORT", stage=CoordinatorStage.INGESTION, node_key="DS_HSR", idempotency_key="K1")
        t_inv1 = TaskContext(task_id="T_INV_1", execution_id="EXEC-SORT", stage=CoordinatorStage.INVENTORY_EVALUATION, node_key="DS_HSR:SKU_A", dependencies=("T_INGEST",), idempotency_key="K2")
        t_inv2 = TaskContext(task_id="T_INV_2", execution_id="EXEC-SORT", stage=CoordinatorStage.INVENTORY_EVALUATION, node_key="DS_HSR:SKU_B", dependencies=("T_INGEST",), idempotency_key="K3")
        t_proc = TaskContext(task_id="T_PROC", execution_id="EXEC-SORT", stage=CoordinatorStage.PROCUREMENT_PLANNING, node_key="DS_HSR", dependencies=("T_INV_1", "T_INV_2"), idempotency_key="K4")

        dag.add_task(t_ingest)
        dag.add_task(t_inv1)
        dag.add_task(t_inv2)
        dag.add_task(t_proc)

        # Topological order check
        order = dag.topological_sort()
        self.assertEqual(order[0], "T_INGEST")
        self.assertIn("T_INV_1", order[1:3])
        self.assertIn("T_INV_2", order[1:3])
        self.assertEqual(order[3], "T_PROC")

        # Initial ready tasks: only T_INGEST has no dependencies
        ready = dag.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "T_INGEST")

        # Complete T_INGEST -> both T_INV_1 and T_INV_2 become parallel ready!
        dag.transition_task("T_INGEST", TaskStatus.COMPLETED)
        parallel_ready = dag.get_parallel_ready_tasks()
        self.assertEqual(len(parallel_ready), 2)
        ready_ids = {t.task_id for t in parallel_ready}
        self.assertEqual(ready_ids, {"T_INV_1", "T_INV_2"})

        # Complete one inventory task -> T_PROC is NOT ready yet
        dag.transition_task("T_INV_1", TaskStatus.COMPLETED)
        self.assertEqual(len(dag.get_ready_tasks()), 1)
        self.assertEqual(dag.get_ready_tasks()[0].task_id, "T_INV_2")

        # Complete second inventory task -> T_PROC becomes ready
        dag.transition_task("T_INV_2", TaskStatus.COMPLETED)
        self.assertEqual(len(dag.get_ready_tasks()), 1)
        self.assertEqual(dag.get_ready_tasks()[0].task_id, "T_PROC")

    def test_skip_task_and_descendants(self):
        dag = ExecutionDAG("EXEC-SKIP")
        t1 = TaskContext(task_id="T1", execution_id="EXEC-SKIP", stage=CoordinatorStage.INGESTION, node_key="N1", idempotency_key="K1")
        t2 = TaskContext(task_id="T2", execution_id="EXEC-SKIP", stage=CoordinatorStage.INVENTORY_EVALUATION, node_key="N2", dependencies=("T1",), idempotency_key="K2")
        t3 = TaskContext(task_id="T3", execution_id="EXEC-SKIP", stage=CoordinatorStage.PROCUREMENT_PLANNING, node_key="N3", dependencies=("T2",), idempotency_key="K3")

        dag.add_task(t1)
        dag.add_task(t2)
        dag.add_task(t3)

        # Skip T1 -> cascades to T1, T2, T3
        skipped = dag.skip_task_and_descendants("T1", reason="No data available")
        self.assertEqual(len(skipped), 3)
        self.assertEqual(dag.get_task("T1").status, TaskStatus.SKIPPED)
        self.assertEqual(dag.get_task("T2").status, TaskStatus.SKIPPED)
        self.assertEqual(dag.get_task("T3").status, TaskStatus.SKIPPED)
        self.assertTrue(dag.is_completed())

    def test_build_replenishment_dag_multi_store_sku(self):
        req = CoordinatorRequest(
            request_id="REQ-DAG-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT", "DS_BLR_INDIRANAGAR"),
            target_skus=("SKU_MILK_1L", "SKU_CURD_400G"),
            correlation_id="CORR-DAG-001",
        )
        dag = build_replenishment_dag(req)

        # Per store: 1 INGESTION, 2 INVENTORY_EVALUATION, 1 PROCUREMENT_PLANNING, 1 HUMAN_APPROVAL_GATE = 5 tasks per store * 2 stores = 10 tasks
        self.assertEqual(dag.task_count, 10)

        # Verify topological sort passes without errors or cycles
        order = dag.topological_sort()
        self.assertEqual(len(order), 10)

        # Verify N-to-1 aggregation provenance on procurement task
        proc_hsr = dag.get_task(make_task_id(req.request_id, CoordinatorStage.PROCUREMENT_PLANNING, "DS_BLR_HSR_LAYOUT"))
        self.assertEqual(proc_hsr.provenance["aggregation_type"], "N_TO_1_STORE_PROCUREMENT")
        self.assertEqual(proc_hsr.provenance["evaluated_sku_count"], 2)
        self.assertEqual(len(proc_hsr.dependencies), 2)


class TestThinAdapters(unittest.TestCase):
    """Verifies thin agent adapter delegation and output structures."""

    def test_data_loader_adapter_direct_payload(self):
        adapter = DataLoaderAdapter()
        inp = _make_test_daily_input()
        task = TaskContext(
            task_id="TASK-LOAD",
            execution_id="EXEC-1",
            stage=CoordinatorStage.INGESTION,
            node_key="DS_BLR_HSR_LAYOUT",
            idempotency_key="KEY-LOAD",
            input_payload={
                "store_id": "DS_BLR_HSR_LAYOUT",
                "operating_date": "2024-11-15",
                "daily_agent_inputs": [inp.model_dump()],
            },
        )
        self.assertTrue(adapter.validate_task_input(task))
        output = adapter.execute_task(task)
        self.assertEqual(output["count"], 1)
        self.assertEqual(output["source"], "DIRECT_PAYLOAD")

    def test_inventory_engine_adapter_execution(self):
        adapter = InventoryEngineAdapter()
        inp = _make_test_daily_input(p50=40.0, on_hand=10.0)
        task = TaskContext(
            task_id="TASK-INV",
            execution_id="EXEC-1",
            stage=CoordinatorStage.INVENTORY_EVALUATION,
            node_key="DS_BLR_HSR_LAYOUT:SKU_MILK_1L",
            idempotency_key="KEY-INV",
            input_payload={"daily_agent_input": inp.model_dump()},
        )
        self.assertTrue(adapter.validate_task_input(task))
        output = adapter.execute_task(task)
        self.assertTrue(output["reorder_triggered"])
        self.assertGreater(output["suggested_replenishment_qty"], 0.0)

    def test_procurement_engine_adapter_execution(self):
        # 1. Run inventory to get calculation result
        inv_adapter = InventoryEngineAdapter()
        inp = _make_test_daily_input(p50=40.0, on_hand=10.0)
        t_inv = TaskContext(
            task_id="TASK-INV",
            execution_id="EXEC-1",
            stage=CoordinatorStage.INVENTORY_EVALUATION,
            node_key="DS_BLR_HSR_LAYOUT:SKU_MILK_1L",
            idempotency_key="KEY-INV",
            input_payload={"daily_agent_input": inp.model_dump()},
        )
        inv_out = inv_adapter.execute_task(t_inv)

        # 2. Run procurement adapter
        proc_adapter = ProcurementEngineAdapter()
        t_proc = TaskContext(
            task_id="TASK-PROC",
            execution_id="EXEC-1",
            stage=CoordinatorStage.PROCUREMENT_PLANNING,
            node_key="DS_BLR_HSR_LAYOUT",
            idempotency_key="KEY-PROC",
            input_payload={
                "operating_date": "2024-11-15",
                "calculation_results": [inv_out["calculation_result"]],
                "risk_assessments": [inv_out["risk_assessment"]],
            },
        )
        self.assertTrue(proc_adapter.validate_task_input(t_proc))
        proc_out = proc_adapter.execute_task(t_proc)
        self.assertEqual(proc_out["draft_po_count"], 1)
        self.assertEqual(len(proc_out["purchase_orders"]), 1)
        draft_po = proc_out["purchase_orders"][0]
        self.assertEqual(draft_po["status"], PurchaseOrderStatus.PENDING_APPROVAL.value)
        self.assertIsNone(draft_po["approved_by"])

    def test_approval_workflow_adapter_suspension_governance(self):
        """Verifies that in the absence of human credentials, approval adapter suspends task."""
        adapter = ApprovalWorkflowAdapter()
        draft_po = _make_test_draft_po()

        task = TaskContext(
            task_id="TASK-APPROVAL-SUSPEND",
            execution_id="EXEC-1",
            stage=CoordinatorStage.HUMAN_APPROVAL_GATE,
            node_key=draft_po.store_id,
            idempotency_key="KEY-APP-1",
            input_payload={"purchase_order": draft_po.model_dump()},
        )
        self.assertTrue(adapter.validate_task_input(task))
        output = adapter.execute_task(task)

        # Invariant: PO is NOT auto-approved; output action is SUSPENDED awaiting human signoff
        self.assertEqual(output["action"], "SUSPENDED")
        self.assertEqual(output["action_required"], "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(output["current_status"], PurchaseOrderStatus.PENDING_APPROVAL.value)

    def test_approval_workflow_adapter_human_decision_execution(self):
        """Verifies that when human approval credentials are provided, approval succeeds."""
        adapter = ApprovalWorkflowAdapter()
        draft_po = _make_test_draft_po()

        task = TaskContext(
            task_id="TASK-APPROVAL-PASS",
            execution_id="EXEC-1",
            stage=CoordinatorStage.HUMAN_APPROVAL_GATE,
            node_key=draft_po.store_id,
            idempotency_key="KEY-APP-2",
            input_payload={
                "purchase_order": draft_po.model_dump(),
                "approved_by": "OPERATIONS_DIRECTOR_01",
                "approval_timestamp": "2024-11-15T11:30:00Z",
                "approval_reason": "Verified replenishment budget for HSR Layout dairy items.",
            },
        )
        output = adapter.execute_task(task)
        self.assertEqual(output["action"], "APPROVED")
        self.assertEqual(output["status"], "APPROVED")
        self.assertEqual(output["approved_by"], "OPERATIONS_DIRECTOR_01")
        self.assertEqual(output["approval_timestamp"], "2024-11-15T11:30:00Z")

    def test_fulfilment_service_adapter_transmission(self):
        """Verifies FulfilmentServiceAdapter handles FULFILMENT_TRANSMISSION stage."""
        draft_po = _make_test_draft_po()
        approved_po = approve_purchase_order(
            po=draft_po,
            approved_by="LEAD_OPERATOR",
            approval_timestamp="2024-11-15T11:30:00Z",
            approval_reason="Approved for immediate vendor transmission",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fulfilment_test.db"
            adapter = FulfilmentServiceAdapter(db_path=db_path)
            try:
                task = TaskContext(
                    task_id="TASK-TX-1",
                    execution_id="EXEC-1",
                    stage=CoordinatorStage.FULFILMENT_TRANSMISSION,
                    node_key=approved_po.po_id,
                    idempotency_key="KEY-TX-1",
                    input_payload={"purchase_order": approved_po.model_dump()},
                )
                self.assertTrue(adapter.validate_task_input(task))
                output = adapter.execute_task(task)

                self.assertIn("fo_id", output)
                self.assertEqual(output["status"], "ACKNOWLEDGED")
                self.assertEqual(output["transmission_status"], "CONFIRMED_ACCEPTANCE")
                self.assertIsNotNone(output["supplier_reference"])
            finally:
                adapter.service.close()

    def test_tracking_service_adapter_execution(self):
        """Verifies TrackingServiceAdapter registers and advances tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fulfilment_tracking_test.db"
            service = FulfilmentService(db_path=db_path)
            try:
                # Setup FO with shipment dispatch
                draft_po = _make_test_draft_po()
                approved_po = approve_purchase_order(
                    po=draft_po,
                    approved_by="LEAD_OPERATOR",
                    approval_timestamp="2024-11-15T11:30:00Z",
                )
                fo = service.create_order(approved_po)
                fo, _ = service.transmit_purchase_order(
                    fo_id=fo.fo_id,
                    po=approved_po,
                    idempotency_key="TX-TEST-TRACK-1",
                    timestamp="2024-11-15T11:35:00Z",
                )
                asn = AdvanceShipmentNotice(
                    shipment_id="SH-TEST-1",
                    fo_id=fo.fo_id,
                    po_id=approved_po.po_id,
                    supplier_id=approved_po.supplier_id,
                    dispatch_at="2024-11-15T12:00:00Z",
                    expected_arrival_at="2024-11-15T16:00:00Z",
                    lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
                    carrier="BLR_COLD_EXPRESS",
                )
                fo = service.dispatch_shipment(fo.fo_id, asn=asn)

                # Test tracking task
                task = TaskContext(
                    task_id="TASK-TRACK-1",
                    execution_id="EXEC-1",
                    stage=CoordinatorStage.SHIPMENT_TRACKING,
                    node_key="SH-TEST-1",
                    idempotency_key="KEY-TRACK-1",
                    input_payload={
                        "fo_id": fo.fo_id,
                        "shipment_id": "SH-TEST-1",
                        "carrier_id": "BLR_COLD_EXPRESS",
                        "scenario": TrackingScenario.NORMAL_DELIVERY.value,
                        "target_status": ShipmentTrackingStatus.OUT_FOR_DELIVERY.value,
                    },
                )
                self.assertTrue(adapter := TrackingServiceAdapter(tracking_service=service.tracking_service))
                self.assertTrue(adapter.validate_task_input(task))
                output = adapter.execute_task(task)
                self.assertEqual(output["shipment_id"], "SH-TEST-1")
                self.assertEqual(output["current_status"], ShipmentTrackingStatus.OUT_FOR_DELIVERY.value)
            finally:
                service.close()

    def test_circularity_adapter_prototype_disclosure(self):
        """Verifies CircularityAdapter computes markdowns and explicitly identifies prototype status."""
        adapter = CircularityAdapter()
        self.assertEqual(adapter.IMPLEMENTATION_STATUS, "PROTOTYPE_WRAPPER_STEP4")
        self.assertFalse(adapter.IS_PRODUCTION_GRADE)

        # Moderate overstock near expiry: closing_stock=15, demand=5/day, remaining_sl=1 -> overstock=10 -> markdown 40%
        task = TaskContext(
            task_id="TASK-CIRC-1",
            execution_id="EXEC-1",
            stage=CoordinatorStage.CIRCULARITY_DISPOSITION,
            node_key="DS_BLR_HSR_LAYOUT:SKU_MILK_1L",
            idempotency_key="KEY-CIRC-1",
            input_payload={
                "store_id": "DS_BLR_HSR_LAYOUT",
                "sku_id": "SKU_MILK_1L",
                "closing_stock": 15.0,
                "p50_demand": 5.0,
                "remaining_shelf_life_days": 1,
            },
        )
        self.assertTrue(adapter.validate_task_input(task))
        output = adapter.execute_task(task)

        self.assertEqual(output["implementation_status"], "PROTOTYPE_WRAPPER_STEP4")
        self.assertFalse(output["is_production_grade"])
        self.assertGreater(output["overstock_risk_units"], 0.0)
        self.assertEqual(output["markdown_discount_pct"], 40.0)
        self.assertEqual(output["circularity_action"], "DYNAMIC_MARKDOWN_40")
        self.assertTrue(output["food_rescue_eligible"])

        # High overstock near expiry: closing_stock=50, demand=5/day, remaining_sl=1 -> overstock=45 (> 3*demand) -> divert to food bank
        task_high = task.model_copy(update={"input_payload": {**task.input_payload, "closing_stock": 50.0}})
        output_high = adapter.execute_task(task_high)
        self.assertEqual(output_high["markdown_discount_pct"], 50.0)
        self.assertEqual(output_high["circularity_action"], "DIVERT_TO_FOOD_BANK")
        self.assertTrue(output_high["food_rescue_eligible"])


if __name__ == "__main__":
    unittest.main()
