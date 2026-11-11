"""
Integration & Orchestration Tests: Multi-Agent Coordinator (Phase 6.3)
======================================================================
Comprehensive test suite verifying:
1. CoordinatorRepository SQLite durability, optimistic concurrency, conflict protection,
   idempotent replay, and crash/restart recovery.
2. Operational execution modes:
   - DRY_RUN
   - AUTOMATED_TO_APPROVAL
   - FULL_END_TO_END
   - STEP_BY_STEP
3. Mandatory Human Approval Governance:
   - Coordinator NEVER auto-approves PurchaseOrders.
   - Safe suspension at HUMAN_APPROVAL_GATE.
   - Resumption with verified human approval credentials (transmitting order).
   - Resumption with human rejection (clean pipeline termination).
4. End-to-end commercial lifecycle:
   - Ingestion -> Inventory Evaluation -> Procurement Planning -> Approval Gate
     -> Fulfilment Transmission -> Shipment Tracking -> Dock Receiving -> Inventory Posting
5. Dynamic task generation, parent-child provenance, and prototype circularity disposition.
6. Failure isolation, task retry, execution cancellation, and JSON serialization.
"""

from datetime import datetime as dt_datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional
import unittest

from src.agents import (
    ApprovalWorkflowAdapter,
    CircularityAdapter,
    CoordinatorAuditRecord,
    CoordinatorExecutionMode,
    CoordinatorRepository,
    CoordinatorRequest,
    CoordinatorResult,
    CoordinatorStage,
    DataLoaderAdapter,
    ExecutionDAG,
    FulfilmentServiceAdapter,
    IdempotencyConflictError,
    InventoryEngineAdapter,
    MultiAgentCoordinator,
    ProcurementEngineAdapter,
    StaleExecutionError,
    TaskContext,
    TaskStatus,
    TrackingServiceAdapter,
    build_replenishment_dag,
    make_idempotency_key,
    make_task_id,
)
from src.agents.common.config import Lead2Config, get_default_config
from src.agents.common.schemas import (
    DailyAgentInput,
    DemandForecast,
    InventoryPosition,
    InventorySnapshot,
    InventorySnapshotBasis,
)
from src.agents.coordinator.registry import AgentRegistry, TaskDelegationError
from src.agents.fulfilment.schemas import (
    AdvanceShipmentNotice,
    DockInspectionLine,
    FulfilmentOrder,
    GoodsReceiptNote,
)
from src.agents.fulfilment.service import FulfilmentService, ShipmentTrackingService
from src.agents.fulfilment.tracking import ShipmentTrackingStatus, TrackingScenario
from src.agents.procurement.approval import approve_purchase_order
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
)


def _make_daily_input(
    store_id: str = "DS_BLR_HSR_LAYOUT",
    sku_id: str = "SKU_MILK_1L",
    date: str = "2024-11-15",
    p50: float = 30.0,
    on_hand: float = 10.0,
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


class TestCoordinatorRepository(unittest.TestCase):
    """Verifies SQLite repository persistence, concurrency, and crash recovery."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_coordinator.db"
        self.repo = CoordinatorRepository(db_path=self.db_path)

    def tearDown(self) -> None:
        self.repo.close()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_create_and_get_execution(self) -> None:
        """Tests inserting and reading coordinator execution header."""
        req = CoordinatorRequest(
            request_id="REQ-TEST-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            correlation_id="CORR-001",
        )
        created = self.repo.create_execution(req)
        self.assertEqual(created["request_id"], "REQ-TEST-001")
        self.assertEqual(created["status"], TaskStatus.RUNNING)
        self.assertEqual(created["version"], 1)

        fetched = self.repo.get_execution("REQ-TEST-001")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["request_id"], "REQ-TEST-001")
        self.assertEqual(fetched["operating_date"], "2024-11-15")

    def test_idempotent_replay_and_conflict_detection(self) -> None:
        """Tests that identical request replays safely while conflicting payload raises IdempotencyConflictError."""
        req = CoordinatorRequest(
            request_id="REQ-IDEM-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            correlation_id="CORR-001",
        )
        first = self.repo.create_execution(req)
        second = self.repo.create_execution(req)
        self.assertEqual(first["execution_id"], second["execution_id"])

        conflicting_req = CoordinatorRequest(
            request_id="REQ-IDEM-001",
            operating_date="2024-11-16",  # Different operating date
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            correlation_id="CORR-001",
        )
        with self.assertRaises(IdempotencyConflictError):
            self.repo.create_execution(conflicting_req)

    def test_optimistic_concurrency_stale_update(self) -> None:
        """Tests that optimistic concurrency version check rejects stale updates."""
        req = CoordinatorRequest(
            request_id="REQ-CONCUR-001",
            operating_date="2024-11-15",
            correlation_id="CORR-001",
        )
        self.repo.create_execution(req)

        # Valid update at expected_version=1 -> increments to 2
        updated = self.repo.update_execution(
            execution_id="REQ-CONCUR-001",
            status=TaskStatus.SUSPENDED,
            current_stage=CoordinatorStage.HUMAN_APPROVAL_GATE,
            expected_version=1,
        )
        self.assertEqual(updated["version"], 2)
        self.assertEqual(updated["status"], TaskStatus.SUSPENDED)

        # Stale update with expected_version=1 must fail
        with self.assertRaises(StaleExecutionError):
            self.repo.update_execution(
                execution_id="REQ-CONCUR-001",
                status=TaskStatus.COMPLETED,
                current_stage=CoordinatorStage.COMPLETED,
                expected_version=1,
            )

    def test_task_persistence_and_lifecycle(self) -> None:
        """Tests inserting, querying, and updating TaskContext instances."""
        req = CoordinatorRequest(
            request_id="REQ-TASK-001",
            operating_date="2024-11-15",
            correlation_id="CORR-001",
        )
        self.repo.create_execution(req)

        task = TaskContext(
            task_id="TASK:REQ-TASK-001:INGESTION:DS_BLR_HSR_LAYOUT",
            execution_id="REQ-TASK-001",
            stage=CoordinatorStage.INGESTION,
            node_key="DS_BLR_HSR_LAYOUT",
            status=TaskStatus.PENDING,
            idempotency_key="COORD:REQ-TASK-001:INGESTION:DS_BLR_HSR_LAYOUT",
            input_payload={"store_id": "DS_BLR_HSR_LAYOUT"},
        )
        self.repo.save_task(task)

        retrieved = self.repo.get_task(task.task_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.status, TaskStatus.PENDING)

        # Transition task to RUNNING then COMPLETED
        running = retrieved.with_transition(TaskStatus.RUNNING)
        self.repo.save_task(running)

        completed = running.with_transition(TaskStatus.COMPLETED, output_payload={"loaded": True})
        self.repo.save_task(completed)

        final_task = self.repo.get_task(task.task_id)
        self.assertEqual(final_task.status, TaskStatus.COMPLETED)
        self.assertEqual(final_task.output_payload, {"loaded": True})

    def test_crash_recovery_resets_running_tasks(self) -> None:
        """Tests that recover_interrupted_executions safely recovers crashed runs."""
        req = CoordinatorRequest(
            request_id="REQ-CRASH-001",
            operating_date="2024-11-15",
            correlation_id="CORR-001",
        )
        self.repo.create_execution(req, initial_status=TaskStatus.RUNNING)

        task = TaskContext(
            task_id="TASK:REQ-CRASH-001:EVAL:DS_HSR:SKU_MILK",
            execution_id="REQ-CRASH-001",
            stage=CoordinatorStage.INVENTORY_EVALUATION,
            node_key="DS_HSR:SKU_MILK",
            status=TaskStatus.RUNNING,
            idempotency_key="KEY-CRASH-1",
        )
        self.repo.save_task(task)

        recovered_ids = self.repo.recover_interrupted_executions()
        self.assertIn("REQ-CRASH-001", recovered_ids)

        recovered_exec = self.repo.get_execution("REQ-CRASH-001")
        self.assertEqual(recovered_exec["status"], TaskStatus.FAILED)

        recovered_task = self.repo.get_task(task.task_id)
        self.assertEqual(recovered_task.status, TaskStatus.FAILED)
        self.assertIn("Interrupted", recovered_task.error_message or "")


class TestCoordinatorOrchestration(unittest.TestCase):
    """Verifies end-to-end multi-agent orchestration, modes, approval, and downstream execution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.coord_db = Path(self.temp_dir.name) / "coord.db"
        self.fulfilment_db = Path(self.temp_dir.name) / "fulfilment.db"

        self.config = Lead2Config(
            coordinator_db_path=self.coord_db,
            fulfilment_db_path=self.fulfilment_db,
        )

        self.coordinator = MultiAgentCoordinator(config=self.config)
        self.daily_input = _make_daily_input()

    def tearDown(self) -> None:
        if hasattr(self, "coordinator"):
            self.coordinator.close()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_dry_run_mode_omits_human_approval_and_downstream(self) -> None:
        """Verifies DRY_RUN evaluates through procurement planning and safely skips approval/downstream."""
        req = CoordinatorRequest(
            request_id="REQ-DRYRUN-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.DRY_RUN,
            correlation_id="CORR-DRYRUN-001",
        )

        result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )

        self.assertEqual(result.overall_status, TaskStatus.COMPLETED)
        self.assertIn(result.current_stage, (CoordinatorStage.COMPLETED, CoordinatorStage.PROCUREMENT_PLANNING))
        self.assertEqual(result.evaluated_sku_count, 1)
        self.assertGreaterEqual(len(result.draft_po_ids), 1)
        self.assertEqual(result.approved_po_ids, ())
        self.assertEqual(result.transmitted_fo_ids, ())

        # Verify HUMAN_APPROVAL_GATE task was marked SKIPPED
        tasks = self.coordinator.get_execution_tasks(req.request_id)
        approval_tasks = [t for t in tasks if t.stage == CoordinatorStage.HUMAN_APPROVAL_GATE]
        self.assertEqual(len(approval_tasks), 1)
        self.assertEqual(approval_tasks[0].status, TaskStatus.SKIPPED)

    def test_automated_to_approval_suspends_and_never_auto_approves(self) -> None:
        """Verifies strict governance: pipeline automatically suspends at HUMAN_APPROVAL_GATE."""
        req = CoordinatorRequest(
            request_id="REQ-AUTO-SUSP-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-AUTO-001",
        )

        result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )

        # Invariant: Must suspend at HUMAN_APPROVAL_GATE awaiting human credentials
        self.assertEqual(result.overall_status, TaskStatus.SUSPENDED)
        self.assertEqual(result.current_stage, CoordinatorStage.HUMAN_APPROVAL_GATE)
        self.assertEqual(result.suspended_tasks, 1)
        self.assertGreaterEqual(len(result.draft_po_ids), 1)
        self.assertEqual(result.approved_po_ids, ())

        # Verify task output contains mandatory escalation/suspension action
        tasks = self.coordinator.get_execution_tasks(req.request_id)
        approval_tasks = [t for t in tasks if t.stage == CoordinatorStage.HUMAN_APPROVAL_GATE]
        self.assertEqual(len(approval_tasks), 1)
        self.assertEqual(approval_tasks[0].status, TaskStatus.SUSPENDED)
        self.assertEqual(
            approval_tasks[0].output_payload.get("action_required"),
            "HUMAN_APPROVAL_REQUIRED",
        )

    def test_human_approval_resume_and_gateway_transmission(self) -> None:
        """Verifies resuming a suspended execution with valid human approval creates and transmits FO."""
        req = CoordinatorRequest(
            request_id="REQ-RESUME-APPROVE-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-RESUME-001",
        )

        initial_result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )
        self.assertEqual(initial_result.overall_status, TaskStatus.SUSPENDED)
        draft_po_id = initial_result.draft_po_ids[0]

        # Operator supplies verified human review credentials
        human_decisions = [
            {
                "po_id": draft_po_id,
                "action": "APPROVED",
                "approved_by": "OPERATOR_ALICE",
                "approval_timestamp": "2024-11-15T10:00:00Z",
                "approval_reason": "Verified replenishment against morning stock check",
            }
        ]

        resumed_result = self.coordinator.resume_execution(
            execution_id=req.request_id,
            human_decisions=human_decisions,
        )

        self.assertEqual(resumed_result.overall_status, TaskStatus.COMPLETED)
        self.assertIn(draft_po_id, resumed_result.approved_po_ids)
        self.assertGreaterEqual(len(resumed_result.transmitted_fo_ids), 1)

        # Verify that FulfilmentTransmission task completed
        tasks = self.coordinator.get_execution_tasks(req.request_id)
        trans_tasks = [t for t in tasks if t.stage == CoordinatorStage.FULFILMENT_TRANSMISSION]
        self.assertEqual(len(trans_tasks), 1)
        self.assertEqual(trans_tasks[0].status, TaskStatus.COMPLETED)
        self.assertEqual(trans_tasks[0].output_payload.get("transmission_status"), "CONFIRMED_ACCEPTANCE")

    def test_human_rejection_resume_terminates_cleanly(self) -> None:
        """Verifies resuming with human rejection marks PO rejected and terminates without transmission."""
        req = CoordinatorRequest(
            request_id="REQ-RESUME-REJECT-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-REJECT-001",
        )

        initial_result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )
        draft_po_id = initial_result.draft_po_ids[0]

        # Operator supplies verified human rejection
        human_decisions = [
            {
                "po_id": draft_po_id,
                "action": "REJECTED",
                "rejected_by": "OPERATOR_ALICE",
                "rejection_timestamp": "2024-11-15T10:00:00Z",
                "rejection_reason": "Excess dairy buffer identified in secondary chillers",
            }
        ]

        resumed_result = self.coordinator.resume_execution(
            execution_id=req.request_id,
            human_decisions=human_decisions,
        )

        self.assertEqual(resumed_result.overall_status, TaskStatus.COMPLETED)
        self.assertEqual(resumed_result.approved_po_ids, ())
        self.assertEqual(resumed_result.transmitted_fo_ids, ())

        # Verify no FULFILMENT_TRANSMISSION tasks were created
        tasks = self.coordinator.get_execution_tasks(req.request_id)
        trans_tasks = [t for t in tasks if t.stage == CoordinatorStage.FULFILMENT_TRANSMISSION]
        self.assertEqual(len(trans_tasks), 0)

    def test_step_by_step_execution_mode(self) -> None:
        """Verifies STEP_BY_STEP pauses after each operational stage for operator inspection."""
        req = CoordinatorRequest(
            request_id="REQ-STEP-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.STEP_BY_STEP,
            correlation_id="CORR-STEP-001",
        )

        # Step 1: Ingestion
        res1 = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )
        self.assertEqual(res1.overall_status, TaskStatus.SUSPENDED)
        self.assertEqual(res1.current_stage, CoordinatorStage.INGESTION)

        # Step 2: Inventory Evaluation
        res2 = self.coordinator.resume_execution(execution_id=req.request_id)
        self.assertEqual(res2.overall_status, TaskStatus.SUSPENDED)
        self.assertEqual(res2.current_stage, CoordinatorStage.INVENTORY_EVALUATION)

        # Step 3: Procurement Planning
        res3 = self.coordinator.resume_execution(execution_id=req.request_id)
        self.assertEqual(res3.overall_status, TaskStatus.SUSPENDED)
        self.assertEqual(res3.current_stage, CoordinatorStage.PROCUREMENT_PLANNING)
        self.assertGreaterEqual(len(res3.draft_po_ids), 1)

        # Step 4: Human Approval Gate (suspends awaiting human decision)
        res4 = self.coordinator.resume_execution(execution_id=req.request_id)
        self.assertEqual(res4.overall_status, TaskStatus.SUSPENDED)
        self.assertEqual(res4.current_stage, CoordinatorStage.HUMAN_APPROVAL_GATE)

    def test_full_end_to_end_dock_receiving_and_inventory_posting(self) -> None:
        """
        Verifies complete lifecycle orchestration from demand through inventory ledger posting.
        """
        req = CoordinatorRequest(
            request_id="REQ-E2E-FULL-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-E2E-001",
        )

        # Step A: Run to approval gate
        initial_res = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )
        draft_po_id = initial_res.draft_po_ids[0]

        # Step B: Resume with approval + operational inputs for tracking, dock receiving, and posting
        human_decisions = [
            {
                "po_id": draft_po_id,
                "action": "APPROVED",
                "approved_by": "MGR_JOHN",
                "approval_timestamp": "2024-11-15T10:00:00Z",
            }
        ]

        expected_fo_id = f"FO-{draft_po_id.replace('PO-', '')}"
        shipment_id = f"SHP-{draft_po_id.replace('PO-', '')}-1"
        grn_id = f"GRN-{draft_po_id.replace('PO-', '')}-1"

        valid_grn = {
            "grn_id": grn_id,
            "po_id": draft_po_id,
            "fo_id": expected_fo_id,
            "supplier_id": "SIM_SUPP_DAIRY_BACKUP",
            "store_id": "DS_BLR_HSR_LAYOUT",
            "shipment_id": shipment_id,
            "receipt_at": "2024-11-15T14:30:00Z",
            "receiver_id": "RCV_DOCK_101",
            "inspection_lines": [
                {
                    "sku_id": "SKU_MILK_1L",
                    "delivered_quantity": 24.0,
                    "accepted_quantity": 24.0,
                    "rejected_quantity": 0.0,
                    "unit_cost": 30.0,
                    "remaining_shelf_life_days": 5,
                    "total_shelf_life_days": 7,
                    "batch_references": ["BATCH-MILK-20241115"],
                }
            ],
        }

        op_inputs = {
            "target_tracking_status": ShipmentTrackingStatus.DELIVERED.value,
            "grn": valid_grn,
            "dock_receiver_id": "RCV_DOCK_101",
            "inventory_poster_id": "INV_POSTER_001",
        }

        final_res = self.coordinator.resume_execution(
            execution_id=req.request_id,
            human_decisions=human_decisions,
            operational_inputs=op_inputs,
        )

        self.assertEqual(final_res.overall_status, TaskStatus.COMPLETED)
        self.assertIn(draft_po_id, final_res.approved_po_ids)
        self.assertIn(expected_fo_id, final_res.transmitted_fo_ids)
        self.assertIn(grn_id, final_res.grn_ids)
        self.assertGreaterEqual(final_res.posted_inventory_entry_count, 1)

    def test_prototype_circularity_disposition(self) -> None:
        """Verifies prototype circularity adapter execution and Step 4 math."""
        circ_item = {
            "store_id": "DS_BLR_HSR_LAYOUT",
            "sku_id": "SKU_MILK_1L",
            "closing_stock": 50.0,
            "p50_demand": 5.0,
            "remaining_shelf_life_days": 1,
            "original_retail_price": 60.0,
        }

        task = TaskContext(
            task_id="TASK:CIRC-001:CIRCULARITY:DS_HSR:SKU_MILK",
            execution_id="CIRC-001",
            stage=CoordinatorStage.CIRCULARITY_DISPOSITION,
            node_key="DS_BLR_HSR_LAYOUT:SKU_MILK_1L",
            idempotency_key="KEY-CIRC-001",
            input_payload=circ_item,
        )

        adapter = CircularityAdapter()
        output = adapter.execute_task(task)

        # Omega_t = max(0, 50 - 5*1) = 45.0
        self.assertEqual(output["overstock_risk_units"], 45.0)
        self.assertEqual(output["markdown_discount_pct"], 50.0)  # > 3*5 triggers food bank (50%)
        self.assertTrue(output["food_rescue_eligible"])

    def test_failure_isolation_on_single_task(self) -> None:
        """Verifies that an error on one node does not crash the coordinator process or corrupt others."""
        req = CoordinatorRequest(
            request_id="REQ-FAIL-ISO-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-FAIL-001",
        )

        # Intentionally invalid daily input (negative demand triggering validation error)
        bad_input = {
            "store_id": "DS_BLR_HSR_LAYOUT",
            "sku_id": "SKU_MILK_1L",
            "date": "2024-11-15",
            "invalid_field": 123,
        }

        result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [bad_input]},
        )

        self.assertEqual(result.overall_status, TaskStatus.FAILED)
        self.assertGreaterEqual(len(result.errors), 1)

    def test_execution_cancellation(self) -> None:
        """Verifies that cancelling an execution halts all pending tasks."""
        req = CoordinatorRequest(
            request_id="REQ-CANCEL-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-CANCEL-001",
        )

        initial = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )
        self.assertEqual(initial.overall_status, TaskStatus.SUSPENDED)

        cancelled_result = self.coordinator.cancel_execution(
            execution_id=req.request_id,
            reason="Market emergency halt",
        )

        self.assertEqual(cancelled_result.overall_status, TaskStatus.CANCELLED)

        # Resuming a cancelled execution must raise ValueError
        with self.assertRaises(ValueError):
            self.coordinator.resume_execution(execution_id=req.request_id)

    def test_json_round_trip_and_serialization(self) -> None:
        """Verifies full JSON serialization and deserialization of CoordinatorResult and DAG."""
        req = CoordinatorRequest(
            request_id="REQ-SERIAL-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.DRY_RUN,
            correlation_id="CORR-SERIAL-001",
        )

        result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )

        # Test Pydantic V2 JSON serialization
        json_str = result.model_dump_json()
        reconstructed = CoordinatorResult.model_validate_json(json_str)

        self.assertEqual(result.execution_id, reconstructed.execution_id)
        self.assertEqual(result.overall_status, reconstructed.overall_status)
        self.assertEqual(result.draft_po_ids, reconstructed.draft_po_ids)
        self.assertEqual(result.total_tasks, reconstructed.total_tasks)

    def test_bounded_concurrency_parallel_execution(self) -> None:
        """Verifies multi-SKU parallel bounded concurrency execution across independent ready tasks."""
        skus = ("SKU_MILK_1L", "SKU_CURD_400G", "SKU_PANEER_200G", "SKU_BUTTER_100G")
        req = CoordinatorRequest(
            request_id="REQ-PARALLEL-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=skus,
            execution_mode=CoordinatorExecutionMode.DRY_RUN,
            correlation_id="CORR-PARALLEL-001",
        )

        inputs = [_make_daily_input(sku_id=sku, p50=25.0, on_hand=5.0) for sku in skus]
        result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [d.model_dump() for d in inputs]},
        )

        self.assertEqual(result.overall_status, TaskStatus.COMPLETED)
        self.assertEqual(result.evaluated_sku_count, 4)
        tasks = self.coordinator.get_execution_tasks(req.request_id)
        eval_tasks = [t for t in tasks if t.stage == CoordinatorStage.INVENTORY_EVALUATION]
        self.assertEqual(len(eval_tasks), 4)
        for t in eval_tasks:
            self.assertEqual(t.status, TaskStatus.COMPLETED)

    def test_idempotency_conflict_on_different_operational_inputs(self) -> None:
        """Verifies that reusing the same request_id with differing operational_inputs raises IdempotencyConflictError."""
        req = CoordinatorRequest(
            request_id="REQ-OP-CONFLICT-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.DRY_RUN,
            correlation_id="CORR-OP-001",
        )

        # First run succeeds
        self.coordinator.start_execution(
            request=req,
            operational_inputs={"param_key": "VALUE_A", "daily_agent_inputs": [self.daily_input.model_dump()]},
        )

        # Second run with differing operational_inputs under the same request_id MUST raise IdempotencyConflictError
        with self.assertRaises(IdempotencyConflictError):
            self.coordinator.start_execution(
                request=req,
                operational_inputs={"param_key": "VALUE_B_CONFLICT", "daily_agent_inputs": [self.daily_input.model_dump()]},
            )

    def test_crash_recovery_boundary_4_domain_pre_check(self) -> None:
        """
        Verifies Boundary 4 crash recovery: if an external domain side-effect succeeded
        (order acknowledged, dock intake recorded, or inventory posted) but coordinator crashed,
        retrying task execution reuses domain state safely without duplicate insertion or illegal transitions.
        """
        adapter = FulfilmentServiceAdapter(db_path=self.fulfilment_db)
        try:
            # 1. Setup approved PO
            item = PurchaseOrderLineItem(
                store_id="DS_BLR_HSR_LAYOUT",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                raw_suggested_qty=24.0,
                case_pack_rounded_qty=24.0,
                final_order_qty=24.0,
                case_pack_size=12,
                moq=10.0,
                unit_retail_price=50.0,
                unit_procurement_cost=30.0,
                total_line_cost=720.0,
                requires_escalation=False,
                escalation_reasons=[],
            )
            draft_po = PurchaseOrder(
                po_id="PO-TEST-B4-001",
                created_date="2024-11-15",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                supplier_type=SupplierType.PRIMARY,
                store_id="DS_BLR_HSR_LAYOUT",
                urgency=PurchaseOrderUrgency.STANDARD,
                status=PurchaseOrderStatus.PENDING_APPROVAL,
                items=[item],
                total_order_qty=24.0,
                total_order_cost=720.0,
                requires_escalation=False,
                escalation_reasons=[],
                approved_by=None,
                approval_timestamp=None,
            )
            po = approve_purchase_order(
                po=draft_po,
                approved_by="DIR_OPERATIONS",
                approval_timestamp="2024-11-15T09:00:00Z",
                approval_reason="Verified replenishment budget",
            )

            fo_id = "FO-TEST-B4-001"
            shp_id = "SHP-TEST-B4-001"
            grn_id = "GRN-TEST-B4-001"

            tx_task = TaskContext(
                task_id="TASK:TX:B4",
                execution_id="EXEC-B4",
                stage=CoordinatorStage.FULFILMENT_TRANSMISSION,
                node_key=po.po_id,
                idempotency_key="KEY-TX-B4",
                input_payload={
                    "purchase_order": po.model_dump(),
                    "fo_id": fo_id,
                    "asn": {
                        "shipment_id": shp_id,
                        "fo_id": fo_id,
                        "po_id": po.po_id,
                        "supplier_id": po.supplier_id,
                        "dispatch_at": "2024-11-15T10:00:00Z",
                        "expected_arrival_at": "2024-11-15T14:00:00Z",
                        "lines": [{"sku_id": "SKU_MILK_1L", "shipped_quantity": 24.0}],
                        "carrier": "BLR_COLD_EXPRESS",
                    },
                },
            )

            # Step 1: Initial transmission
            out_tx1 = adapter.execute_task(tx_task)
            self.assertEqual(out_tx1["status"], "IN_TRANSIT")

            # Step 2: Retry transmission (simulating coordinator crash before persisting output)
            # Must safely reuse existing order and NOT raise InvalidStateTransitionError
            out_tx2 = adapter.execute_task(tx_task)
            self.assertEqual(out_tx2["fo_id"], fo_id)
            self.assertEqual(out_tx2["status"], "IN_TRANSIT")
            self.assertIn(shp_id, out_tx2["shipment_ids"])

            # Step 3: Dock Intake
            grn_payload = {
                "grn_id": grn_id,
                "po_id": po.po_id,
                "fo_id": fo_id,
                "supplier_id": po.supplier_id,
                "store_id": po.store_id,
                "shipment_id": shp_id,
                "receipt_at": "2024-11-15T14:30:00Z",
                "receiver_id": "RCV_101",
                "inspection_lines": [
                    {
                        "sku_id": "SKU_MILK_1L",
                        "delivered_quantity": 24.0,
                        "accepted_quantity": 24.0,
                        "rejected_quantity": 0.0,
                        "unit_cost": 30.0,
                        "remaining_shelf_life_days": 5,
                        "total_shelf_life_days": 7,
                        "batch_references": ["BATCH-B4-001"],
                    }
                ],
            }
            dock_task = TaskContext(
                task_id="TASK:DOCK:B4",
                execution_id="EXEC-B4",
                stage=CoordinatorStage.DOCK_RECEIVING,
                node_key=grn_id,
                idempotency_key="KEY-DOCK-B4",
                input_payload={"fo_id": fo_id, "grn": grn_payload},
            )
            out_dock1 = adapter.execute_task(dock_task)
            self.assertEqual(out_dock1["grn_id"], grn_id)

            # Step 4: Retry dock intake (must reuse existing GRN without throwing duplicate GRN error)
            out_dock2 = adapter.execute_task(dock_task)
            self.assertEqual(out_dock2["grn_id"], grn_id)
            self.assertEqual(out_dock2["total_accepted_quantity"], 24.0)

            # Step 5: Inventory Posting
            post_task = TaskContext(
                task_id="TASK:POST:B4",
                execution_id="EXEC-B4",
                stage=CoordinatorStage.INVENTORY_POSTING,
                node_key=grn_id,
                idempotency_key="KEY-POST-B4",
                input_payload={"fo_id": fo_id, "grn_id": grn_id},
            )
            out_post1 = adapter.execute_task(post_task)
            self.assertGreater(out_post1["posted_entries_count"], 0)

            # Step 6: Retry inventory posting (must reuse existing entries without duplicate ledger insertion)
            out_post2 = adapter.execute_task(post_task)
            self.assertEqual(out_post2["posted_entries_count"], out_post1["posted_entries_count"])
            self.assertEqual(out_post2["total_posted_quantity"], out_post1["total_posted_quantity"])

        finally:
            adapter.service.close()

    def test_multi_store_failure_isolation(self) -> None:
        """Verifies DAG failure isolation: an adapter error on Store A does not stop Store B tasks."""
        req = CoordinatorRequest(
            request_id="REQ-MULTI-STORE-ISO",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT", "DS_BLR_KORAMANGALA"),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.DRY_RUN,
            correlation_id="CORR-ISO-001",
        )

        valid_hsr = _make_daily_input(store_id="DS_BLR_HSR_LAYOUT", p50=30.0, on_hand=10.0)
        # Store Koramangala is missing its forecast entirely, which will cause an error on its Ingestion / Evaluation
        result = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [valid_hsr.model_dump()]},
        )

        # Store HSR tasks must have executed and completed
        tasks = self.coordinator.get_execution_tasks(req.request_id)
        hsr_eval = next((t for t in tasks if t.stage == CoordinatorStage.INVENTORY_EVALUATION and "HSR" in t.node_key), None)
        self.assertIsNotNone(hsr_eval)
        self.assertEqual(hsr_eval.status, TaskStatus.COMPLETED)

    def test_retry_limit_exhaustion_and_attempt_history(self) -> None:
        """Verifies retry_failed_tasks enforces max_retries and preserves attempt_history in task provenance."""
        req = CoordinatorRequest(
            request_id="REQ-RETRY-LIMIT-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.DRY_RUN,
            correlation_id="CORR-RETRY-001",
        )

        # Force failure with corrupted payload
        bad_input = {"store_id": "DS_BLR_HSR_LAYOUT", "sku_id": "SKU_MILK_1L", "bad": True}
        initial_res = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [bad_input]},
        )
        self.assertEqual(initial_res.overall_status, TaskStatus.FAILED)

        # Retry 1
        res1 = self.coordinator.retry_failed_tasks(req.request_id, max_retries=2)
        self.assertEqual(res1.overall_status, TaskStatus.FAILED)

        # Retry 2 (exhausts max_retries=2)
        res2 = self.coordinator.retry_failed_tasks(req.request_id, max_retries=2)
        self.assertEqual(res2.overall_status, TaskStatus.FAILED)

        tasks = self.coordinator.get_execution_tasks(req.request_id)
        failed_tasks = [t for t in tasks if t.status == TaskStatus.FAILED]
        self.assertGreater(len(failed_tasks), 0)
        task = failed_tasks[0]
        self.assertGreaterEqual(task.retry_count, 2)
        self.assertIn("attempt_history", task.provenance)
        self.assertGreaterEqual(len(task.provenance["attempt_history"]), 2)

    def test_cancellation_preserves_completed_and_reports_cancelled_tasks(self) -> None:
        """Verifies cancellation reports cancelled_tasks metric and preserves previously completed tasks."""
        req = CoordinatorRequest(
            request_id="REQ-METRIC-CANCEL-001",
            operating_date="2024-11-15",
            target_stores=("DS_BLR_HSR_LAYOUT",),
            target_skus=("SKU_MILK_1L",),
            execution_mode=CoordinatorExecutionMode.AUTOMATED_TO_APPROVAL,
            correlation_id="CORR-CANCEL-METRIC",
        )

        initial = self.coordinator.start_execution(
            request=req,
            operational_inputs={"daily_agent_inputs": [self.daily_input.model_dump()]},
        )
        self.assertEqual(initial.overall_status, TaskStatus.SUSPENDED)
        self.assertGreater(initial.completed_tasks, 0)

        cancelled = self.coordinator.cancel_execution(req.request_id, reason="Store emergency evacuation")
        self.assertEqual(cancelled.overall_status, TaskStatus.CANCELLED)
        self.assertGreater(cancelled.completed_tasks, 0)
        self.assertGreaterEqual(cancelled.cancelled_tasks, 1)

    def test_thread_safe_concurrent_repository_operations(self) -> None:
        """Verifies that concurrent multi-threaded writes to CoordinatorRepository are fully thread-safe."""
        import concurrent.futures

        def _worker_operation(worker_idx: int) -> bool:
            w_req = CoordinatorRequest(
                request_id=f"REQ-THREAD-{worker_idx}",
                operating_date="2024-11-15",
                target_stores=("DS_BLR_HSR_LAYOUT",),
                target_skus=("SKU_MILK_1L",),
                correlation_id=f"CORR-THREAD-{worker_idx}",
            )
            self.coordinator.repository.create_execution(w_req)
            t = TaskContext(
                task_id=f"TASK:THREAD:{worker_idx}",
                execution_id=w_req.request_id,
                stage=CoordinatorStage.INGESTION,
                node_key="DS_BLR_HSR_LAYOUT",
                status=TaskStatus.COMPLETED,
                idempotency_key=f"KEY:THREAD:{worker_idx}",
            )
            self.coordinator.repository.save_task(t)
            rec = CoordinatorAuditRecord(
                execution_id=w_req.request_id,
                event_id=f"EVT:THREAD:{worker_idx}:{time.time_ns()}",
                timestamp="2024-11-15T10:00:00Z",
                stage=CoordinatorStage.INGESTION,
                action="THREAD_TEST",
                actor_id=f"WORKER_{worker_idx}",
                details={"worker": worker_idx},
                payload_fingerprint="fp123",
            )
            self.coordinator.repository.save_audit_record(rec)
            fetched = self.coordinator.repository.get_execution(w_req.request_id)
            return fetched is not None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_worker_operation, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 10)
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()
