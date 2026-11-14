"""
Unit and Integration Tests for Runtime Click CLI (agy-pipeline)
==============================================================
Tests CLI commands: run, status, approve, inventory, retry, cancel,
output formatting (text and JSON), and human governance security checks.
"""

import json
import os
import shutil
import tempfile
import uuid
import pytest
from click.testing import CliRunner

from src.runtime.config import RuntimeConfig
from src.runtime.context import ApplicationContext
from src.runtime.cli import main


@pytest.fixture
def cli_env():
    """Sets up an isolated temporary database environment for CLI testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_runtime_cli_")
    coord_db = os.path.join(temp_dir, "coordinator.db")
    fulfil_db = os.path.join(temp_dir, "fulfilment.db")

    cfg = RuntimeConfig(
        env="testing",
        log_level="INFO",
        coordinator_db_path=coord_db,
        fulfilment_db_path=fulfil_db,
        operator_api_key="cli-secret-test-key",
        authorized_operators=["admin", "ops_lead", "lead2_supervisor"],
    )
    ctx = ApplicationContext(config=cfg)
    ctx.startup()

    runner = CliRunner()
    yield {
        "runner": runner,
        "ctx": ctx,
        "cfg": cfg,
        "temp_dir": temp_dir,
    }

    ctx.shutdown()
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_help(cli_env):
    """Verifies that main --help displays usage and all registered commands."""
    runner = cli_env["runner"]
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output
    assert "approve" in result.output
    assert "inventory" in result.output
    assert "serve" in result.output
    assert "retry" in result.output
    assert "cancel" in result.output


def test_cli_run_dry_run_json(cli_env):
    """Verifies run command in DRY_RUN mode with JSON formatting."""
    runner = cli_env["runner"]
    ctx = cli_env["ctx"]
    req_id = f"REQ-CLI-DRY-{uuid.uuid4().hex[:8]}"

    args = [
        "run",
        "--mode", "DRY_RUN",
        "--date", "2024-11-15",
        "--stores", "DS_BLR_HSR_LAYOUT",
        "--skus", "SKU_MILK_1L",
        "--request-id", req_id,
        "--format", "json",
    ]
    result = runner.invoke(main, args, obj=ctx)
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["request_id"] == req_id
    assert data["overall_status"] == "COMPLETED"
    assert data["completed_tasks"] >= 3
    assert len(data["draft_po_ids"]) >= 1


def test_cli_run_text_output(cli_env):
    """Verifies run command text output formatting."""
    runner = cli_env["runner"]
    ctx = cli_env["ctx"]
    req_id = f"REQ-CLI-TXT-{uuid.uuid4().hex[:8]}"

    args = [
        "run",
        "--mode", "DRY_RUN",
        "--request-id", req_id,
        "--format", "text",
    ]
    result = runner.invoke(main, args, obj=ctx)
    assert result.exit_code == 0
    assert "Execution:" in result.output
    assert "Status:          COMPLETED" in result.output
    assert "Draft POs:" in result.output


def test_cli_status_and_tasks(cli_env):
    """Verifies checking execution status and DAG task breakdown."""
    runner = cli_env["runner"]
    ctx = cli_env["ctx"]
    req_id = f"REQ-CLI-STAT-{uuid.uuid4().hex[:8]}"

    # Run execution first
    run_res = runner.invoke(
        main,
        ["run", "--mode", "DRY_RUN", "--request-id", req_id, "--format", "json"],
        obj=ctx,
    )
    assert run_res.exit_code == 0
    exec_id = json.loads(run_res.output)["execution_id"]

    # Status without tasks
    status_res = runner.invoke(main, ["status", "-e", exec_id], obj=ctx)
    assert status_res.exit_code == 0
    assert exec_id in status_res.output
    assert "DAG Task Breakdown" not in status_res.output

    # Status with --tasks
    status_tasks_res = runner.invoke(main, ["status", "-e", exec_id, "--tasks"], obj=ctx)
    assert status_tasks_res.exit_code == 0
    assert "DAG Task Breakdown:" in status_tasks_res.output
    assert "INGESTION" in status_tasks_res.output
    assert "PROCUREMENT_PLANNING" in status_tasks_res.output

    # Status with JSON output
    status_json_res = runner.invoke(main, ["status", "-e", exec_id, "--tasks", "--format", "json"], obj=ctx)
    assert status_json_res.exit_code == 0
    data = json.loads(status_json_res.output)
    assert data["result"]["execution_id"] == exec_id
    assert len(data["tasks"]) >= 3

    # Non-existent execution returns error
    missing_res = runner.invoke(main, ["status", "-e", "EXEC-DOES-NOT-EXIST"], obj=ctx)
    assert missing_res.exit_code == 1
    assert "not found" in missing_res.output


def test_cli_approve_flow_and_security(cli_env):
    """Verifies human approval governance and security in the CLI."""
    runner = cli_env["runner"]
    ctx = cli_env["ctx"]
    req_id = f"REQ-CLI-APP-{uuid.uuid4().hex[:8]}"

    # 1. Run in AUTOMATED_TO_APPROVAL mode -> suspends at approval gate
    run_res = runner.invoke(
        main,
        ["run", "--mode", "AUTOMATED_TO_APPROVAL", "--request-id", req_id, "--format", "json"],
        obj=ctx,
    )
    assert run_res.exit_code == 0
    exec_data = json.loads(run_res.output)
    exec_id = exec_data["execution_id"]
    assert exec_data["overall_status"] == "SUSPENDED"
    po_id = exec_data["draft_po_ids"][0]

    # 2. Missing/wrong API key -> exit 1
    bad_key_res = runner.invoke(
        main,
        [
            "approve",
            "-e", exec_id,
            "-p", po_id,
            "-a", "admin",
            "-r", "Approval justification",
            "--api-key", "wrong-key",
        ],
        obj=ctx,
    )
    assert bad_key_res.exit_code == 1
    assert "Forbidden: Invalid or missing operator security key" in bad_key_res.output

    # 3. Unauthorized operator -> exit 1
    bad_actor_res = runner.invoke(
        main,
        [
            "approve",
            "-e", exec_id,
            "-p", po_id,
            "-a", "unauthorized_user_1",
            "-r", "Approval justification",
            "--api-key", "cli-secret-test-key",
        ],
        obj=ctx,
    )
    assert bad_actor_res.exit_code == 1
    assert "Forbidden: Operator 'unauthorized_user_1' is not authorized" in bad_actor_res.output

    # 4. Empty or invalid reason -> exit 1
    empty_reason_res = runner.invoke(
        main,
        [
            "approve",
            "-e", exec_id,
            "-p", po_id,
            "-a", "admin",
            "-r", "  ",
            "--api-key", "cli-secret-test-key",
        ],
        obj=ctx,
    )
    assert empty_reason_res.exit_code == 1
    assert "Approval reason must be a descriptive non-empty string" in empty_reason_res.output

    # 5. Valid authorized approval -> exit 0
    valid_res = runner.invoke(
        main,
        [
            "approve",
            "-e", exec_id,
            "-p", po_id,
            "-a", "admin",
            "-r", "Verified stock and demand spikes for milk",
            "--api-key", "cli-secret-test-key",
        ],
        obj=ctx,
    )
    assert valid_res.exit_code == 0
    assert f"Purchase Order '{po_id}' successfully APPROVED by 'admin'" in valid_res.output


def test_cli_cancel(cli_env):
    """Verifies cancelling a suspended execution via CLI."""
    runner = cli_env["runner"]
    ctx = cli_env["ctx"]
    req_id = f"REQ-CLI-CNC-{uuid.uuid4().hex[:8]}"

    # Start suspended execution
    run_res = runner.invoke(
        main,
        ["run", "--mode", "AUTOMATED_TO_APPROVAL", "--request-id", req_id, "--format", "json"],
        obj=ctx,
    )
    exec_id = json.loads(run_res.output)["execution_id"]

    # Cancel execution
    cancel_res = runner.invoke(
        main,
        ["cancel", "-e", exec_id, "-r", "Cancelled due to warehouse maintenance"],
        obj=ctx,
    )
    assert cancel_res.exit_code == 0
    assert f"Execution '{exec_id}' CANCELLED" in cancel_res.output


def test_cli_retry(cli_env):
    """Verifies retry command on completed/failed executions via CLI."""
    runner = cli_env["runner"]
    ctx = cli_env["ctx"]
    req_id = f"REQ-CLI-RTR-{uuid.uuid4().hex[:8]}"

    run_res = runner.invoke(
        main,
        ["run", "--mode", "DRY_RUN", "--request-id", req_id, "--format", "json"],
        obj=ctx,
    )
    exec_id = json.loads(run_res.output)["execution_id"]

    retry_res = runner.invoke(main, ["retry", "-e", exec_id], obj=ctx)
    assert retry_res.exit_code == 0
    assert f"Retry triggered for execution '{exec_id}'" in retry_res.output


def test_cli_inventory(cli_env):
    """Verifies querying inventory balances and batch expiry via CLI."""
    runner = cli_env["runner"]
    ctx = cli_env["ctx"]

    # Text output
    inv_text_res = runner.invoke(main, ["inventory", "-s", "DS_BLR_HSR_LAYOUT"], obj=ctx)
    assert inv_text_res.exit_code == 0
    assert "Operational Inventory Balances for Store: DS_BLR_HSR_LAYOUT" in inv_text_res.output

    # JSON output with SKU filter
    inv_json_res = runner.invoke(
        main,
        ["inventory", "-s", "DS_BLR_HSR_LAYOUT", "-k", "SKU_MILK_1L", "--format", "json"],
        obj=ctx,
    )
    assert inv_json_res.exit_code == 0
    data = json.loads(inv_json_res.output)
    assert data["store_id"] == "DS_BLR_HSR_LAYOUT"
    assert data["sku_id"] == "SKU_MILK_1L"

    # Invalid store
    bad_store_res = runner.invoke(main, ["inventory", "-s", "NON_EXISTENT_STORE"], obj=ctx)
    assert bad_store_res.exit_code == 1
    assert "Invalid store" in bad_store_res.output

    # Invalid SKU
    bad_sku_res = runner.invoke(
        main,
        ["inventory", "-s", "DS_BLR_HSR_LAYOUT", "-k", "NON_EXISTENT_SKU"],
        obj=ctx,
    )
    assert bad_sku_res.exit_code == 1
    assert "Invalid SKU" in bad_sku_res.output

