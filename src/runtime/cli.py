"""
Operational Command-Line Interface (CLI)
========================================
Provides headless and operational management commands for:
- run: Triggering replenishment runs (DRY_RUN, AUTOMATED_TO_APPROVAL, etc.)
- status: Checking execution state and task DAG breakdown
- approve: Submitting verified human approval sign-off on draft Purchase Orders
- inventory: Querying store balances and batch perishability from the ledger
- serve: Launching the FastAPI REST API server
- cancel: Cancelling active or suspended runs
- retry: Retrying eligible failed tasks
"""

from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, List, Optional
import click

from .config import RuntimeConfig, get_runtime_config
from .context import ApplicationContext, get_application_context
from ..agents.coordinator.schemas import (
    CoordinatorExecutionMode,
    CoordinatorRequest,
    CoordinatorResult,
    TaskContext,
    TaskStatus,
)
from ..agents.coordinator.repository import IdempotencyConflictError
from ..agents.common.config import CANONICAL_SKUS, CANONICAL_STORES


def _get_context(ctx_obj: Optional[Any] = None) -> ApplicationContext:
    """Helper retrieving active or global ApplicationContext."""
    if ctx_obj and isinstance(ctx_obj, ApplicationContext):
        app_ctx = ctx_obj
    else:
        app_ctx = get_application_context()
    if not app_ctx.is_initialized:
        app_ctx.startup()
    return app_ctx


@click.group()
@click.option("--env-file", type=click.Path(exists=True), default=None, help="Optional .env file path")
@click.pass_context
def main(ctx: click.Context, env_file: Optional[str] = None) -> None:
    """Autonomous Demand-Consumption Replenishment Pipeline CLI (agy-pipeline)."""
    if ctx.obj is None:
        cfg = RuntimeConfig.load_from_env(env_file=env_file)
        app_ctx = ApplicationContext(config=cfg)
        ctx.obj = app_ctx


@main.command("run")
@click.option("--date", "-d", "operating_date", default="2024-11-15", help="Operating calculation date (YYYY-MM-DD)")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["DRY_RUN", "AUTOMATED_TO_APPROVAL", "FULL_END_TO_END", "STEP_BY_STEP"], case_sensitive=False),
    default="AUTOMATED_TO_APPROVAL",
    help="Execution mode (default: AUTOMATED_TO_APPROVAL)",
)
@click.option("--stores", "-s", default="DS_BLR_HSR_LAYOUT", help="Comma-separated store IDs")
@click.option("--skus", "-k", default="SKU_MILK_1L", help="Comma-separated SKU IDs")
@click.option("--request-id", default=None, help="Optional unique request ID")
@click.option("--correlation-id", default=None, help="Optional correlation ID")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.pass_obj
def run_command(
    app_ctx: ApplicationContext,
    operating_date: str,
    mode: str,
    stores: str,
    skus: str,
    request_id: Optional[str],
    correlation_id: Optional[str],
    output_format: str,
) -> None:
    """Trigger a new replenishment orchestration run."""
    ctx = _get_context(app_ctx)
    coordinator = ctx.get_coordinator()

    store_list = tuple(s.strip() for s in stores.split(",") if s.strip())
    sku_list = tuple(k.strip() for k in skus.split(",") if k.strip())

    req_id = request_id or f"REQ-{operating_date.replace('-', '')}-{int(datetime.now(timezone.utc).timestamp())}"
    corr_id = correlation_id or f"CORR-{req_id}"

    try:
        request = CoordinatorRequest(
            request_id=req_id,
            operating_date=operating_date,
            target_stores=store_list,
            target_skus=sku_list,
            execution_mode=CoordinatorExecutionMode(mode.upper()),
            correlation_id=corr_id,
        )
    except Exception as e:
        click.secho(f"Invalid request parameters: {e}", fg="red", err=True)
        sys.exit(1)

    try:
        result = coordinator.start_execution(request)
        if output_format == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            click.secho(f"\nExecution: {result.execution_id}", fg="cyan", bold=True)
            click.echo(f"  Status:          {result.overall_status.value}")
            click.echo(f"  Stage:           {result.current_stage.value}")
            click.echo(f"  Operating Date:  {result.operating_date}")
            click.echo(f"  Tasks (Total):   {result.total_tasks} (Completed: {result.completed_tasks}, Suspended: {result.suspended_tasks}, Failed: {result.failed_tasks})")
            click.echo(f"  Evaluated SKUs:  {result.evaluated_sku_count}")
            click.echo(f"  Draft POs:       {list(result.draft_po_ids)}")
            click.echo(f"  Approved POs:    {list(result.approved_po_ids)}")
            click.echo(f"  Transmitted FOs: {list(result.transmitted_fo_ids)}")
            click.echo(f"  GRNs Created:    {list(result.grn_ids)}")
            click.echo(f"  Posted Entries:  {result.posted_inventory_entry_count}")
            click.echo(f"  Duration:        {result.execution_duration_ms:.1f}ms\n")
            if result.overall_status == TaskStatus.SUSPENDED:
                click.secho("  [!] Execution suspended at HUMAN_APPROVAL_GATE. Awaiting human operator sign-off.", fg="yellow", bold=True)
    except IdempotencyConflictError as err:
        click.secho(f"Idempotency Conflict: {err}", fg="red", err=True)
        sys.exit(1)
    except Exception as err:
        click.secho(f"Execution failed: {err}", fg="red", err=True)
        sys.exit(1)


@main.command("status")
@click.option("--execution-id", "-e", required=True, help="Execution run identifier")
@click.option("--tasks", "-t", is_flag=True, default=False, help="Display detailed task breakdown")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.pass_obj
def status_command(
    app_ctx: ApplicationContext,
    execution_id: str,
    tasks: bool,
    output_format: str,
) -> None:
    """Check current status and results of an execution run."""
    ctx = _get_context(app_ctx)
    coordinator = ctx.get_coordinator()

    result = coordinator.get_execution(execution_id)
    if result is None:
        click.secho(f"Error: Execution '{execution_id}' not found.", fg="red", err=True)
        sys.exit(1)

    task_list = coordinator.get_execution_tasks(execution_id) if tasks else []

    if output_format == "json":
        data = {
            "result": result.model_dump(),
            "tasks": [t.model_dump() for t in task_list] if tasks else None,
        }
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.secho(f"\nExecution: {result.execution_id}", fg="cyan", bold=True)
        click.echo(f"  Status:          {result.overall_status.value}")
        click.echo(f"  Stage:           {result.current_stage.value}")
        click.echo(f"  Draft POs:       {list(result.draft_po_ids)}")
        click.echo(f"  Approved POs:    {list(result.approved_po_ids)}")
        click.echo(f"  Transmitted FOs: {list(result.transmitted_fo_ids)}")
        click.echo(f"  Posted Entries:  {result.posted_inventory_entry_count}")

        if tasks and task_list:
            click.secho("\nDAG Task Breakdown:", fg="cyan")
            for t in task_list:
                status_color = "green" if t.status == TaskStatus.COMPLETED else ("yellow" if t.status == TaskStatus.SUSPENDED else "red")
                click.secho(f"  [{t.status.value}]", fg=status_color, nl=False)
                click.echo(f" {t.stage.value:<25} Node: {t.node_key:<35} Task: {t.task_id}")
                if t.error_message:
                    click.secho(f"       Error: {t.error_message}", fg="red")
        click.echo("")


@main.command("approve")
@click.option("--execution-id", "-e", required=True, help="Execution run identifier")
@click.option("--po-id", "-p", required=True, help="Draft Purchase Order identifier")
@click.option("--actor", "-a", required=True, help="Authorized human approver ID")
@click.option("--reason", "-r", required=True, help="Approval rationale / business justification")
@click.option("--timestamp", default=None, help="Optional approval timestamp (default: UTC now)")
@click.option("--api-key", default=None, help="Optional operator security key")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.pass_obj
def approve_command(
    app_ctx: ApplicationContext,
    execution_id: str,
    po_id: str,
    actor: str,
    reason: str,
    timestamp: Optional[str],
    api_key: Optional[str],
    output_format: str,
) -> None:
    """Submit authorized human approval for a pending Purchase Order."""
    ctx = _get_context(app_ctx)

    # 1. Security Check: Operator Authorization & Key Validation
    if ctx.config.operator_api_key:
        provided_key = api_key or os.environ.get("OPERATOR_API_KEY")
        if not provided_key or provided_key != ctx.config.operator_api_key:
            click.secho("Forbidden: Invalid or missing operator security key (--api-key / OPERATOR_API_KEY).", fg="red", err=True)
            sys.exit(1)

    if ctx.config.authorized_operators:
        if actor not in ctx.config.authorized_operators:
            click.secho(f"Forbidden: Operator '{actor}' is not authorized to sign off Purchase Orders.", fg="red", err=True)
            sys.exit(1)

    clean_reason = reason.strip()
    if len(clean_reason) < 3:
        click.secho("Error: Approval reason must be a descriptive non-empty string.", fg="red", err=True)
        sys.exit(1)

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    decisions = [
        {
            "po_id": po_id,
            "action": "APPROVED",
            "approved_by": actor,
            "approval_timestamp": ts,
            "approval_reason": clean_reason,
        }
    ]

    coordinator = ctx.get_coordinator()
    try:
        result = coordinator.resume_execution(
            execution_id=execution_id,
            human_decisions=decisions,
        )
        if output_format == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            click.secho(f"\nPurchase Order '{po_id}' successfully APPROVED by '{actor}'.", fg="green", bold=True)
            click.echo(f"  Execution Status: {result.overall_status.value}")
            click.echo(f"  Current Stage:    {result.current_stage.value}")
            click.echo(f"  Approved POs:     {list(result.approved_po_ids)}")
            click.echo(f"  Transmitted FOs:  {list(result.transmitted_fo_ids)}\n")
    except Exception as err:
        click.secho(f"Approval resumption failed: {err}", fg="red", err=True)
        sys.exit(1)


@main.command("inventory")
@click.option("--store", "-s", required=True, help="Dark store identifier (e.g. DS_BLR_HSR_LAYOUT)")
@click.option("--sku", "-k", default=None, help="Optional SKU identifier (e.g. SKU_MILK_1L)")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.pass_obj
def inventory_command(
    app_ctx: ApplicationContext,
    store: str,
    sku: Optional[str],
    output_format: str,
) -> None:
    """Query current operational inventory ledger balances and batch expiry."""
    ctx = _get_context(app_ctx)
    inv_repo = ctx.get_inventory_repository()

    if store not in CANONICAL_STORES:
        click.secho(f"Invalid store '{store}'. Must be one of: {CANONICAL_STORES}", fg="red", err=True)
        sys.exit(1)
    if sku and sku not in CANONICAL_SKUS:
        click.secho(f"Invalid SKU '{sku}'. Must be one of: {CANONICAL_SKUS}", fg="red", err=True)
        sys.exit(1)

    store_balances = inv_repo.get_store_balances(store_id=store)
    if sku:
        store_balances = [sb for sb in store_balances if sb.sku_id == sku]
    batch_balances = inv_repo.get_batch_balances(store_id=store, sku_id=sku)

    if output_format == "json":
        data = {
            "store_id": store,
            "sku_id": sku,
            "store_balances": [sb.model_dump() for sb in store_balances],
            "batch_balances": [bb.model_dump() for bb in batch_balances],
        }
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.secho(f"\nOperational Inventory Balances for Store: {store}", fg="cyan", bold=True)
        if not store_balances:
            click.echo("  No stock currently recorded in ledger.")
        else:
            for sb in store_balances:
                click.echo(f"  SKU: {sb.sku_id:<20} On Hand: {sb.on_hand_stock:>8.1f} units | Total Cost: ₹{sb.total_cost_inr:>10.2f} | Batches: {sb.active_batch_count}")

        if batch_balances:
            click.secho("\nActive Batches (FEFO Order):", fg="cyan")
            for bb in batch_balances:
                exp_label = "EXPIRED" if bb.is_expired else f"{bb.current_remaining_shelf_life_days}d remaining"
                exp_color = "red" if bb.is_expired else "green"
                click.echo(f"  Batch: {bb.batch_id:<22} SKU: {bb.sku_id:<16} Qty: {bb.on_hand_quantity:>6.1f} | Expiry: {bb.expiry_date} (", nl=False)
                click.secho(exp_label, fg=exp_color, nl=False)
                click.echo(")")
        click.echo("")


@main.command("retry")
@click.option("--execution-id", "-e", required=True, help="Execution run identifier")
@click.option("--task-ids", default=None, help="Optional comma-separated specific task IDs to retry")
@click.option("--max-retries", default=3, help="Maximum retry attempts (default: 3)")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.pass_obj
def retry_command(
    app_ctx: ApplicationContext,
    execution_id: str,
    task_ids: Optional[str],
    max_retries: int,
    output_format: str,
) -> None:
    """Retry eligible failed tasks in an execution DAG."""
    ctx = _get_context(app_ctx)
    coordinator = ctx.get_coordinator()

    t_ids = [t.strip() for t in task_ids.split(",") if t.strip()] if task_ids else None
    try:
        result = coordinator.retry_failed_tasks(
            execution_id=execution_id,
            task_ids=t_ids,
            max_retries=max_retries,
        )
        if output_format == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            click.secho(f"\nRetry triggered for execution '{execution_id}'.", fg="green")
            click.echo(f"  Status:          {result.overall_status.value}")
            click.echo(f"  Completed Tasks: {result.completed_tasks}")
            click.echo(f"  Failed Tasks:    {result.failed_tasks}\n")
    except Exception as err:
        click.secho(f"Retry operation failed: {err}", fg="red", err=True)
        sys.exit(1)


@main.command("cancel")
@click.option("--execution-id", "-e", required=True, help="Execution run identifier")
@click.option("--reason", "-r", default="Cancelled via CLI", help="Cancellation rationale")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.pass_obj
def cancel_command(
    app_ctx: ApplicationContext,
    execution_id: str,
    reason: str,
    output_format: str,
) -> None:
    """Cancel an active or suspended execution run."""
    ctx = _get_context(app_ctx)
    coordinator = ctx.get_coordinator()

    try:
        result = coordinator.cancel_execution(execution_id=execution_id, reason=reason)
        if output_format == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            click.secho(f"\nExecution '{execution_id}' CANCELLED.", fg="yellow", bold=True)
            click.echo(f"  Status:          {result.overall_status.value}")
            click.echo(f"  Cancelled Tasks: {result.cancelled_tasks}\n")
    except Exception as err:
        click.secho(f"Cancel operation failed: {err}", fg="red", err=True)
        sys.exit(1)


@main.command("serve")
@click.option("--host", "-h", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
@click.option("--port", "-p", default=8000, type=int, help="Bind port (default: 8000)")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload for development")
def serve_command(host: str, port: int, reload: bool) -> None:
    """Launch the FastAPI REST API server via Uvicorn."""
    import uvicorn
    click.secho(f"Starting Demand-Consumption Pipeline API server on {host}:{port}...", fg="green", bold=True)
    uvicorn.run(
        "src.runtime.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
