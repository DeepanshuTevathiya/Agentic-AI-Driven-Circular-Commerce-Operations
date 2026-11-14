"""
FastAPI REST API Module
=======================
Exposes standardized, secure REST endpoints for operational replenishment runs,
DAG task monitoring, human approval resumption, failed task retries, cancellation,
inventory ledger inspection, and health/readiness probes.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from .config import RuntimeConfig, get_runtime_config
from .context import ApplicationContext, get_application_context
from ..agents.coordinator.schemas import (
    CoordinatorRequest,
    CoordinatorResult,
    TaskContext,
    TaskStatus,
)
from ..agents.coordinator.repository import IdempotencyConflictError, ExecutionNotFoundError
from ..agents.procurement.approval import ApprovalGovernanceError, InvalidStatusTransitionError
from ..agents.common.config import CANONICAL_SKUS, CANONICAL_STORES

logger = logging.getLogger("pipeline.api")


# -----------------------------------------------------------------------------
# Request & Response Contracts
# -----------------------------------------------------------------------------

class StartExecutionPayload(BaseModel):
    """Payload to initiate a coordinator execution run."""
    model_config = ConfigDict(extra="forbid")

    request: CoordinatorRequest
    operational_inputs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional operational inputs such as custom daily inputs or tracking overrides",
    )


class HumanDecisionPayload(BaseModel):
    """Payload for human approval or rejection review on a draft Purchase Order."""
    model_config = ConfigDict(extra="forbid")

    po_id: str = Field(..., min_length=3, description="Purchase Order identifier")
    action: str = Field(..., pattern="^(APPROVED|REJECTED)$", description="Review decision action")
    approved_by: Optional[str] = Field(default=None, description="Authorized human approver ID")
    approval_timestamp: Optional[str] = Field(default=None, description="ISO 8601 UTC approval timestamp")
    approval_reason: Optional[str] = Field(default=None, description="Human approval rationale")
    rejected_by: Optional[str] = Field(default=None, description="Authorized human reviewer ID")
    rejection_timestamp: Optional[str] = Field(default=None, description="ISO 8601 UTC rejection timestamp")
    rejection_reason: Optional[str] = Field(default=None, description="Human rejection rationale")


class ResumeExecutionPayload(BaseModel):
    """Payload to resume a suspended execution with human decisions."""
    model_config = ConfigDict(extra="forbid")

    decisions: List[HumanDecisionPayload] = Field(
        ...,
        min_length=1,
        description="List of verified human decisions on pending purchase orders",
    )
    operational_inputs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional downstream operational overrides (e.g. carrier tracking, dock GRN)",
    )


class RetryTasksPayload(BaseModel):
    """Payload to retry failed DAG tasks."""
    model_config = ConfigDict(extra="forbid")

    task_ids: Optional[List[str]] = Field(default=None, description="Optional specific task IDs to retry")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts allowed")


class CancelExecutionPayload(BaseModel):
    """Payload to cancel an active or suspended execution."""
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="Cancelled by operator", min_length=3, description="Cancellation reason")


# -----------------------------------------------------------------------------
# Security & Authorization Helper
# -----------------------------------------------------------------------------

def _verify_operator_security(
    actor_id: str,
    config: RuntimeConfig,
    x_api_key: Optional[str] = None,
    authorization: Optional[str] = None,
    x_operator_id: Optional[str] = None,
) -> None:
    """
    Enforces the configurable operator authentication and authorization boundary:
    1. If OPERATOR_API_KEY is configured, mandates a valid API key or Bearer token.
    2. If X-Operator-Id header is provided, mandates that the caller-supplied actor
       in payload matches the authenticated operator identity (cannot be replaced).
    3. Validates that the actor_id is present in authorized_operators.
    """
    if config.operator_api_key:
        token = None
        if x_api_key:
            token = x_api_key.strip()
        elif authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()

        if not token or token != config.operator_api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid or missing operator security key (X-API-Key / Authorization header).",
            )

    if x_operator_id and x_operator_id.strip():
        auth_op = x_operator_id.strip()
        if auth_op != actor_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Caller-supplied actor '{actor_id}' does not match authenticated operator identity '{auth_op}'.",
            )
        if config.authorized_operators and auth_op not in config.authorized_operators:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Authenticated operator '{auth_op}' is not authorized for human approval operations.",
            )

    if config.authorized_operators:
        if actor_id not in config.authorized_operators:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Actor '{actor_id}' is not authorized for human approval operations.",
            )


# -----------------------------------------------------------------------------
# FastAPI Application Factory with Lifespan
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that initializes and tears down ApplicationContext."""
    ctx = getattr(app.state, "context", None)
    if ctx is None:
        ctx = get_application_context()
        app.state.context = ctx

    if not ctx.is_initialized:
        ctx.startup()
    yield
    if ctx.is_initialized:
        ctx.shutdown()


def create_app(context: Optional[ApplicationContext] = None) -> FastAPI:
    """Constructs and configures the FastAPI application instance."""
    app = FastAPI(
        title="Autonomous Demand-Consumption Replenishment Pipeline",
        description=(
            "Central operational API for autonomous demand forecasting, inventory evaluation, "
            "procurement planning, human approval gating, fulfilment transmission, shipment tracking, "
            "dock receiving, and operational inventory ledger postings."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    if context is not None:
        app.state.context = context

    def get_ctx() -> ApplicationContext:
        ctx = getattr(app.state, "context", None)
        if ctx is None:
            ctx = get_application_context()
            app.state.context = ctx
        if not ctx.is_initialized:
            ctx.startup()
        return ctx

    # -------------------------------------------------------------------------
    # Health & Readiness Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/v1/health", tags=["System"])
    async def get_health() -> Dict[str, Any]:
        """Liveness check returning service health and timestamp."""
        return {
            "status": "HEALTHY",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    @app.get("/api/v1/ready", tags=["System"])
    async def get_readiness(ctx: ApplicationContext = Depends(get_ctx)) -> JSONResponse:
        """Readiness check validating databases, datasets, and model artifacts."""
        readiness = ctx.verify_readiness()
        if not readiness["ready"]:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=readiness,
            )
        return JSONResponse(status_code=status.HTTP_200_OK, content=readiness)

    # -------------------------------------------------------------------------
    # Execution Lifecycle Endpoints
    # -------------------------------------------------------------------------

    @app.post("/api/v1/executions", response_model=CoordinatorResult, status_code=status.HTTP_200_OK, tags=["Executions"])
    async def start_execution(
        payload: StartExecutionPayload,
        ctx: ApplicationContext = Depends(get_ctx),
    ) -> CoordinatorResult:
        """
        Initiates a coordinator execution run or idempotently replays an existing run.
        """
        coordinator = ctx.get_coordinator()
        try:
            result = coordinator.start_execution(
                request=payload.request,
                operational_inputs=payload.operational_inputs,
            )
            return result
        except IdempotencyConflictError as err:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Idempotency conflict: {err}",
            )
        except Exception as err:
            logger.error(f"Error starting execution: {err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Execution request failed: {err}",
            )

    @app.get("/api/v1/executions/{execution_id}", response_model=CoordinatorResult, tags=["Executions"])
    async def get_execution_status(
        execution_id: str,
        ctx: ApplicationContext = Depends(get_ctx),
    ) -> CoordinatorResult:
        """Retrieves consolidated outcome, metrics, and audit history for an execution."""
        coordinator = ctx.get_coordinator()
        result = coordinator.get_execution(execution_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found.",
            )
        return result

    @app.get("/api/v1/executions/{execution_id}/tasks", response_model=List[TaskContext], tags=["Executions"])
    async def get_execution_tasks(
        execution_id: str,
        ctx: ApplicationContext = Depends(get_ctx),
    ) -> List[TaskContext]:
        """Retrieves all planned, executing, and completed tasks in the execution DAG."""
        coordinator = ctx.get_coordinator()
        tasks = coordinator.get_execution_tasks(execution_id)
        if not tasks:
            # Check if execution even exists
            if coordinator.get_execution(execution_id) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Execution '{execution_id}' not found.",
                )
        return tasks

    @app.post("/api/v1/executions/{execution_id}/resume", response_model=CoordinatorResult, tags=["Executions"])
    async def resume_execution(
        execution_id: str,
        payload: ResumeExecutionPayload,
        ctx: ApplicationContext = Depends(get_ctx),
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
        authorization: Optional[str] = Header(None, alias="Authorization"),
        x_operator_id: Optional[str] = Header(None, alias="X-Operator-Id"),
    ) -> CoordinatorResult:
        """
        Resumes a paused (SUSPENDED) execution by applying verified human review decisions.
        Mandates Phase 4 human approval security and operator authorization.
        """
        coordinator = ctx.get_coordinator()

        # 1. Check execution existence
        existing = coordinator.get_execution(execution_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found.",
            )

        # 2. Validate human governance decisions
        validated_decisions: List[Dict[str, Any]] = []
        for dec in payload.decisions:
            if dec.action == "APPROVED":
                if not dec.approved_by or not dec.approval_timestamp:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Mandatory Governance: Approval for PO '{dec.po_id}' requires 'approved_by' and 'approval_timestamp'.",
                    )
                _verify_operator_security(
                    actor_id=dec.approved_by,
                    config=ctx.config,
                    x_api_key=x_api_key,
                    authorization=authorization,
                    x_operator_id=x_operator_id,
                )
                validated_decisions.append(dec.model_dump())

            elif dec.action == "REJECTED":
                if not dec.rejected_by or not dec.rejection_timestamp or not dec.rejection_reason:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Mandatory Governance: Rejection for PO '{dec.po_id}' requires 'rejected_by', 'rejection_timestamp', and 'rejection_reason'.",
                    )
                _verify_operator_security(
                    actor_id=dec.rejected_by,
                    config=ctx.config,
                    x_api_key=x_api_key,
                    authorization=authorization,
                    x_operator_id=x_operator_id,
                )
                validated_decisions.append(dec.model_dump())

        # 3. Resume DAG execution
        try:
            result = coordinator.resume_execution(
                execution_id=execution_id,
                human_decisions=validated_decisions,
                operational_inputs=payload.operational_inputs,
            )
            return result
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            )
        except Exception as err:
            logger.error(f"Error resuming execution: {err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Resume operation failed: {err}",
            )

    @app.post("/api/v1/executions/{execution_id}/retry", response_model=CoordinatorResult, tags=["Executions"])
    async def retry_failed_tasks(
        execution_id: str,
        payload: RetryTasksPayload,
        ctx: ApplicationContext = Depends(get_ctx),
    ) -> CoordinatorResult:
        """Retries eligible tasks in FAILED status within retry limits and resumes the DAG."""
        coordinator = ctx.get_coordinator()
        existing = coordinator.get_execution(execution_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found.",
            )

        try:
            result = coordinator.retry_failed_tasks(
                execution_id=execution_id,
                task_ids=payload.task_ids,
                max_retries=payload.max_retries,
            )
            return result
        except Exception as err:
            logger.error(f"Error retrying tasks: {err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Retry operation failed: {err}",
            )

    @app.post("/api/v1/executions/{execution_id}/cancel", response_model=CoordinatorResult, tags=["Executions"])
    async def cancel_execution(
        execution_id: str,
        payload: CancelExecutionPayload,
        ctx: ApplicationContext = Depends(get_ctx),
    ) -> CoordinatorResult:
        """Cancels an active or suspended execution and marks non-terminal tasks as CANCELLED."""
        coordinator = ctx.get_coordinator()
        existing = coordinator.get_execution(execution_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found.",
            )

        try:
            result = coordinator.cancel_execution(
                execution_id=execution_id,
                reason=payload.reason,
            )
            return result
        except Exception as err:
            logger.error(f"Error cancelling execution: {err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Cancel operation failed: {err}",
            )

    # -------------------------------------------------------------------------
    # Inventory Ledger Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/v1/inventory/balances", tags=["Inventory"])
    async def get_inventory_balances(
        store_id: str = Query(..., description="Dark store identifier (e.g. DS_BLR_HSR_LAYOUT)"),
        sku_id: Optional[str] = Query(None, description="Optional SKU identifier (e.g. SKU_MILK_1L)"),
        ctx: ApplicationContext = Depends(get_ctx),
    ) -> Dict[str, Any]:
        """Queries authoritative store-level balances and FEFO batch perishability tracking."""
        if store_id not in CANONICAL_STORES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid store_id '{store_id}'. Must be one of canonical stores: {CANONICAL_STORES}",
            )
        if sku_id is not None and sku_id not in CANONICAL_SKUS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sku_id '{sku_id}'. Must be one of canonical SKUs: {CANONICAL_SKUS}",
            )

        inv_repo = ctx.get_inventory_repository()
        store_balances = inv_repo.get_store_balances(store_id=store_id)
        if sku_id:
            store_balances = [sb for sb in store_balances if sb.sku_id == sku_id]
        batch_balances = inv_repo.get_batch_balances(store_id=store_id, sku_id=sku_id)

        return {
            "store_id": store_id,
            "sku_id": sku_id,
            "store_balances": [sb.model_dump() for sb in store_balances],
            "batch_balances": [bb.model_dump() for bb in batch_balances],
        }

    return app


# Default app instance
app = create_app()
