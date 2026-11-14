"""
Unit and Integration Tests for Runtime FastAPI REST API
======================================================
Tests health, readiness, execution lifecycle, human approval security,
task breakdown, cancel/retry, and inventory ledger endpoints.
"""

import os
import shutil
import tempfile
import uuid
import pytest
from fastapi.testclient import TestClient

from src.runtime.config import RuntimeConfig
from src.runtime.context import ApplicationContext
from src.runtime.api import create_app
from src.agents.coordinator.schemas import (
    CoordinatorExecutionMode,
    CoordinatorRequest,
    TaskStatus,
)


@pytest.fixture
def runtime_env():
    """Sets up an isolated temporary database environment for API testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_runtime_api_")
    coord_db = os.path.join(temp_dir, "coordinator.db")
    fulfil_db = os.path.join(temp_dir, "fulfilment.db")

    cfg = RuntimeConfig(
        env="testing",
        log_level="INFO",
        coordinator_db_path=coord_db,
        fulfilment_db_path=fulfil_db,
        operator_api_key="test-secret-key-12345",
        authorized_operators=["admin", "ops_lead", "lead2_supervisor"],
    )
    ctx = ApplicationContext(config=cfg)
    ctx.startup()

    app = create_app(context=ctx)

    with TestClient(app) as client:
        yield {
            "client": client,
            "ctx": ctx,
            "cfg": cfg,
            "temp_dir": temp_dir,
        }

    ctx.shutdown()
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_health_endpoint(runtime_env):
    """Verifies GET /api/v1/health returns 200 OK with HEALTHY status."""
    client = runtime_env["client"]
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data


def test_openapi_documentation_accessible(runtime_env):
    """Verifies OpenAPI schema and Swagger UI documentation endpoints are accessible."""
    client = runtime_env["client"]
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    spec = openapi_resp.json()
    assert spec["info"]["title"] == "Autonomous Demand-Consumption Replenishment Pipeline"
    assert "/api/v1/executions" in spec["paths"]
    assert "/api/v1/inventory/balances" in spec["paths"]


def test_readiness_endpoint(runtime_env):
    """Verifies GET /api/v1/ready validates databases and components."""
    client = runtime_env["client"]
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["checks"]["coordinator_db"] == "OK"
    assert data["checks"]["fulfilment_db"] == "OK"


def test_start_execution_dry_run(runtime_env):
    """Verifies POST /api/v1/executions executes a dry run DAG to completion."""
    client = runtime_env["client"]
    req_id = f"REQ-API-DRY-{uuid.uuid4().hex[:8]}"
    payload = {
        "request": {
            "request_id": req_id,
            "operating_date": "2024-11-15",
            "target_stores": ["DS_BLR_HSR_LAYOUT"],
            "target_skus": ["SKU_MILK_1L"],
            "execution_mode": "DRY_RUN",
            "correlation_id": f"CORR-{req_id}",
        }
    }
    resp = client.post("/api/v1/executions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == req_id
    assert data["overall_status"] == "COMPLETED"
    assert data["completed_tasks"] >= 3
    assert len(data["draft_po_ids"]) >= 1


def test_get_execution_and_tasks(runtime_env):
    """Verifies retrieval of execution state and task breakdown."""
    client = runtime_env["client"]
    req_id = f"REQ-API-STATUS-{uuid.uuid4().hex[:8]}"
    payload = {
        "request": {
            "request_id": req_id,
            "operating_date": "2024-11-15",
            "target_stores": ["DS_BLR_HSR_LAYOUT"],
            "target_skus": ["SKU_MILK_1L"],
            "execution_mode": "DRY_RUN",
            "correlation_id": f"CORR-{req_id}",
        }
    }
    post_resp = client.post("/api/v1/executions", json=payload)
    assert post_resp.status_code == 200
    exec_id = post_resp.json()["execution_id"]

    # Retrieve execution details
    get_resp = client.get(f"/api/v1/executions/{exec_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["execution_id"] == exec_id

    # Retrieve task list
    tasks_resp = client.get(f"/api/v1/executions/{exec_id}/tasks")
    assert tasks_resp.status_code == 200
    tasks = tasks_resp.json()
    assert len(tasks) >= 3
    stages = [t["stage"] for t in tasks]
    assert "INGESTION" in stages
    assert "PROCUREMENT_PLANNING" in stages

    # Non-existent execution 404
    missing_resp = client.get("/api/v1/executions/EXEC-DOES-NOT-EXIST")
    assert missing_resp.status_code == 404

    missing_tasks = client.get("/api/v1/executions/EXEC-DOES-NOT-EXIST/tasks")
    assert missing_tasks.status_code == 404


def test_human_approval_resume_flow_and_security(runtime_env):
    """Verifies that human approval enforces security boundaries and resumes execution."""
    client = runtime_env["client"]
    req_id = f"REQ-API-LIVE-{uuid.uuid4().hex[:8]}"
    payload = {
        "request": {
            "request_id": req_id,
            "operating_date": "2024-11-15",
            "target_stores": ["DS_BLR_HSR_LAYOUT"],
            "target_skus": ["SKU_MILK_1L"],
            "execution_mode": "AUTOMATED_TO_APPROVAL",
            "correlation_id": f"CORR-{req_id}",
        }
    }
    start_resp = client.post("/api/v1/executions", json=payload)
    assert start_resp.status_code == 200
    exec_data = start_resp.json()
    exec_id = exec_data["execution_id"]
    assert exec_data["overall_status"] == "SUSPENDED"
    assert len(exec_data["draft_po_ids"]) >= 1
    po_id = exec_data["draft_po_ids"][0]

    resume_url = f"/api/v1/executions/{exec_id}/resume"

    # 1. Missing API Key -> 403 Forbidden
    unauthed_payload = {
        "decisions": [
            {
                "po_id": po_id,
                "action": "APPROVED",
                "approved_by": "admin",
                "approval_timestamp": "2024-11-15T12:00:00Z",
                "approval_reason": "Verified for morning rush",
            }
        ]
    }
    resp_no_key = client.post(resume_url, json=unauthed_payload)
    assert resp_no_key.status_code == 403
    assert "Invalid or missing operator security key" in resp_no_key.json()["detail"]

    # 2. Invalid API Key -> 403 Forbidden
    resp_bad_key = client.post(
        resume_url,
        json=unauthed_payload,
        headers={"X-API-Key": "wrong-secret-key"},
    )
    assert resp_bad_key.status_code == 403

    # 3. Unauthorized operator ID -> 403 Forbidden
    unauthorized_actor_payload = {
        "decisions": [
            {
                "po_id": po_id,
                "action": "APPROVED",
                "approved_by": "unauthorized_user_xyz",
                "approval_timestamp": "2024-11-15T12:00:00Z",
                "approval_reason": "Trying to bypass approval",
            }
        ]
    }
    resp_bad_actor = client.post(
        resume_url,
        json=unauthorized_actor_payload,
        headers={"X-API-Key": "test-secret-key-12345"},
    )
    assert resp_bad_actor.status_code == 403
    assert "not authorized for human approval operations" in resp_bad_actor.json()["detail"]

    # 3b. Authenticated operator identity replacement attempt -> 403 Forbidden
    spoof_payload = {
        "decisions": [
            {
                "po_id": po_id,
                "action": "APPROVED",
                "approved_by": "admin",
                "approval_timestamp": "2024-11-15T12:00:00Z",
                "approval_reason": "Spoofing identity attempt",
            }
        ]
    }
    resp_spoof = client.post(
        resume_url,
        json=spoof_payload,
        headers={"X-API-Key": "test-secret-key-12345", "X-Operator-Id": "ops_lead"},
    )
    assert resp_spoof.status_code == 403
    assert "does not match authenticated operator identity" in resp_spoof.json()["detail"]

    # 4. Incomplete governance payload -> 422 Unprocessable Entity
    incomplete_payload = {
        "decisions": [
            {
                "po_id": po_id,
                "action": "APPROVED",
                # missing approved_by and approval_timestamp
            }
        ]
    }
    resp_incomplete = client.post(
        resume_url,
        json=incomplete_payload,
        headers={"X-API-Key": "test-secret-key-12345"},
    )
    assert resp_incomplete.status_code == 422

    # 5. Valid Authorized Approval -> 200 OK and resumes pipeline
    valid_payload = {
        "decisions": [
            {
                "po_id": po_id,
                "action": "APPROVED",
                "approved_by": "admin",
                "approval_timestamp": "2024-11-15T12:00:00Z",
                "approval_reason": "Verified replenishment stock levels",
            }
        ]
    }
    resp_valid = client.post(
        resume_url,
        json=valid_payload,
        headers={"X-API-Key": "test-secret-key-12345"},
    )
    assert resp_valid.status_code == 200
    resumed_data = resp_valid.json()
    assert po_id in resumed_data["approved_po_ids"]
    assert len(resumed_data["transmitted_fo_ids"]) >= 1


def test_execution_cancellation(runtime_env):
    """Verifies cancelling a suspended execution transitions non-terminal tasks to CANCELLED."""
    client = runtime_env["client"]
    req_id = f"REQ-API-CANCEL-{uuid.uuid4().hex[:8]}"
    payload = {
        "request": {
            "request_id": req_id,
            "operating_date": "2024-11-15",
            "target_stores": ["DS_BLR_HSR_LAYOUT"],
            "target_skus": ["SKU_MILK_1L"],
            "execution_mode": "AUTOMATED_TO_APPROVAL",
            "correlation_id": f"CORR-{req_id}",
        }
    }
    start_resp = client.post("/api/v1/executions", json=payload)
    assert start_resp.status_code == 200
    exec_id = start_resp.json()["execution_id"]

    cancel_resp = client.post(
        f"/api/v1/executions/{exec_id}/cancel",
        json={"reason": "Operator cancelled due to weather disruption"},
    )
    assert cancel_resp.status_code == 200
    data = cancel_resp.json()
    assert data["overall_status"] == "CANCELLED"
    assert data["cancelled_tasks"] >= 1


def test_execution_retry_and_missing(runtime_env):
    """Verifies retry endpoint for executions and 404 for non-existent ones."""
    client = runtime_env["client"]
    # 404 for missing execution
    resp_missing = client.post(
        "/api/v1/executions/EXEC-NOT-FOUND/retry",
        json={"max_retries": 2},
    )
    assert resp_missing.status_code == 404

    # Dry run execution (no failed tasks) -> retry returns completed result
    req_id = f"REQ-API-RETRY-{uuid.uuid4().hex[:8]}"
    payload = {
        "request": {
            "request_id": req_id,
            "operating_date": "2024-11-15",
            "target_stores": ["DS_BLR_HSR_LAYOUT"],
            "target_skus": ["SKU_MILK_1L"],
            "execution_mode": "DRY_RUN",
            "correlation_id": f"CORR-{req_id}",
        }
    }
    start_resp = client.post("/api/v1/executions", json=payload)
    exec_id = start_resp.json()["execution_id"]

    resp_retry = client.post(
        f"/api/v1/executions/{exec_id}/retry",
        json={"max_retries": 3},
    )
    assert resp_retry.status_code == 200


def test_inventory_balances_endpoint(runtime_env):
    """Verifies GET /api/v1/inventory/balances queries store and batch balances."""
    client = runtime_env["client"]

    # Valid store query
    resp = client.get("/api/v1/inventory/balances?store_id=DS_BLR_HSR_LAYOUT")
    assert resp.status_code == 200
    data = resp.json()
    assert data["store_id"] == "DS_BLR_HSR_LAYOUT"
    assert "store_balances" in data
    assert "batch_balances" in data

    # Valid store and SKU query
    resp_sku = client.get("/api/v1/inventory/balances?store_id=DS_BLR_HSR_LAYOUT&sku_id=SKU_MILK_1L")
    assert resp_sku.status_code == 200
    data_sku = resp_sku.json()
    assert data_sku["sku_id"] == "SKU_MILK_1L"

    # Invalid store query -> 400 Bad Request
    resp_invalid_store = client.get("/api/v1/inventory/balances?store_id=INVALID_STORE_ID")
    assert resp_invalid_store.status_code == 400

    # Invalid SKU query -> 400 Bad Request
    resp_invalid_sku = client.get("/api/v1/inventory/balances?store_id=DS_BLR_HSR_LAYOUT&sku_id=INVALID_SKU")
    assert resp_invalid_sku.status_code == 400


def test_idempotency_conflict_endpoint(runtime_env):
    """Verifies that posting a conflicting request with same request_id returns 409 Conflict."""
    client = runtime_env["client"]
    req_id = f"REQ-API-IDEMP-{uuid.uuid4().hex[:8]}"
    payload1 = {
        "request": {
            "request_id": req_id,
            "operating_date": "2024-11-15",
            "target_stores": ["DS_BLR_HSR_LAYOUT"],
            "target_skus": ["SKU_MILK_1L"],
            "execution_mode": "DRY_RUN",
            "correlation_id": f"CORR-{req_id}",
        }
    }
    resp1 = client.post("/api/v1/executions", json=payload1)
    assert resp1.status_code == 200

    # Same request payload should be idempotent (200 OK)
    resp_idempotent = client.post("/api/v1/executions", json=payload1)
    assert resp_idempotent.status_code == 200

    # Altered payload with same request_id causes conflict (409 Conflict)
    payload2 = {
        "request": {
            "request_id": req_id,
            "operating_date": "2024-11-16",  # Different date!
            "target_stores": ["DS_BLR_HSR_LAYOUT"],
            "target_skus": ["SKU_MILK_1L"],
            "execution_mode": "DRY_RUN",
            "correlation_id": f"CORR-{req_id}",
        }
    }
    resp_conflict = client.post("/api/v1/executions", json=payload2)
    assert resp_conflict.status_code == 409
    assert "Idempotency conflict" in resp_conflict.json()["detail"]

