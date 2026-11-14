"""
Lead 2 Multi-Agent Coordinator Persistent SQLite Repository
===========================================================
Durable SQLite persistence for coordinator executions, DAG tasks, and immutable audit logs.
Provides atomic aggregate storage, optimistic concurrency control via versioning,
idempotent request replay, conflict protection, and crash/restart recovery.

GOVERNANCE & ARCHITECTURAL INVARIANTS:
1. IMMUTABILITY: Coordinator audit records are strictly append-only.
2. CONCURRENCY: Optimistic concurrency control using execution and task versioning rejects stale updates.
3. IDEMPOTENCY: Replaying the same request_id with identical payload returns the existing execution;
   conflicting payloads with the same request_id raise IdempotencyConflictError.
4. ISOLATION: The coordinator does not write directly to operational domain tables (fulfilment,
   procurement, inventory); it persists only coordinator orchestration aggregates.
"""

from datetime import datetime as dt_datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
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
from ..common.config import DEFAULT_COORDINATOR_DB_PATH


class CoordinatorPersistenceError(Exception):
    """Base exception for Coordinator repository and persistence operations."""
    pass


class ExecutionNotFoundError(CoordinatorPersistenceError):
    """Raised when an execution_id is not found in the repository."""
    pass


class TaskNotFoundError(CoordinatorPersistenceError):
    """Raised when a task_id is not found in the repository."""
    pass


class StaleExecutionError(CoordinatorPersistenceError):
    """Raised when an optimistic concurrency version conflict occurs on execution update."""
    pass


class IdempotencyConflictError(CoordinatorPersistenceError):
    """Raised when a request_id is reused with a different or conflicting payload."""
    pass


def _now_utc_str() -> str:
    """Helper returning current UTC ISO 8601 string."""
    return dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CoordinatorRepository:
    """
    Durable SQLite repository for Coordinator orchestration aggregates.
    Manages coordinator_executions, coordinator_tasks, and coordinator_audit_log.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_COORDINATOR_DB_PATH
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        """Creates and configures a safe SQLite connection with foreign keys and WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
            check_same_thread=False,
        )
        conn.execute("PRAGMA foreign_keys = ON;")
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)};")
        conn.row_factory = sqlite3.Row
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Returns the active connection, reconnecting if closed."""
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn

    def _init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                # 1. Coordinator Executions Header
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS coordinator_executions (
                        execution_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        operating_date TEXT NOT NULL,
                        execution_mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_stage TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        result_json TEXT,
                        operational_inputs_json TEXT,
                        payload_fingerprint TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coord_exec_request_id ON coordinator_executions(request_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coord_exec_status ON coordinator_executions(status);"
                )

                # Migration: ensure operational_inputs_json exists on preexisting databases
                cursor = conn.execute("PRAGMA table_info(coordinator_executions);")
                col_names = [r["name"] for r in cursor.fetchall()]
                if "operational_inputs_json" not in col_names:
                    conn.execute("ALTER TABLE coordinator_executions ADD COLUMN operational_inputs_json TEXT;")

                # 2. Coordinator Tasks
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS coordinator_tasks (
                        task_id TEXT PRIMARY KEY,
                        execution_id TEXT NOT NULL REFERENCES coordinator_executions(execution_id) ON DELETE CASCADE,
                        stage TEXT NOT NULL,
                        node_key TEXT NOT NULL,
                        status TEXT NOT NULL,
                        dependencies_json TEXT NOT NULL,
                        input_payload_json TEXT NOT NULL,
                        output_payload_json TEXT,
                        error_message TEXT,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        idempotency_key TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        parent_task_ids_json TEXT NOT NULL DEFAULT '[]',
                        child_task_ids_json TEXT NOT NULL DEFAULT '[]',
                        source_node_keys_json TEXT NOT NULL DEFAULT '[]',
                        provenance_json TEXT NOT NULL DEFAULT '{}',
                        version INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coord_task_exec_id ON coordinator_tasks(execution_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coord_task_status ON coordinator_tasks(status);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coord_task_stage ON coordinator_tasks(stage);"
                )

                # 3. Coordinator Audit Trail (Append-only)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS coordinator_audit_log (
                        event_id TEXT PRIMARY KEY,
                        execution_id TEXT NOT NULL REFERENCES coordinator_executions(execution_id) ON DELETE CASCADE,
                        timestamp TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        action TEXT NOT NULL,
                        actor_id TEXT NOT NULL,
                        task_id TEXT,
                        details_json TEXT NOT NULL,
                        payload_fingerprint TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coord_audit_exec_id ON coordinator_audit_log(execution_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coord_audit_timestamp ON coordinator_audit_log(timestamp);"
                )

    def close(self) -> None:
        """Closes the active database connection if open."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                finally:
                    self._conn = None

    def __enter__(self) -> "CoordinatorRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Execution Aggregates
    # -------------------------------------------------------------------------

    def create_execution(
        self,
        request: CoordinatorRequest,
        execution_id: Optional[str] = None,
        initial_status: TaskStatus = TaskStatus.RUNNING,
        initial_stage: CoordinatorStage = CoordinatorStage.INGESTION,
        operational_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registers a new execution run or idempotently returns existing run.
        
        Raises:
            IdempotencyConflictError: If request_id already exists with a different payload.
        """
        with self._lock:
            conn = self._get_connection()
            payload_to_fingerprint = {
                "request": request.model_dump(),
                "operational_inputs": operational_inputs or {},
            }
            incoming_fingerprint = compute_payload_fingerprint(payload_to_fingerprint)
            exec_id = execution_id or request.request_id

            with conn:
                cursor = conn.execute(
                    "SELECT * FROM coordinator_executions WHERE request_id = ?;",
                    (request.request_id,),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    stored_fp = existing["payload_fingerprint"]
                    if stored_fp != incoming_fingerprint:
                        raise IdempotencyConflictError(
                            f"Execution for request_id '{request.request_id}' already exists with different payload "
                            f"(stored fingerprint: '{stored_fp}', incoming: '{incoming_fingerprint}')."
                        )
                    return self._row_to_execution_dict(existing)

                now = _now_utc_str()
                op_json = json.dumps(operational_inputs, default=str) if operational_inputs is not None else None
                conn.execute(
                    """
                    INSERT INTO coordinator_executions (
                        execution_id, request_id, operating_date, execution_mode,
                        status, current_stage, request_json, result_json,
                        operational_inputs_json, payload_fingerprint, version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 1, ?, ?);
                    """,
                    (
                        exec_id,
                        request.request_id,
                        request.operating_date,
                        request.execution_mode.value,
                        initial_status.value,
                        initial_stage.value,
                        request.model_dump_json(),
                        op_json,
                        incoming_fingerprint,
                        now,
                        now,
                    ),
                )

            return self.get_execution(exec_id)  # type: ignore[return-value]

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an execution record by execution_id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM coordinator_executions WHERE execution_id = ?;",
                (execution_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_execution_dict(row)

    def get_execution_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an execution record by request_id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM coordinator_executions WHERE request_id = ?;",
                (request_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_execution_dict(row)

    def update_execution(
        self,
        execution_id: str,
        status: TaskStatus,
        current_stage: CoordinatorStage,
        result: Optional[CoordinatorResult] = None,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Updates an execution record with optimistic concurrency protection.

        Raises:
            ExecutionNotFoundError: If execution_id does not exist.
            StaleExecutionError: If expected_version does not match persisted version.
        """
        with self._lock:
            conn = self._get_connection()
            now = _now_utc_str()
            result_json = result.model_dump_json() if result is not None else None

            with conn:
                if expected_version is not None:
                    cursor = conn.execute(
                        """
                        UPDATE coordinator_executions
                        SET status = ?, current_stage = ?, result_json = COALESCE(?, result_json),
                            version = version + 1, updated_at = ?
                        WHERE execution_id = ? AND version = ?;
                        """,
                        (
                            status.value,
                            current_stage.value,
                            result_json,
                            now,
                            execution_id,
                            expected_version,
                        ),
                    )
                    if cursor.rowcount == 0:
                        # Check if execution even exists
                        check = conn.execute(
                            "SELECT version FROM coordinator_executions WHERE execution_id = ?;",
                            (execution_id,),
                        ).fetchone()
                        if check is None:
                            raise ExecutionNotFoundError(f"Execution '{execution_id}' does not exist.")
                        raise StaleExecutionError(
                            f"Optimistic lock conflict on execution '{execution_id}': "
                            f"expected version {expected_version}, but database is at version {check['version']}."
                        )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE coordinator_executions
                        SET status = ?, current_stage = ?, result_json = COALESCE(?, result_json),
                            version = version + 1, updated_at = ?
                        WHERE execution_id = ?;
                        """,
                        (
                            status.value,
                            current_stage.value,
                            result_json,
                            now,
                            execution_id,
                        ),
                    )
                    if cursor.rowcount == 0:
                        raise ExecutionNotFoundError(f"Execution '{execution_id}' does not exist.")

            return self.get_execution(execution_id)  # type: ignore[return-value]

    def _row_to_execution_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Parses execution SQLite row to structured dictionary."""
        op_inputs = None
        if "operational_inputs_json" in row.keys() and row["operational_inputs_json"] is not None:
            try:
                op_inputs = json.loads(row["operational_inputs_json"])
            except Exception:
                op_inputs = None

        return {
            "execution_id": row["execution_id"],
            "request_id": row["request_id"],
            "operating_date": row["operating_date"],
            "execution_mode": CoordinatorExecutionMode(row["execution_mode"]),
            "status": TaskStatus(row["status"]),
            "current_stage": CoordinatorStage(row["current_stage"]),
            "request": CoordinatorRequest.model_validate_json(row["request_json"]),
            "result": (
                CoordinatorResult.model_validate_json(row["result_json"])
                if row["result_json"] is not None
                else None
            ),
            "operational_inputs": op_inputs,
            "payload_fingerprint": row["payload_fingerprint"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # -------------------------------------------------------------------------
    # Tasks Persistence
    # -------------------------------------------------------------------------

    def save_task(self, task: TaskContext) -> None:
        """Saves or updates a single TaskContext."""
        self.save_tasks([task])

    def save_tasks(self, tasks: Sequence[TaskContext]) -> None:
        """
        Inserts or updates a sequence of TaskContexts within a transaction.
        """
        if not tasks:
            return

        with self._lock:
            conn = self._get_connection()
            now = _now_utc_str()

            with conn:
                for task in tasks:
                    conn.execute(
                        """
                        INSERT INTO coordinator_tasks (
                            task_id, execution_id, stage, node_key, status,
                            dependencies_json, input_payload_json, output_payload_json,
                            error_message, retry_count, idempotency_key,
                            started_at, completed_at, parent_task_ids_json,
                            child_task_ids_json, source_node_keys_json, provenance_json,
                            version, created_at, updated_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?
                        )
                        ON CONFLICT(task_id) DO UPDATE SET
                            status = excluded.status,
                            dependencies_json = excluded.dependencies_json,
                            input_payload_json = excluded.input_payload_json,
                            output_payload_json = excluded.output_payload_json,
                            error_message = excluded.error_message,
                            retry_count = excluded.retry_count,
                            started_at = excluded.started_at,
                            completed_at = excluded.completed_at,
                            parent_task_ids_json = excluded.parent_task_ids_json,
                            child_task_ids_json = excluded.child_task_ids_json,
                            source_node_keys_json = excluded.source_node_keys_json,
                            provenance_json = excluded.provenance_json,
                            version = coordinator_tasks.version + 1,
                            updated_at = excluded.updated_at;
                        """,
                        (
                            task.task_id,
                            task.execution_id,
                            task.stage.value,
                            task.node_key,
                            task.status.value,
                            json.dumps(list(task.dependencies)),
                            json.dumps(task.input_payload, default=str),
                            json.dumps(task.output_payload, default=str) if task.output_payload is not None else None,
                            task.error_message,
                            task.retry_count,
                            task.idempotency_key,
                            task.started_at,
                            task.completed_at,
                            json.dumps(list(task.parent_task_ids)),
                            json.dumps(list(task.child_task_ids)),
                            json.dumps(list(task.source_node_keys)),
                            json.dumps(task.provenance, default=str),
                            now,
                            now,
                        ),
                    )

    def get_task(self, task_id: str) -> Optional[TaskContext]:
        """Retrieves a single TaskContext by task_id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM coordinator_tasks WHERE task_id = ?;",
                (task_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_task_context(row)

    def get_tasks_for_execution(self, execution_id: str) -> List[TaskContext]:
        """Retrieves all TaskContexts for an execution, ordered by creation time."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM coordinator_tasks WHERE execution_id = ? ORDER BY created_at ASC, task_id ASC;",
                (execution_id,),
            )
            return [self._row_to_task_context(row) for row in cursor.fetchall()]

    def _row_to_task_context(self, row: sqlite3.Row) -> TaskContext:
        """Reconstructs TaskContext from a database row."""
        return TaskContext(
            task_id=row["task_id"],
            execution_id=row["execution_id"],
            stage=CoordinatorStage(row["stage"]),
            node_key=row["node_key"],
            status=TaskStatus(row["status"]),
            dependencies=tuple(json.loads(row["dependencies_json"])),
            input_payload=json.loads(row["input_payload_json"]),
            output_payload=json.loads(row["output_payload_json"]) if row["output_payload_json"] is not None else None,
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            idempotency_key=row["idempotency_key"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            parent_task_ids=tuple(json.loads(row["parent_task_ids_json"])),
            child_task_ids=tuple(json.loads(row["child_task_ids_json"])),
            source_node_keys=tuple(json.loads(row["source_node_keys_json"])),
            provenance=json.loads(row["provenance_json"]),
        )

    # -------------------------------------------------------------------------
    # Immutable Audit Log
    # -------------------------------------------------------------------------

    def save_audit_record(self, record: CoordinatorAuditRecord) -> None:
        """Appends a single immutable audit record."""
        self.save_audit_records([record])

    def save_audit_records(self, records: Sequence[CoordinatorAuditRecord]) -> None:
        """Appends a sequence of immutable audit records within a transaction."""
        if not records:
            return

        with self._lock:
            conn = self._get_connection()
            now = _now_utc_str()

            with conn:
                for rec in records:
                    conn.execute(
                        """
                        INSERT INTO coordinator_audit_log (
                            event_id, execution_id, timestamp, stage, action,
                            actor_id, task_id, details_json, payload_fingerprint, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            rec.event_id,
                            rec.execution_id,
                            rec.timestamp,
                            rec.stage.value,
                            rec.action,
                            rec.actor_id,
                            rec.task_id,
                            json.dumps(rec.details, default=str),
                            rec.payload_fingerprint,
                            now,
                        ),
                    )

    def get_audit_records_for_execution(self, execution_id: str) -> List[CoordinatorAuditRecord]:
        """Retrieves all audit records for an execution, ordered chronologically."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM coordinator_audit_log WHERE execution_id = ? ORDER BY timestamp ASC, created_at ASC;",
                (execution_id,),
            )
            records: List[CoordinatorAuditRecord] = []
            for row in cursor.fetchall():
                records.append(
                    CoordinatorAuditRecord(
                        execution_id=row["execution_id"],
                        event_id=row["event_id"],
                        timestamp=row["timestamp"],
                        stage=CoordinatorStage(row["stage"]),
                        action=row["action"],
                        actor_id=row["actor_id"],
                        task_id=row["task_id"],
                        details=json.loads(row["details_json"]),
                        payload_fingerprint=row["payload_fingerprint"],
                    )
                )
            return records

    # Convenience alias
    get_audit_log = get_audit_records_for_execution

    # -------------------------------------------------------------------------
    # Crash / Restart Recovery
    # -------------------------------------------------------------------------

    def recover_interrupted_executions(self) -> List[str]:
        """
        Identifies executions left in RUNNING status (e.g. from an abrupt crash or restart)
        and transitions any stuck RUNNING tasks to FAILED so they can be retried or inspected.
        Updates execution status to FAILED or SUSPENDED.
        Returns the list of recovered execution IDs.
        """
        with self._lock:
            conn = self._get_connection()
            now = _now_utc_str()
            recovered_ids: List[str] = []

            with conn:
                cursor = conn.execute(
                    "SELECT execution_id, current_stage FROM coordinator_executions WHERE status = ?;",
                    (TaskStatus.RUNNING.value,),
                )
                interrupted = cursor.fetchall()

                for row in interrupted:
                    eid = row["execution_id"]
                    current_stage = CoordinatorStage(row["current_stage"])
                    recovered_ids.append(eid)

                    # Reset any tasks stuck in RUNNING to FAILED
                    conn.execute(
                        """
                        UPDATE coordinator_tasks
                        SET status = ?, error_message = 'Interrupted by coordinator crash/restart',
                            completed_at = ?, updated_at = ?
                        WHERE execution_id = ? AND status = ?;
                        """,
                        (
                            TaskStatus.FAILED.value,
                            now,
                            now,
                            eid,
                            TaskStatus.RUNNING.value,
                        ),
                    )

                    # Update execution status to FAILED
                    conn.execute(
                        """
                        UPDATE coordinator_executions
                        SET status = ?, updated_at = ?, version = version + 1
                        WHERE execution_id = ?;
                        """,
                        (
                            TaskStatus.FAILED.value,
                            now,
                            eid,
                        ),
                    )

                    # Write audit event
                    audit_fp = compute_payload_fingerprint({"execution_id": eid, "reason": "CRASH_RECOVERY"})
                    conn.execute(
                        """
                        INSERT INTO coordinator_audit_log (
                            event_id, execution_id, timestamp, stage, action,
                            actor_id, task_id, details_json, payload_fingerprint, created_at
                        ) VALUES (?, ?, ?, ?, 'CRASH_RECOVERY_EXECUTED', 'COORDINATOR_RECOVERY', NULL, ?, ?, ?);
                        """,
                        (
                            f"EVT:{eid}:CRASH_RECOVERY",
                            eid,
                            now,
                            current_stage.value,
                            json.dumps({"message": "Recovered from unexpected process interruption"}),
                            audit_fp,
                            now,
                        ),
                    )

            return recovered_ids
