"""
Lead 2 Fulfilment Persistence & Repository Module
================================================
Durable SQLite persistence for the fulfilment domain.
Provides atomic aggregate storage, relational indexing, append-only event auditing,
scoped idempotency protection, optimistic concurrency control, and dynamic pipeline queries.

GOVERNANCE RULES:
1. ATOMICITY: Aggregate header, lines, append-only events, idempotency tokens, and
   shipment/GRN references commit together or roll back together.
2. IMMUTABILITY: Historical audit events and source approved PurchaseOrders are never mutated.
3. CONCURRENCY: Stale concurrent updates are detected and rejected using sequence/version checks.
4. PIPELINE FIDELITY: Pipeline quantities are derived strictly on-the-fly from active confirmed
   orders without maintaining a separate mutable counter.
"""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from src.agents.common.config import DEFAULT_FULFILMENT_DB_PATH
from src.agents.fulfilment.schemas import (
    DiscrepancyReason,
    DiscrepancyResolution,
    DiscrepancyResolutionType,
    DockInspectionLine,
    FulfilmentActorType,
    FulfilmentEvent,
    FulfilmentEventType,
    FulfilmentOrder,
    FulfilmentOrderLine,
    FulfilmentOrderStatus,
    GoodsReceiptNote,
    QualityHoldResolution,
    compute_payload_fingerprint,
)
from src.agents.fulfilment.state_machine import (
    FulfilmentGovernanceError,
    validate_inbound_fulfilment_order,
)
from .tracking import (
    InvalidTrackingEventError,
    ShipmentNotFoundError,
    ShipmentTrackingEvent,
    ShipmentTrackingRecord,
    ShipmentTrackingStatus,
    TrackingIdempotencyConflictError,
    TrackingLocation,
)


class FulfilmentPersistenceError(Exception):
    """Base exception for fulfilment persistence and repository operations."""
    pass


class FulfilmentOrderNotFoundError(FulfilmentPersistenceError):
    """Raised when an expected FulfilmentOrder is not found in persistence."""
    pass


class DuplicateFulfilmentOrderError(FulfilmentPersistenceError):
    """Raised when attempting to insert a FulfilmentOrder whose fo_id already exists."""
    pass


class StaleUpdateError(FulfilmentPersistenceError):
    """Raised when an update fails because the persisted record was modified concurrently."""
    pass


class IdempotencyConflictError(FulfilmentPersistenceError):
    """Raised when an idempotency key is reused with a conflicting payload or event type."""
    pass


# Active statuses whose outstanding confirmed quantities contribute to inbound pipeline
ACTIVE_PIPELINE_STATUSES: Tuple[str, ...] = (
    FulfilmentOrderStatus.ACKNOWLEDGED.value,
    FulfilmentOrderStatus.PARTIALLY_ACKNOWLEDGED.value,
    FulfilmentOrderStatus.IN_TRANSIT.value,
    FulfilmentOrderStatus.PARTIALLY_RECEIVED.value,
    FulfilmentOrderStatus.DELIVERY_EXCEPTION.value,
)


class FulfilmentRepository:
    """
    Durable SQLite repository for FulfilmentOrder aggregates.
    Enforces relational constraints, optimistic concurrency versioning,
    append-only audit history, and dynamic pipeline calculation.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        timeout: float = 10.0,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_FULFILMENT_DB_PATH
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None

        # Auto-create parent directory if file-based database
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        """Configures safe SQLite connection with foreign keys and WAL mode."""
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
        """Returns the active connection, reconnecting if previously closed."""
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn

    def _init_db(self) -> None:
        """Creates the relational schema tables and indexes if they do not exist."""
        conn = self._get_connection()
        with conn:
            # 1. Fulfilment Order Header & Aggregate Snapshot
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_orders (
                    fo_id TEXT PRIMARY KEY,
                    po_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL,
                    supplier_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expected_delivery_at TEXT,
                    cancellation_reason TEXT,
                    cancellation_category TEXT,
                    cancelled_by TEXT,
                    cancellation_timestamp TEXT,
                    supplier_cancellation_ref TEXT,
                    shipment_cancellation_ref TEXT,
                    order_wide_cancellation_scope INTEGER NOT NULL DEFAULT 0,
                    closed_by TEXT,
                    closure_timestamp TEXT,
                    closure_reason TEXT,
                    closed_unfulfilled_qty REAL,
                    open_exception_reason TEXT,
                    open_exception_timestamp TEXT,
                    resumable_status TEXT,
                    is_simulation INTEGER NOT NULL DEFAULT 1,
                    version INTEGER NOT NULL DEFAULT 1,
                    aggregate_payload TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fo_store_status ON fulfilment_orders (store_id, status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fo_po_id ON fulfilment_orders (po_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fo_supplier_status ON fulfilment_orders (supplier_id, status);")

            # 2. Line Items Table (Relational Balances)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_order_lines (
                    fo_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    ordered_quantity REAL NOT NULL,
                    unit_cost REAL NOT NULL,
                    supplier_accepted_quantity REAL NOT NULL DEFAULT 0.0,
                    supplier_declined_quantity REAL NOT NULL DEFAULT 0.0,
                    supplier_unconfirmed_quantity REAL NOT NULL DEFAULT 0.0,
                    shipped_quantity REAL NOT NULL DEFAULT 0.0,
                    delivered_quantity REAL NOT NULL DEFAULT 0.0,
                    accepted_quantity REAL NOT NULL DEFAULT 0.0,
                    rejected_quantity REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (fo_id, sku_id),
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fol_sku ON fulfilment_order_lines (sku_id);")

            # 3. Append-Only Audit Events Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_events (
                    event_id TEXT PRIMARY KEY,
                    fo_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT NOT NULL,
                    related_reference_id TEXT,
                    payload_fingerprint TEXT,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE RESTRICT,
                    UNIQUE (fo_id, sequence_number),
                    UNIQUE (fo_id, idempotency_key)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_fo_seq ON fulfilment_events (fo_id, sequence_number);")

            # 4. Scoped Idempotency Records
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_idempotency_records (
                    idempotency_key TEXT PRIMARY KEY,
                    fo_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_fingerprint TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE
                );
                """
            )

            # 5. Shipment Identifiers & Resolution Evidence
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_shipment_ids (
                    fo_id TEXT NOT NULL,
                    shipment_id TEXT NOT NULL,
                    is_resolved INTEGER NOT NULL DEFAULT 0,
                    resolution_evidence TEXT,
                    PRIMARY KEY (fo_id, shipment_id),
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE
                );
                """
            )

            # 6. Goods Receipt Note Identifiers
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_grn_ids (
                    fo_id TEXT NOT NULL,
                    grn_id TEXT NOT NULL,
                    PRIMARY KEY (fo_id, grn_id),
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE
                );
                """
            )

            # 7. Gateway Transmission Records (Phase 5.4.2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_transmissions (
                    idempotency_key TEXT PRIMARY KEY,
                    fo_id TEXT NOT NULL,
                    po_id TEXT NOT NULL,
                    transmission_id TEXT,
                    correlation_id TEXT NOT NULL,
                    payload_fingerprint TEXT NOT NULL,
                    transmission_status TEXT NOT NULL,
                    transmitted_at TEXT NOT NULL,
                    completed_at TEXT,
                    supplier_reference TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transmissions_fo ON fulfilment_transmissions (fo_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transmissions_po ON fulfilment_transmissions (po_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transmissions_status ON fulfilment_transmissions (transmission_status);")

            # 8. Shipment Tracking Projection Records (Phase 5.5.2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shipment_tracking_records (
                    shipment_id TEXT PRIMARY KEY,
                    fo_id TEXT NOT NULL,
                    carrier_id TEXT NOT NULL,
                    current_status TEXT NOT NULL,
                    scenario TEXT DEFAULT 'NORMAL_DELIVERY',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    estimated_delivery_at TEXT,
                    actual_delivery_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (fo_id, shipment_id) REFERENCES fulfilment_shipment_ids(fo_id, shipment_id) ON DELETE RESTRICT
                );
                """
            )
            # Safe forward-compatible schema evolution if table already exists without scenario column
            try:
                cur_cols = conn.execute("PRAGMA table_info(shipment_tracking_records);").fetchall()
                col_names = {c["name"] for c in cur_cols}
                if "scenario" not in col_names and "shipment_id" in col_names:
                    conn.execute("ALTER TABLE shipment_tracking_records ADD COLUMN scenario TEXT DEFAULT 'NORMAL_DELIVERY';")
            except Exception:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_records_fo ON shipment_tracking_records (fo_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_records_carrier ON shipment_tracking_records (carrier_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_records_status ON shipment_tracking_records (current_status);")

            # 9. Shipment Tracking Events History (Phase 5.5.2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shipment_tracking_events (
                    event_id TEXT PRIMARY KEY,
                    shipment_id TEXT NOT NULL,
                    carrier_id TEXT NOT NULL,
                    provider_event_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    received_timestamp TEXT NOT NULL,
                    location_json TEXT,
                    exception_reason TEXT,
                    exception_details_json TEXT,
                    notes TEXT,
                    raw_payload_json TEXT,
                    payload_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (shipment_id) REFERENCES shipment_tracking_records(shipment_id) ON DELETE RESTRICT,
                    UNIQUE (carrier_id, shipment_id, provider_event_id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_events_shipment_ts ON shipment_tracking_events (shipment_id, event_timestamp ASC);")

            # 10. Goods Receipt Notes Header Table (Phase 5.6.1)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_goods_receipt_notes (
                    grn_id TEXT PRIMARY KEY,
                    fo_id TEXT NOT NULL,
                    po_id TEXT NOT NULL,
                    shipment_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL,
                    receipt_at TEXT NOT NULL,
                    receiver_id TEXT NOT NULL,
                    supervisor_signoff_by TEXT,
                    supervisor_signoff_at TEXT,
                    max_dock_discrepancy_cost_inr REAL NOT NULL,
                    rejected_goods_disposition TEXT,
                    notes TEXT,
                    total_delivered_quantity REAL NOT NULL,
                    total_accepted_quantity REAL NOT NULL,
                    total_rejected_quantity REAL NOT NULL,
                    total_accepted_cost REAL NOT NULL,
                    total_rejected_cost REAL NOT NULL,
                    payload_fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE RESTRICT,
                    FOREIGN KEY (fo_id, shipment_id) REFERENCES fulfilment_shipment_ids(fo_id, shipment_id) ON DELETE RESTRICT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_fo ON fulfilment_goods_receipt_notes (fo_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_shipment ON fulfilment_goods_receipt_notes (shipment_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_po ON fulfilment_goods_receipt_notes (po_id);")

            # 11. Goods Receipt Dock Inspection Lines Table (Phase 5.6.1)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_grn_inspection_lines (
                    grn_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL,
                    delivered_quantity REAL NOT NULL,
                    accepted_quantity REAL NOT NULL,
                    rejected_quantity REAL NOT NULL,
                    damaged_quantity REAL NOT NULL,
                    excess_quantity REAL NOT NULL,
                    short_shelf_life_quantity REAL NOT NULL,
                    unit_cost REAL NOT NULL,
                    remaining_shelf_life_days INTEGER,
                    total_shelf_life_days INTEGER,
                    discrepancy_reasons TEXT,
                    batch_references TEXT,
                    damaged_line_cost REAL NOT NULL,
                    rejected_line_cost REAL NOT NULL,
                    PRIMARY KEY (grn_id, sku_id),
                    FOREIGN KEY (grn_id) REFERENCES fulfilment_goods_receipt_notes(grn_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_lines_sku ON fulfilment_grn_inspection_lines (sku_id);")

            # 12. Discrepancy & Rejected-Goods Resolutions Table (Phase 5.6.2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_discrepancy_resolutions (
                    resolution_id TEXT PRIMARY KEY,
                    fo_id TEXT NOT NULL,
                    shipment_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL,
                    resolution_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    discrepancy_reason TEXT NOT NULL,
                    disposition_reference TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE,
                    FOREIGN KEY (fo_id, shipment_id) REFERENCES fulfilment_shipment_ids(fo_id, shipment_id) ON DELETE RESTRICT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_disc_res_fo ON fulfilment_discrepancy_resolutions (fo_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_disc_res_shipment ON fulfilment_discrepancy_resolutions (shipment_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_disc_res_sku ON fulfilment_discrepancy_resolutions (sku_id);")

            # 13. Quality Hold & Reinspection Resolutions Table (Phase 5.6.2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_quality_hold_resolutions (
                    resolution_id TEXT PRIMARY KEY,
                    fo_id TEXT NOT NULL,
                    shipment_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL,
                    resolution_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    disposition_reference TEXT NOT NULL,
                    inspector_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    verified_remaining_shelf_life_days INTEGER,
                    total_shelf_life_days INTEGER,
                    inspection_notes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE,
                    FOREIGN KEY (fo_id, shipment_id) REFERENCES fulfilment_shipment_ids(fo_id, shipment_id) ON DELETE RESTRICT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qh_res_fo ON fulfilment_quality_hold_resolutions (fo_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qh_res_shipment ON fulfilment_quality_hold_resolutions (shipment_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qh_res_sku ON fulfilment_quality_hold_resolutions (sku_id);")

            # 14. GRN Inventory Posting Tasks & Durable Workflow Records (Phase 5.7.2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulfilment_grn_postings (
                    posting_task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL DEFAULT 'GRN_RECEIPT',
                    resolution_id TEXT,
                    grn_id TEXT NOT NULL,
                    fo_id TEXT NOT NULL,
                    po_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_fingerprint TEXT NOT NULL,
                    total_accepted_quantity REAL NOT NULL,
                    posted_quantity REAL NOT NULL DEFAULT 0.0,
                    error_message TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    posting_ids_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE CASCADE,
                    FOREIGN KEY (grn_id) REFERENCES fulfilment_goods_receipt_notes(grn_id) ON DELETE CASCADE
                );
                """
            )
            # Safe schema upgrade check if table was created in an earlier session
            cur_post_cols = [r["name"] for r in conn.execute("PRAGMA table_info(fulfilment_grn_postings);").fetchall()]
            if cur_post_cols and "task_type" not in cur_post_cols:
                conn.execute("ALTER TABLE fulfilment_grn_postings ADD COLUMN task_type TEXT NOT NULL DEFAULT 'GRN_RECEIPT';")
            if cur_post_cols and "resolution_id" not in cur_post_cols:
                conn.execute("ALTER TABLE fulfilment_grn_postings ADD COLUMN resolution_id TEXT;")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_postings_grn ON fulfilment_grn_postings (grn_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_postings_fo ON fulfilment_grn_postings (fo_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_postings_status ON fulfilment_grn_postings (status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_postings_res ON fulfilment_grn_postings (resolution_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_grn_postings_type ON fulfilment_grn_postings (task_type);")

    # -------------------------------------------------------------------------
    # Core Repository CRUD Operations
    # -------------------------------------------------------------------------

    def _create_order_internal(self, conn: sqlite3.Connection, fo: FulfilmentOrder) -> None:
        """Internal worker executing INSERT statements for an order inside an active transaction."""
        # 1. Duplicate ID check
        cur = conn.execute("SELECT 1 FROM fulfilment_orders WHERE fo_id = ?", (fo.fo_id,))
        if cur.fetchone():
            raise DuplicateFulfilmentOrderError(f"FulfilmentOrder '{fo.fo_id}' already exists in persistence.")

        # 2. Check idempotency keys in event history
        for event in fo.event_history:
            cur = conn.execute(
                "SELECT fo_id, event_type, payload_fingerprint FROM fulfilment_idempotency_records WHERE idempotency_key = ?",
                (event.idempotency_key,),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["fo_id"] != fo.fo_id
                    or existing["event_type"] != event.event_type.value
                    or (existing["payload_fingerprint"] and existing["payload_fingerprint"] != event.payload_fingerprint)
                ):
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: key '{event.idempotency_key}' previously recorded for fo_id "
                        f"'{existing['fo_id']}' with event '{existing['event_type']}', cannot be reused for '{event.event_type.value}'."
                    )

        # 3. Insert Header
        payload_json = fo.model_dump_json()
        conn.execute(
            """
            INSERT INTO fulfilment_orders (
                fo_id, po_id, store_id, supplier_id, supplier_type, status,
                created_at, updated_at, expected_delivery_at,
                cancellation_reason, cancellation_category, cancelled_by, cancellation_timestamp,
                supplier_cancellation_ref, shipment_cancellation_ref, order_wide_cancellation_scope,
                closed_by, closure_timestamp, closure_reason, closed_unfulfilled_qty,
                open_exception_reason, open_exception_timestamp, resumable_status,
                is_simulation, version, aggregate_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                fo.fo_id, fo.po_id, fo.store_id, fo.supplier_id, fo.supplier_type.value, fo.status.value,
                fo.created_at, fo.updated_at, fo.expected_delivery_at,
                fo.cancellation_reason,
                fo.cancellation_category.value if fo.cancellation_category else None,
                fo.cancelled_by, fo.cancellation_timestamp,
                fo.supplier_cancellation_ref, fo.shipment_cancellation_ref,
                1 if fo.order_wide_cancellation_scope else 0,
                fo.closed_by, fo.closure_timestamp, fo.closure_reason, fo.closed_unfulfilled_qty,
                fo.open_exception_reason, fo.open_exception_timestamp,
                fo.resumable_status.value if fo.resumable_status else None,
                1 if fo.is_simulation else 0,
                payload_json,
            ),
        )

        # 4. Insert Line items
        for line in fo.lines:
            conn.execute(
                """
                INSERT INTO fulfilment_order_lines (
                    fo_id, sku_id, category, ordered_quantity, unit_cost,
                    supplier_accepted_quantity, supplier_declined_quantity, supplier_unconfirmed_quantity,
                    shipped_quantity, delivered_quantity, accepted_quantity, rejected_quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fo.fo_id, line.sku_id, line.category, line.ordered_quantity, line.unit_cost,
                    line.supplier_accepted_quantity, line.supplier_declined_quantity, line.supplier_unconfirmed_quantity,
                    line.shipped_quantity, line.delivered_quantity, line.accepted_quantity, line.rejected_quantity,
                ),
            )

        # 5. Insert Events & Scoped Idempotency Records
        for event in fo.event_history:
            conn.execute(
                """
                INSERT INTO fulfilment_events (
                    event_id, fo_id, sequence_number, idempotency_key, event_type,
                    actor_type, actor_id, timestamp, previous_status, new_status,
                    related_reference_id, payload_fingerprint, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id, event.fo_id, event.sequence_number, event.idempotency_key, event.event_type.value,
                    event.actor_type.value, event.actor_id, event.timestamp,
                    event.previous_status.value if event.previous_status else None,
                    event.new_status.value,
                    event.related_reference_id, event.payload_fingerprint,
                    json.dumps(event.details),
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO fulfilment_idempotency_records (
                    idempotency_key, fo_id, event_type, payload_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.idempotency_key, event.fo_id, event.event_type.value,
                    event.payload_fingerprint, event.timestamp,
                ),
            )

        # 6. Insert Shipment records
        resolved_set = set(fo.resolved_shipment_ids)
        for s_id in fo.shipment_ids:
            is_res = 1 if s_id in resolved_set else 0
            evidence = fo.shipment_resolution_evidence.get(s_id)
            conn.execute(
                """
                INSERT INTO fulfilment_shipment_ids (fo_id, shipment_id, is_resolved, resolution_evidence)
                VALUES (?, ?, ?, ?)
                """,
                (fo.fo_id, s_id, is_res, evidence),
            )

        # 7. Insert GRN records
        for g_id in fo.grn_ids:
            conn.execute(
                "INSERT INTO fulfilment_grn_ids (fo_id, grn_id) VALUES (?, ?)",
                (fo.fo_id, g_id),
            )

        # 8. Insert Discrepancy Resolutions
        for res in fo.discrepancy_resolutions:
            cur = conn.execute(
                "SELECT fo_id, shipment_id, sku_id, resolution_type, quantity, disposition_reference "
                "FROM fulfilment_discrepancy_resolutions WHERE resolution_id = ?",
                (res.resolution_id,),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["fo_id"] != res.fo_id
                    or existing["shipment_id"] != res.shipment_id
                    or existing["sku_id"] != res.sku_id
                    or existing["resolution_type"] != res.resolution_type.value
                    or abs(existing["quantity"] - res.quantity) > 1e-4
                    or existing["disposition_reference"] != res.disposition_reference
                ):
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: resolution_id '{res.resolution_id}' already exists with differing attributes."
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO fulfilment_discrepancy_resolutions (
                        resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                        quantity, discrepancy_reason, disposition_reference, actor_id,
                        timestamp, notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        res.resolution_id, res.fo_id, res.shipment_id, res.sku_id, res.resolution_type.value,
                        res.quantity, res.discrepancy_reason.value, res.disposition_reference, res.actor_id,
                        res.timestamp, res.notes, res.timestamp,
                    ),
                )

        # 9. Insert Quality Hold Resolutions
        for qhr in fo.quality_hold_resolutions:
            cur = conn.execute(
                "SELECT fo_id, shipment_id, sku_id, resolution_type, quantity, disposition_reference "
                "FROM fulfilment_quality_hold_resolutions WHERE resolution_id = ?",
                (qhr.resolution_id,),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["fo_id"] != qhr.fo_id
                    or existing["shipment_id"] != qhr.shipment_id
                    or existing["sku_id"] != qhr.sku_id
                    or existing["resolution_type"] != qhr.resolution_type.value
                    or abs(existing["quantity"] - qhr.quantity) > 1e-4
                    or existing["disposition_reference"] != qhr.disposition_reference
                ):
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: quality hold resolution_id '{qhr.resolution_id}' already exists with differing attributes."
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO fulfilment_quality_hold_resolutions (
                        resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                        quantity, disposition_reference, inspector_id, timestamp,
                        verified_remaining_shelf_life_days, total_shelf_life_days,
                        inspection_notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        qhr.resolution_id, qhr.fo_id, qhr.shipment_id, qhr.sku_id, qhr.resolution_type.value,
                        qhr.quantity, qhr.disposition_reference, qhr.inspector_id, qhr.timestamp,
                        qhr.verified_remaining_shelf_life_days, qhr.total_shelf_life_days,
                        qhr.inspection_notes, qhr.timestamp,
                    ),
                )

    def _save_grn_internal(self, conn: sqlite3.Connection, grn: GoodsReceiptNote) -> None:
        """Saves a GoodsReceiptNote and its inspection lines inside an active transaction."""
        payload_dict = {
            "grn_id": grn.grn_id,
            "fo_id": grn.fo_id,
            "po_id": grn.po_id,
            "shipment_id": grn.shipment_id,
            "store_id": grn.store_id,
            "supplier_id": grn.supplier_id,
            "receipt_at": grn.receipt_at,
            "receiver_id": grn.receiver_id,
            "supervisor_signoff_by": grn.supervisor_signoff_by,
            "supervisor_signoff_at": grn.supervisor_signoff_at,
            "max_dock_discrepancy_cost_inr": grn.max_dock_discrepancy_cost_inr,
            "rejected_goods_disposition": grn.rejected_goods_disposition,
            "notes": grn.notes,
            "lines": [
                {
                    "sku_id": l.sku_id,
                    "delivered_quantity": l.delivered_quantity,
                    "accepted_quantity": l.accepted_quantity,
                    "rejected_quantity": l.rejected_quantity,
                    "damaged_quantity": l.damaged_quantity,
                    "excess_quantity": l.excess_quantity,
                    "short_shelf_life_quantity": l.short_shelf_life_quantity,
                    "remaining_shelf_life_days": l.remaining_shelf_life_days,
                    "total_shelf_life_days": l.total_shelf_life_days,
                    "batch_references": list(l.batch_references),
                    "discrepancy_reasons": [r.value for r in l.discrepancy_reasons],
                }
                for l in grn.inspection_lines
            ],
        }
        fp = compute_payload_fingerprint(payload_dict)

        # 1. Idempotency check: does grn_id already exist?
        cur = conn.execute(
            "SELECT payload_fingerprint FROM fulfilment_goods_receipt_notes WHERE grn_id = ?",
            (grn.grn_id,),
        )
        row = cur.fetchone()
        if row:
            if row["payload_fingerprint"] == fp:
                # Exact match - idempotent no-op
                return
            raise IdempotencyConflictError(
                f"Idempotency conflict: GoodsReceiptNote '{grn.grn_id}' previously recorded with fingerprint "
                f"'{row['payload_fingerprint']}', incoming payload has '{fp}'."
            )

        # 2. Insert Header
        payload_json = grn.model_dump_json()
        conn.execute(
            """
            INSERT INTO fulfilment_goods_receipt_notes (
                grn_id, fo_id, po_id, shipment_id, store_id, supplier_id,
                receipt_at, receiver_id, supervisor_signoff_by, supervisor_signoff_at,
                max_dock_discrepancy_cost_inr, rejected_goods_disposition, notes,
                total_delivered_quantity, total_accepted_quantity, total_rejected_quantity,
                total_accepted_cost, total_rejected_cost, payload_fingerprint,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                grn.grn_id, grn.fo_id, grn.po_id, grn.shipment_id, grn.store_id, grn.supplier_id,
                grn.receipt_at, grn.receiver_id, grn.supervisor_signoff_by, grn.supervisor_signoff_at,
                grn.max_dock_discrepancy_cost_inr, grn.rejected_goods_disposition, grn.notes,
                grn.total_delivered_quantity, grn.total_accepted_quantity, grn.total_rejected_quantity,
                grn.total_accepted_cost, grn.total_rejected_cost, fp,
                payload_json, grn.receipt_at,
            ),
        )

        # 3. Insert Line Items
        for line in grn.inspection_lines:
            conn.execute(
                """
                INSERT INTO fulfilment_grn_inspection_lines (
                    grn_id, sku_id, delivered_quantity, accepted_quantity, rejected_quantity,
                    damaged_quantity, excess_quantity, short_shelf_life_quantity, unit_cost,
                    remaining_shelf_life_days, total_shelf_life_days, discrepancy_reasons,
                    batch_references, damaged_line_cost, rejected_line_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    grn.grn_id, line.sku_id, line.delivered_quantity, line.accepted_quantity, line.rejected_quantity,
                    line.damaged_quantity, line.excess_quantity, line.short_shelf_life_quantity, line.unit_cost,
                    line.remaining_shelf_life_days, line.total_shelf_life_days,
                    json.dumps([r.value for r in line.discrepancy_reasons]),
                    json.dumps(list(line.batch_references)),
                    line.damaged_line_cost, line.rejected_line_cost,
                ),
            )

    def _update_order_internal(
        self,
        conn: sqlite3.Connection,
        fo: FulfilmentOrder,
        expected_version: Optional[int] = None,
        grn: Optional[GoodsReceiptNote] = None,
    ) -> None:
        """Internal worker executing UPDATE statements for an order inside an active transaction."""
        # 1. Fetch current persistence version & event count
        cur = conn.execute("SELECT version FROM fulfilment_orders WHERE fo_id = ?", (fo.fo_id,))
        row = cur.fetchone()
        if not row:
            raise FulfilmentOrderNotFoundError(f"FulfilmentOrder '{fo.fo_id}' not found in persistence.")

        current_version = row["version"]
        if expected_version is not None and expected_version != current_version:
            raise StaleUpdateError(
                f"Stale update conflict for '{fo.fo_id}': expected version {expected_version}, "
                f"but database is currently at version {current_version}."
            )

        # Verify event count has not regressed
        cur_seq = conn.execute(
            "SELECT COUNT(*) as count FROM fulfilment_events WHERE fo_id = ?",
            (fo.fo_id,),
        )
        seq_row = cur_seq.fetchone()
        db_event_count = seq_row["count"] or 0
        if len(fo.event_history) < db_event_count:
            raise StaleUpdateError(
                f"Stale update conflict for '{fo.fo_id}': incoming order has {len(fo.event_history)} events, "
                f"but database already contains {db_event_count} events."
            )

        # 2. Check idempotency keys in event history against global registry
        for event in fo.event_history:
            cur = conn.execute(
                "SELECT fo_id, event_type, payload_fingerprint FROM fulfilment_idempotency_records WHERE idempotency_key = ?",
                (event.idempotency_key,),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["fo_id"] != fo.fo_id
                    or existing["event_type"] != event.event_type.value
                    or (existing["payload_fingerprint"] and existing["payload_fingerprint"] != event.payload_fingerprint)
                ):
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: key '{event.idempotency_key}' previously recorded for fo_id "
                        f"'{existing['fo_id']}' with event '{existing['event_type']}', cannot be reused for '{event.event_type.value}'."
                    )

        # 3. Optimistic atomic update of header
        target_version_condition = current_version if expected_version is None else expected_version
        payload_json = fo.model_dump_json()
        cur_upd = conn.execute(
            """
            UPDATE fulfilment_orders SET
                po_id = ?, store_id = ?, supplier_id = ?, supplier_type = ?, status = ?,
                created_at = ?, updated_at = ?, expected_delivery_at = ?,
                cancellation_reason = ?, cancellation_category = ?, cancelled_by = ?, cancellation_timestamp = ?,
                supplier_cancellation_ref = ?, shipment_cancellation_ref = ?, order_wide_cancellation_scope = ?,
                closed_by = ?, closure_timestamp = ?, closure_reason = ?, closed_unfulfilled_qty = ?,
                open_exception_reason = ?, open_exception_timestamp = ?, resumable_status = ?,
                is_simulation = ?, version = version + 1, aggregate_payload = ?
            WHERE fo_id = ? AND version = ?
            """,
            (
                fo.po_id, fo.store_id, fo.supplier_id, fo.supplier_type.value, fo.status.value,
                fo.created_at, fo.updated_at, fo.expected_delivery_at,
                fo.cancellation_reason,
                fo.cancellation_category.value if fo.cancellation_category else None,
                fo.cancelled_by, fo.cancellation_timestamp,
                fo.supplier_cancellation_ref, fo.shipment_cancellation_ref,
                1 if fo.order_wide_cancellation_scope else 0,
                fo.closed_by, fo.closure_timestamp, fo.closure_reason, fo.closed_unfulfilled_qty,
                fo.open_exception_reason, fo.open_exception_timestamp,
                fo.resumable_status.value if fo.resumable_status else None,
                1 if fo.is_simulation else 0,
                payload_json,
                fo.fo_id, target_version_condition,
            ),
        )
        if cur_upd.rowcount == 0:
            raise StaleUpdateError(
                f"Concurrent update conflict: FulfilmentOrder '{fo.fo_id}' was concurrently modified "
                f"by another transaction (expected version {target_version_condition})."
            )

        # 4. Refresh lines table
        conn.execute("DELETE FROM fulfilment_order_lines WHERE fo_id = ?", (fo.fo_id,))
        for line in fo.lines:
            conn.execute(
                """
                INSERT INTO fulfilment_order_lines (
                    fo_id, sku_id, category, ordered_quantity, unit_cost,
                    supplier_accepted_quantity, supplier_declined_quantity, supplier_unconfirmed_quantity,
                    shipped_quantity, delivered_quantity, accepted_quantity, rejected_quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fo.fo_id, line.sku_id, line.category, line.ordered_quantity, line.unit_cost,
                    line.supplier_accepted_quantity, line.supplier_declined_quantity, line.supplier_unconfirmed_quantity,
                    line.shipped_quantity, line.delivered_quantity, line.accepted_quantity, line.rejected_quantity,
                ),
            )

        # 5. Append new events (append-only enforcement!)
        cur_evts = conn.execute(
            "SELECT sequence_number, event_id FROM fulfilment_events WHERE fo_id = ?",
            (fo.fo_id,),
        )
        existing_evts = {r["sequence_number"]: r["event_id"] for r in cur_evts.fetchall()}

        for event in fo.event_history:
            if event.sequence_number in existing_evts:
                # Non-repudiation: existing events can NEVER be altered or renamed
                if existing_evts[event.sequence_number] != event.event_id:
                    raise FulfilmentPersistenceError(
                        f"Audit History Tampering: Sequence number {event.sequence_number} originally had id "
                        f"'{existing_evts[event.sequence_number]}', but incoming order has '{event.event_id}'."
                    )
            else:
                # Append new event
                conn.execute(
                    """
                    INSERT INTO fulfilment_events (
                        event_id, fo_id, sequence_number, idempotency_key, event_type,
                        actor_type, actor_id, timestamp, previous_status, new_status,
                        related_reference_id, payload_fingerprint, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id, event.fo_id, event.sequence_number, event.idempotency_key, event.event_type.value,
                        event.actor_type.value, event.actor_id, event.timestamp,
                        event.previous_status.value if event.previous_status else None,
                        event.new_status.value,
                        event.related_reference_id, event.payload_fingerprint,
                        json.dumps(event.details),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO fulfilment_idempotency_records (
                        idempotency_key, fo_id, event_type, payload_fingerprint, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.idempotency_key, event.fo_id, event.event_type.value,
                        event.payload_fingerprint, event.timestamp,
                    ),
                )

        # 6. Refresh Shipments table without violating child FK constraints (e.g. tracking records)
        resolved_set = set(fo.resolved_shipment_ids)
        current_shipment_ids = set(fo.shipment_ids)
        for s_id in fo.shipment_ids:
            is_res = 1 if s_id in resolved_set else 0
            evidence = fo.shipment_resolution_evidence.get(s_id)
            conn.execute(
                """
                INSERT INTO fulfilment_shipment_ids (fo_id, shipment_id, is_resolved, resolution_evidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(fo_id, shipment_id) DO UPDATE SET
                    is_resolved = excluded.is_resolved,
                    resolution_evidence = excluded.resolution_evidence
                """,
                (fo.fo_id, s_id, is_res, evidence),
            )
        if current_shipment_ids:
            placeholders = ",".join("?" for _ in current_shipment_ids)
            conn.execute(
                f"DELETE FROM fulfilment_shipment_ids WHERE fo_id = ? AND shipment_id NOT IN ({placeholders})",
                (fo.fo_id, *current_shipment_ids),
            )
        else:
            conn.execute("DELETE FROM fulfilment_shipment_ids WHERE fo_id = ?", (fo.fo_id,))

        # 7. Refresh GRNs table
        current_grn_ids = set(fo.grn_ids)
        for g_id in fo.grn_ids:
            conn.execute(
                """
                INSERT INTO fulfilment_grn_ids (fo_id, grn_id)
                VALUES (?, ?)
                ON CONFLICT(fo_id, grn_id) DO NOTHING
                """,
                (fo.fo_id, g_id),
            )
        if current_grn_ids:
            placeholders = ",".join("?" for _ in current_grn_ids)
            conn.execute(
                f"DELETE FROM fulfilment_grn_ids WHERE fo_id = ? AND grn_id NOT IN ({placeholders})",
                (fo.fo_id, *current_grn_ids),
            )
        else:
            conn.execute("DELETE FROM fulfilment_grn_ids WHERE fo_id = ?", (fo.fo_id,))

        # 8. Save Goods Receipt Note if provided
        if grn is not None:
            self._save_grn_internal(conn, grn)

        # 9. Insert/Update Discrepancy Resolutions
        for res in fo.discrepancy_resolutions:
            cur = conn.execute(
                "SELECT fo_id, shipment_id, sku_id, resolution_type, quantity, disposition_reference "
                "FROM fulfilment_discrepancy_resolutions WHERE resolution_id = ?",
                (res.resolution_id,),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["fo_id"] != res.fo_id
                    or existing["shipment_id"] != res.shipment_id
                    or existing["sku_id"] != res.sku_id
                    or existing["resolution_type"] != res.resolution_type.value
                    or abs(existing["quantity"] - res.quantity) > 1e-4
                    or existing["disposition_reference"] != res.disposition_reference
                ):
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: resolution_id '{res.resolution_id}' already exists with differing attributes."
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO fulfilment_discrepancy_resolutions (
                        resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                        quantity, discrepancy_reason, disposition_reference, actor_id,
                        timestamp, notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        res.resolution_id, res.fo_id, res.shipment_id, res.sku_id, res.resolution_type.value,
                        res.quantity, res.discrepancy_reason.value, res.disposition_reference, res.actor_id,
                        res.timestamp, res.notes, res.timestamp,
                    ),
                )

        # 10. Insert/Update Quality Hold Resolutions
        for qhr in fo.quality_hold_resolutions:
            cur = conn.execute(
                "SELECT fo_id, shipment_id, sku_id, resolution_type, quantity, disposition_reference "
                "FROM fulfilment_quality_hold_resolutions WHERE resolution_id = ?",
                (qhr.resolution_id,),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["fo_id"] != qhr.fo_id
                    or existing["shipment_id"] != qhr.shipment_id
                    or existing["sku_id"] != qhr.sku_id
                    or existing["resolution_type"] != qhr.resolution_type.value
                    or abs(existing["quantity"] - qhr.quantity) > 1e-4
                    or existing["disposition_reference"] != qhr.disposition_reference
                ):
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: quality hold resolution_id '{qhr.resolution_id}' already exists with differing attributes."
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO fulfilment_quality_hold_resolutions (
                        resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                        quantity, disposition_reference, inspector_id, timestamp,
                        verified_remaining_shelf_life_days, total_shelf_life_days,
                        inspection_notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        qhr.resolution_id, qhr.fo_id, qhr.shipment_id, qhr.sku_id, qhr.resolution_type.value,
                        qhr.quantity, qhr.disposition_reference, qhr.inspector_id, qhr.timestamp,
                        qhr.verified_remaining_shelf_life_days, qhr.total_shelf_life_days,
                        qhr.inspection_notes, qhr.timestamp,
                    ),
                )

    def create(self, fo: FulfilmentOrder) -> FulfilmentOrder:
        """
        Atomically persists a newly initialized FulfilmentOrder aggregate.

        Raises:
            DuplicateFulfilmentOrderError: if fo_id already exists.
            IdempotencyConflictError: if an idempotency key collides with conflicting payload/type.
            FulfilmentPersistenceError: on foreign key or integrity violations.
            FulfilmentGovernanceError: on invalid or tampered order models.
        """
        if not isinstance(fo, FulfilmentOrder):
            raise TypeError(f"Expected FulfilmentOrder instance, got {type(fo)}.")
        validated_fo = validate_inbound_fulfilment_order(fo)

        conn = self._get_connection()
        try:
            with conn:
                self._create_order_internal(conn, validated_fo)
        except (DuplicateFulfilmentOrderError, IdempotencyConflictError, FulfilmentGovernanceError):
            raise
        except sqlite3.IntegrityError as err:
            raise FulfilmentPersistenceError(f"Database constraint violation on create '{fo.fo_id}': {err}") from err

        return validated_fo

    def update(
        self,
        fo: FulfilmentOrder,
        expected_version: Optional[int] = None,
        grn: Optional[GoodsReceiptNote] = None,
    ) -> FulfilmentOrder:
        """
        Atomically updates an existing FulfilmentOrder aggregate with optimistic concurrency control.

        Args:
            fo: The updated FulfilmentOrder aggregate root.
            expected_version: Optional integer indicating the expected version in persistence.
                              If omitted, matches the current version in persistence.
            grn: Optional GoodsReceiptNote to persist atomically with the order update.

        Raises:
            FulfilmentOrderNotFoundError: if fo_id does not exist.
            StaleUpdateError: if expected_version does not match current DB version or if events regressed.
            IdempotencyConflictError: on conflicting idempotency key reuse or conflicting GRN payload.
            FulfilmentPersistenceError: on constraint or tampering violations.
            FulfilmentGovernanceError: on invalid or tampered order models.
        """
        if not isinstance(fo, FulfilmentOrder):
            raise TypeError(f"Expected FulfilmentOrder instance, got {type(fo)}.")
        if grn is not None and not isinstance(grn, GoodsReceiptNote):
            raise TypeError(f"Expected GoodsReceiptNote instance for grn, got {type(grn)}.")
        validated_fo = validate_inbound_fulfilment_order(fo)

        conn = self._get_connection()
        try:
            with conn:
                self._update_order_internal(conn, validated_fo, expected_version=expected_version, grn=grn)
        except (FulfilmentOrderNotFoundError, StaleUpdateError, IdempotencyConflictError, FulfilmentPersistenceError, FulfilmentGovernanceError):
            raise
        except sqlite3.IntegrityError as err:
            raise FulfilmentPersistenceError(f"Database constraint violation on update '{fo.fo_id}': {err}") from err

        return validated_fo

    def save_goods_receipt(
        self,
        fo: FulfilmentOrder,
        grn: GoodsReceiptNote,
        expected_version: Optional[int] = None,
    ) -> Tuple[FulfilmentOrder, GoodsReceiptNote]:
        """
        Atomically updates the FulfilmentOrder aggregate and persists the GoodsReceiptNote
        and its inspection lines in a single SQLite transaction.
        """
        if not isinstance(fo, FulfilmentOrder):
            raise TypeError(f"Expected FulfilmentOrder instance, got {type(fo)}.")
        if not isinstance(grn, GoodsReceiptNote):
            raise TypeError(f"Expected GoodsReceiptNote instance, got {type(grn)}.")
        validated_fo = validate_inbound_fulfilment_order(fo)

        conn = self._get_connection()
        try:
            with conn:
                self._update_order_internal(conn, validated_fo, expected_version=expected_version, grn=grn)
        except (FulfilmentOrderNotFoundError, StaleUpdateError, IdempotencyConflictError, FulfilmentPersistenceError, FulfilmentGovernanceError):
            raise
        except sqlite3.IntegrityError as err:
            raise FulfilmentPersistenceError(f"Database constraint violation on saving GRN '{grn.grn_id}': {err}") from err

        return validated_fo, grn

    def save_goods_receipt_note(self, grn: GoodsReceiptNote) -> GoodsReceiptNote:
        """Atomically persists a standalone GoodsReceiptNote and its inspection lines."""
        if not isinstance(grn, GoodsReceiptNote):
            raise TypeError(f"Expected GoodsReceiptNote instance, got {type(grn)}.")
        conn = self._get_connection()
        try:
            with conn:
                self._save_grn_internal(conn, grn)
        except IdempotencyConflictError:
            raise
        except sqlite3.IntegrityError as err:
            raise FulfilmentPersistenceError(f"Database constraint violation on saving GRN '{grn.grn_id}': {err}") from err
        return grn

    def persist_supersession(
        self,
        superseded_fo: FulfilmentOrder,
        replacement_fo: FulfilmentOrder,
        expected_superseded_version: Optional[int] = None,
    ) -> Tuple[FulfilmentOrder, FulfilmentOrder]:
        """
        Atomically persists the supersession of an existing order and the creation
        of a replacement order within a single SQLite transaction.

        If either operation fails, both roll back completely.
        """
        if not isinstance(superseded_fo, FulfilmentOrder):
            raise TypeError(f"Expected FulfilmentOrder for superseded_fo, got {type(superseded_fo)}.")
        if not isinstance(replacement_fo, FulfilmentOrder):
            raise TypeError(f"Expected FulfilmentOrder for replacement_fo, got {type(replacement_fo)}.")
        val_superseded = validate_inbound_fulfilment_order(superseded_fo)
        val_replacement = validate_inbound_fulfilment_order(replacement_fo)

        conn = self._get_connection()
        try:
            with conn:
                self._update_order_internal(conn, val_superseded, expected_version=expected_superseded_version)
                self._create_order_internal(conn, val_replacement)
        except (FulfilmentOrderNotFoundError, StaleUpdateError, DuplicateFulfilmentOrderError, IdempotencyConflictError, FulfilmentPersistenceError, FulfilmentGovernanceError):
            raise
        except sqlite3.IntegrityError as err:
            raise FulfilmentPersistenceError(f"Database constraint violation during supersession: {err}") from err

        return val_superseded, val_replacement

    def save(
        self,
        fo: FulfilmentOrder,
        expected_version: Optional[int] = None,
    ) -> FulfilmentOrder:
        """
        Convenience method: updates if fo_id already exists, otherwise creates.
        """
        conn = self._get_connection()
        cur = conn.execute("SELECT 1 FROM fulfilment_orders WHERE fo_id = ?", (fo.fo_id,))
        if cur.fetchone():
            return self.update(fo, expected_version=expected_version)
        else:
            if expected_version is not None:
                raise StaleUpdateError(f"Cannot save with expected_version {expected_version}: order '{fo.fo_id}' does not exist.")
            return self.create(fo)

    def get(self, fo_id: str) -> Optional[FulfilmentOrder]:
        """
        Retrieves and validates a FulfilmentOrder aggregate by identifier.
        Returns None if not found.
        """
        clean_id = fo_id.strip() if fo_id else ""
        if not clean_id:
            return None

        conn = self._get_connection()
        cur = conn.execute("SELECT aggregate_payload FROM fulfilment_orders WHERE fo_id = ?", (clean_id,))
        row = cur.fetchone()
        if not row:
            return None
        return FulfilmentOrder.model_validate_json(row["aggregate_payload"])

    def get_or_raise(self, fo_id: str) -> FulfilmentOrder:
        """Retrieves and validates a FulfilmentOrder aggregate, raising FulfilmentOrderNotFoundError if missing."""
        order = self.get(fo_id)
        if order is None:
            raise FulfilmentOrderNotFoundError(f"FulfilmentOrder '{fo_id}' was not found in persistence.")
        return order

    def get_version(self, fo_id: str) -> Optional[int]:
        """Returns the current optimistic concurrency version integer for the order, or None if missing."""
        clean_id = fo_id.strip() if fo_id else ""
        if not clean_id:
            return None
        conn = self._get_connection()
        cur = conn.execute("SELECT version FROM fulfilment_orders WHERE fo_id = ?", (clean_id,))
        row = cur.fetchone()
        return row["version"] if row else None

    def get_by_po_id(self, po_id: str) -> List[FulfilmentOrder]:
        """Returns all FulfilmentOrder aggregates associated with a given PurchaseOrder identifier."""
        clean_po = po_id.strip() if po_id else ""
        if not clean_po:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT aggregate_payload FROM fulfilment_orders WHERE po_id = ? ORDER BY created_at ASC, fo_id ASC",
            (clean_po,),
        )
        return [FulfilmentOrder.model_validate_json(row["aggregate_payload"]) for row in cur.fetchall()]

    def get_goods_receipt_note(self, grn_id: str) -> Optional[GoodsReceiptNote]:
        """Retrieves a persisted GoodsReceiptNote by its identifier."""
        clean_id = grn_id.strip() if grn_id else ""
        if not clean_id:
            return None
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT payload_json FROM fulfilment_goods_receipt_notes WHERE grn_id = ?",
            (clean_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return GoodsReceiptNote.model_validate_json(row["payload_json"])

    def get_goods_receipt_note_or_raise(self, grn_id: str) -> GoodsReceiptNote:
        """Retrieves a persisted GoodsReceiptNote or raises FulfilmentPersistenceError."""
        grn = self.get_goods_receipt_note(grn_id)
        if grn is None:
            raise FulfilmentPersistenceError(f"GoodsReceiptNote '{grn_id}' not found in persistence.")
        return grn

    def get_grns_for_order(self, fo_id: str) -> List[GoodsReceiptNote]:
        """Retrieves all GoodsReceiptNotes recorded for a given FulfilmentOrder."""
        clean_id = fo_id.strip() if fo_id else ""
        if not clean_id:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT payload_json FROM fulfilment_goods_receipt_notes WHERE fo_id = ? ORDER BY receipt_at ASC, grn_id ASC",
            (clean_id,),
        )
        return [GoodsReceiptNote.model_validate_json(r["payload_json"]) for r in cur.fetchall()]

    def get_grns_for_shipment(self, shipment_id: str) -> List[GoodsReceiptNote]:
        """Retrieves all GoodsReceiptNotes recorded for a given shipment."""
        clean_id = shipment_id.strip() if shipment_id else ""
        if not clean_id:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT payload_json FROM fulfilment_goods_receipt_notes WHERE shipment_id = ? ORDER BY receipt_at ASC, grn_id ASC",
            (clean_id,),
        )
        return [GoodsReceiptNote.model_validate_json(r["payload_json"]) for r in cur.fetchall()]

    def get_discrepancy_resolutions_for_order(self, fo_id: str) -> List[DiscrepancyResolution]:
        """Retrieves all discrepancy resolutions recorded for a given FulfilmentOrder."""
        clean_id = fo_id.strip() if fo_id else ""
        if not clean_id:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                   quantity, discrepancy_reason, disposition_reference, actor_id,
                   timestamp, notes
            FROM fulfilment_discrepancy_resolutions
            WHERE fo_id = ?
            ORDER BY timestamp ASC, resolution_id ASC
            """,
            (clean_id,),
        )
        results = []
        for row in cur.fetchall():
            results.append(
                DiscrepancyResolution(
                    resolution_id=row["resolution_id"],
                    fo_id=row["fo_id"],
                    shipment_id=row["shipment_id"],
                    sku_id=row["sku_id"],
                    resolution_type=DiscrepancyResolutionType(row["resolution_type"]),
                    quantity=row["quantity"],
                    discrepancy_reason=DiscrepancyReason(row["discrepancy_reason"]),
                    disposition_reference=row["disposition_reference"],
                    actor_id=row["actor_id"],
                    timestamp=row["timestamp"],
                    notes=row["notes"],
                )
            )
        return results

    def get_discrepancy_resolutions_for_shipment(self, shipment_id: str) -> List[DiscrepancyResolution]:
        """Retrieves all discrepancy resolutions recorded for a given shipment."""
        clean_id = shipment_id.strip() if shipment_id else ""
        if not clean_id:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                   quantity, discrepancy_reason, disposition_reference, actor_id,
                   timestamp, notes
            FROM fulfilment_discrepancy_resolutions
            WHERE shipment_id = ?
            ORDER BY timestamp ASC, resolution_id ASC
            """,
            (clean_id,),
        )
        results = []
        for row in cur.fetchall():
            results.append(
                DiscrepancyResolution(
                    resolution_id=row["resolution_id"],
                    fo_id=row["fo_id"],
                    shipment_id=row["shipment_id"],
                    sku_id=row["sku_id"],
                    resolution_type=DiscrepancyResolutionType(row["resolution_type"]),
                    quantity=row["quantity"],
                    discrepancy_reason=DiscrepancyReason(row["discrepancy_reason"]),
                    disposition_reference=row["disposition_reference"],
                    actor_id=row["actor_id"],
                    timestamp=row["timestamp"],
                    notes=row["notes"],
                )
            )
        return results

    def get_quality_hold_resolutions_for_order(self, fo_id: str) -> List[QualityHoldResolution]:
        """Retrieves all quality hold resolutions recorded for a given FulfilmentOrder."""
        clean_id = fo_id.strip() if fo_id else ""
        if not clean_id:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                   quantity, disposition_reference, inspector_id, timestamp,
                   verified_remaining_shelf_life_days, total_shelf_life_days, inspection_notes
            FROM fulfilment_quality_hold_resolutions
            WHERE fo_id = ?
            ORDER BY timestamp ASC, resolution_id ASC
            """,
            (clean_id,),
        )
        results = []
        for row in cur.fetchall():
            results.append(
                QualityHoldResolution(
                    resolution_id=row["resolution_id"],
                    fo_id=row["fo_id"],
                    shipment_id=row["shipment_id"],
                    sku_id=row["sku_id"],
                    resolution_type=DiscrepancyResolutionType(row["resolution_type"]),
                    quantity=row["quantity"],
                    disposition_reference=row["disposition_reference"],
                    inspector_id=row["inspector_id"],
                    timestamp=row["timestamp"],
                    verified_remaining_shelf_life_days=row["verified_remaining_shelf_life_days"],
                    total_shelf_life_days=row["total_shelf_life_days"],
                    inspection_notes=row["inspection_notes"],
                )
            )
        return results

    def get_quality_hold_resolutions_for_shipment(self, shipment_id: str) -> List[QualityHoldResolution]:
        """Retrieves all quality hold resolutions recorded for a given shipment."""
        clean_id = shipment_id.strip() if shipment_id else ""
        if not clean_id:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT resolution_id, fo_id, shipment_id, sku_id, resolution_type,
                   quantity, disposition_reference, inspector_id, timestamp,
                   verified_remaining_shelf_life_days, total_shelf_life_days, inspection_notes
            FROM fulfilment_quality_hold_resolutions
            WHERE shipment_id = ?
            ORDER BY timestamp ASC, resolution_id ASC
            """,
            (clean_id,),
        )
        results = []
        for row in cur.fetchall():
            results.append(
                QualityHoldResolution(
                    resolution_id=row["resolution_id"],
                    fo_id=row["fo_id"],
                    shipment_id=row["shipment_id"],
                    sku_id=row["sku_id"],
                    resolution_type=DiscrepancyResolutionType(row["resolution_type"]),
                    quantity=row["quantity"],
                    disposition_reference=row["disposition_reference"],
                    inspector_id=row["inspector_id"],
                    timestamp=row["timestamp"],
                    verified_remaining_shelf_life_days=row["verified_remaining_shelf_life_days"],
                    total_shelf_life_days=row["total_shelf_life_days"],
                    inspection_notes=row["inspection_notes"],
                )
            )
        return results

    def record_inventory_posting_task(self, task_data: Dict[str, Any]) -> None:
        """Records or updates a durable inventory posting task."""
        conn = self._get_connection()
        posting_ids = task_data.get("posting_ids", [])
        if "posting_ids_json" in task_data:
            posting_ids_val = task_data["posting_ids_json"]
        elif isinstance(posting_ids, (list, tuple)):
            posting_ids_val = json.dumps(list(posting_ids))
        else:
            posting_ids_val = json.dumps([])

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fulfilment_grn_postings (
                    posting_task_id, task_type, resolution_id, grn_id, fo_id, po_id, store_id, status,
                    idempotency_key, payload_fingerprint, total_accepted_quantity,
                    posted_quantity, error_message, retry_count, posting_ids_json,
                    created_at, completed_at
                ) VALUES (
                    :posting_task_id, :task_type, :resolution_id, :grn_id, :fo_id, :po_id, :store_id, :status,
                    :idempotency_key, :payload_fingerprint, :total_accepted_quantity,
                    :posted_quantity, :error_message, :retry_count, :posting_ids_json,
                    :created_at, :completed_at
                );
                """,
                {
                    "posting_task_id": task_data["posting_task_id"],
                    "task_type": task_data.get("task_type", "GRN_RECEIPT"),
                    "resolution_id": task_data.get("resolution_id"),
                    "grn_id": task_data["grn_id"],
                    "fo_id": task_data["fo_id"],
                    "po_id": task_data["po_id"],
                    "store_id": task_data["store_id"],
                    "status": task_data["status"],
                    "idempotency_key": task_data["idempotency_key"],
                    "payload_fingerprint": task_data["payload_fingerprint"],
                    "total_accepted_quantity": float(task_data["total_accepted_quantity"]),
                    "posted_quantity": float(task_data.get("posted_quantity", 0.0)),
                    "error_message": task_data.get("error_message"),
                    "retry_count": int(task_data.get("retry_count", 0)),
                    "posting_ids_json": posting_ids_val,
                    "created_at": task_data["created_at"],
                    "completed_at": task_data.get("completed_at"),
                },
            )

    def update_inventory_posting_task(
        self,
        posting_task_id: str,
        status: str,
        posted_quantity: float,
        posting_ids: List[str],
        error_message: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> None:
        """Updates the status and result of an existing inventory posting task."""
        clean_id = posting_task_id.strip() if posting_task_id else ""
        if not clean_id:
            return
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                UPDATE fulfilment_grn_postings
                SET status = ?,
                    posted_quantity = ?,
                    posting_ids_json = ?,
                    error_message = ?,
                    completed_at = ?
                WHERE posting_task_id = ?;
                """,
                (
                    status,
                    float(posted_quantity),
                    json.dumps(posting_ids),
                    error_message,
                    completed_at,
                    clean_id,
                ),
            )

    def get_inventory_posting_task(self, posting_task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an inventory posting task by its ID."""
        clean_id = posting_task_id.strip() if posting_task_id else ""
        if not clean_id:
            return None
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM fulfilment_grn_postings WHERE posting_task_id = ?",
            (clean_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_inventory_posting_task_for_grn(self, grn_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent inventory posting task for a given GRN."""
        clean_id = grn_id.strip() if grn_id else ""
        if not clean_id:
            return None
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM fulfilment_grn_postings WHERE grn_id = ? ORDER BY created_at DESC",
            (clean_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_inventory_posting_task_for_resolution(self, resolution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent inventory posting task for a given quality hold resolution."""
        clean_id = resolution_id.strip() if resolution_id else ""
        if not clean_id:
            return None
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM fulfilment_grn_postings WHERE resolution_id = ? ORDER BY created_at DESC",
            (clean_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_pending_or_failed_inventory_posting_tasks(self) -> List[Dict[str, Any]]:
        """Retrieves all pending or failed inventory posting tasks for recovery."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT * FROM fulfilment_grn_postings
            WHERE status IN ('PENDING', 'FAILED')
            ORDER BY created_at ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def list(
        self,
        store_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        status: Optional[Union[FulfilmentOrderStatus, str]] = None,
        statuses: Optional[Iterable[Union[FulfilmentOrderStatus, str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[FulfilmentOrder]:
        """
        Lists FulfilmentOrder aggregates matching optional store, supplier, and status filters.
        """
        conn = self._get_connection()
        query = "SELECT aggregate_payload FROM fulfilment_orders WHERE 1=1"
        params: List[Any] = []

        if store_id:
            query += " AND store_id = ?"
            params.append(store_id.strip())
        if supplier_id:
            query += " AND supplier_id = ?"
            params.append(supplier_id.strip())
        if status:
            val = status.value if isinstance(status, FulfilmentOrderStatus) else str(status).strip()
            query += " AND status = ?"
            params.append(val)
        elif statuses:
            stat_vals = [s.value if isinstance(s, FulfilmentOrderStatus) else str(s).strip() for s in statuses]
            placeholders = ", ".join("?" for _ in stat_vals)
            query += f" AND status IN ({placeholders})"
            params.extend(stat_vals)

        query += " ORDER BY created_at ASC, fo_id ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
            if offset is not None:
                query += " OFFSET ?"
                params.append(offset)

        cur = conn.execute(query, params)
        return [FulfilmentOrder.model_validate_json(row["aggregate_payload"]) for row in cur.fetchall()]

    def count(
        self,
        store_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        status: Optional[Union[FulfilmentOrderStatus, str]] = None,
    ) -> int:
        """Returns count of matching fulfilment orders."""
        conn = self._get_connection()
        query = "SELECT COUNT(*) as count FROM fulfilment_orders WHERE 1=1"
        params: List[Any] = []

        if store_id:
            query += " AND store_id = ?"
            params.append(store_id.strip())
        if supplier_id:
            query += " AND supplier_id = ?"
            params.append(supplier_id.strip())
        if status:
            val = status.value if isinstance(status, FulfilmentOrderStatus) else str(status).strip()
            query += " AND status = ?"
            params.append(val)

        cur = conn.execute(query, params)
        row = cur.fetchone()
        return row["count"] if row else 0

    def get_events(self, fo_id: str) -> List[FulfilmentEvent]:
        """
        Retrieves all audit events for an order directly from the relational events table.
        """
        clean_id = fo_id.strip() if fo_id else ""
        if not clean_id:
            return []
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM fulfilment_events WHERE fo_id = ? ORDER BY sequence_number ASC",
            (clean_id,),
        )
        events: List[FulfilmentEvent] = []
        for row in cur.fetchall():
            events.append(
                FulfilmentEvent(
                    event_id=row["event_id"],
                    idempotency_key=row["idempotency_key"],
                    fo_id=row["fo_id"],
                    event_type=FulfilmentEventType(row["event_type"]),
                    actor_type=FulfilmentActorType(row["actor_type"]),
                    actor_id=row["actor_id"],
                    timestamp=row["timestamp"],
                    previous_status=FulfilmentOrderStatus(row["previous_status"]) if row["previous_status"] else None,
                    new_status=FulfilmentOrderStatus(row["new_status"]),
                    sequence_number=row["sequence_number"],
                    related_reference_id=row["related_reference_id"],
                    payload_fingerprint=row["payload_fingerprint"],
                    details=json.loads(row["details_json"]),
                )
            )
        return events

    # -------------------------------------------------------------------------
    # Pipeline Calculations (Derived Dynamically from Persisted Aggregates)
    # -------------------------------------------------------------------------

    def get_confirmed_pipeline_quantity(
        self,
        store_id: Optional[str] = None,
        sku_id: Optional[str] = None,
    ) -> float:
        """
        Derives confirmed outstanding inbound pipeline units directly from persisted
        fulfilment aggregates and their line balances.

        GOVERNANCE RULES:
        - Only active fulfilment statuses count (ACKNOWLEDGED, PARTIALLY_ACKNOWLEDGED,
          IN_TRANSIT, PARTIALLY_RECEIVED, DELIVERY_EXCEPTION).
        - Strictly zero for unacknowledged, rejected, cancelled, closed, or received orders.
        - For each line: outstanding_confirmed_qty = max(0.0, supplier_accepted_quantity - delivered_quantity).
        - Unconfirmed, declined, rejected, and already received quantities are strictly excluded.
        - Does NOT maintain a separate mutable counter.
        """
        conn = self._get_connection()
        status_placeholders = ", ".join("?" for _ in ACTIVE_PIPELINE_STATUSES)
        query = f"""
        SELECT COALESCE(SUM(
            CASE 
                WHEN fol.supplier_accepted_quantity > fol.delivered_quantity 
                THEN (fol.supplier_accepted_quantity - fol.delivered_quantity) 
                ELSE 0.0 
            END
        ), 0.0) as total_pipeline_qty
        FROM fulfilment_orders fo
        JOIN fulfilment_order_lines fol ON fo.fo_id = fol.fo_id
        WHERE fo.status IN ({status_placeholders})
        """
        params: List[Any] = list(ACTIVE_PIPELINE_STATUSES)

        if store_id:
            query += " AND fo.store_id = ?"
            params.append(store_id.strip())
        if sku_id:
            query += " AND fol.sku_id = ?"
            params.append(sku_id.strip())

        cur = conn.execute(query, params)
        row = cur.fetchone()
        return round(float(row["total_pipeline_qty"]), 4) if row else 0.0

    def get_sku_pipeline_quantities(
        self,
        store_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Returns a mapping of sku_id -> confirmed outstanding pipeline quantity.
        """
        conn = self._get_connection()
        status_placeholders = ", ".join("?" for _ in ACTIVE_PIPELINE_STATUSES)
        query = f"""
        SELECT fol.sku_id, SUM(
            CASE 
                WHEN fol.supplier_accepted_quantity > fol.delivered_quantity 
                THEN (fol.supplier_accepted_quantity - fol.delivered_quantity) 
                ELSE 0.0 
            END
        ) as sku_pipeline_qty
        FROM fulfilment_orders fo
        JOIN fulfilment_order_lines fol ON fo.fo_id = fol.fo_id
        WHERE fo.status IN ({status_placeholders})
        """
        params: List[Any] = list(ACTIVE_PIPELINE_STATUSES)

        if store_id:
            query += " AND fo.store_id = ?"
            params.append(store_id.strip())

        query += " GROUP BY fol.sku_id"

        cur = conn.execute(query, params)
        return {
            row["sku_id"]: round(float(row["sku_pipeline_qty"]), 4)
            for row in cur.fetchall()
            if row["sku_pipeline_qty"] > 0
        }

    def get_pipeline_breakdown(
        self,
        store_id: Optional[str] = None,
        sku_id: Optional[str] = None,
    ) -> Dict[Tuple[str, str], float]:
        """
        Returns mapping of (store_id, sku_id) -> confirmed outstanding pipeline quantity.
        """
        conn = self._get_connection()
        status_placeholders = ", ".join("?" for _ in ACTIVE_PIPELINE_STATUSES)
        query = f"""
        SELECT fo.store_id, fol.sku_id, SUM(
            CASE 
                WHEN fol.supplier_accepted_quantity > fol.delivered_quantity 
                THEN (fol.supplier_accepted_quantity - fol.delivered_quantity) 
                ELSE 0.0 
            END
        ) as line_pipeline_qty
        FROM fulfilment_orders fo
        JOIN fulfilment_order_lines fol ON fo.fo_id = fol.fo_id
        WHERE fo.status IN ({status_placeholders})
        """
        params: List[Any] = list(ACTIVE_PIPELINE_STATUSES)

        if store_id:
            query += " AND fo.store_id = ?"
            params.append(store_id.strip())
        if sku_id:
            query += " AND fol.sku_id = ?"
            params.append(sku_id.strip())

        query += " GROUP BY fo.store_id, fol.sku_id"

        cur = conn.execute(query, params)
        return {
            (row["store_id"], row["sku_id"]): round(float(row["line_pipeline_qty"]), 4)
            for row in cur.fetchall()
            if row["line_pipeline_qty"] > 0
        }

    # -------------------------------------------------------------------------
    # Gateway Transmission Operations (Phase 5.4.2)
    # -------------------------------------------------------------------------

    def record_transmission_intent(
        self,
        fo_id: str,
        po_id: str,
        idempotency_key: str,
        payload_fingerprint: str,
        correlation_id: str,
        timestamp: str,
    ) -> None:
        """
        Durable two-phase commit: records the intention to transmit before
        calling the external supplier gateway.
        """
        conn = self._get_connection()
        with conn:
            cur = conn.execute(
                "SELECT fo_id, payload_fingerprint, transmission_status FROM fulfilment_transmissions WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            existing = cur.fetchone()
            if existing:
                if existing["fo_id"] != fo_id or existing["payload_fingerprint"] != payload_fingerprint:
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: key '{idempotency_key}' previously recorded for fo_id "
                        f"'{existing['fo_id']}' with fingerprint '{existing['payload_fingerprint']}', "
                        f"cannot be reused for fo_id '{fo_id}' with fingerprint '{payload_fingerprint}'."
                    )
                return

            try:
                conn.execute(
                    """
                    INSERT INTO fulfilment_transmissions (
                        idempotency_key, fo_id, po_id, transmission_id, correlation_id,
                        payload_fingerprint, transmission_status, transmitted_at,
                        completed_at, supplier_reference, error_code, error_message,
                        result_json, created_at
                    ) VALUES (?, ?, ?, NULL, ?, ?, 'INITIATED', ?, NULL, NULL, NULL, NULL, NULL, ?)
                    """,
                    (
                        idempotency_key, fo_id, po_id, correlation_id,
                        payload_fingerprint, timestamp, timestamp,
                    ),
                )
            except sqlite3.IntegrityError:
                # Concurrent race condition: verify existing record has matching fingerprint
                cur2 = conn.execute(
                    "SELECT fo_id, payload_fingerprint FROM fulfilment_transmissions WHERE idempotency_key = ?",
                    (idempotency_key,),
                )
                row2 = cur2.fetchone()
                if row2 and (row2["fo_id"] != fo_id or row2["payload_fingerprint"] != payload_fingerprint):
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: key '{idempotency_key}' previously recorded for fo_id "
                        f"'{row2['fo_id']}' with fingerprint '{row2['payload_fingerprint']}', "
                        f"cannot be reused for fo_id '{fo_id}' with fingerprint '{payload_fingerprint}'."
                    )

    def record_transmission_outcome(
        self,
        idempotency_key: str,
        status: str,
        result: Any,
        updated_fo: Optional[FulfilmentOrder] = None,
        expected_version: Optional[int] = None,
    ) -> Optional[FulfilmentOrder]:
        """
        Atomically updates the transmission record and, if updated_fo is provided,
        the FulfilmentOrder aggregate within a single SQLite transaction.
        """
        conn = self._get_connection()
        with conn:
            result_json = result.model_dump_json() if hasattr(result, "model_dump_json") else json.dumps(result)
            conn.execute(
                """
                UPDATE fulfilment_transmissions SET
                    transmission_id = ?,
                    transmission_status = ?,
                    completed_at = ?,
                    supplier_reference = ?,
                    error_code = ?,
                    error_message = ?,
                    result_json = ?
                WHERE idempotency_key = ?
                """,
                (
                    getattr(result, "transmission_id", None),
                    status,
                    getattr(result, "transmitted_at", None),
                    getattr(result, "supplier_reference", None),
                    getattr(result, "error_code", None),
                    getattr(result, "error_message", None),
                    result_json,
                    idempotency_key,
                ),
            )

            if updated_fo is not None:
                self._update_order_internal(conn, updated_fo, expected_version=expected_version)

        return updated_fo

    def get_transmission(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a transmission record by its idempotency key."""
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM fulfilment_transmissions WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_transmissions_for_order(self, fo_id: str) -> List[Dict[str, Any]]:
        """Retrieves all transmission records associated with a given fo_id."""
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM fulfilment_transmissions WHERE fo_id = ? ORDER BY created_at ASC",
            (fo_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Shipment Tracking Persistence (Phase 5.5.2)
    # -------------------------------------------------------------------------

    def create_tracking_record(
        self,
        fo_id: str,
        record: ShipmentTrackingRecord,
    ) -> ShipmentTrackingRecord:
        """
        Atomically persists a newly initialized ShipmentTrackingRecord projection
        and its initial tracking events in SQLite.
        """
        conn = self._get_connection()
        with conn:
            cur = conn.execute(
                "SELECT fo_id, carrier_id FROM shipment_tracking_records WHERE shipment_id = ?",
                (record.shipment_id,),
            )
            existing = cur.fetchone()
            if existing:
                if existing["fo_id"] == fo_id and existing["carrier_id"] == record.carrier_id:
                    return self.get_tracking_record_or_raise(record.shipment_id)
                raise TrackingIdempotencyConflictError(
                    f"Tracking record for shipment '{record.shipment_id}' already registered for "
                    f"fo_id '{existing['fo_id']}' with carrier '{existing['carrier_id']}'."
                )

            try:
                conn.execute(
                    """
                    INSERT INTO shipment_tracking_records (
                        shipment_id, fo_id, carrier_id, current_status, scenario, created_at,
                        updated_at, estimated_delivery_at, actual_delivery_at, attempt_count, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        record.shipment_id,
                        fo_id,
                        record.carrier_id,
                        record.current_status.value,
                        record.scenario or "NORMAL_DELIVERY",
                        record.created_at,
                        record.updated_at,
                        record.estimated_delivery_at,
                        record.actual_delivery_at,
                        record.attempt_count,
                    ),
                )
            except sqlite3.IntegrityError as err:
                cur_check = conn.execute(
                    "SELECT fo_id, carrier_id FROM shipment_tracking_records WHERE shipment_id = ?",
                    (record.shipment_id,),
                )
                existing_after = cur_check.fetchone()
                if existing_after:
                    if existing_after["fo_id"] == fo_id and existing_after["carrier_id"] == record.carrier_id:
                        return self.get_tracking_record_or_raise(record.shipment_id)
                    raise TrackingIdempotencyConflictError(
                        f"Tracking record for shipment '{record.shipment_id}' already registered for "
                        f"fo_id '{existing_after['fo_id']}' with carrier '{existing_after['carrier_id']}'."
                    ) from err
                raise FulfilmentPersistenceError(
                    f"Database constraint violation on creating tracking for shipment '{record.shipment_id}': {err}"
                ) from err

            for evt in record.events:
                self._insert_tracking_event_internal(conn, evt)

        return self.get_tracking_record_or_raise(record.shipment_id)

    def _insert_tracking_event_internal(
        self,
        conn: sqlite3.Connection,
        event: ShipmentTrackingEvent,
    ) -> None:
        """
        Internal worker inserting a single tracking event inside an active transaction.
        Enforces uniqueness of provider_event_id and detects conflicting payloads.
        """
        cur = conn.execute(
            """
            SELECT payload_fingerprint, status FROM shipment_tracking_events
            WHERE carrier_id = ? AND shipment_id = ? AND provider_event_id = ?
            """,
            (event.carrier_id, event.shipment_id, event.provider_event_id),
        )
        row = cur.fetchone()
        if row:
            if row["payload_fingerprint"] == event.fingerprint:
                return
            raise TrackingIdempotencyConflictError(
                f"Conflicting provider_event_id '{event.provider_event_id}' for shipment '{event.shipment_id}': "
                f"existing event has fingerprint '{row['payload_fingerprint']}', incoming has '{event.fingerprint}'."
            )

        loc_json = event.location.model_dump_json() if event.location else None
        exc_details_json = json.dumps(event.exception_details) if event.exception_details else None
        raw_json = json.dumps(event.raw_payload) if event.raw_payload else None
        event_id = f"TRK-EVT-{event.carrier_id}-{event.shipment_id}-{event.provider_event_id}"

        try:
            conn.execute(
                """
                INSERT INTO shipment_tracking_events (
                    event_id, shipment_id, carrier_id, provider_event_id, status,
                    event_timestamp, received_timestamp, location_json, exception_reason,
                    exception_details_json, notes, raw_payload_json, payload_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event.shipment_id,
                    event.carrier_id,
                    event.provider_event_id,
                    event.status.value,
                    event.event_timestamp,
                    event.received_timestamp,
                    loc_json,
                    event.exception_reason,
                    exc_details_json,
                    event.notes,
                    raw_json,
                    event.fingerprint,
                    event.received_timestamp,
                ),
            )
        except sqlite3.IntegrityError as err:
            cur2 = conn.execute(
                """
                SELECT payload_fingerprint FROM shipment_tracking_events
                WHERE carrier_id = ? AND shipment_id = ? AND provider_event_id = ?
                """,
                (event.carrier_id, event.shipment_id, event.provider_event_id),
            )
            row2 = cur2.fetchone()
            if row2 and row2["payload_fingerprint"] == event.fingerprint:
                return
            raise TrackingIdempotencyConflictError(
                f"Conflicting provider_event_id '{event.provider_event_id}' on concurrent insert for '{event.shipment_id}'."
            ) from err

    def update_tracking_record(
        self,
        record: ShipmentTrackingRecord,
        new_event: Optional[ShipmentTrackingEvent] = None,
        expected_version: Optional[int] = None,
    ) -> ShipmentTrackingRecord:
        """
        Atomically updates the tracking record projection and inserts any new event.
        Enforces optimistic concurrency control using 'version'.
        """
        conn = self._get_connection()
        with conn:
            cur = conn.execute(
                "SELECT version FROM shipment_tracking_records WHERE shipment_id = ?",
                (record.shipment_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ShipmentNotFoundError(f"Shipment '{record.shipment_id}' does not exist in tracking persistence.")

            current_ver = row["version"]
            if expected_version is not None and expected_version != current_ver:
                raise StaleUpdateError(
                    f"Stale update conflict for shipment tracking '{record.shipment_id}': "
                    f"expected version {expected_version}, but database is at version {current_ver}."
                )

            target_version = current_ver if expected_version is None else expected_version

            if new_event is not None:
                self._insert_tracking_event_internal(conn, new_event)

            cur_upd = conn.execute(
                """
                UPDATE shipment_tracking_records SET
                    current_status = ?,
                    updated_at = ?,
                    estimated_delivery_at = ?,
                    actual_delivery_at = ?,
                    attempt_count = ?,
                    version = version + 1
                WHERE shipment_id = ? AND version = ?
                """,
                (
                    record.current_status.value,
                    record.updated_at,
                    record.estimated_delivery_at,
                    record.actual_delivery_at,
                    record.attempt_count,
                    record.shipment_id,
                    target_version,
                ),
            )
            if cur_upd.rowcount == 0:
                raise StaleUpdateError(
                    f"Concurrent update conflict: shipment tracking '{record.shipment_id}' was concurrently modified "
                    f"by another transaction (expected version {target_version})."
                )

        return self.get_tracking_record_or_raise(record.shipment_id)

    def get_tracking_record(self, shipment_id: str) -> Optional[ShipmentTrackingRecord]:
        """
        Reconstructs the complete ShipmentTrackingRecord and its chronological event history.
        """
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM shipment_tracking_records WHERE shipment_id = ?",
            (shipment_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        evts_cur = conn.execute(
            """
            SELECT * FROM shipment_tracking_events
            WHERE shipment_id = ?
            ORDER BY event_timestamp ASC, received_timestamp ASC, provider_event_id ASC
            """,
            (shipment_id,),
        )
        event_rows = evts_cur.fetchall()

        events: List[ShipmentTrackingEvent] = []
        for er in event_rows:
            loc = TrackingLocation.model_validate_json(er["location_json"]) if er["location_json"] else None
            exc_det = json.loads(er["exception_details_json"]) if er["exception_details_json"] else None
            raw_pl = json.loads(er["raw_payload_json"]) if er["raw_payload_json"] else None

            evt = ShipmentTrackingEvent(
                shipment_id=er["shipment_id"],
                carrier_id=er["carrier_id"],
                provider_event_id=er["provider_event_id"],
                status=ShipmentTrackingStatus(er["status"]),
                event_timestamp=er["event_timestamp"],
                received_timestamp=er["received_timestamp"],
                location=loc,
                exception_reason=er["exception_reason"],
                exception_details=exc_det,
                notes=er["notes"],
                raw_payload=raw_pl,
            )
            events.append(evt)

        return ShipmentTrackingRecord(
            shipment_id=row["shipment_id"],
            carrier_id=row["carrier_id"],
            current_status=ShipmentTrackingStatus(row["current_status"]),
            events=tuple(events),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            estimated_delivery_at=row["estimated_delivery_at"],
            actual_delivery_at=row["actual_delivery_at"],
            attempt_count=row["attempt_count"],
            scenario=row["scenario"] if "scenario" in row.keys() else None,
        )

    def get_tracking_record_or_raise(self, shipment_id: str) -> ShipmentTrackingRecord:
        """Retrieves the tracking record for a shipment, raising ShipmentNotFoundError if missing."""
        rec = self.get_tracking_record(shipment_id)
        if rec is None:
            raise ShipmentNotFoundError(f"Shipment '{shipment_id}' not found in tracking persistence.")
        return rec

    def get_tracking_version(self, shipment_id: str) -> Optional[int]:
        """Returns the current optimistic concurrency version for a shipment tracking record."""
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT version FROM shipment_tracking_records WHERE shipment_id = ?",
            (shipment_id,),
        )
        row = cur.fetchone()
        return row["version"] if row else None

    def get_tracking_events(self, shipment_id: str) -> Tuple[ShipmentTrackingEvent, ...]:
        """Retrieves the chronological tracking event history for a shipment."""
        rec = self.get_tracking_record(shipment_id)
        return rec.events if rec else ()

    def get_tracking_records_for_order(self, fo_id: str) -> List[ShipmentTrackingRecord]:
        """Retrieves all tracking records associated with a given FulfilmentOrder."""
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT shipment_id FROM shipment_tracking_records WHERE fo_id = ? ORDER BY created_at ASC",
            (fo_id,),
        )
        shipment_ids = [r["shipment_id"] for r in cur.fetchall()]
        records = []
        for sid in shipment_ids:
            rec = self.get_tracking_record(sid)
            if rec:
                records.append(rec)
        return records

    # -------------------------------------------------------------------------
    # Lifecycle & Cleanup
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Closes the underlying database connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FulfilmentRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

