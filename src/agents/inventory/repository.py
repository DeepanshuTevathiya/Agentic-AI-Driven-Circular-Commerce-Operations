"""
Operational Inventory Repository & Durable Ledger Persistence (Phase 5.7.1)
===========================================================================
Durable SQLite persistence for operational dark-store inventory ledger.
Provides:
- Immutable append-only inventory ledger recording atomic stock movements.
- Durable inventory batch records linking physical FulfilmentBatch to inventory.
- Dynamic balance derivation (SUM of ledger entries) preventing counter drift.
- Strict FEFO (First-Expiring, First-Out) perishability tracking.
- Scoped idempotency deduplication with conflict detection.
- Audit reconciliation comparing GRN dock receipts against posted stock.
- Clean integration with existing Fulfilment database architecture.

ARCHITECTURAL PRINCIPLES:
1. IMMUTABILITY: Ledger entries are append-only; stock movements are never updated or deleted.
2. BALANCE FIDELITY: On-hand stock is derived on-the-fly from SUM(quantity) across ledger entries.
   Counters are never independently maintained to eliminate drift.
3. QUALITY BOUNDARY: Only accepted, sellable FulfilmentBatch quantities can be posted.
   Damaged, excess, or unaccepted goods are strictly prohibited from entering sellable stock.
4. ATOMIC TRANSACTIONS: Ledger entries, batch records, and idempotency tokens commit together
   or roll back together.
5. ISOLATION: Empty ledger returns 0.0 stock. No opening stock is assumed or invented.
"""

from datetime import date as dt_date, datetime as dt_datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.agents.common.config import CANONICAL_SKUS, CANONICAL_STORES, DEFAULT_FULFILMENT_DB_PATH
from src.agents.common.schemas import (
    InventoryBatch,
    InventoryPosition,
)
from src.agents.fulfilment.schemas import (
    FulfilmentBatch,
    GoodsReceiptNote,
    generate_grn_fulfilment_batches,
    validate_and_normalize_iso_timestamp,
)
from .schemas import (
    BatchBalance,
    DuplicateBatchPostingError,
    GrnLinePostingReconciliation,
    GrnPostingReconciliation,
    IdempotencyConflictError,
    InventoryBatchRecord,
    InventoryGovernanceError,
    InventoryLedgerEntry,
    InventoryPersistenceError,
    InventoryPostingReason,
    InventoryTransactionType,
    StoreSkuBalance,
    compute_payload_fingerprint,
)


class OperationalInventoryRepository:
    """
    Durable SQLite repository for the operational inventory ledger and batch tracking.
    Enforces relational integrity, dynamic stock balance derivation, and append-only audits.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        timeout: float = 30.0,
        fulfilment_repo: Optional[Any] = None,
    ) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        elif fulfilment_repo is not None:
            self.db_path = Path(fulfilment_repo.db_path)
        else:
            self.db_path = DEFAULT_FULFILMENT_DB_PATH

        self.timeout = timeout
        self._fulfilment_repo = fulfilment_repo
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        """Creates a safe SQLite connection with foreign keys enabled and WAL mode."""
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
        """Returns the active SQLite connection."""
        with self._lock:
            if self._fulfilment_repo is not None and str(self.db_path) == ":memory:":
                # Share in-memory connection if fulfilment repo is provided
                return self._fulfilment_repo._get_connection()
            if self._conn is None:
                self._conn = self._create_connection()
            return self._conn

    def _init_db(self) -> None:
        """Initializes relational tables and indexes for the inventory ledger."""
        conn = self._get_connection()
        with self._lock:
            with conn:
                # 1. Prerequisite fulfilment tables (if not already created by FulfilmentRepository)
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
                        created_at TEXT NOT NULL
                    );
                    """
                )

                # 2. Immutable Operational Inventory Ledger Table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inventory_ledger_entries (
                        posting_id TEXT PRIMARY KEY,
                        transaction_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        store_id TEXT NOT NULL,
                        sku_id TEXT NOT NULL,
                        batch_id TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        unit_cost REAL NOT NULL,
                        total_cost REAL NOT NULL,
                        remaining_shelf_life_days INTEGER,
                        expiry_date TEXT,
                        fo_id TEXT,
                        shipment_id TEXT,
                        grn_id TEXT,
                        fulfilment_batch_id TEXT,
                        actor_type TEXT NOT NULL,
                        actor_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        payload_fingerprint TEXT NOT NULL,
                        notes TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE RESTRICT,
                        FOREIGN KEY (grn_id) REFERENCES fulfilment_goods_receipt_notes(grn_id) ON DELETE RESTRICT
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_ledger_store_sku ON inventory_ledger_entries (store_id, sku_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_ledger_batch ON inventory_ledger_entries (store_id, sku_id, batch_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_ledger_grn ON inventory_ledger_entries (grn_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_ledger_fo ON inventory_ledger_entries (fo_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_ledger_idem ON inventory_ledger_entries (idempotency_key);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_ledger_timestamp ON inventory_ledger_entries (timestamp ASC);")
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_inv_ledger_unique_batch_posting
                    ON inventory_ledger_entries (grn_id, fulfilment_batch_id)
                    WHERE grn_id IS NOT NULL AND fulfilment_batch_id IS NOT NULL;
                    """
                )

                # 3. Durable Inventory Batches Table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inventory_batches (
                        batch_id TEXT NOT NULL,
                        store_id TEXT NOT NULL,
                        sku_id TEXT NOT NULL,
                        initial_units REAL NOT NULL,
                        receipt_date TEXT NOT NULL,
                        expiry_date TEXT NOT NULL,
                        remaining_shelf_life_days_at_receipt INTEGER NOT NULL,
                        unit_cost REAL NOT NULL,
                        fo_id TEXT NOT NULL,
                        grn_id TEXT NOT NULL,
                        fulfilment_batch_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (store_id, sku_id, batch_id),
                        FOREIGN KEY (fo_id) REFERENCES fulfilment_orders(fo_id) ON DELETE RESTRICT,
                        FOREIGN KEY (grn_id) REFERENCES fulfilment_goods_receipt_notes(grn_id) ON DELETE RESTRICT,
                        UNIQUE (grn_id, fulfilment_batch_id)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_batches_store_sku ON inventory_batches (store_id, sku_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_batches_expiry ON inventory_batches (store_id, sku_id, expiry_date ASC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_batches_grn ON inventory_batches (grn_id);")

                # 4. Inventory Posting Idempotency Table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inventory_posting_idempotency_records (
                        idempotency_key TEXT PRIMARY KEY,
                        operation_type TEXT NOT NULL,
                        store_id TEXT NOT NULL,
                        grn_id TEXT,
                        fo_id TEXT,
                        payload_fingerprint TEXT NOT NULL,
                        posting_ids_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_idem_store ON inventory_posting_idempotency_records (store_id);")

    def _resolve_grn(self, grn: Union[GoodsReceiptNote, str]) -> GoodsReceiptNote:
        """Resolves a GoodsReceiptNote instance either directly or by querying SQLite."""
        if isinstance(grn, GoodsReceiptNote):
            return grn

        clean_id = grn.strip() if isinstance(grn, str) else ""
        if not clean_id:
            raise InventoryGovernanceError("GRN identifier must be a non-empty string.")

        if self._fulfilment_repo is not None:
            grn_obj = self._fulfilment_repo.get_goods_receipt_note(clean_id)
            if grn_obj is not None:
                return grn_obj

        conn = self._get_connection()
        cur = conn.execute(
            "SELECT payload_json FROM fulfilment_goods_receipt_notes WHERE grn_id = ?",
            (clean_id,),
        )
        row = cur.fetchone()
        if not row:
            raise InventoryGovernanceError(f"Goods Receipt Note '{clean_id}' not found in persistence.")
        return GoodsReceiptNote.model_validate_json(row["payload_json"])

    def post_inbound_receipt(
        self,
        grn: Union[GoodsReceiptNote, str],
        batches: Optional[Sequence[FulfilmentBatch]] = None,
        idempotency_key: str = "",
        actor_id: str = "",
        actor_type: str = "STORE_RECEIVER",
        timestamp: Optional[str] = None,
        reason: str = InventoryPostingReason.GOODS_RECEIPT.value,
        notes: Optional[str] = None,
    ) -> Tuple[InventoryLedgerEntry, ...]:
        """
        Posts verified inbound receipt stock movements from an accepted GoodsReceiptNote.

        GOVERNANCE RULES:
        1. Only accepted, sellable stock can be posted. Rejected, damaged, or unaccepted goods
           are strictly forbidden from entering the ledger.
        2. If grn.total_accepted_quantity == 0, raises InventoryGovernanceError.
        3. Quantities posted for a line cannot exceed the line's accepted_quantity.
        4. Duplicate batch postings for the same FulfilmentBatch raise DuplicateBatchPostingError.
        5. Idempotent replays return identical ledger entries without duplicating movements.
        6. Conflicting reuse of idempotency_key raises IdempotencyConflictError.
        7. Stock balances are derived dynamically from ledger entries, ensuring zero counter drift.
        """
        clean_idem = idempotency_key.strip() if idempotency_key else ""
        if not clean_idem:
            raise InventoryGovernanceError("Field 'idempotency_key' must be a non-empty string.")

        clean_actor = actor_id.strip() if actor_id else ""
        if not clean_actor:
            raise InventoryGovernanceError("Field 'actor_id' must be a non-empty string.")

        clean_actor_type = actor_type.strip() if actor_type else "STORE_RECEIVER"
        clean_reason = reason.strip() if reason else InventoryPostingReason.GOODS_RECEIPT.value
        clean_notes = notes.strip() if notes and notes.strip() else None

        now_utc = dt_datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        posting_timestamp = (
            validate_and_normalize_iso_timestamp(timestamp, "timestamp", error_cls=InventoryGovernanceError)
            if timestamp
            else now_utc
        )

        grn_obj = self._resolve_grn(grn)

        # Validate store
        if grn_obj.store_id not in CANONICAL_STORES:
            raise InventoryGovernanceError(f"GRN store_id '{grn_obj.store_id}' must be one of {CANONICAL_STORES}")

        # Quality check: reject if total accepted quantity is zero and no QA reinspection release exists
        has_qa_release = False
        try:
            conn_pre = self._get_connection()
            fo_cur = conn_pre.execute("SELECT aggregate_payload FROM fulfilment_orders WHERE fo_id = ?", (grn_obj.fo_id,)).fetchone()
            if fo_cur:
                fo_dict = json.loads(fo_cur["aggregate_payload"])
                for qhr in fo_dict.get("quality_hold_resolutions", []):
                    if qhr.get("resolution_type") == "REINSPECTION_RELEASE":
                        has_qa_release = True
                        break
        except Exception:
            pass

        if grn_obj.total_accepted_quantity <= 1e-4 and not has_qa_release:
            raise InventoryGovernanceError(
                f"Goods Receipt Note '{grn_obj.grn_id}' has 0 accepted sellable units "
                f"(total_delivered={grn_obj.total_delivered_quantity}, total_rejected={grn_obj.total_rejected_quantity}). "
                "Rejected, damaged, or unaccepted goods cannot be posted to operational inventory."
            )

        # Determine batches to post
        if batches is None:
            batches_to_post = generate_grn_fulfilment_batches(grn_obj)
            if not batches_to_post:
                raise InventoryGovernanceError(
                    f"No accepted batches could be generated from GRN '{grn_obj.grn_id}'."
                )
        else:
            batches_to_post = tuple(batches)
            if not batches_to_post:
                raise InventoryGovernanceError("Batches sequence cannot be empty when explicitly provided.")

        # Build canonical payload dictionary and SHA-256 fingerprint
        canonical_batches_payload = [
            {
                "batch_id": b.batch_id,
                "sku_id": b.sku_id,
                "units": float(b.units),
                "remaining_shelf_life_days": int(b.remaining_shelf_life_days),
                "receipt_date": str(b.receipt_date),
                "supplier_lot_ref": b.supplier_lot_ref,
            }
            for b in sorted(batches_to_post, key=lambda x: (x.sku_id, x.batch_id))
        ]
        request_payload = {
            "operation": "POST_INBOUND_RECEIPT",
            "grn_id": grn_obj.grn_id,
            "fo_id": grn_obj.fo_id,
            "store_id": grn_obj.store_id,
            "batches": canonical_batches_payload,
            "actor_id": clean_actor,
            "actor_type": clean_actor_type,
            "reason": clean_reason,
        }
        req_fingerprint = compute_payload_fingerprint(request_payload)

        with self._lock:
            conn = self._get_connection()

            # 1. Idempotency Check
            cur_idem = conn.execute(
                "SELECT payload_fingerprint, posting_ids_json FROM inventory_posting_idempotency_records WHERE idempotency_key = ?",
                (clean_idem,),
            ).fetchone()
            if cur_idem:
                if cur_idem["payload_fingerprint"] == req_fingerprint:
                    # Exact replay - fetch and return existing ledger records
                    posting_ids = json.loads(cur_idem["posting_ids_json"])
                    placeholders = ",".join("?" for _ in posting_ids)
                    cur_entries = conn.execute(
                        f"""
                        SELECT * FROM inventory_ledger_entries
                        WHERE posting_id IN ({placeholders})
                        ORDER BY timestamp ASC, posting_id ASC
                        """,
                        posting_ids,
                    ).fetchall()
                    return tuple(self._row_to_ledger_entry(r) for r in cur_entries)
                else:
                    raise IdempotencyConflictError(
                        f"Idempotency key '{clean_idem}' was previously used with a different posting payload."
                    )

            # 2. Line & Batch Pre-Validation
            grn_lines = {line.sku_id: line for line in grn_obj.inspection_lines}
            seen_request_batches = set()

            for b in batches_to_post:
                if b.grn_id != grn_obj.grn_id:
                    raise InventoryGovernanceError(
                        f"Batch '{b.batch_id}' specifies grn_id='{b.batch_id}', expected '{grn_obj.grn_id}'."
                    )
                if b.fo_id != grn_obj.fo_id:
                    raise InventoryGovernanceError(
                        f"Batch '{b.batch_id}' specifies fo_id='{b.fo_id}', expected '{grn_obj.fo_id}'."
                    )
                if b.sku_id not in grn_lines:
                    raise InventoryGovernanceError(
                        f"Batch '{b.batch_id}' references SKU '{b.sku_id}' not found in GRN '{grn_obj.grn_id}' inspection lines."
                    )
                if b.batch_id in seen_request_batches:
                    raise InventoryGovernanceError(
                        f"Duplicate batch_id '{b.batch_id}' within the same inbound posting request."
                    )
                seen_request_batches.add(b.batch_id)

                # Check if batch was already posted previously
                existing_batch = conn.execute(
                    "SELECT posting_id FROM inventory_ledger_entries WHERE grn_id = ? AND fulfilment_batch_id = ?",
                    (grn_obj.grn_id, b.batch_id),
                ).fetchone()
                if existing_batch:
                    raise DuplicateBatchPostingError(
                        f"FulfilmentBatch '{b.batch_id}' from GRN '{grn_obj.grn_id}' has already been posted "
                        f"(posting_id='{existing_batch['posting_id']}')."
                    )

            # 3. Check cumulative quantities per SKU against accepted limits
            skus_in_batches = {b.sku_id for b in batches_to_post}
            for sku in skus_in_batches:
                line = grn_lines[sku]
                incoming_units = round(sum(b.units for b in batches_to_post if b.sku_id == sku), 4)

                prior_posted_row = conn.execute(
                    """
                    SELECT COALESCE(SUM(quantity), 0.0) FROM inventory_ledger_entries
                    WHERE grn_id = ? AND sku_id = ? AND transaction_type = 'INBOUND_RECEIPT'
                    """,
                    (grn_obj.grn_id, sku),
                ).fetchone()
                prior_posted = float(prior_posted_row[0]) if prior_posted_row else 0.0

                # Check for any QA reinspection releases on the parent fulfilment order
                released_qty = 0.0
                try:
                    fo_row = conn.execute("SELECT aggregate_payload FROM fulfilment_orders WHERE fo_id = ?", (grn_obj.fo_id,)).fetchone()
                    if fo_row:
                        fo_payload = json.loads(fo_row["aggregate_payload"])
                        for qhr in fo_payload.get("quality_hold_resolutions", []):
                            if qhr.get("sku_id") == sku and qhr.get("resolution_type") == "REINSPECTION_RELEASE":
                                released_qty += float(qhr.get("quantity", 0.0))
                except Exception:
                    pass

                max_allowable = round(line.accepted_quantity + released_qty, 4)
                total_after = round(prior_posted + incoming_units, 4)
                if total_after > max_allowable + 1e-4:
                    raise InventoryGovernanceError(
                        f"[{sku}] Posting quantity {incoming_units} would bring total posted units to {total_after}, "
                        f"exceeding line accepted_quantity of {max_allowable} (dock accepted: {line.accepted_quantity}, "
                        f"reinspection released: {released_qty}) on GRN '{grn_obj.grn_id}'. "
                        f"Damaged ({line.damaged_quantity}), excess ({line.excess_quantity}), or rejected stock cannot be posted."
                    )

            # 4. Construct Models for Ledger Entries and Batch Records
            transaction_id = f"TXN-{grn_obj.grn_id}"
            created_at = now_utc
            ledger_entries: List[InventoryLedgerEntry] = []
            batch_records: List[Dict[str, Any]] = []

            for b in batches_to_post:
                line = grn_lines[b.sku_id]
                posting_id = f"POST-{grn_obj.grn_id}-{b.batch_id}"

                receipt_dt = dt_date.fromisoformat(b.receipt_date)
                expiry_dt = receipt_dt + timedelta(days=b.remaining_shelf_life_days)
                expiry_date_str = expiry_dt.isoformat()

                unit_cost = line.unit_cost
                total_cost = round(b.units * unit_cost, 2)

                entry = InventoryLedgerEntry(
                    posting_id=posting_id,
                    transaction_id=transaction_id,
                    idempotency_key=clean_idem,
                    store_id=grn_obj.store_id,
                    sku_id=b.sku_id,
                    batch_id=b.batch_id,
                    transaction_type=InventoryTransactionType.INBOUND_RECEIPT,
                    quantity=b.units,
                    unit_cost=unit_cost,
                    total_cost=total_cost,
                    remaining_shelf_life_days=b.remaining_shelf_life_days,
                    expiry_date=expiry_date_str,
                    fo_id=grn_obj.fo_id,
                    shipment_id=grn_obj.shipment_id,
                    grn_id=grn_obj.grn_id,
                    fulfilment_batch_id=b.batch_id,
                    actor_type=clean_actor_type,
                    actor_id=clean_actor,
                    timestamp=posting_timestamp,
                    reason=clean_reason,
                    payload_fingerprint=req_fingerprint,
                    notes=clean_notes,
                    created_at=created_at,
                )
                ledger_entries.append(entry)

                batch_record_data = {
                    "batch_id": b.batch_id,
                    "store_id": grn_obj.store_id,
                    "sku_id": b.sku_id,
                    "initial_units": b.units,
                    "receipt_date": b.receipt_date,
                    "expiry_date": expiry_date_str,
                    "remaining_shelf_life_days_at_receipt": b.remaining_shelf_life_days,
                    "unit_cost": unit_cost,
                    "fo_id": grn_obj.fo_id,
                    "grn_id": grn_obj.grn_id,
                    "fulfilment_batch_id": b.batch_id,
                    "created_at": created_at,
                }
                batch_records.append(batch_record_data)

            # 5. Atomic SQLite Persistence
            try:
                with conn:
                    # Insert ledger entries
                    for entry in ledger_entries:
                        conn.execute(
                            """
                            INSERT INTO inventory_ledger_entries (
                                posting_id, transaction_id, idempotency_key, store_id, sku_id,
                                batch_id, transaction_type, quantity, unit_cost, total_cost,
                                remaining_shelf_life_days, expiry_date, fo_id, shipment_id,
                                grn_id, fulfilment_batch_id, actor_type, actor_id, timestamp,
                                reason, payload_fingerprint, notes, created_at
                            ) VALUES (
                                :posting_id, :transaction_id, :idempotency_key, :store_id, :sku_id,
                                :batch_id, :transaction_type, :quantity, :unit_cost, :total_cost,
                                :remaining_shelf_life_days, :expiry_date, :fo_id, :shipment_id,
                                :grn_id, :fulfilment_batch_id, :actor_type, :actor_id, :timestamp,
                                :reason, :payload_fingerprint, :notes, :created_at
                            );
                            """,
                            {
                                **entry.model_dump(),
                                "transaction_type": entry.transaction_type.value,
                            },
                        )

                    # Insert inventory batch records
                    for b_data in batch_records:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO inventory_batches (
                                batch_id, store_id, sku_id, initial_units, receipt_date,
                                expiry_date, remaining_shelf_life_days_at_receipt, unit_cost,
                                fo_id, grn_id, fulfilment_batch_id, created_at
                            ) VALUES (
                                :batch_id, :store_id, :sku_id, :initial_units, :receipt_date,
                                :expiry_date, :remaining_shelf_life_days_at_receipt, :unit_cost,
                                :fo_id, :grn_id, :fulfilment_batch_id, :created_at
                            );
                            """,
                            b_data,
                        )

                    # Insert idempotency token
                    posting_ids_json = json.dumps([e.posting_id for e in ledger_entries])
                    conn.execute(
                        """
                        INSERT INTO inventory_posting_idempotency_records (
                            idempotency_key, operation_type, store_id, grn_id, fo_id,
                            payload_fingerprint, posting_ids_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            clean_idem,
                            "POST_INBOUND_RECEIPT",
                            grn_obj.store_id,
                            grn_obj.grn_id,
                            grn_obj.fo_id,
                            req_fingerprint,
                            posting_ids_json,
                            created_at,
                        ),
                    )

            except sqlite3.IntegrityError as err:
                err_msg = str(err)
                if "idx_inv_ledger_unique_batch_posting" in err_msg or "fulfilment_batch_id" in err_msg:
                    raise DuplicateBatchPostingError(
                        f"Database constraint violation: batch already posted from GRN '{grn_obj.grn_id}': {err}"
                    ) from err
                if "idempotency_key" in err_msg:
                    raise IdempotencyConflictError(
                        f"Concurrent or conflicting idempotency key '{clean_idem}': {err}"
                    ) from err
                raise InventoryPersistenceError(f"Database integrity error during inventory posting: {err}") from err

            return tuple(ledger_entries)

    def _row_to_ledger_entry(self, row: sqlite3.Row) -> InventoryLedgerEntry:
        """Helper to convert a sqlite3.Row to an InventoryLedgerEntry model."""
        return InventoryLedgerEntry(
            posting_id=row["posting_id"],
            transaction_id=row["transaction_id"],
            idempotency_key=row["idempotency_key"],
            store_id=row["store_id"],
            sku_id=row["sku_id"],
            batch_id=row["batch_id"],
            transaction_type=InventoryTransactionType(row["transaction_type"]),
            quantity=row["quantity"],
            unit_cost=row["unit_cost"],
            total_cost=row["total_cost"],
            remaining_shelf_life_days=row["remaining_shelf_life_days"],
            expiry_date=row["expiry_date"],
            fo_id=row["fo_id"],
            shipment_id=row["shipment_id"],
            grn_id=row["grn_id"],
            fulfilment_batch_id=row["fulfilment_batch_id"],
            actor_type=row["actor_type"],
            actor_id=row["actor_id"],
            timestamp=row["timestamp"],
            reason=row["reason"],
            payload_fingerprint=row["payload_fingerprint"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def get_ledger_history(
        self,
        store_id: Optional[str] = None,
        sku_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        fo_id: Optional[str] = None,
        grn_id: Optional[str] = None,
        transaction_type: Optional[Union[InventoryTransactionType, str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[InventoryLedgerEntry, ...]:
        """
        Queries append-only ledger history with flexible filtering by store, SKU,
        batch, FO, GRN, transaction type, or time window.
        """
        conn = self._get_connection()
        query = "SELECT * FROM inventory_ledger_entries WHERE 1=1"
        params: List[Any] = []

        if store_id:
            query += " AND store_id = ?"
            params.append(store_id.strip())
        if sku_id:
            query += " AND sku_id = ?"
            params.append(sku_id.strip())
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id.strip())
        if fo_id:
            query += " AND fo_id = ?"
            params.append(fo_id.strip())
        if grn_id:
            query += " AND grn_id = ?"
            params.append(grn_id.strip())
        if transaction_type:
            tt_val = transaction_type.value if isinstance(transaction_type, InventoryTransactionType) else str(transaction_type)
            query += " AND transaction_type = ?"
            params.append(tt_val.strip())
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.strip())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.strip())

        query += " ORDER BY timestamp ASC, posting_id ASC"

        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(int(limit))

        cur = conn.execute(query, params)
        return tuple(self._row_to_ledger_entry(r) for r in cur.fetchall())

    def get_on_hand_balance(self, store_id: str, sku_id: str) -> float:
        """
        Derives current authoritative on-hand stock for a store and SKU from the immutable ledger.
        Returns 0.0 if the ledger contains no entries (zero opening stock).
        """
        clean_store = store_id.strip() if store_id else ""
        clean_sku = sku_id.strip() if sku_id else ""
        if not clean_store or not clean_sku:
            return 0.0

        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT COALESCE(SUM(quantity), 0.0)
            FROM inventory_ledger_entries
            WHERE store_id = ? AND sku_id = ?
            """,
            (clean_store, clean_sku),
        )
        row = cur.fetchone()
        return round(float(row[0]), 4) if row else 0.0

    def get_store_balances(self, store_id: str) -> List[StoreSkuBalance]:
        """
        Derives aggregated stock positions for all active SKUs at a store from the ledger.
        Excludes SKUs with zero or negative derived stock balances.
        """
        clean_store = store_id.strip() if store_id else ""
        if not clean_store:
            return []

        conn = self._get_connection()
        # Find distinct SKUs with movements
        sku_rows = conn.execute(
            """
            SELECT sku_id,
                   COALESCE(SUM(quantity), 0.0) as on_hand_stock,
                   COALESCE(SUM(total_cost), 0.0) as total_cost_inr,
                   MAX(timestamp) as last_movement_at
            FROM inventory_ledger_entries
            WHERE store_id = ?
            GROUP BY sku_id
            HAVING SUM(quantity) > 1e-4
            ORDER BY sku_id ASC
            """,
            (clean_store,),
        ).fetchall()

        results: List[StoreSkuBalance] = []
        for r in sku_rows:
            sku = r["sku_id"]
            # Count active batches
            batch_count_row = conn.execute(
                """
                SELECT COUNT(DISTINCT batch_id)
                FROM inventory_ledger_entries
                WHERE store_id = ? AND sku_id = ?
                GROUP BY batch_id
                HAVING SUM(quantity) > 1e-4
                """,
                (clean_store, sku),
            ).fetchall()
            active_batches = len(batch_count_row)

            results.append(
                StoreSkuBalance(
                    store_id=clean_store,
                    sku_id=sku,
                    on_hand_stock=round(r["on_hand_stock"], 4),
                    total_cost_inr=max(0.0, round(r["total_cost_inr"], 2)),
                    active_batch_count=active_batches,
                    last_movement_at=r["last_movement_at"],
                )
            )
        return results

    def get_batch_balance(
        self,
        store_id: str,
        sku_id: str,
        batch_id: str,
        as_of_date: Optional[str] = None,
    ) -> Optional[BatchBalance]:
        """
        Derives the current on-hand stock and perishability status for a single batch.
        """
        clean_store = store_id.strip() if store_id else ""
        clean_sku = sku_id.strip() if sku_id else ""
        clean_batch = batch_id.strip() if batch_id else ""
        if not clean_store or not clean_sku or not clean_batch:
            return None

        conn = self._get_connection()
        batch_row = conn.execute(
            """
            SELECT b.*, COALESCE(SUM(l.quantity), 0.0) as on_hand
            FROM inventory_batches b
            LEFT JOIN inventory_ledger_entries l
                ON b.store_id = l.store_id AND b.sku_id = l.sku_id AND b.batch_id = l.batch_id
            WHERE b.store_id = ? AND b.sku_id = ? AND b.batch_id = ?
            GROUP BY b.store_id, b.sku_id, b.batch_id
            """,
            (clean_store, clean_sku, clean_batch),
        ).fetchone()

        if not batch_row:
            return None

        on_hand = round(float(batch_row["on_hand"]), 4)
        receipt_date_str = batch_row["receipt_date"]
        expiry_date_str = batch_row["expiry_date"]

        ref_date = dt_date.fromisoformat(as_of_date) if as_of_date else dt_datetime.now(timezone.utc).date()
        exp_date = dt_date.fromisoformat(expiry_date_str)
        diff_days = (exp_date - ref_date).days
        is_expired = diff_days <= 0
        current_remaining = max(0, diff_days)

        return BatchBalance(
            store_id=clean_store,
            sku_id=clean_sku,
            batch_id=clean_batch,
            on_hand_quantity=on_hand,
            receipt_date=receipt_date_str,
            expiry_date=expiry_date_str,
            remaining_shelf_life_days_at_receipt=batch_row["remaining_shelf_life_days_at_receipt"],
            current_remaining_shelf_life_days=current_remaining,
            is_expired=is_expired,
            unit_cost=batch_row["unit_cost"],
            fo_id=batch_row["fo_id"],
            grn_id=batch_row["grn_id"],
            fulfilment_batch_id=batch_row["fulfilment_batch_id"],
        )

    def get_batch_balances(
        self,
        store_id: str,
        sku_id: Optional[str] = None,
        as_of_date: Optional[str] = None,
        include_zero_balance: bool = False,
    ) -> List[BatchBalance]:
        """
        Derives batch balances for a store (and optional SKU), ordered by FEFO (expiry_date ASC).
        """
        clean_store = store_id.strip() if store_id else ""
        if not clean_store:
            return []

        conn = self._get_connection()
        query = """
            SELECT b.*, COALESCE(SUM(l.quantity), 0.0) as on_hand
            FROM inventory_batches b
            LEFT JOIN inventory_ledger_entries l
                ON b.store_id = l.store_id AND b.sku_id = l.sku_id AND b.batch_id = l.batch_id
            WHERE b.store_id = ?
        """
        params: List[Any] = [clean_store]

        if sku_id:
            query += " AND b.sku_id = ?"
            params.append(sku_id.strip())

        query += " GROUP BY b.store_id, b.sku_id, b.batch_id"

        if not include_zero_balance:
            query += " HAVING COALESCE(SUM(l.quantity), 0.0) > 1e-4"

        query += " ORDER BY b.expiry_date ASC, b.batch_id ASC"

        rows = conn.execute(query, params).fetchall()
        ref_date = dt_date.fromisoformat(as_of_date) if as_of_date else dt_datetime.now(timezone.utc).date()

        results: List[BatchBalance] = []
        for r in rows:
            on_hand = round(float(r["on_hand"]), 4)
            exp_date = dt_date.fromisoformat(r["expiry_date"])
            diff_days = (exp_date - ref_date).days
            is_expired = diff_days <= 0
            current_remaining = max(0, diff_days)

            results.append(
                BatchBalance(
                    store_id=r["store_id"],
                    sku_id=r["sku_id"],
                    batch_id=r["batch_id"],
                    on_hand_quantity=on_hand,
                    receipt_date=r["receipt_date"],
                    expiry_date=r["expiry_date"],
                    remaining_shelf_life_days_at_receipt=r["remaining_shelf_life_days_at_receipt"],
                    current_remaining_shelf_life_days=current_remaining,
                    is_expired=is_expired,
                    unit_cost=r["unit_cost"],
                    fo_id=r["fo_id"],
                    grn_id=r["grn_id"],
                    fulfilment_batch_id=r["fulfilment_batch_id"],
                )
            )
        return results

    def get_expiring_batches(
        self,
        store_id: str,
        as_of_date: str,
        days_threshold: int = 2,
    ) -> List[BatchBalance]:
        """
        Returns all active batches at a store expiring within days_threshold of as_of_date.
        Ordered by FEFO (earliest expiry date first).
        """
        clean_date = as_of_date.strip() if as_of_date else ""
        if not clean_date:
            raise ValueError("as_of_date must be provided in YYYY-MM-DD format.")

        active_batches = self.get_batch_balances(
            store_id=store_id,
            as_of_date=clean_date,
            include_zero_balance=False,
        )

        ref_date = dt_date.fromisoformat(clean_date)
        expiring: List[BatchBalance] = []
        for b in active_batches:
            exp_date = dt_date.fromisoformat(b.expiry_date)
            remaining_days = (exp_date - ref_date).days
            if remaining_days <= days_threshold:
                expiring.append(b)

        return expiring

    def reconcile_grn_postings(self, grn: Union[GoodsReceiptNote, str]) -> GrnPostingReconciliation:
        """
        Reconciles physical dock receipt quantities from a GoodsReceiptNote against
        posted quantities in the operational inventory ledger.
        Surfaces line-level balances, posted batch IDs, and remaining unposted units.
        """
        grn_obj = self._resolve_grn(grn)
        conn = self._get_connection()

        # Query all ledger entries for this GRN
        cur_entries = conn.execute(
            """
            SELECT * FROM inventory_ledger_entries
            WHERE grn_id = ?
            ORDER BY timestamp ASC, posting_id ASC
            """,
            (grn_obj.grn_id,),
        ).fetchall()
        entries = [self._row_to_ledger_entry(r) for r in cur_entries]

        # Group posted quantities and batch IDs by SKU
        posted_by_sku: Dict[str, List[InventoryLedgerEntry]] = {}
        for e in entries:
            posted_by_sku.setdefault(e.sku_id, []).append(e)

        # Check parent FO for any QA reinspection releases on this shipment / GRN
        released_by_sku: Dict[str, float] = {}
        try:
            fo_cur = conn.execute(
                "SELECT aggregate_payload FROM fulfilment_orders WHERE fo_id = ?",
                (grn_obj.fo_id,),
            ).fetchone()
            if fo_cur:
                fo_dict = json.loads(fo_cur["aggregate_payload"])
                for qhr in fo_dict.get("quality_hold_resolutions", []):
                    if (
                        qhr.get("resolution_type") == "REINSPECTION_RELEASE"
                        and qhr.get("shipment_id") == grn_obj.shipment_id
                    ):
                        sku = qhr.get("sku_id")
                        if sku:
                            released_by_sku[sku] = round(
                                released_by_sku.get(sku, 0.0) + float(qhr.get("quantity", 0.0)),
                                4,
                            )
        except Exception:
            pass

        line_reconciliations: List[GrnLinePostingReconciliation] = []
        total_posted = 0.0

        for line in grn_obj.inspection_lines:
            line_entries = posted_by_sku.get(line.sku_id, [])
            line_posted_qty = round(sum(e.quantity for e in line_entries if e.transaction_type == InventoryTransactionType.INBOUND_RECEIPT), 4)
            rel_qty = released_by_sku.get(line.sku_id, 0.0)
            line_effective_accepted = round(line.accepted_quantity + rel_qty, 4)
            unposted_qty = max(0.0, round(line_effective_accepted - line_posted_qty, 4))
            posted_batch_ids = tuple(e.batch_id for e in line_entries)
            is_line_fully_posted = (
                line_effective_accepted > 0
                and line_posted_qty >= line_effective_accepted - 1e-4
            )

            total_posted += line_posted_qty
            line_reconciliations.append(
                GrnLinePostingReconciliation(
                    sku_id=line.sku_id,
                    accepted_quantity=line.accepted_quantity,
                    rejected_quantity=line.rejected_quantity,
                    posted_quantity=line_posted_qty,
                    unposted_accepted_quantity=unposted_qty,
                    batch_references=line.batch_references,
                    posted_batch_ids=posted_batch_ids,
                    is_fully_posted=is_line_fully_posted,
                )
            )

        total_effective_accepted = round(grn_obj.total_accepted_quantity + sum(released_by_sku.values()), 4)
        is_fully_posted = (
            total_effective_accepted > 0
            and all(
                lr.is_fully_posted
                for lr in line_reconciliations
                if (lr.accepted_quantity + released_by_sku.get(lr.sku_id, 0.0)) > 0
            )
        )

        return GrnPostingReconciliation(
            grn_id=grn_obj.grn_id,
            fo_id=grn_obj.fo_id,
            store_id=grn_obj.store_id,
            supplier_id=grn_obj.supplier_id,
            receipt_at=grn_obj.receipt_at,
            total_accepted_quantity=grn_obj.total_accepted_quantity,
            total_posted_quantity=round(total_posted, 4),
            is_fully_posted=is_fully_posted,
            lines=tuple(line_reconciliations),
        )

    def get_inventory_position(
        self,
        store_id: str,
        sku_id: str,
        reserved_stock: float = 0.0,
        on_order_stock: float = 0.0,
    ) -> InventoryPosition:
        """
        Derives an InventoryPosition model combining authoritative on-hand stock
        from the ledger with active reservations and in-transit orders.
        """
        on_hand = self.get_on_hand_balance(store_id, sku_id)
        return InventoryPosition(
            on_hand_stock=on_hand,
            reserved_stock=reserved_stock,
            on_order_stock=on_order_stock,
        )

    def get_active_inventory_batches(
        self,
        store_id: str,
        sku_id: str,
        as_of_date: Optional[str] = None,
    ) -> Tuple[InventoryBatch, ...]:
        """
        Retrieves active inventory batches converted to the core InventoryBatch model
        for immediate interoperability with the Phase 3 InventoryEngine.
        """
        batches = self.get_batch_balances(
            store_id=store_id,
            sku_id=sku_id,
            as_of_date=as_of_date,
            include_zero_balance=False,
        )
        return tuple(
            InventoryBatch(
                batch_id=b.batch_id,
                units=b.on_hand_quantity,
                remaining_shelf_life_days=b.current_remaining_shelf_life_days if b.current_remaining_shelf_life_days is not None else 0,
                creation_date=b.receipt_date,
            )
            for b in batches
        )

    def close(self) -> None:
        """Closes the active database connection if open."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
