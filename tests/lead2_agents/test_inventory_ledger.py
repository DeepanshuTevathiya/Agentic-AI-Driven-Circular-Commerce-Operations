"""
Unit & Integration Tests: Operational Inventory Ledger & Durable Posting Persistence
=====================================================================================
Phase 5.7.1 Verification Test Suite:
1. Schema integrity, frozen immutability, and validation invariants.
2. Inbound receipt posting from accepted GoodsReceiptNote and FulfilmentBatch minting.
3. Strict quality gate enforcement: rejected, damaged, or excess goods never posted.
4. Total rejection (0 accepted units) blocks posting into sellable inventory.
5. Over-posting protection and cumulative line balance enforcement.
6. Duplicate batch posting prevention (DuplicateBatchPostingError).
7. Duplicate batch IDs within a single posting request rejected.
8. Scoped idempotency: exact replays return identical entries without duplicate movements.
9. Conflicting idempotency key reuse raises IdempotencyConflictError.
10. Dynamic on-hand balance derivation from SUM(quantity); zero counter drift.
11. Empty ledger returns 0.0 on-hand stock (zero assumed opening stock).
12. Batch balance tracking and expiration dates.
13. FEFO (First-Expiring, First-Out) query ordering and get_expiring_batches().
14. Audit reconciliation query (reconcile_grn_postings).
15. InventoryPosition and core InventoryBatch interoperability with InventoryEngine.
16. Persistence recovery across repository restarts.
17. Transaction atomicity and rollback on errors.
18. Source PurchaseOrder immutability preserved.
"""

from datetime import date as dt_date, timedelta
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from pydantic import ValidationError

from src.agents.common.config import CANONICAL_SKUS, CANONICAL_STORES
from src.agents.common.schemas import InventoryBatch, InventoryPosition
from src.agents.fulfilment.repository import FulfilmentRepository
from src.agents.fulfilment.schemas import (
    AdvanceShipmentNotice,
    DiscrepancyReason,
    DockInspectionLine,
    FulfilmentBatch,
    FulfilmentOrder,
    GoodsReceiptNote,
    ShipmentLineItem,
    generate_grn_fulfilment_batches,
)
from src.agents.inventory import (
    BatchBalance,
    DuplicateBatchPostingError,
    GrnPostingReconciliation,
    IdempotencyConflictError,
    InventoryBatchRecord,
    InventoryEngine,
    InventoryGovernanceError,
    InventoryLedgerEntry,
    InventoryPersistenceError,
    InventoryPostingReason,
    InventoryTransactionType,
    OperationalInventoryRepository,
    StoreSkuBalance,
    compute_payload_fingerprint,
    validate_batch_inventory,
)
from src.agents.procurement.approval import approve_purchase_order
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
)


def _make_approved_po(
    po_id: str = "PO-20261115-DS_BLR_KORAMANGALA-DAIRY-01",
    store_id: str = "DS_BLR_KORAMANGALA",
    supplier_id: str = "SIM_SUPP_DAIRY_PRIMARY",
    sku_id: str = "SKU_MILK_1L",
    final_order_qty: float = 24.0,
    unit_cost: float = 30.0,
) -> PurchaseOrder:
    """Helper to construct an approved PurchaseOrder."""
    item = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id=sku_id,
        category="Dairy",
        raw_suggested_qty=20.0,
        case_pack_rounded_qty=24.0,
        final_order_qty=final_order_qty,
        case_pack_size=12,
        moq=10.0,
        unit_retail_price=50.0,
        unit_procurement_cost=unit_cost,
        total_line_cost=round(final_order_qty * unit_cost, 2),
        requires_escalation=False,
        escalation_reasons=[],
    )
    po = PurchaseOrder(
        po_id=po_id,
        created_date="2026-11-15",
        store_id=store_id,
        supplier_id=supplier_id,
        supplier_type=SupplierType.PRIMARY,
        items=[item],
        total_order_qty=final_order_qty,
        total_order_cost=round(final_order_qty * unit_cost, 2),
        urgency=PurchaseOrderUrgency.STANDARD,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        requires_escalation=False,
        escalation_reasons=[],
        audit_trail=(),
    )
    return approve_purchase_order(
        po=po,
        approved_by="OPS_MANAGER_ALOK",
        approval_timestamp="2026-11-15T10:00:00Z",
        approval_reason="Approved for fulfilment.",
    )


def _make_grn(
    grn_id: str = "GRN-FO-001-01",
    fo_id: str = "FO-001",
    po_id: str = "PO-20261115-DS_BLR_KORAMANGALA-DAIRY-01",
    shipment_id: str = "SHP-FO-001-01",
    store_id: str = "DS_BLR_KORAMANGALA",
    supplier_id: str = "SIM_SUPP_DAIRY_PRIMARY",
    sku_id: str = "SKU_MILK_1L",
    delivered_qty: float = 24.0,
    accepted_qty: float = 24.0,
    rejected_qty: float = 0.0,
    damaged_qty: float = 0.0,
    unit_cost: float = 30.0,
    remaining_shelf_life_days: int = 5,
    total_shelf_life_days: int = 7,
    batch_references: tuple = ("BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-01",),
) -> GoodsReceiptNote:
    """Helper to construct a valid GoodsReceiptNote."""
    line = DockInspectionLine(
        sku_id=sku_id,
        delivered_quantity=delivered_qty,
        accepted_quantity=accepted_qty,
        rejected_quantity=rejected_qty,
        damaged_quantity=damaged_qty,
        unit_cost=unit_cost,
        remaining_shelf_life_days=remaining_shelf_life_days if accepted_qty > 0 else None,
        total_shelf_life_days=total_shelf_life_days,
        batch_references=batch_references if accepted_qty > 0 else (),
    )
    return GoodsReceiptNote(
        grn_id=grn_id,
        fo_id=fo_id,
        po_id=po_id,
        shipment_id=shipment_id,
        store_id=store_id,
        supplier_id=supplier_id,
        receipt_at="2026-11-15T14:00:00Z",
        receiver_id="RECEIVER_MEENA",
        inspection_lines=(line,),
    )


class TestOperationalInventoryLedger(unittest.TestCase):
    """Comprehensive test suite for Phase 5.7.1 Operational Inventory Ledger."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_inventory.db"
        self.fulfilment_repo = FulfilmentRepository(db_path=self.db_path)
        self.inventory_repo = OperationalInventoryRepository(
            db_path=self.db_path,
            fulfilment_repo=self.fulfilment_repo,
        )

    def tearDown(self):
        self.inventory_repo.close()
        self.fulfilment_repo.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _persist_po_and_grn(self, po: PurchaseOrder, grn: GoodsReceiptNote) -> None:
        """Helper to persist prerequisites into fulfilment database tables."""
        conn = self.fulfilment_repo._get_connection()
        with conn:
            # Insert fulfilment order stub
            conn.execute(
                """
                INSERT OR IGNORE INTO fulfilment_orders (
                    fo_id, po_id, store_id, supplier_id, supplier_type, status,
                    created_at, updated_at, aggregate_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    grn.fo_id,
                    po.po_id,
                    po.store_id,
                    po.supplier_id,
                    po.supplier_type.value,
                    "IN_TRANSIT",
                    "2026-11-15T10:00:00Z",
                    "2026-11-15T10:00:00Z",
                    po.model_dump_json(),
                ),
            )
            # Insert shipment mapping
            conn.execute(
                """
                INSERT OR IGNORE INTO fulfilment_shipment_ids (fo_id, shipment_id)
                VALUES (?, ?);
                """,
                (grn.fo_id, grn.shipment_id),
            )
            # Insert GRN
            self.fulfilment_repo.save_goods_receipt_note(grn)

    # -------------------------------------------------------------------------
    # 1. Schema Integrity & Validation Invariants
    # -------------------------------------------------------------------------

    def test_schema_immutability_and_validation(self):
        """Validates frozen immutability and invariant enforcement on ledger models."""
        entry = InventoryLedgerEntry(
            posting_id="POST-001",
            transaction_id="TXN-001",
            idempotency_key="IDEM-001",
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            batch_id="BATCH-001",
            transaction_type=InventoryTransactionType.INBOUND_RECEIPT,
            quantity=10.0,
            unit_cost=30.0,
            total_cost=300.0,
            remaining_shelf_life_days=5,
            expiry_date="2026-11-20",
            fo_id="FO-001",
            shipment_id="SHP-001",
            grn_id="GRN-001",
            fulfilment_batch_id="BATCH-001",
            actor_type="STORE_RECEIVER",
            actor_id="USER_01",
            timestamp="2026-11-15T14:00:00Z",
            reason=InventoryPostingReason.GOODS_RECEIPT.value,
            payload_fingerprint="abc123def456",
            notes="Dock receipt",
            created_at="2026-11-15T14:05:00Z",
        )
        # Verify frozen immutability
        with self.assertRaises(ValidationError):
            entry.quantity = 15.0  # type: ignore

        # Zero quantity rejected
        with self.assertRaises(ValidationError):
            InventoryLedgerEntry(
                posting_id="POST-002",
                transaction_id="TXN-002",
                idempotency_key="IDEM-002",
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                batch_id="BATCH-002",
                transaction_type=InventoryTransactionType.INBOUND_RECEIPT,
                quantity=0.0,
                unit_cost=30.0,
                total_cost=0.0,
                actor_id="USER_01",
                timestamp="2026-11-15T14:00:00Z",
                reason="TEST",
                payload_fingerprint="fp",
                created_at="2026-11-15T14:00:00Z",
            )

        # Inbound receipt with negative quantity rejected
        with self.assertRaises(ValidationError):
            InventoryLedgerEntry(
                posting_id="POST-003",
                transaction_id="TXN-003",
                idempotency_key="IDEM-003",
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                batch_id="BATCH-003",
                transaction_type=InventoryTransactionType.INBOUND_RECEIPT,
                quantity=-5.0,
                unit_cost=30.0,
                total_cost=-150.0,
                actor_id="USER_01",
                timestamp="2026-11-15T14:00:00Z",
                reason="TEST",
                payload_fingerprint="fp",
                created_at="2026-11-15T14:00:00Z",
            )

        # Total cost mismatch rejected
        with self.assertRaises(ValidationError):
            InventoryLedgerEntry(
                posting_id="POST-004",
                transaction_id="TXN-004",
                idempotency_key="IDEM-004",
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                batch_id="BATCH-004",
                transaction_type=InventoryTransactionType.INBOUND_RECEIPT,
                quantity=10.0,
                unit_cost=30.0,
                total_cost=250.0,  # Expected 300.0
                actor_id="USER_01",
                timestamp="2026-11-15T14:00:00Z",
                reason="TEST",
                payload_fingerprint="fp",
                created_at="2026-11-15T14:00:00Z",
            )

    # -------------------------------------------------------------------------
    # 2. Inbound Receipt Posting & Batch Tracking
    # -------------------------------------------------------------------------

    def test_post_inbound_receipt_success(self):
        """Verifies successful posting of accepted GRN batches into ledger and inventory batches."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        entries = self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-POST-001",
            actor_id="RECEIVER_MEENA",
            actor_type="STORE_RECEIVER",
            reason=InventoryPostingReason.GOODS_RECEIPT.value,
        )

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.store_id, "DS_BLR_KORAMANGALA")
        self.assertEqual(entry.sku_id, "SKU_MILK_1L")
        self.assertEqual(entry.quantity, 24.0)
        self.assertEqual(entry.unit_cost, 30.0)
        self.assertEqual(entry.total_cost, 720.0)
        self.assertEqual(entry.transaction_type, InventoryTransactionType.INBOUND_RECEIPT)
        self.assertEqual(entry.grn_id, grn.grn_id)
        self.assertEqual(entry.fo_id, grn.fo_id)
        self.assertEqual(entry.batch_id, "BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-01")

        # Verify on-hand stock derived dynamically
        on_hand = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(on_hand, 24.0)

        # Verify batch balance
        batch_bal = self.inventory_repo.get_batch_balance(
            "DS_BLR_KORAMANGALA",
            "SKU_MILK_1L",
            "BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-01",
            as_of_date="2026-11-15",
        )
        self.assertIsNotNone(batch_bal)
        self.assertEqual(batch_bal.on_hand_quantity, 24.0)
        self.assertEqual(batch_bal.expiry_date, "2026-11-20")
        self.assertEqual(batch_bal.current_remaining_shelf_life_days, 5)
        self.assertFalse(batch_bal.is_expired)

    def test_post_inbound_receipt_multiple_batches_per_line(self):
        """Verifies multi-batch receipt posting (e.g. 2 batches of 12 units for 24 total units)."""
        po = _make_approved_po()
        grn = _make_grn(
            batch_references=(
                "BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-A",
                "BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-B",
            )
        )
        self._persist_po_and_grn(po, grn)

        entries = self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-POST-MULTI-BATCH",
            actor_id="RECEIVER_MEENA",
        )

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].quantity, 12.0)
        self.assertEqual(entries[1].quantity, 12.0)
        self.assertEqual(entries[0].batch_id, "BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-A")
        self.assertEqual(entries[1].batch_id, "BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-B")

        # Total on-hand should be 24.0
        on_hand = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(on_hand, 24.0)

        # Store balance summary should show 2 active batches
        store_bals = self.inventory_repo.get_store_balances("DS_BLR_KORAMANGALA")
        self.assertEqual(len(store_bals), 1)
        self.assertEqual(store_bals[0].active_batch_count, 2)
        self.assertEqual(store_bals[0].on_hand_stock, 24.0)

    # -------------------------------------------------------------------------
    # 3. Quality Gate Enforcement & Rejection Blocking
    # -------------------------------------------------------------------------

    def test_zero_accepted_units_grn_rejected(self):
        """Verifies GRN with 0 accepted units cannot be posted into sellable inventory."""
        po = _make_approved_po()
        # 100% damaged line
        line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=0.0,
            rejected_quantity=24.0,
            damaged_quantity=24.0,
            unit_cost=30.0,
            total_shelf_life_days=7,
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        grn = GoodsReceiptNote(
            grn_id="GRN-FO-001-ALL-DAMAGED",
            fo_id="FO-001",
            po_id=po.po_id,
            shipment_id="SHP-FO-001-01",
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            receipt_at="2026-11-15T14:00:00Z",
            receiver_id="RECEIVER_MEENA",
            inspection_lines=(line,),
        )
        self._persist_po_and_grn(po, grn)

        with self.assertRaises(InventoryGovernanceError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn=grn,
                idempotency_key="IDEM-POST-DAMAGED",
                actor_id="RECEIVER_MEENA",
            )
        self.assertIn("0 accepted sellable units", str(ctx.exception))

        # Ledger remains completely empty
        on_hand = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(on_hand, 0.0)

    def test_partial_rejection_only_accepted_units_posted(self):
        """Verifies partial dock rejection: only accepted units post, rejected/damaged units blocked."""
        po = _make_approved_po()
        grn = _make_grn(
            delivered_qty=24.0,
            accepted_qty=16.0,
            rejected_qty=8.0,
            damaged_qty=8.0,
            batch_references=("BATCH-PARTIAL-01",),
        )
        self._persist_po_and_grn(po, grn)

        entries = self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-POST-PARTIAL",
            actor_id="RECEIVER_MEENA",
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].quantity, 16.0)

        # On-hand stock reflects ONLY accepted 16.0 units, NOT delivered 24.0
        on_hand = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(on_hand, 16.0)

    def test_over_posting_beyond_accepted_quantity_rejected(self):
        """Verifies that attempting to post more than accepted_quantity raises InventoryGovernanceError."""
        po = _make_approved_po()
        grn = _make_grn(delivered_qty=24.0, accepted_qty=16.0, rejected_qty=8.0)
        self._persist_po_and_grn(po, grn)

        # Attempt to post 24 units when only 16 were accepted
        custom_batch = FulfilmentBatch(
            batch_id="BATCH-OVER-01",
            sku_id="SKU_MILK_1L",
            units=24.0,
            remaining_shelf_life_days=5,
            receipt_date="2026-11-15",
            fo_id=grn.fo_id,
            grn_id=grn.grn_id,
        )

        with self.assertRaises(InventoryGovernanceError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn=grn,
                batches=(custom_batch,),
                idempotency_key="IDEM-POST-OVER",
                actor_id="RECEIVER_MEENA",
            )
        self.assertIn("exceeding line accepted_quantity", str(ctx.exception))

    def test_duplicate_batch_posting_rejected(self):
        """Verifies duplicate batch posting for the same FulfilmentBatch raises DuplicateBatchPostingError."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        # First posting succeeds
        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-POST-01",
            actor_id="RECEIVER_MEENA",
        )

        # Second posting with different idempotency key but same batch
        with self.assertRaises(DuplicateBatchPostingError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn=grn,
                idempotency_key="IDEM-POST-02",
                actor_id="RECEIVER_MEENA",
            )
        self.assertIn("has already been posted", str(ctx.exception))

    def test_duplicate_batch_id_in_same_request_rejected(self):
        """Verifies duplicate batch_id within the same posting request is rejected."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        batch_a = FulfilmentBatch(
            batch_id="BATCH-DUP-01",
            sku_id="SKU_MILK_1L",
            units=12.0,
            remaining_shelf_life_days=5,
            receipt_date="2026-11-15",
            fo_id=grn.fo_id,
            grn_id=grn.grn_id,
        )
        batch_b = FulfilmentBatch(
            batch_id="BATCH-DUP-01",  # Same batch_id
            sku_id="SKU_MILK_1L",
            units=12.0,
            remaining_shelf_life_days=5,
            receipt_date="2026-11-15",
            fo_id=grn.fo_id,
            grn_id=grn.grn_id,
        )

        with self.assertRaises(InventoryGovernanceError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn=grn,
                batches=(batch_a, batch_b),
                idempotency_key="IDEM-POST-DUP-BATCHES",
                actor_id="RECEIVER_MEENA",
            )
        self.assertIn("Duplicate batch_id", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Scoped Idempotency & Conflict Detection
    # -------------------------------------------------------------------------

    def test_exact_idempotent_replay(self):
        """Verifies exact retry with identical key and payload returns identical records with no duplication."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        # Initial call
        entries_1 = self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-REPLAY-01",
            actor_id="RECEIVER_MEENA",
        )
        # Exact replay
        entries_2 = self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-REPLAY-01",
            actor_id="RECEIVER_MEENA",
        )

        self.assertEqual(len(entries_1), len(entries_2))
        self.assertEqual(entries_1[0].posting_id, entries_2[0].posting_id)
        self.assertEqual(entries_1[0].quantity, entries_2[0].quantity)

        # Verify on-hand stock is still exactly 24.0 (no double addition)
        on_hand = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(on_hand, 24.0)

    def test_conflicting_idempotency_key_reuse_rejected(self):
        """Verifies reuse of idempotency key with conflicting payload raises IdempotencyConflictError."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        # Initial call with actor RECEIVER_MEENA
        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-CONFLICT-01",
            actor_id="RECEIVER_MEENA",
        )

        # Conflicting reuse with different actor
        with self.assertRaises(IdempotencyConflictError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn=grn,
                idempotency_key="IDEM-CONFLICT-01",
                actor_id="RECEIVER_ALOK",  # Conflict!
            )
        self.assertIn("previously used with a different posting payload", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 5. Dynamic Balances & Zero Opening Stock
    # -------------------------------------------------------------------------

    def test_empty_ledger_returns_zero_stock(self):
        """Verifies empty ledger returns 0.0 stock with no invented opening inventory."""
        for sku in CANONICAL_SKUS:
            balance = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", sku)
            self.assertEqual(balance, 0.0)

        store_bals = self.inventory_repo.get_store_balances("DS_BLR_KORAMANGALA")
        self.assertEqual(store_bals, [])

    def test_ledger_history_query(self):
        """Verifies querying ledger history by store, SKU, batch, or transaction type."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-HIST-01",
            actor_id="RECEIVER_MEENA",
        )

        # Query all
        all_hist = self.inventory_repo.get_ledger_history(store_id="DS_BLR_KORAMANGALA")
        self.assertEqual(len(all_hist), 1)

        # Filter by SKU
        sku_hist = self.inventory_repo.get_ledger_history(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
        )
        self.assertEqual(len(sku_hist), 1)

        # Filter by different SKU returns empty
        empty_hist = self.inventory_repo.get_ledger_history(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_CURD_400G",
        )
        self.assertEqual(len(empty_hist), 0)

    # -------------------------------------------------------------------------
    # 6. Expiration & FEFO Perishability Tracking
    # -------------------------------------------------------------------------

    def test_fefo_batch_ordering_and_expiring_batches(self):
        """Verifies First-Expiring, First-Out (FEFO) ordering and get_expiring_batches()."""
        po = _make_approved_po()

        # Batch 1: Expiring in 2 days (receipt 2026-11-15, shelf-life 2 days -> expiry 2026-11-17)
        batch_1 = FulfilmentBatch(
            batch_id="BATCH-FEFO-SHORT",
            sku_id="SKU_MILK_1L",
            units=10.0,
            remaining_shelf_life_days=2,
            receipt_date="2026-11-15",
            fo_id="FO-001",
            grn_id="GRN-FEFO-01",
        )
        line_1 = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=10.0,
            accepted_quantity=10.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=2,
            total_shelf_life_days=3,  # 2/3 = 66.7% > 60%
            batch_references=("BATCH-FEFO-SHORT",),
        )
        grn_1 = GoodsReceiptNote(
            grn_id="GRN-FEFO-01",
            fo_id="FO-001",
            po_id=po.po_id,
            shipment_id="SHP-FEFO-01",
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            receipt_at="2026-11-15T10:00:00Z",
            receiver_id="RECEIVER_MEENA",
            inspection_lines=(line_1,),
        )
        self._persist_po_and_grn(po, grn_1)
        self.inventory_repo.post_inbound_receipt(
            grn=grn_1,
            batches=(batch_1,),
            idempotency_key="IDEM-FEFO-01",
            actor_id="RECEIVER_MEENA",
        )

        # Batch 2: Expiring in 6 days (receipt 2026-11-15, shelf-life 6 days -> expiry 2026-11-21)
        batch_2 = FulfilmentBatch(
            batch_id="BATCH-FEFO-LONG",
            sku_id="SKU_MILK_1L",
            units=14.0,
            remaining_shelf_life_days=6,
            receipt_date="2026-11-15",
            fo_id="FO-001",
            grn_id="GRN-FEFO-02",
        )
        line_2 = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=14.0,
            accepted_quantity=14.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=6,
            total_shelf_life_days=7,
            batch_references=("BATCH-FEFO-LONG",),
        )
        grn_2 = GoodsReceiptNote(
            grn_id="GRN-FEFO-02",
            fo_id="FO-001",
            po_id=po.po_id,
            shipment_id="SHP-FEFO-02",
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            receipt_at="2026-11-15T12:00:00Z",
            receiver_id="RECEIVER_MEENA",
            inspection_lines=(line_2,),
        )
        self._persist_po_and_grn(po, grn_2)
        self.inventory_repo.post_inbound_receipt(
            grn=grn_2,
            batches=(batch_2,),
            idempotency_key="IDEM-FEFO-02",
            actor_id="RECEIVER_MEENA",
        )

        # 1. Verify get_batch_balances orders by FEFO (expiry_date ASC)
        all_batches = self.inventory_repo.get_batch_balances("DS_BLR_KORAMANGALA", as_of_date="2026-11-15")
        self.assertEqual(len(all_batches), 2)
        self.assertEqual(all_batches[0].batch_id, "BATCH-FEFO-SHORT")
        self.assertEqual(all_batches[0].expiry_date, "2026-11-17")
        self.assertEqual(all_batches[1].batch_id, "BATCH-FEFO-LONG")
        self.assertEqual(all_batches[1].expiry_date, "2026-11-21")

        # 2. Verify get_expiring_batches with threshold of 2 days on 2026-11-15
        expiring = self.inventory_repo.get_expiring_batches(
            store_id="DS_BLR_KORAMANGALA",
            as_of_date="2026-11-15",
            days_threshold=2,
        )
        self.assertEqual(len(expiring), 1)
        self.assertEqual(expiring[0].batch_id, "BATCH-FEFO-SHORT")

        # 3. On 2026-11-18, BATCH-FEFO-SHORT should be expired (is_expired=True)
        expired_batch = self.inventory_repo.get_batch_balance(
            "DS_BLR_KORAMANGALA",
            "SKU_MILK_1L",
            "BATCH-FEFO-SHORT",
            as_of_date="2026-11-18",
        )
        self.assertIsNotNone(expired_batch)
        self.assertTrue(expired_batch.is_expired)
        self.assertEqual(expired_batch.current_remaining_shelf_life_days, 0)

    # -------------------------------------------------------------------------
    # 7. Audit Reconciliation Query
    # -------------------------------------------------------------------------

    def test_reconcile_grn_postings(self):
        """Verifies GRN-to-ledger audit reconciliation for fully and partially posted receipts."""
        po = _make_approved_po()
        grn = _make_grn(
            batch_references=("BATCH-REC-A", "BATCH-REC-B"),
            delivered_qty=24.0,
            accepted_qty=24.0,
        )
        self._persist_po_and_grn(po, grn)

        # Before posting: 0 posted units
        recon_pre = self.inventory_repo.reconcile_grn_postings(grn)
        self.assertEqual(recon_pre.total_accepted_quantity, 24.0)
        self.assertEqual(recon_pre.total_posted_quantity, 0.0)
        self.assertFalse(recon_pre.is_fully_posted)
        self.assertEqual(recon_pre.lines[0].unposted_accepted_quantity, 24.0)

        # Post only the first batch (12 units)
        batch_a = FulfilmentBatch(
            batch_id="BATCH-REC-A",
            sku_id="SKU_MILK_1L",
            units=12.0,
            remaining_shelf_life_days=5,
            receipt_date="2026-11-15",
            fo_id=grn.fo_id,
            grn_id=grn.grn_id,
        )
        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            batches=(batch_a,),
            idempotency_key="IDEM-REC-01",
            actor_id="RECEIVER_MEENA",
        )

        # After partial posting
        recon_mid = self.inventory_repo.reconcile_grn_postings(grn)
        self.assertEqual(recon_mid.total_posted_quantity, 12.0)
        self.assertFalse(recon_mid.is_fully_posted)
        self.assertEqual(recon_mid.lines[0].unposted_accepted_quantity, 12.0)
        self.assertEqual(recon_mid.lines[0].posted_batch_ids, ("BATCH-REC-A",))

        # Post the second batch (12 units)
        batch_b = FulfilmentBatch(
            batch_id="BATCH-REC-B",
            sku_id="SKU_MILK_1L",
            units=12.0,
            remaining_shelf_life_days=5,
            receipt_date="2026-11-15",
            fo_id=grn.fo_id,
            grn_id=grn.grn_id,
        )
        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            batches=(batch_b,),
            idempotency_key="IDEM-REC-02",
            actor_id="RECEIVER_MEENA",
        )

        # After full posting
        recon_post = self.inventory_repo.reconcile_grn_postings(grn)
        self.assertEqual(recon_post.total_posted_quantity, 24.0)
        self.assertTrue(recon_post.is_fully_posted)
        self.assertEqual(recon_post.lines[0].unposted_accepted_quantity, 0.0)
        self.assertEqual(set(recon_post.lines[0].posted_batch_ids), {"BATCH-REC-A", "BATCH-REC-B"})

    # -------------------------------------------------------------------------
    # 8. Interoperability with InventoryEngine & InventoryPosition
    # -------------------------------------------------------------------------

    def test_interoperability_with_inventory_engine(self):
        """Verifies active batches can be extracted and supplied directly to InventoryEngine."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-ENGINE-01",
            actor_id="RECEIVER_MEENA",
        )

        # Extract active batches as core InventoryBatch models
        batches = self.inventory_repo.get_active_inventory_batches(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            as_of_date="2026-11-15",
        )
        self.assertEqual(len(batches), 1)
        self.assertIsInstance(batches[0], InventoryBatch)
        self.assertEqual(batches[0].units, 24.0)
        self.assertEqual(batches[0].remaining_shelf_life_days, 5)

        # Verify validate_batch_inventory validates extracted batches
        is_valid, validation_errors = validate_batch_inventory(
            batches=list(batches),
            available_physical_stock=24.0,
            require_full_coverage=True,
        )
        self.assertTrue(is_valid)
        self.assertEqual(validation_errors, [])

        # Verify get_inventory_position
        inv_pos = self.inventory_repo.get_inventory_position(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            reserved_stock=4.0,
            on_order_stock=10.0,
        )
        self.assertEqual(inv_pos.on_hand_stock, 24.0)
        self.assertEqual(inv_pos.reserved_stock, 4.0)
        self.assertEqual(inv_pos.net_available_stock, 20.0)
        self.assertEqual(inv_pos.on_order_stock, 10.0)

    # -------------------------------------------------------------------------
    # 9. Persistence Recovery Across Restarts
    # -------------------------------------------------------------------------

    def test_persistence_recovery_across_restarts(self):
        """Verifies state is completely recovered from SQLite upon repository restart."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-RESTART-01",
            actor_id="RECEIVER_MEENA",
        )

        # Close current repository connections
        self.inventory_repo.close()
        self.fulfilment_repo.close()

        # Reopen with fresh repository instances pointing to same SQLite file
        reopened_fulfilment = FulfilmentRepository(db_path=self.db_path)
        reopened_inventory = OperationalInventoryRepository(
            db_path=self.db_path,
            fulfilment_repo=reopened_fulfilment,
        )

        try:
            # Check on-hand balance survives restart
            on_hand = reopened_inventory.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
            self.assertEqual(on_hand, 24.0)

            # Check batch balance survives restart
            batch_bal = reopened_inventory.get_batch_balance(
                "DS_BLR_KORAMANGALA",
                "SKU_MILK_1L",
                "BATCH-DS_BLR_KORAMANGALA-SKU_MILK_1L-20261115-01",
                as_of_date="2026-11-15",
            )
            self.assertIsNotNone(batch_bal)
            self.assertEqual(batch_bal.on_hand_quantity, 24.0)

            # Check idempotency deduplication survives restart
            replayed = reopened_inventory.post_inbound_receipt(
                grn=grn,
                idempotency_key="IDEM-RESTART-01",
                actor_id="RECEIVER_MEENA",
            )
            self.assertEqual(len(replayed), 1)
            self.assertEqual(replayed[0].quantity, 24.0)
            self.assertEqual(
                reopened_inventory.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"),
                24.0,
            )
        finally:
            reopened_inventory.close()
            reopened_fulfilment.close()

    # -------------------------------------------------------------------------
    # 10. PurchaseOrder Immutability
    # -------------------------------------------------------------------------

    def test_purchase_order_immutability(self):
        """Verifies that approved PurchaseOrder remains strictly immutable throughout posting."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        # Verify PO is frozen
        with self.assertRaises(ValidationError):
            po.total_order_qty = 999.0  # type: ignore

        self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-PO-IMMUTABLE",
            actor_id="RECEIVER_MEENA",
        )

        # PO attributes remain identical
        self.assertEqual(po.total_order_qty, 24.0)
        self.assertEqual(po.status, PurchaseOrderStatus.APPROVED)

    # -------------------------------------------------------------------------
    # 11. String GRN ID Resolution & Multi-SKU Receipts
    # -------------------------------------------------------------------------

    def test_post_inbound_receipt_by_string_grn_id(self):
        """Verifies resolving GRN by string grn_id directly from the database."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        # Post using string grn_id instead of GoodsReceiptNote instance
        entries = self.inventory_repo.post_inbound_receipt(
            grn=grn.grn_id,
            idempotency_key="IDEM-STR-GRN",
            actor_id="RECEIVER_MEENA",
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].grn_id, grn.grn_id)

    def test_post_inbound_receipt_nonexistent_grn_raises(self):
        """Verifies attempting to post a non-existent string grn_id raises InventoryGovernanceError."""
        with self.assertRaises(InventoryGovernanceError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn="GRN-NONEXISTENT",
                idempotency_key="IDEM-FAIL",
                actor_id="RECEIVER_MEENA",
            )
        self.assertIn("not found in persistence", str(ctx.exception))

    def test_post_inbound_receipt_multi_sku(self):
        """Verifies receipt posting with multiple distinct SKUs in the same GRN."""
        po = _make_approved_po()
        line_milk = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            batch_references=("BATCH-MULTI-MILK",),
        )
        line_curd = DockInspectionLine(
            sku_id="SKU_CURD_400G",
            delivered_quantity=12.0,
            accepted_quantity=12.0,
            rejected_quantity=0.0,
            unit_cost=25.0,
            remaining_shelf_life_days=4,
            total_shelf_life_days=6,
            batch_references=("BATCH-MULTI-CURD",),
        )
        grn = GoodsReceiptNote(
            grn_id="GRN-MULTI-SKU",
            fo_id="FO-001",
            po_id=po.po_id,
            shipment_id="SHP-MULTI-01",
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            receipt_at="2026-11-15T14:00:00Z",
            receiver_id="RECEIVER_MEENA",
            inspection_lines=(line_milk, line_curd),
        )
        self._persist_po_and_grn(po, grn)

        entries = self.inventory_repo.post_inbound_receipt(
            grn=grn,
            idempotency_key="IDEM-MULTI-SKU",
            actor_id="RECEIVER_MEENA",
        )
        self.assertEqual(len(entries), 2)

        # Check balances for both SKUs
        milk_bal = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        curd_bal = self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_CURD_400G")
        self.assertEqual(milk_bal, 24.0)
        self.assertEqual(curd_bal, 12.0)

        # Store summary shows 2 distinct active SKUs
        store_bals = self.inventory_repo.get_store_balances("DS_BLR_KORAMANGALA")
        self.assertEqual(len(store_bals), 2)
        skus_in_store = {b.sku_id for b in store_bals}
        self.assertEqual(skus_in_store, {"SKU_MILK_1L", "SKU_CURD_400G"})

    def test_post_inbound_receipt_mismatched_batch_references(self):
        """Verifies batches referencing wrong fo_id, grn_id, or non-matching SKU are rejected."""
        po = _make_approved_po()
        grn = _make_grn()
        self._persist_po_and_grn(po, grn)

        # Mismatched fo_id
        bad_fo_batch = FulfilmentBatch(
            batch_id="BATCH-BAD-FO",
            sku_id="SKU_MILK_1L",
            units=10.0,
            remaining_shelf_life_days=5,
            receipt_date="2026-11-15",
            fo_id="FO-WRONG",
            grn_id=grn.grn_id,
        )
        with self.assertRaises(InventoryGovernanceError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn=grn,
                batches=(bad_fo_batch,),
                idempotency_key="IDEM-BAD-FO",
                actor_id="RECEIVER_MEENA",
            )
        self.assertIn("specifies fo_id='FO-WRONG'", str(ctx.exception))

        # Mismatched SKU
        bad_sku_batch = FulfilmentBatch(
            batch_id="BATCH-BAD-SKU",
            sku_id="SKU_CURD_400G",  # Not in this GRN
            units=10.0,
            remaining_shelf_life_days=4,
            receipt_date="2026-11-15",
            fo_id=grn.fo_id,
            grn_id=grn.grn_id,
        )
        with self.assertRaises(InventoryGovernanceError) as ctx:
            self.inventory_repo.post_inbound_receipt(
                grn=grn,
                batches=(bad_sku_batch,),
                idempotency_key="IDEM-BAD-SKU",
                actor_id="RECEIVER_MEENA",
            )
        self.assertIn("not found in GRN", str(ctx.exception))

    def test_nonexistent_queries_return_zero_or_none(self):
        """Verifies querying non-existent entities returns safe defaults (0.0 or None)."""
        self.assertEqual(
            self.inventory_repo.get_on_hand_balance("", ""),
            0.0,
        )
        self.assertEqual(
            self.inventory_repo.get_on_hand_balance("DS_BLR_KORAMANGALA", "NON_EXISTENT_SKU"),
            0.0,
        )
        self.assertIsNone(
            self.inventory_repo.get_batch_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L", "NON_EXISTENT_BATCH"),
        )
        self.assertEqual(
            self.inventory_repo.get_store_balances(""),
            [],
        )
        self.assertEqual(
            self.inventory_repo.get_batch_balances(""),
            [],
        )


if __name__ == "__main__":
    unittest.main()
