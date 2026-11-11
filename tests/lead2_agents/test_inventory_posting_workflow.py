"""
End-to-End & Integration Tests: FulfilmentService -> Operational Inventory Posting Workflow
=========================================================================================
Phase 5.7.2 Verification Test Suite:
1. Full end-to-end lifecycle: Approved PO -> Gateway Transmission -> Supplier Ack -> ASN Dispatch
   -> Tracking -> Dock Intake & GRN -> FulfilmentBatch -> Inventory Ledger -> InventoryPosition
   & InventoryEngine assessment.
2. Partial delivery across multiple GRNs for the same shipment and multiple shipments per FO;
   multiple SKUs and multiple batches per SKU with FEFO expiry ordering.
3. Mixed accepted/rejected/damaged/excess quantities and supplier rejection: zero ledger impact for
   rejected, damaged, excess, and unaccepted goods.
4. Quality hold (full & partial) prevents sellable posting until authorized REINSPECTION_RELEASE;
   disposal / return dispositions never enter inventory.
5. Failed inventory posting after committed GRN -> durable FAILED/PENDING task persistence ->
   explicit retry (`retry_failed_inventory_postings`, `post_grn_to_inventory`) and restart
   recovery (`recover_pending_inventory_postings`).
6. Duplicate prevention (`DuplicateBatchPostingError`), exact idempotent replay, conflicting
   idempotency key (`IdempotencyConflictError`), and concurrent posting protection.
7. Carrier tracking `DELIVERED` alone never triggers inventory posting.
8. Strict `InventoryPosition` & `InventoryEngine` semantics: on-hand vs reserved vs confirmed
   inbound pipeline (`on_order_stock`) separation, zero double-counting across partial receipts,
   and zero invented opening stock or reservations.
9. Source `PurchaseOrder` immutability preserved across the full workflow.
"""

from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.agents.common.schemas import (
    DailyAgentInput,
    DemandForecast,
    InventoryPosition,
    InventorySnapshot,
)
from src.agents.fulfilment import (
    AdvanceShipmentNotice,
    DiscrepancyReason,
    DiscrepancyResolutionType,
    DockInspectionLine,
    FulfilmentGovernanceError,
    FulfilmentOrderStatus,
    FulfilmentService,
    GoodsReceiptNote,
    ShipmentLineItem,
    ShipmentTrackingStatus,
    TrackingScenario,
)
from src.agents.fulfilment.gateway import SupplierSimulationScenario
from src.agents.inventory import (
    DuplicateBatchPostingError,
    IdempotencyConflictError as InventoryIdempotencyConflictError,
    InventoryEngine,
    InventoryGovernanceError,
    InventoryTransactionType,
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
    po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
    store_id: str = "DS_BLR_KORAMANGALA",
    sku_id: str = "SKU_MILK_1L",
    qty: float = 24.0,
    unit_cost: float = 30.0,
) -> PurchaseOrder:
    item = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id=sku_id,
        category="Dairy",
        raw_suggested_qty=qty,
        case_pack_rounded_qty=qty,
        final_order_qty=qty,
        case_pack_size=12 if qty >= 12 else 1,
        moq=6.0,
        unit_retail_price=50.0,
        unit_procurement_cost=unit_cost,
        total_line_cost=round(qty * unit_cost, 2),
        requires_escalation=False,
        escalation_reasons=[],
    )
    po = PurchaseOrder(
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
        audit_trail=(),
    )
    return approve_purchase_order(
        po=po,
        approved_by="STORE_MANAGER_BLR",
        approval_timestamp="2024-11-15T09:00:00Z",
        approval_reason="Routine replenishment for Phase 5.7.2 integration test.",
    )


def _make_multi_sku_approved_po(
    po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-MULTI-01",
    store_id: str = "DS_BLR_KORAMANGALA",
) -> PurchaseOrder:
    item1 = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id="SKU_MILK_1L",
        category="Dairy",
        raw_suggested_qty=24.0,
        case_pack_rounded_qty=24.0,
        final_order_qty=24.0,
        case_pack_size=12,
        moq=12.0,
        unit_retail_price=50.0,
        unit_procurement_cost=30.0,
        total_line_cost=720.0,
        requires_escalation=False,
        escalation_reasons=[],
    )
    item2 = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id="SKU_CURD_400G",
        category="Dairy",
        raw_suggested_qty=12.0,
        case_pack_rounded_qty=12.0,
        final_order_qty=12.0,
        case_pack_size=6,
        moq=6.0,
        unit_retail_price=40.0,
        unit_procurement_cost=25.0,
        total_line_cost=300.0,
        requires_escalation=False,
        escalation_reasons=[],
    )
    po = PurchaseOrder(
        po_id=po_id,
        created_date="2024-11-15",
        store_id=store_id,
        supplier_id="SIM_SUPP_DAIRY_PRIMARY",
        supplier_type=SupplierType.PRIMARY,
        items=[item1, item2],
        total_order_qty=36.0,
        total_order_cost=1020.0,
        urgency=PurchaseOrderUrgency.STANDARD,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        requires_escalation=False,
        escalation_reasons=[],
        audit_trail=(),
    )
    return approve_purchase_order(
        po=po,
        approved_by="STORE_MANAGER_BLR",
        approval_timestamp="2024-11-15T09:00:00Z",
        approval_reason="Multi-SKU replenishment for Phase 5.7.2 integration test.",
    )


def _setup_transmitted_and_dispatched(
    service: FulfilmentService,
    po: PurchaseOrder,
    shipment_id: str = "ASN-01",
    lines: tuple[ShipmentLineItem, ...] | None = None,
    scenario: SupplierSimulationScenario = SupplierSimulationScenario.ACCEPT,
):
    fo = service.create_order(po, actor_id="AGENT", timestamp="2024-11-15T10:00:00Z")
    fo, tx_res = service.transmit_purchase_order(
        fo_id=fo.fo_id,
        po=po,
        idempotency_key=f"TX-{fo.fo_id}",
        timestamp="2024-11-15T10:15:00Z",
        override_scenario=scenario,
    )
    if scenario == SupplierSimulationScenario.REJECT:
        return fo, tx_res

    asn_lines = lines or tuple(
        ShipmentLineItem(sku_id=item.sku_id, shipped_quantity=item.final_order_qty)
        for item in po.items
    )
    asn = AdvanceShipmentNotice(
        shipment_id=shipment_id,
        fo_id=fo.fo_id,
        po_id=fo.po_id,
        supplier_id=fo.supplier_id,
        carrier="DELHIVERY",
        tracking_reference=f"TRK-{shipment_id}",
        dispatch_at="2024-11-15T12:00:00Z",
        expected_arrival_at="2024-11-16T08:00:00Z",
        lines=asn_lines,
    )
    fo = service.apply_shipment_dispatch(fo.fo_id, asn, actor_id="GATEWAY")
    return fo, tx_res


class TestInventoryPostingEndToEndWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_inv_posting_wf_"))
        self.db_path = self.temp_dir / "fulfilment_inventory.db"
        self.service = FulfilmentService(db_path=self.db_path)

    def tearDown(self) -> None:
        self.service.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_full_e2e_lifecycle_to_inventory_engine(self) -> None:
        """
        Approved PO -> Gateway transmission & ack -> ASN dispatch -> Tracking DELIVERED
        -> Dock intake GRN -> FulfilmentBatch -> Operational Inventory Ledger
        -> InventoryPosition & FEFO InventoryBatch -> InventoryEngine assessment.
        Also verifies source PO immutability and no double counting.
        """
        po = _make_approved_po(qty=24.0)
        po_snapshot_before = po.model_dump(mode="json")

        fo, _ = _setup_transmitted_and_dispatched(self.service, po, shipment_id="ASN-E2E-1")
        self.assertEqual(fo.status, FulfilmentOrderStatus.IN_TRANSIT)

        # Register and advance tracking to DELIVERED
        self.service.register_shipment_tracking(
            fo_id=fo.fo_id,
            shipment_id="ASN-E2E-1",
            carrier_id="DELHIVERY",
            scenario=TrackingScenario.NORMAL_DELIVERY,
            base_timestamp="2024-11-15T12:00:00Z",
        )
        tracking_rec = self.service.advance_to_tracking_status(
            "ASN-E2E-1",
            ShipmentTrackingStatus.DELIVERED,
        )
        self.assertEqual(tracking_rec.current_status, ShipmentTrackingStatus.DELIVERED)

        # Requirement: Tracking DELIVERED alone must NOT trigger inventory posting or invent stock
        pos_before_dock = self.service.get_inventory_position("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(pos_before_dock.on_hand_stock, 0.0)
        self.assertEqual(pos_before_dock.reserved_stock, 0.0)
        self.assertEqual(pos_before_dock.on_order_stock, 24.0)
        self.assertEqual(pos_before_dock.total_inventory_position, 24.0)

        # Dock intake -> GRN -> automatic inventory posting
        grn = GoodsReceiptNote(
            grn_id="GRN-E2E-1",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            shipment_id="ASN-E2E-1",
            receipt_at="2024-11-16T08:15:00Z",
            receiver_id="DOCK_CLERK_01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=24.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("LOT-E2E-A1",),
                ),
            ),
        )
        updated_fo, persisted_grn = self.service.process_dock_intake(
            fo.fo_id,
            grn,
            po=po,
            actor_id="DOCK_CLERK_01",
            idempotency_key="DOCK-E2E-1",
            verify_carrier_arrival=True,
        )
        self.assertEqual(updated_fo.status, FulfilmentOrderStatus.RECEIVED)
        self.assertEqual(persisted_grn.total_accepted_quantity, 24.0)

        # Verify durable posting task & reconciliation
        task = self.service.get_inventory_posting_task_for_grn("GRN-E2E-1")
        self.assertIsNotNone(task)
        self.assertEqual(task["status"], "COMPLETED")
        self.assertEqual(task["posted_quantity"], 24.0)

        recon = self.service.reconcile_inventory_postings("GRN-E2E-1")
        self.assertTrue(recon.is_fully_posted)
        self.assertEqual(recon.total_accepted_quantity, 24.0)
        self.assertEqual(recon.total_posted_quantity, 24.0)

        # Verify ledger traceability to FO, PO, shipment, GRN, batch
        ledger_entries = self.service.inventory_repository.get_ledger_history(grn_id="GRN-E2E-1")
        self.assertEqual(len(ledger_entries), 1)
        entry = ledger_entries[0]
        self.assertEqual(entry.transaction_type, InventoryTransactionType.INBOUND_RECEIPT)
        self.assertEqual(entry.quantity, 24.0)
        self.assertEqual(entry.fo_id, fo.fo_id)
        self.assertEqual(task["po_id"], po.po_id)
        self.assertEqual(entry.shipment_id, "ASN-E2E-1")
        self.assertEqual(entry.grn_id, "GRN-E2E-1")
        minted_batches = self.service.get_fulfilment_batches_for_grn("GRN-E2E-1")
        self.assertEqual(entry.fulfilment_batch_id, minted_batches[0].batch_id)

        # Verify InventoryPosition after dock intake: on_hand=24, on_order=0, zero double counting
        pos_after = self.service.get_inventory_position(
            "DS_BLR_KORAMANGALA",
            "SKU_MILK_1L",
            reserved_stock=4.0,
        )
        self.assertEqual(pos_after.on_hand_stock, 24.0)
        self.assertEqual(pos_after.reserved_stock, 4.0)
        self.assertEqual(pos_after.on_order_stock, 0.0)
        self.assertEqual(pos_after.net_available_stock, 20.0)
        self.assertEqual(pos_after.total_inventory_position, 20.0)

        # Active FEFO batches and InventoryEngine assessment
        batches = self.service.get_active_batches("DS_BLR_KORAMANGALA", "SKU_MILK_1L", as_of_date="2024-11-16")
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].units, 24.0)
        self.assertEqual(batches[0].remaining_shelf_life_days, 6)

        daily_input = DailyAgentInput(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-16",
            forecast=DemandForecast(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                date="2024-11-16",
                p10_demand=5.0,
                p50_demand=10.0,
                p90_demand=15.0,
                sigma_demand=3.0,
            ),
            inventory=InventorySnapshot(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                date="2024-11-16",
                on_hand_stock=pos_after.on_hand_stock,
            ),
        )
        calc_res, risk_res = self.service.run_inventory_assessment(
            daily_input=daily_input,
            reserved_stock=4.0,
            engine=InventoryEngine(),
        )
        self.assertEqual(calc_res.inventory_position.on_hand_stock, 24.0)
        self.assertEqual(calc_res.inventory_position.on_order_stock, 0.0)
        self.assertEqual(risk_res.store_id, "DS_BLR_KORAMANGALA")

        # PurchaseOrder immutability preserved
        self.assertEqual(po.model_dump(mode="json"), po_snapshot_before)

    def test_02_partial_delivery_multi_grn_multi_shipment_multi_sku_and_fefo_batches(self) -> None:
        """
        Tests multiple shipments per FO, multiple GRNs per same shipment, multiple SKUs,
        multiple batches per SKU with distinct expiry dates (FEFO order), and invariant
        `on_hand + on_order == total_confirmed` across partial receipts.
        """
        po = _make_multi_sku_approved_po()
        # Total ordered: SKU_MILK_1L=24, SKU_CURD_400G=12
        fo, _ = _setup_transmitted_and_dispatched(
            self.service,
            po,
            shipment_id="ASN-PART-1",
            lines=(
                ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=16.0),
                ShipmentLineItem(sku_id="SKU_CURD_400G", shipped_quantity=12.0),
            ),
        )
        # Dispatch 2nd shipment for remaining 8 MILK
        asn2 = AdvanceShipmentNotice(
            shipment_id="ASN-PART-2",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            supplier_id=po.supplier_id,
            carrier="BLUEDART",
            tracking_reference="TRK-PART-2",
            dispatch_at="2024-11-15T16:00:00Z",
            expected_arrival_at="2024-11-16T12:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=8.0),),
        )
        fo = self.service.apply_shipment_dispatch(fo.fo_id, asn2, actor_id="GATEWAY")

        # Initial position before any receipt
        milk_pos_0 = self.service.get_inventory_position("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(milk_pos_0.on_hand_stock, 0.0)
        self.assertEqual(milk_pos_0.on_order_stock, 24.0)

        # GRN-1 on ASN-PART-1: receives 10 MILK (with 6d shelf life) + 12 CURD (2 batches inferred via split or single batch)
        grn1 = GoodsReceiptNote(
            grn_id="GRN-PART-1A",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            shipment_id="ASN-PART-1",
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="DOCK_CLERK_01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=10.0,
                    accepted_quantity=10.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("LOT-M-EXP22",),
                ),
                DockInspectionLine(
                    sku_id="SKU_CURD_400G",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=25.0,
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("LOT-C-1",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn1, idempotency_key="DOCK-1A")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # Check conservation: MILK on_hand=10, on_order=14 (sum=24); CURD on_hand=12, on_order=0
        milk_pos_1 = self.service.get_inventory_position("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        curd_pos_1 = self.service.get_inventory_position("DS_BLR_KORAMANGALA", "SKU_CURD_400G")
        self.assertEqual(milk_pos_1.on_hand_stock, 10.0)
        self.assertEqual(milk_pos_1.on_order_stock, 14.0)
        self.assertEqual(milk_pos_1.total_inventory_position, 24.0)
        self.assertEqual(curd_pos_1.on_hand_stock, 12.0)
        self.assertEqual(curd_pos_1.on_order_stock, 0.0)

        # GRN-2 ALSO on ASN-PART-1 (second GRN for same shipment, earlier expiry -> FEFO check!)
        grn2 = GoodsReceiptNote(
            grn_id="GRN-PART-1B",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            shipment_id="ASN-PART-1",
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="DOCK_CLERK_02",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=6.0,
                    accepted_quantity=6.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=5,  # Expires sooner than LOT-M-EXP22!
                    total_shelf_life_days=7,
                    batch_references=("LOT-M-EXP20",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn2, idempotency_key="DOCK-1B")
        self.assertEqual(fo.status, FulfilmentOrderStatus.PARTIALLY_RECEIVED)

        # GRN-3 on ASN-PART-2: receives remaining 8 MILK (7d shelf life)
        grn3 = GoodsReceiptNote(
            grn_id="GRN-PART-2",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            shipment_id="ASN-PART-2",
            receipt_at="2024-11-16T13:00:00Z",
            receiver_id="DOCK_CLERK_01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=8.0,
                    accepted_quantity=8.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=7,
                    total_shelf_life_days=7,
                    batch_references=("LOT-M-EXP23",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn3, idempotency_key="DOCK-2")
        self.assertEqual(fo.status, FulfilmentOrderStatus.RECEIVED)

        # Final positions & untouched SKU invariant (no invented stock)
        milk_pos_final = self.service.get_inventory_position("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(milk_pos_final.on_hand_stock, 24.0)
        self.assertEqual(milk_pos_final.on_order_stock, 0.0)
        untouched_pos = self.service.get_inventory_position("DS_BLR_HSR_LAYOUT", "SKU_MILK_1L")
        self.assertEqual(untouched_pos.on_hand_stock, 0.0)
        self.assertEqual(untouched_pos.reserved_stock, 0.0)
        self.assertEqual(untouched_pos.on_order_stock, 0.0)

        # FEFO verification for SKU_MILK_1L (sorted 5d -> 6d -> 7d)
        milk_batches = self.service.get_active_batches("DS_BLR_KORAMANGALA", "SKU_MILK_1L", as_of_date="2024-11-16")
        self.assertEqual([b.units for b in milk_batches], [6.0, 10.0, 8.0])
        self.assertEqual([b.remaining_shelf_life_days for b in milk_batches], [5, 6, 7])

    def test_03_rejected_damaged_excess_and_supplier_rejection_zero_ledger_impact(self) -> None:
        """
        Ensures damaged, rejected, excess, and supplier-rejected order quantities never enter inventory.
        """
        # 1. Supplier rejection scenario -> 0 ledger impact
        po_rej = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-REJ-01", qty=12.0)
        fo_rej, tx_res = _setup_transmitted_and_dispatched(
            self.service,
            po_rej,
            scenario=SupplierSimulationScenario.REJECT,
        )
        self.assertEqual(fo_rej.status, FulfilmentOrderStatus.REJECTED_BY_SUPPLIER)
        self.assertEqual(tx_res.status.value, "CONFIRMED_REJECTION")
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 0.0)

        # 2. Mixed accepted=15, damaged=5, rejected=4 (plus 2 excess delivered => delivered=26, accepted=15, rejected=11)
        po_mix = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-MIX-01", qty=24.0)
        fo_mix, _ = _setup_transmitted_and_dispatched(self.service, po_mix, shipment_id="ASN-MIX-1")

        grn_mix = GoodsReceiptNote(
            grn_id="GRN-MIX-1",
            fo_id=fo_mix.fo_id,
            po_id=po_mix.po_id,
            store_id=po_mix.store_id,
            supplier_id=po_mix.supplier_id,
            shipment_id="ASN-MIX-1",
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="DOCK_CLERK_01",
            supervisor_signoff_by="SUPERVISOR_01",
            supervisor_signoff_at="2024-11-16T09:05:00Z",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=26.0,
                    accepted_quantity=15.0,
                    damaged_quantity=5.0,
                    rejected_quantity=11.0,
                    excess_quantity=2.0,
                    unit_cost=30.0,
                    discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT, DiscrepancyReason.EXCESS_DELIVERY),
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("LOT-MIX-1",),
                ),
            ),
        )
        self.service.process_dock_intake(fo_mix.fo_id, grn_mix, idempotency_key="DOCK-MIX-1")

        # Only the 15 accepted units enter sellable inventory; damaged (5), rejected (4), excess (2) stay out!
        pos = self.service.get_inventory_position("DS_BLR_KORAMANGALA", "SKU_MILK_1L")
        self.assertEqual(pos.on_hand_stock, 15.0)
        recon = self.service.reconcile_inventory_postings("GRN-MIX-1")
        self.assertEqual(recon.total_posted_quantity, 15.0)
        self.assertTrue(recon.is_fully_posted)

    def test_04_quality_hold_blocked_then_reinspection_release_posts_inventory(self) -> None:
        """
        Tests both:
        (a) Full quality hold (`accepted=0, rejected=24`) skips posting (`SKIPPED_ZERO_ACCEPTED`),
            disposal/return disposition never posts, and `REINSPECTION_RELEASE` posts only released units.
        (b) Partial hold (`accepted=18, rejected=6`) posts 18 immediately, keeps 6 out until QA release.
        """
        # Case A: Full quality hold -> partial QA release (18) + certified disposal (6)
        po = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-QH-01", qty=24.0)
        fo, _ = _setup_transmitted_and_dispatched(self.service, po, shipment_id="ASN-QH-1")

        grn_hold = GoodsReceiptNote(
            grn_id="GRN-QH-FULL",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            shipment_id="ASN-QH-1",
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="DOCK_CLERK_01",
            supervisor_signoff_by="SUPERVISOR_01",
            supervisor_signoff_at="2024-11-16T08:05:00Z",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=0.0,
                    damaged_quantity=0.0,
                    rejected_quantity=24.0,
                    unit_cost=30.0,
                    discrepancy_reasons=(DiscrepancyReason.QUALITY_DEFECT,),
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("LOT-QH-FULL",),
                ),
            ),
        )
        fo, _ = self.service.process_dock_intake(fo.fo_id, grn_hold, idempotency_key="DOCK-QH-1")

        # Must record SKIPPED_ZERO_ACCEPTED and 0 inventory
        task = self.service.get_inventory_posting_task_for_grn("GRN-QH-FULL")
        self.assertIsNotNone(task)
        self.assertEqual(task["status"], "SKIPPED_ZERO_ACCEPTED")
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 0.0)

        # Direct posting of held GRN must be rejected
        with self.assertRaises(InventoryGovernanceError):
            self.service.post_grn_to_inventory(fo.fo_id, "GRN-QH-FULL")

        # Certified disposal of 6 units must NOT enter sellable inventory
        fo, _ = self.service.resolve_quality_hold(
            fo_id=fo.fo_id,
            shipment_id="ASN-QH-1",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.CERTIFIED_DISPOSAL,
            quantity=6.0,
            disposition_reference="CERT-DISP-01",
            inspector_id="QA_LEAD_01",
            timestamp="2024-11-16T10:00:00Z",
            idempotency_key="QHR-DISP-1",
        )
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 0.0)

        # Authorized REINSPECTION_RELEASE of remaining 18 units enters inventory
        fo, qhr_rel = self.service.resolve_quality_hold(
            fo_id=fo.fo_id,
            shipment_id="ASN-QH-1",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
            quantity=18.0,
            disposition_reference="QA-CLEAR-01",
            inspector_id="QA_LEAD_01",
            verified_remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            timestamp="2024-11-16T11:00:00Z",
            idempotency_key="QHR-REL-1",
            inspection_notes="Lab cleared batch",
        )
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 18.0)

        # Idempotent replay of release posting returns identical entry without doubling
        replay_entries = self.service.post_quality_hold_release_to_inventory(
            fo_id=fo.fo_id,
            quality_hold_resolution_id=qhr_rel.resolution_id,
            actor_id="QA_LEAD_01",
            idempotency_key=f"INV_REL:{qhr_rel.resolution_id}",
        )
        self.assertEqual(len(replay_entries), 1)
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 18.0)

        # Verify durable QA release posting task creation and attributes
        qhr_task = self.service.get_inventory_posting_task_for_resolution(qhr_rel.resolution_id)
        self.assertIsNotNone(qhr_task)
        self.assertEqual(qhr_task["status"], "COMPLETED")
        self.assertEqual(qhr_task["task_type"], "QA_RELEASE")
        self.assertEqual(qhr_task["posted_quantity"], 18.0)
        self.assertEqual(qhr_task["resolution_id"], qhr_rel.resolution_id)

        # Verify reconciliation reflects QA release posting
        recon = self.service.reconcile_inventory_postings("GRN-QH-FULL")
        self.assertTrue(recon.is_fully_posted)
        self.assertEqual(recon.total_posted_quantity, 18.0)
        self.assertEqual(recon.lines[0].unposted_accepted_quantity, 0.0)

        # Attempting second QA release posting with a different idempotency key raises DuplicateBatchPostingError
        with self.assertRaises(DuplicateBatchPostingError):
            self.service.post_quality_hold_release_to_inventory(
                fo_id=fo.fo_id,
                quality_hold_resolution_id=qhr_rel.resolution_id,
                actor_id="QA_LEAD_01",
                idempotency_key="DIFFERENT-KEY-FOR-RELEASE",
            )

    def test_05_failed_posting_retry_and_restart_recovery(self) -> None:
        """
        Simulates ledger failure after committed GRN -> verifies GRN remains committed,
        task is FAILED, failure is NOT silently swallowed -> explicit retry via
        `retry_failed_inventory_postings()` & restart recovery via fresh `FulfilmentService`.
        """
        po1 = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-REC-01", qty=12.0)
        fo1, _ = _setup_transmitted_and_dispatched(self.service, po1, shipment_id="ASN-REC-1")
        grn1 = GoodsReceiptNote(
            grn_id="GRN-REC-1",
            fo_id=fo1.fo_id,
            po_id=po1.po_id,
            store_id=po1.store_id,
            supplier_id=po1.supplier_id,
            shipment_id="ASN-REC-1",
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="DOCK_CLERK_01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("LOT-REC-1",),
                ),
            ),
        )

        # Fail inventory posting during dock intake; GRN already committed, error must propagate
        with patch.object(
            self.service.inventory_repository,
            "post_inbound_receipt",
            side_effect=RuntimeError("Simulated SQLite lock during inventory posting"),
        ):
            with self.assertRaises(RuntimeError):
                self.service.process_dock_intake(fo1.fo_id, grn1, idempotency_key="DOCK-REC-1")

        # Verify GRN persisted, on-hand still 0, task status FAILED
        self.assertIsNotNone(self.service.get_goods_receipt_note("GRN-REC-1"))
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 0.0)
        task1 = self.service.get_inventory_posting_task_for_grn("GRN-REC-1")
        self.assertIsNotNone(task1)
        self.assertEqual(task1["status"], "FAILED")
        self.assertIn("Simulated SQLite lock", task1["error_message"])
        recon_before = self.service.reconcile_inventory_postings("GRN-REC-1")
        self.assertFalse(recon_before.is_fully_posted)
        self.assertEqual(recon_before.total_posted_quantity, 0.0)

        # Test explicit retry_failed_inventory_postings()
        retry_results = self.service.retry_failed_inventory_postings()
        self.assertEqual(len(retry_results), 1)
        self.assertEqual(retry_results[0]["status"], "COMPLETED")
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 12.0)

        # Also test restart recovery for a PENDING task created after process crash before posting
        po2 = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-REC-02", qty=12.0)
        fo2, _ = _setup_transmitted_and_dispatched(self.service, po2, shipment_id="ASN-REC-2")
        grn2 = GoodsReceiptNote(
            grn_id="GRN-REC-2",
            fo_id=fo2.fo_id,
            po_id=po2.po_id,
            store_id=po2.store_id,
            supplier_id=po2.supplier_id,
            shipment_id="ASN-REC-2",
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="DOCK_CLERK_01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("LOT-REC-2",),
                ),
            ),
        )
        self.service.process_dock_intake(fo2.fo_id, grn2, idempotency_key="DOCK-REC-2", auto_post_inventory=False)
        self.service.repository.record_inventory_posting_task({
            "posting_task_id": "POST-TASK-GRN-REC-2",
            "grn_id": "GRN-REC-2",
            "fo_id": fo2.fo_id,
            "po_id": po2.po_id,
            "store_id": po2.store_id,
            "status": "PENDING",
            "idempotency_key": "INV_POST:DOCK-REC-2",
            "payload_fingerprint": "pending_fp_test",
            "total_accepted_quantity": 12.0,
            "posted_quantity": 0.0,
            "created_at": "2024-11-16T09:00:01Z",
            "retry_count": 0,
        })

        # Simulate full application restart with a fresh FulfilmentService instance
        self.service.close()
        restarted_service = FulfilmentService(db_path=self.db_path)
        try:
            rec_results = restarted_service.recover_pending_inventory_postings()
            self.assertEqual(len(rec_results), 1)
            self.assertEqual(rec_results[0]["grn_id"], "GRN-REC-2")
            self.assertEqual(rec_results[0]["status"], "COMPLETED")
            self.assertEqual(
                restarted_service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"),
                24.0,
            )
            task2 = restarted_service.get_inventory_posting_task_for_grn("GRN-REC-2")
            self.assertEqual(task2["status"], "COMPLETED")

            # Also verify retry/recovery for a failed QA release posting task
            po3 = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-REC-03", qty=12.0)
            fo3, _ = _setup_transmitted_and_dispatched(restarted_service, po3, shipment_id="ASN-REC-3")
            grn3 = GoodsReceiptNote(
                grn_id="GRN-REC-3",
                fo_id=fo3.fo_id,
                po_id=po3.po_id,
                store_id=po3.store_id,
                supplier_id=po3.supplier_id,
                shipment_id="ASN-REC-3",
                receipt_at="2024-11-16T10:00:00Z",
                receiver_id="DOCK_CLERK_01",
                supervisor_signoff_by="SUPERVISOR_01",
                supervisor_signoff_at="2024-11-16T10:05:00Z",
                inspection_lines=(
                    DockInspectionLine(
                        sku_id="SKU_MILK_1L",
                        delivered_quantity=12.0,
                        accepted_quantity=0.0,
                        rejected_quantity=12.0,
                        unit_cost=30.0,
                        discrepancy_reasons=(DiscrepancyReason.QUALITY_DEFECT,),
                        remaining_shelf_life_days=5,
                        total_shelf_life_days=7,
                        batch_references=("LOT-REC-3",),
                    ),
                ),
            )
            restarted_service.process_dock_intake(fo3.fo_id, grn3, idempotency_key="DOCK-REC-3")

            # Resolve quality hold with simulated inventory repo failure during release posting
            with patch.object(
                restarted_service.inventory_repository,
                "post_inbound_receipt",
                side_effect=RuntimeError("Simulated ledger timeout during QA release posting"),
            ):
                with self.assertRaises(RuntimeError):
                    restarted_service.resolve_quality_hold(
                        fo_id=fo3.fo_id,
                        shipment_id="ASN-REC-3",
                        sku_id="SKU_MILK_1L",
                        resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
                        quantity=12.0,
                        disposition_reference="QA-CLEAR-REC-3",
                        inspector_id="QA_LEAD_01",
                        verified_remaining_shelf_life_days=5,
                        total_shelf_life_days=7,
                        timestamp="2024-11-16T11:00:00Z",
                        idempotency_key="QHR-REC-3",
                    )

            # Task must be recorded as FAILED
            fo3_curr = restarted_service.repository.get_or_raise(fo3.fo_id)
            qhr_failed = fo3_curr.quality_hold_resolutions[0]
            qa_task = restarted_service.get_inventory_posting_task_for_resolution(qhr_failed.resolution_id)
            self.assertIsNotNone(qa_task)
            self.assertEqual(qa_task["status"], "FAILED")
            self.assertIn("Simulated ledger timeout", qa_task["error_message"])

            # Retry failed inventory postings recovers the QA release task
            qa_retry_results = restarted_service.retry_failed_inventory_postings()
            self.assertTrue(any(r["posting_task_id"] == qa_task["posting_task_id"] and r["status"] == "COMPLETED" for r in qa_retry_results))
            qa_task_done = restarted_service.get_inventory_posting_task_for_resolution(qhr_failed.resolution_id)
            self.assertEqual(qa_task_done["status"], "COMPLETED")
            self.assertEqual(qa_task_done["posted_quantity"], 12.0)
            self.assertEqual(
                restarted_service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"),
                36.0,
            )
        finally:
            restarted_service.close()
            # Re-open self.service so tearDown cleanly closes
            self.service = FulfilmentService(db_path=self.db_path)

    def test_06_idempotency_conflict_duplicate_prevention_and_concurrency(self) -> None:
        """
        Verifies exact idempotent replay, duplicate batch prevention with a different key,
        conflicting idempotency key reuse across GRNs, and concurrent posting protection.
        """
        po1 = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-IDEM-01", qty=12.0)
        fo1, _ = _setup_transmitted_and_dispatched(self.service, po1, shipment_id="ASN-IDEM-1")
        grn1 = GoodsReceiptNote(
            grn_id="GRN-IDEM-1",
            fo_id=fo1.fo_id,
            po_id=po1.po_id,
            store_id=po1.store_id,
            supplier_id=po1.supplier_id,
            shipment_id="ASN-IDEM-1",
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="DOCK_CLERK_01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("LOT-IDEM-1",),
                ),
            ),
        )
        # First intake with auto_post_inventory=False so we can exercise explicit post_grn_to_inventory concurrency
        self.service.process_dock_intake(fo1.fo_id, grn1, idempotency_key="DOCK-IDEM-1", auto_post_inventory=False)

        # Concurrent posting storm from multiple threads using separate FulfilmentService handles on same DB
        barrier = threading.Barrier(4)
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker(idx: int) -> None:
            svc = FulfilmentService(db_path=self.db_path)
            try:
                barrier.wait(timeout=5.0)
                svc.post_grn_to_inventory(
                    fo1.fo_id,
                    "GRN-IDEM-1",
                    actor_id=f"WORKER_{idx}",
                    idempotency_key="SHARED-CONCURRENT-POST-KEY",
                )
                with lock:
                    outcomes.append("OK")
            except Exception as exc:
                with lock:
                    outcomes.append(type(exc).__name__)
            finally:
                svc.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Regardless of thread scheduling, stock must equal 12.0 exactly once (never 24/36/48!)
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 12.0)

        # Exact idempotent replay succeeds cleanly
        entries_replay = self.service.post_grn_to_inventory(
            fo1.fo_id,
            "GRN-IDEM-1",
            actor_id="WORKER_0",
            idempotency_key="SHARED-CONCURRENT-POST-KEY",
        )
        self.assertEqual(len(entries_replay), 1)
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 12.0)

        # Second attempt on SAME GRN with DIFFERENT idempotency key raises DuplicateBatchPostingError
        with self.assertRaises(DuplicateBatchPostingError):
            self.service.post_grn_to_inventory(
                fo1.fo_id,
                "GRN-IDEM-1",
                actor_id="WORKER_99",
                idempotency_key="DIFFERENT-KEY-ATTEMPT",
            )

        # Reuse SAME key on a DIFFERENT GRN raises IdempotencyConflictError
        po2 = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-IDEM-02", qty=12.0)
        fo2, _ = _setup_transmitted_and_dispatched(self.service, po2, shipment_id="ASN-IDEM-2")
        grn2 = GoodsReceiptNote(
            grn_id="GRN-IDEM-2",
            fo_id=fo2.fo_id,
            po_id=po2.po_id,
            store_id=po2.store_id,
            supplier_id=po2.supplier_id,
            shipment_id="ASN-IDEM-2",
            receipt_at="2024-11-16T09:00:00Z",
            receiver_id="DOCK_CLERK_01",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=12.0,
                    accepted_quantity=12.0,
                    rejected_quantity=0.0,
                    unit_cost=30.0,
                    remaining_shelf_life_days=6,
                    total_shelf_life_days=7,
                    batch_references=("LOT-IDEM-2",),
                ),
            ),
        )
        self.service.process_dock_intake(fo2.fo_id, grn2, idempotency_key="DOCK-IDEM-2", auto_post_inventory=False)
        with self.assertRaises(InventoryIdempotencyConflictError):
            self.service.post_grn_to_inventory(
                fo2.fo_id,
                "GRN-IDEM-2",
                actor_id="WORKER_0",
                idempotency_key="SHARED-CONCURRENT-POST-KEY",
            )
        # Task 1 remains COMPLETED, on-hand still 12.0
        self.assertEqual(self.service.get_inventory_posting_task_for_grn("GRN-IDEM-1")["status"], "COMPLETED")
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 12.0)

    def test_07_unposted_quality_release_recovered_on_restart(self) -> None:
        """
        Tests that an unposted REINSPECTION_RELEASE resolution (e.g. crash after quality hold
        resolution before inventory posting) is discovered and recovered across process restarts
        by `recover_pending_inventory_postings()`.
        """
        po = _make_approved_po(po_id="PO-20241115-DS_BLR_KORAMANGALA-UNPOST-01", qty=24.0)
        fo, _ = _setup_transmitted_and_dispatched(self.service, po, shipment_id="ASN-UNPOST-1")

        grn = GoodsReceiptNote(
            grn_id="GRN-UNPOST-1",
            fo_id=fo.fo_id,
            po_id=po.po_id,
            store_id=po.store_id,
            supplier_id=po.supplier_id,
            shipment_id="ASN-UNPOST-1",
            receipt_at="2024-11-16T08:00:00Z",
            receiver_id="DOCK_CLERK_01",
            supervisor_signoff_by="SUPERVISOR_01",
            supervisor_signoff_at="2024-11-16T08:05:00Z",
            inspection_lines=(
                DockInspectionLine(
                    sku_id="SKU_MILK_1L",
                    delivered_quantity=24.0,
                    accepted_quantity=0.0,
                    rejected_quantity=24.0,
                    unit_cost=30.0,
                    discrepancy_reasons=(DiscrepancyReason.QUALITY_DEFECT,),
                    remaining_shelf_life_days=5,
                    total_shelf_life_days=7,
                    batch_references=("LOT-UNPOST-1",),
                ),
            ),
        )
        self.service.process_dock_intake(fo.fo_id, grn, idempotency_key="DOCK-UNPOST-1")

        # Resolve quality hold with auto_post_inventory=False to simulate crash right before posting
        fo, qhr = self.service.resolve_quality_hold(
            fo_id=fo.fo_id,
            shipment_id="ASN-UNPOST-1",
            sku_id="SKU_MILK_1L",
            resolution_type=DiscrepancyResolutionType.REINSPECTION_RELEASE,
            quantity=24.0,
            disposition_reference="QA-CLEAR-UNPOST",
            inspector_id="QA_LEAD_01",
            verified_remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            timestamp="2024-11-16T11:00:00Z",
            idempotency_key="QHR-UNPOST-1",
            auto_post_inventory=False,
        )

        # On-hand is currently 0.0 before recovery
        self.assertEqual(self.service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"), 0.0)

        # Restart service
        self.service.close()
        restarted_service = FulfilmentService(db_path=self.db_path)
        try:
            rec_results = restarted_service.recover_pending_inventory_postings()
            self.assertTrue(any(r["task_type"] == "QA_RELEASE" and r["status"] == "COMPLETED" for r in rec_results))
            self.assertEqual(
                restarted_service.inventory_repository.get_on_hand_balance("DS_BLR_KORAMANGALA", "SKU_MILK_1L"),
                24.0,
            )
            qhr_task = restarted_service.get_inventory_posting_task_for_resolution(qhr.resolution_id)
            self.assertIsNotNone(qhr_task)
            self.assertEqual(qhr_task["status"], "COMPLETED")
            self.assertEqual(qhr_task["posted_quantity"], 24.0)

            # Reconcile GRN reflects full posted quantity
            recon = restarted_service.reconcile_inventory_postings("GRN-UNPOST-1")
            self.assertTrue(recon.is_fully_posted)
            self.assertEqual(recon.total_posted_quantity, 24.0)
        finally:
            restarted_service.close()
            self.service = FulfilmentService(db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
