"""
Unit & Integration Tests: Lead 2 Core Procurement Engine
=========================================================
Tests:
1. No reorder recommendation (suggested_replenishment_qty <= 0 produces no PO line).
2. Case-pack rounding across canonical categories.
3. MOQ enforcement and integer multiple adherence.
4. Critical/out-of-stock MOQ handling (Rule 5: allowed without excess flag).
5. Non-critical MOQ exceeding target stock (Rule 6: flagged for manual review).
6. Primary vs Backup supplier selection under critical & coverage conditions.
7. Procurement cost calculations and line-item escalation (threshold > ₹5,000).
8. Purchase Order grouping strictly by (store_id, supplier_id).
9. Order-level escalation for store orders exceeding ₹25,000.
10. Audit reasons and propagation of assumed_zero_pipeline flag.
11. Strict prevention of automatic approval or issuance (MANDATORY HUMAN APPROVAL).
"""

import unittest
from pydantic import ValidationError

from src.agents.common.config import (
    Lead2Config,
    get_default_config,
    CATEGORY_PROCUREMENT_PARAMS,
    CATEGORY_SUPPLIER_PARAMS,
)
from src.agents.common.schemas import (
    InventoryCalculationResult,
    InventoryPosition,
    InventoryRiskAssessment,
    StockRiskLevel,
    ExpiryRiskLevel,
)
from src.agents.procurement.schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierProfile,
    SupplierType,
)
from src.agents.procurement.engine import (
    ProcurementEngine,
    ProcurementRecommendation,
    get_simulated_supplier_profile,
    select_supplier,
    calculate_procurement_quantity,
)


def _make_calc_result(
    store_id: str = "DS_BLR_KORAMANGALA",
    sku_id: str = "SKU_MILK_1L",
    date: str = "2024-11-15",
    average_daily_demand: float = 30.0,
    demand_uncertainty_sigma: float = 5.0,
    on_hand_stock: float = 20.0,
    reserved_stock: float = 0.0,
    on_order_stock: float = 0.0,
    lead_time_mean_days: float = 1.5,
    lead_time_std_days: float = 0.3,
    review_period_days: float = 1.0,
    safety_stock_lead_time: float = 15.0,
    safety_stock_protection: float = 20.0,
    reorder_point: float = 60.0,
    target_stock: float = 95.0,
    suggested_replenishment_qty: float = 75.0,
    reorder_triggered: bool = True,
    assumed_zero_pipeline: bool = True,
) -> InventoryCalculationResult:
    """Helper to construct a valid InventoryCalculationResult for testing."""
    return InventoryCalculationResult(
        store_id=store_id,
        sku_id=sku_id,
        date=date,
        average_daily_demand=average_daily_demand,
        demand_uncertainty_sigma=demand_uncertainty_sigma,
        lead_time_mean_days=lead_time_mean_days,
        lead_time_std_days=lead_time_std_days,
        review_period_days=review_period_days,
        z_score=1.645,
        safety_stock_lead_time=safety_stock_lead_time,
        safety_stock_protection=safety_stock_protection,
        safety_stock=safety_stock_lead_time,
        reorder_point=reorder_point,
        target_stock=target_stock,
        inventory_position=InventoryPosition(
            on_hand_stock=on_hand_stock,
            reserved_stock=reserved_stock,
            on_order_stock=on_order_stock,
        ),
        reorder_triggered=reorder_triggered,
        suggested_replenishment_qty=suggested_replenishment_qty,
        assumed_zero_pipeline=assumed_zero_pipeline,
    )


def _make_risk_assessment(
    calc_result: InventoryCalculationResult,
    stock_risk_level: StockRiskLevel = StockRiskLevel.LOW_STOCK_REORDER,
) -> InventoryRiskAssessment:
    """Helper to construct a valid InventoryRiskAssessment for testing."""
    return InventoryRiskAssessment(
        store_id=calc_result.store_id,
        sku_id=calc_result.sku_id,
        date=calc_result.date,
        stock_risk_level=stock_risk_level,
        reorder_recommended=(stock_risk_level in [StockRiskLevel.CRITICAL_LOW, StockRiskLevel.LOW_STOCK_REORDER]),
        stockout_risk_reasons=["Test stock risk reason"],
        excess_risk_reasons=[],
        excess_quantity=0.0,
        excess_quantity_p10=0.0,
        surge_coverage_ratio_p90=0.8,
        days_of_supply_p50=(
            calc_result.total_inventory_position / calc_result.average_daily_demand
            if calc_result.average_daily_demand > 0 else None
        ),
        expiry_risk_level=ExpiryRiskLevel.NOT_ASSESSABLE,
        at_risk_expiry_units=None,
        expiry_risk_reasons=["Batch data absent"],
        assumed_zero_pipeline=calc_result.assumed_zero_pipeline,
    )


class TestProcurementEngineQuantityAndMOQ(unittest.TestCase):
    def setUp(self):
        self.engine = ProcurementEngine()

    def test_no_reorder_recommendation(self):
        """When suggested_replenishment_qty <= 0, no order line must be generated."""
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            suggested_replenishment_qty=0.0,
            reorder_triggered=False,
        )
        rec = self.engine.generate_recommendation(calc)

        self.assertIsNone(rec.line_item)
        self.assertEqual(rec.final_order_qty, 0.0)
        self.assertEqual(rec.total_line_cost, 0.0)
        self.assertFalse(rec.requires_escalation)

        # Batch generation produces zero POs
        pos, recs, audit = self.engine.generate_purchase_orders([calc])
        self.assertEqual(len(pos), 0)
        self.assertEqual(len(recs), 1)
        self.assertEqual(audit["active_line_items_count"], 0)

    def test_case_pack_rounding(self):
        """Quantities must round upward to exact integer multiples of category case pack."""
        # Bakery: case pack = 10, MOQ = 15
        # raw suggested = 22.0 -> rounded = 30.0 (30 >= 15 so MOQ met)
        packed, final, moq_adj, esc, reasons = calculate_procurement_quantity(
            raw_suggested_qty=22.0,
            case_pack_size=10,
            moq=15.0,
            total_inventory_position=20.0,
            target_stock=42.0,
            is_critical=False,
        )
        self.assertEqual(packed, 30.0)
        self.assertEqual(final, 30.0)
        self.assertEqual(moq_adj, 0.0)
        self.assertFalse(esc)

        # Dairy: case pack = 12, MOQ = 10
        # raw suggested = 15.0 -> rounded = 24.0 (24 >= 10 so MOQ met)
        packed, final, moq_adj, esc, reasons = calculate_procurement_quantity(
            raw_suggested_qty=15.0,
            case_pack_size=12,
            moq=10.0,
            total_inventory_position=30.0,
            target_stock=45.0,
            is_critical=False,
        )
        self.assertEqual(packed, 24.0)
        self.assertEqual(final, 24.0)
        self.assertEqual(moq_adj, 0.0)

        # Meat: case pack = 6, MOQ = 5
        # raw suggested = 7.0 -> rounded = 12.0
        packed, final, moq_adj, esc, reasons = calculate_procurement_quantity(
            raw_suggested_qty=7.0,
            case_pack_size=6,
            moq=5.0,
            total_inventory_position=10.0,
            target_stock=17.0,
            is_critical=False,
        )
        self.assertEqual(packed, 12.0)
        self.assertEqual(final, 12.0)

    def test_moq_enforcement(self):
        """When case pack rounded quantity is below MOQ, order quantity must be raised to MOQ multiple."""
        # Bakery: case pack = 10, MOQ = 15. Raw suggested = 6.0
        # Case pack rounded = 10.0 < 15.0. Smallest multiple >= 15 is 20.0
        packed, final, moq_adj, esc, reasons = calculate_procurement_quantity(
            raw_suggested_qty=6.0,
            case_pack_size=10,
            moq=15.0,
            total_inventory_position=10.0,
            target_stock=16.0,
            is_critical=True,  # Critical allows raising to MOQ
        )
        self.assertEqual(packed, 10.0)
        self.assertEqual(final, 20.0)
        self.assertEqual(moq_adj, 10.0)
        self.assertEqual(final % 10, 0)  # Multiple of case pack
        self.assertGreaterEqual(final, 15.0)  # Adheres to MOQ

    def test_critical_out_of_stock_moq_handling(self):
        """Rule 5: In critical / stockout situations, allow quantity to be raised to MOQ without excess escalation."""
        calc = _make_calc_result(
            sku_id="SKU_BREAD_WHITE",  # Bakery: MOQ 15, case pack 10
            average_daily_demand=2.0,
            demand_uncertainty_sigma=1.0,
            on_hand_stock=0.0,         # Complete stockout!
            suggested_replenishment_qty=6.0,
            target_stock=20.0,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        rec = self.engine.generate_recommendation(
            calc_result=calc,
            risk_assessment=risk,
            unit_retail_price=35.0,
        )
        # Should raise to 20 units (MOQ satisfied)
        self.assertEqual(rec.final_order_qty, 20.0)
        self.assertEqual(rec.case_pack_rounded_qty, 10.0)
        self.assertEqual(rec.moq_adjustment_qty, 10.0)
        # In critical situation, MOQ adjustment does NOT flag excess review
        self.assertFalse(rec.requires_escalation)
        self.assertIsNotNone(rec.line_item)

    def test_non_critical_moq_exceeding_target_stock(self):
        """Rule 6: In non-critical situations, if MOQ adjustment pushes projected inventory above target stock, flag for review."""
        # Non-critical: on_hand is healthy, but replenishment triggered (e.g. LOW_STOCK_REORDER)
        # target_stock = 40.0, ip = 34.0, raw_suggested = 6.0
        calc = _make_calc_result(
            sku_id="SKU_BREAD_WHITE",  # Bakery: MOQ 15, case pack 10
            on_hand_stock=34.0,
            suggested_replenishment_qty=6.0,
            target_stock=40.0,
            safety_stock_lead_time=15.0,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        rec = self.engine.generate_recommendation(
            calc_result=calc,
            risk_assessment=risk,
            unit_retail_price=35.0,
        )
        # Case pack rounds 6.0 -> 10.0. MOQ raises 10.0 -> 20.0
        # Projected inventory = 34.0 + 20.0 = 54.0 > target_stock (40.0)
        self.assertEqual(rec.final_order_qty, 20.0)
        self.assertEqual(rec.case_pack_rounded_qty, 10.0)
        self.assertEqual(rec.moq_adjustment_qty, 10.0)
        # Must be flagged for review!
        self.assertTrue(rec.requires_escalation)
        self.assertTrue(any("MOQ Adjustment Excess" in r for r in rec.escalation_reasons))
        self.assertTrue(rec.line_item.requires_escalation)


class TestProcurementEngineSupplierSelection(unittest.TestCase):
    def setUp(self):
        self.engine = ProcurementEngine()

    def test_primary_supplier_selected_for_non_critical(self):
        """Non-critical replenishment must always select the primary supplier (60% COGS)."""
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            on_hand_stock=25.0,
            average_daily_demand=20.0,
            lead_time_mean_days=1.5,
            suggested_replenishment_qty=40.0,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        supplier, reason = select_supplier(calc, category="Dairy", risk_assessment=risk)
        self.assertEqual(supplier.supplier_type, SupplierType.PRIMARY)
        self.assertEqual(supplier.cogs_ratio, 0.60)
        self.assertIn("Primary vendor selected", reason)
        self.assertIn("non-critical", reason)

    def test_backup_supplier_selected_when_critical_and_coverage_below_lead_time(self):
        """Backup supplier must be selected when situation is critical AND coverage < primary lead time."""
        # On hand = 5 units, daily demand = 20 units -> coverage = 5 / 20 = 0.25 days
        # Dairy primary lead time = 1.5 days. Coverage (0.25d) < Primary LT (1.5d) -> Backup selected!
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            on_hand_stock=5.0,
            average_daily_demand=20.0,
            lead_time_mean_days=1.5,
            safety_stock_lead_time=15.0,
            suggested_replenishment_qty=50.0,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        supplier, reason = select_supplier(calc, category="Dairy", risk_assessment=risk)
        self.assertEqual(supplier.supplier_type, SupplierType.BACKUP)
        self.assertEqual(supplier.cogs_ratio, 0.75)
        self.assertEqual(supplier.lead_time_mean_days, 0.8)
        self.assertIn("Backup vendor selected", reason)
        self.assertIn("critical stock risk", reason)

    def test_primary_supplier_selected_when_critical_but_coverage_exceeds_lead_time(self):
        """When critical but coverage >= primary lead time, prefer primary supplier."""
        # On hand = 40 units, demand = 20 units -> coverage = 2.0 days
        # Meat primary lead time = 1.0 day. Coverage (2.0d) >= Meat Primary LT (1.0d)
        calc = _make_calc_result(
            sku_id="SKU_CHICKEN_500G",
            on_hand_stock=40.0,
            average_daily_demand=20.0,
            lead_time_mean_days=1.0,
            safety_stock_lead_time=45.0,  # on_hand < SS_LT triggers CRITICAL_LOW
            suggested_replenishment_qty=30.0,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        supplier, reason = select_supplier(calc, category="Meat", risk_assessment=risk)
        self.assertEqual(supplier.supplier_type, SupplierType.PRIMARY)
        self.assertEqual(supplier.cogs_ratio, 0.60)
        self.assertIn("Primary vendor selected", reason)
        self.assertIn("sufficiently buffers", reason)

    def test_backup_supplier_lead_time_is_point_eight_days_across_categories(self):
        """Backup supplier profile must have lead_time_mean_days == 0.8 across all categories."""
        for cat in ["Bakery", "Dairy", "Meat", "Produce"]:
            supp = get_simulated_supplier_profile(cat, SupplierType.BACKUP)
            self.assertEqual(supp.lead_time_mean_days, 0.8)
            self.assertEqual(supp.lead_time_std_days, 0.15)
            self.assertEqual(supp.cogs_ratio, 0.75)

    def test_backup_supplier_replenishment_uses_backup_target_stock_and_variability(self):
        """When backup supplier is selected, replenishment is sized to backup target stock (0.8d mean, 0.15d std)."""
        # Dairy: demand = 40.0/day, sigma_D = 8.0, on_hand = 5.0 (coverage = 0.125d < 1.5d, Critical)
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            on_hand_stock=5.0,
            average_daily_demand=40.0,
            target_stock=128.681555,  # Primary target stock
            suggested_replenishment_qty=123.681555,  # Primary suggested
            assumed_zero_pipeline=False,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        rec = self.engine.generate_recommendation(
            calc_result=calc,
            risk_assessment=risk,
            unit_retail_price=50.0,
        )

        # 1. Backup supplier must be selected
        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.BACKUP)
        self.assertEqual(rec.selected_supplier.lead_time_mean_days, 0.8)
        self.assertEqual(rec.selected_supplier.lead_time_std_days, 0.15)
        self.assertEqual(rec.selected_supplier.cogs_ratio, 0.75)

        # 2. Analytical backup target stock:
        # sigma_D = 5.0, d_avg = 40.0, LT = 0.8, R = 1.0, sigma_LT = 0.15, Z = 1.645
        # SS_P = 1.645 * sqrt((0.8 + 1.0) * 25 + 1600 * 0.0225) = 1.645 * sqrt(45 + 36) = 1.645 * 9.0 = 14.805
        # Target stock = 40 * 1.8 + 14.805 = 72.0 + 14.805 = 86.805
        expected_backup_ss_p = 14.805
        expected_backup_target = 86.805
        expected_raw_qty = 86.805 - 5.0  # 81.805

        self.assertAlmostEqual(rec.raw_suggested_qty, expected_raw_qty, places=4)

        # 3. Case-pack rounding (Dairy case pack = 12):
        # ceil(81.805 / 12) * 12 = 84.0 units (vs 132.0 units if primary target stock was used)
        self.assertEqual(rec.case_pack_rounded_qty, 84.0)
        self.assertEqual(rec.final_order_qty, 84.0)
        # Unit procurement cost = 50.0 * 0.75 = 37.50
        self.assertEqual(rec.unit_procurement_cost, 37.5)
        self.assertEqual(rec.total_line_cost, 84.0 * 37.5)


class TestProcurementEngineCostAndEscalation(unittest.TestCase):
    def setUp(self):
        self.engine = ProcurementEngine()

    def test_procurement_cost_calculations(self):
        """Procurement cost = retail_price * cogs_ratio; total_cost = final_qty * unit_cost."""
        # Dairy Primary: cogs_ratio = 0.60. Retail price = ₹50.0. Unit cost = ₹30.0
        # Final qty = 24.0. Total cost = ₹720.0 (non-escalated when pipeline is known)
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            on_hand_stock=30.0,
            suggested_replenishment_qty=20.0,  # Rounds to 24
            assumed_zero_pipeline=False,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        rec = self.engine.generate_recommendation(
            calc_result=calc,
            risk_assessment=risk,
            unit_retail_price=50.0,
        )
        self.assertEqual(rec.unit_retail_price, 50.0)
        self.assertEqual(rec.unit_procurement_cost, 30.0)
        self.assertEqual(rec.final_order_qty, 24.0)
        self.assertEqual(rec.total_line_cost, 720.0)
        self.assertFalse(rec.requires_escalation)

    def test_assumed_zero_pipeline_low_stock_reorder_escalation(self):
        """Assumed-zero pipeline with LOW_STOCK_REORDER must be flagged for manual review."""
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            on_hand_stock=30.0,
            suggested_replenishment_qty=20.0,  # Rounds to 24
            assumed_zero_pipeline=True,       # Simulation assumption flag is True
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        rec = self.engine.generate_recommendation(
            calc_result=calc,
            risk_assessment=risk,
            unit_retail_price=50.0,
        )
        self.assertTrue(rec.requires_escalation)
        self.assertTrue(rec.line_item.requires_escalation)
        self.assertTrue(any("Assumed-Zero Pipeline with LOW_STOCK_REORDER" in r for r in rec.escalation_reasons))

    def test_line_item_cost_escalation_above_5000(self):
        """Line items with total_line_cost > ₹5,000 must be flagged with requires_escalation=True."""
        # Chicken (Meat): retail price = ₹160.0. Primary cost = 0.60 * 160 = ₹96.0
        # Order 60 units (case pack = 6). Total cost = 60 * 96 = ₹5,760 (> ₹5,000)
        calc = _make_calc_result(
            sku_id="SKU_CHICKEN_500G",
            on_hand_stock=10.0,
            suggested_replenishment_qty=58.0,  # Rounds to 60
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        rec = self.engine.generate_recommendation(
            calc_result=calc,
            risk_assessment=risk,
            unit_retail_price=160.0,
        )
        self.assertEqual(rec.total_line_cost, 5760.0)
        self.assertTrue(rec.requires_escalation)
        self.assertTrue(rec.line_item.requires_escalation)
        self.assertTrue(any("exceeds single-line review limit ₹5,000.00" in r for r in rec.escalation_reasons))

    def test_line_item_cost_escalation_boundary_at_exactly_5000(self):
        """Line cost of exactly ₹5,000.00 must NOT trigger line escalation (strictly > ₹5,000)."""
        # Produce: case pack = 10. Qty = 200 units.
        # Retail price = 125.0 / 3.0, Primary cost = round((125.0 / 3.0) * 0.60, 4) = ₹25.0000
        # Total line cost = 200 * 25.00 = ₹5,000.00 exactly
        calc = _make_calc_result(
            sku_id="SKU_POTATO_1KG",
            on_hand_stock=50.0,
            suggested_replenishment_qty=200.0,
            assumed_zero_pipeline=False,
        )
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        rec = self.engine.generate_recommendation(
            calc_result=calc,
            risk_assessment=risk,
            unit_retail_price=125.0 / 3.0,
        )
        self.assertEqual(rec.total_line_cost, 5000.0)
        self.assertFalse(rec.requires_escalation)
        self.assertFalse(rec.line_item.requires_escalation)
        self.assertFalse(any("exceeds single-line review limit" in r for r in rec.escalation_reasons))


class TestProcurementEngineGroupingAndGovernance(unittest.TestCase):
    def setUp(self):
        self.engine = ProcurementEngine()

    def test_purchase_order_grouping_by_store_and_supplier(self):
        """Purchase Orders must be grouped strictly by (store_id, supplier_id)."""
        # Node 1: Store A, Dairy (Primary)
        calc1 = _make_calc_result(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            on_hand_stock=20.0,
            suggested_replenishment_qty=24.0,
        )
        # Node 2: Store A, Dairy (Primary)
        calc2 = _make_calc_result(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_CURD_400G",
            on_hand_stock=20.0,
            suggested_replenishment_qty=12.0,
        )
        # Node 3: Store B, Dairy (Primary)
        calc3 = _make_calc_result(
            store_id="DS_BLR_INDIRANAGAR",
            sku_id="SKU_MILK_1L",
            on_hand_stock=20.0,
            suggested_replenishment_qty=24.0,
        )
        # Node 4: Store A, Meat (Primary)
        calc4 = _make_calc_result(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_CHICKEN_500G",
            on_hand_stock=20.0,
            suggested_replenishment_qty=12.0,
        )

        pos, recs, audit = self.engine.generate_purchase_orders([calc1, calc2, calc3, calc4])

        # Expect 3 Purchase Orders:
        # 1. Store A + Dairy Primary (2 items: Milk + Curd)
        # 2. Store A + Meat Primary (1 item: Chicken)
        # 3. Store B + Dairy Primary (1 item: Milk)
        self.assertEqual(len(pos), 3)

        store_a_dairy = next(p for p in pos if p.store_id == "DS_BLR_KORAMANGALA" and "DAIRY" in p.supplier_id)
        self.assertEqual(len(store_a_dairy.items), 2)
        self.assertEqual(store_a_dairy.total_order_qty, 36.0)

        store_a_meat = next(p for p in pos if p.store_id == "DS_BLR_KORAMANGALA" and "MEAT" in p.supplier_id)
        self.assertEqual(len(store_a_meat.items), 1)

        store_b_dairy = next(p for p in pos if p.store_id == "DS_BLR_INDIRANAGAR" and "DAIRY" in p.supplier_id)
        self.assertEqual(len(store_b_dairy.items), 1)

    def test_order_level_escalation_above_25k(self):
        """Purchase Orders with total_order_cost > ₹25,000 must be flagged with requires_escalation=True."""
        # 300 units of Chicken at ₹96 = ₹28,800
        calc = _make_calc_result(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_CHICKEN_500G",
            on_hand_stock=20.0,
            suggested_replenishment_qty=300.0,
        )
        pos, recs, audit = self.engine.generate_purchase_orders([calc])

        self.assertEqual(len(pos), 1)
        po = pos[0]
        self.assertGreater(po.total_order_cost, 25000.0)
        self.assertTrue(po.requires_escalation)
        self.assertTrue(any("exceeds store-level review threshold ₹25,000.00" in r for r in po.escalation_reasons))

    def test_order_level_escalation_boundary_at_exactly_25000(self):
        """Purchase Orders with total_order_cost of exactly ₹25,000.00 must NOT trigger order-level escalation."""
        produce_skus = ["SKU_BANANA_DZ", "SKU_ONION_1KG", "SKU_POTATO_1KG", "SKU_SPINACH_500G", "SKU_TOMATO_1KG"]
        calcs = [
            _make_calc_result(
                store_id="DS_BLR_KORAMANGALA",
                sku_id=sku,
                on_hand_stock=50.0,
                suggested_replenishment_qty=200.0,
                assumed_zero_pipeline=False,
            )
            for sku in produce_skus
        ]
        price_catalog = {sku: 125.0 / 3.0 for sku in produce_skus}
        pos, recs, audit = self.engine.generate_purchase_orders(
            calcs,
            price_catalog=price_catalog,
        )

        self.assertEqual(len(pos), 1)
        po = pos[0]
        self.assertEqual(po.total_order_cost, 25000.0)
        self.assertFalse(po.requires_escalation)
        self.assertFalse(any("exceeds store-level review threshold" in r for r in po.escalation_reasons))

    def test_assumed_zero_pipeline_flag_propagated(self):
        """When calculation result relied on assumed zero pipeline, flag must be set on the PO."""
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            suggested_replenishment_qty=24.0,
            assumed_zero_pipeline=True,
        )
        pos, recs, audit = self.engine.generate_purchase_orders([calc])

        self.assertEqual(len(pos), 1)
        self.assertTrue(pos[0].assumed_zero_pipeline_flag)
        self.assertEqual(audit["assumed_zero_pipeline_orders_count"], 1)

    def test_mandatory_human_approval_enforcement(self):
        """Strict governance: Every generated PO must be PENDING_APPROVAL and have approved_by=None."""
        calc = _make_calc_result(
            sku_id="SKU_MILK_1L",
            suggested_replenishment_qty=24.0,
        )
        pos, recs, audit = self.engine.generate_purchase_orders([calc])

        self.assertEqual(len(pos), 1)
        po = pos[0]

        # Enforce non-issued state
        self.assertEqual(po.status, PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertIsNone(po.approved_by)
        self.assertIsNone(po.approval_timestamp)
        self.assertTrue(audit["mandatory_human_approval_enforced"])

        # Attempting to construct an APPROVED order without approved_by must fail
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id=po.po_id,
                created_date=po.created_date,
                store_id=po.store_id,
                supplier_id=po.supplier_id,
                supplier_type=po.supplier_type,
                items=po.items,
                total_order_qty=po.total_order_qty,
                total_order_cost=po.total_order_cost,
                requires_escalation=po.requires_escalation,
                escalation_reasons=po.escalation_reasons,
                status=PurchaseOrderStatus.APPROVED,
                approved_by=None,  # Forbidden!
            )
        self.assertIn("Mandatory Governance Violation", str(ctx.exception))


class TestBackupSupplierReorderEvaluation(unittest.TestCase):
    """
    Focused verification for Fix 1: Supplier-Specific Reorder Evaluation for Backup Suppliers.
    Ensures that when a backup supplier is selected (or forced):
    1. Complete inventory evaluation is executed via InventoryEngine.recalculate_for_supplier().
    2. Reorder condition (backup_inventory_position <= backup_reorder_point) is strictly enforced.
    3. If backup ROP is not breached, no replenishment, no line item, and no PO is created.
    4. Primary supplier behavior and calculations remain strictly unchanged.
    """

    def setUp(self):
        self.engine = ProcurementEngine()
        # Analytical reference parameters for Dairy:
        # D_avg = 40.0, sigma_D = 5.0, Z = 1.645
        # Primary: LT = 1.5, sigma_LT = 0.3, R = 1.0 -> SS_LT = 22.162, ROP = 82.162, Target Stock = 126.85
        # Backup: LT = 0.8, sigma_LT = 0.15, R = 1.0 -> SS_LT = 12.310, ROP = 44.310, Target Stock = 86.805
        self.backup_rop = 44.31005280248627
        self.backup_target_stock = 86.805

    def _make_dairy_calc(self, on_hand: float, primary_rop: float = 82.162) -> InventoryCalculationResult:
        return _make_calc_result(
            sku_id="SKU_MILK_1L",
            on_hand_stock=on_hand,
            average_daily_demand=40.0,
            demand_uncertainty_sigma=5.0,
            reorder_point=primary_rop,
            target_stock=126.85,
            suggested_replenishment_qty=max(0.0, 126.85 - on_hand),
            reorder_triggered=(on_hand <= primary_rop),
            assumed_zero_pipeline=False,
        )

    def test_case_a_backup_selected_but_backup_rop_not_breached(self):
        """Case A: Backup supplier selected, but backup ROP not breached -> zero replenishment, no PO."""
        # On hand = 50.0. Coverage = 50 / 40 = 1.25d < 1.5d (Primary LT).
        # Primary ROP = 82.162 (breached!).
        # But Backup ROP = 44.310 -> 50.0 > 44.310 (NOT breached!).
        calc = self._make_dairy_calc(on_hand=50.0)
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        rec = self.engine.generate_recommendation(calc_result=calc, risk_assessment=risk, unit_retail_price=50.0)

        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.BACKUP)
        self.assertFalse(rec.reorder_triggered)
        self.assertEqual(rec.raw_suggested_qty, 0.0)
        self.assertEqual(rec.case_pack_rounded_qty, 0.0)
        self.assertEqual(rec.moq_adjustment_qty, 0.0)
        self.assertEqual(rec.final_order_qty, 0.0)
        self.assertEqual(rec.total_line_cost, 0.0)
        self.assertIsNone(rec.line_item)
        self.assertIn("Backup reorder threshold not breached", rec.supplier_selection_reason)

        # Generating purchase orders must produce ZERO orders
        pos, recs, audit = self.engine.generate_purchase_orders(
            [calc],
            risk_assessments={(calc.store_id, calc.sku_id, calc.date): risk}
        )
        self.assertEqual(len(pos), 0)

    def test_case_b_inventory_position_exactly_equals_backup_rop(self):
        """Case B: Inventory position exactly equals backup ROP -> reorder triggered (equality condition)."""
        calc = self._make_dairy_calc(on_hand=self.backup_rop)
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        rec = self.engine.generate_recommendation(calc_result=calc, risk_assessment=risk, unit_retail_price=50.0)

        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.BACKUP)
        self.assertTrue(rec.reorder_triggered)
        expected_raw = self.backup_target_stock - self.backup_rop  # ~42.495
        self.assertAlmostEqual(rec.raw_suggested_qty, expected_raw, places=4)
        # Dairy case pack = 12 -> ceil(42.495 / 12) * 12 = 48.0
        self.assertEqual(rec.case_pack_rounded_qty, 48.0)
        self.assertEqual(rec.final_order_qty, 48.0)
        self.assertIsNotNone(rec.line_item)

        pos, _, _ = self.engine.generate_purchase_orders(
            [calc],
            risk_assessments={(calc.store_id, calc.sku_id, calc.date): risk}
        )
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0].supplier_type, SupplierType.BACKUP)
        self.assertEqual(pos[0].urgency, PurchaseOrderUrgency.EXPEDITE)

    def test_case_c_inventory_position_below_backup_rop(self):
        """Case C: Inventory position strictly below backup ROP -> replenishment triggered using backup target stock."""
        calc = self._make_dairy_calc(on_hand=20.0)
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        rec = self.engine.generate_recommendation(calc_result=calc, risk_assessment=risk, unit_retail_price=50.0)

        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.BACKUP)
        self.assertTrue(rec.reorder_triggered)
        expected_raw = self.backup_target_stock - 20.0  # 66.805
        self.assertAlmostEqual(rec.raw_suggested_qty, expected_raw, places=4)
        # ceil(66.805 / 12) * 12 = 72.0
        self.assertEqual(rec.final_order_qty, 72.0)
        self.assertIsNotNone(rec.line_item)

    def test_case_d_inventory_position_above_backup_target_stock(self):
        """Case D: Inventory position above backup target stock -> zero replenishment, no negative or unnecessary PO."""
        calc = self._make_dairy_calc(on_hand=90.0)  # 90 > backup target stock (86.805)
        supp = get_simulated_supplier_profile("Dairy", SupplierType.BACKUP)

        rec = self.engine.generate_recommendation(calc_result=calc, supplier=supp, unit_retail_price=50.0)

        self.assertEqual(rec.raw_suggested_qty, 0.0)
        self.assertEqual(rec.final_order_qty, 0.0)
        self.assertIsNone(rec.line_item)
        pos, _, _ = self.engine.generate_purchase_orders([calc])
        self.assertEqual(len(pos), 0)

    def test_case_e_primary_supplier_selected_preserves_baseline(self):
        """Case E: Primary supplier selected -> baseline calculation and reorder quantity remain completely unchanged."""
        calc = self._make_dairy_calc(on_hand=50.0)  # 50 <= primary ROP (82.162)
        # Non-critical -> primary supplier selected
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        rec = self.engine.generate_recommendation(calc_result=calc, risk_assessment=risk, unit_retail_price=50.0)

        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.PRIMARY)
        self.assertTrue(rec.reorder_triggered)
        self.assertEqual(rec.raw_suggested_qty, calc.suggested_replenishment_qty)
        self.assertEqual(rec.unit_procurement_cost, 50.0 * 0.60)
        self.assertIsNotNone(rec.line_item)

    def test_case_f_explicit_backup_supplier_override(self):
        """Case F: Explicit backup supplier override applies the same supplier-specific reorder evaluation."""
        calc = self._make_dairy_calc(on_hand=50.0)  # 50 > backup ROP (44.310)
        backup_supp = get_simulated_supplier_profile("Dairy", SupplierType.BACKUP)

        # Force override to backup supplier
        rec = self.engine.generate_recommendation(calc_result=calc, supplier=backup_supp, unit_retail_price=50.0)

        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.BACKUP)
        self.assertFalse(rec.reorder_triggered)
        self.assertEqual(rec.raw_suggested_qty, 0.0)
        self.assertEqual(rec.final_order_qty, 0.0)
        self.assertIsNone(rec.line_item)

    def test_case_g_moq_and_case_pack_interaction(self):
        """Case G: MOQ and case pack are never applied when supplier reorder is not triggered."""
        calc = self._make_dairy_calc(on_hand=50.0)  # 50 > backup ROP, MOQ is 24
        risk = _make_risk_assessment(calc, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        rec = self.engine.generate_recommendation(calc_result=calc, risk_assessment=risk, unit_retail_price=50.0)

        # Zero replenishment must NEVER be raised to MOQ (24)
        self.assertEqual(rec.raw_suggested_qty, 0.0)
        self.assertEqual(rec.case_pack_rounded_qty, 0.0)
        self.assertEqual(rec.moq_adjustment_qty, 0.0)
        self.assertEqual(rec.final_order_qty, 0.0)
        self.assertIsNone(rec.line_item)

    def test_case_h_po_grouping_with_zero_and_valid_backup_lines(self):
        """Case H: Zero-quantity recommendations do not create empty POs and valid lines group correctly."""
        # Item 1: Store 1, Milk, on_hand = 50.0 (Backup selected, but ROP not breached -> 0 qty)
        calc_s1_milk = _make_calc_result(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            on_hand_stock=50.0,
            average_daily_demand=40.0,
            reorder_point=82.162,
            target_stock=126.85,
            suggested_replenishment_qty=76.85,
            reorder_triggered=True,
        )
        risk_s1_milk = _make_risk_assessment(calc_s1_milk, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        # Item 2: Store 1, Curd, on_hand = 10.0 (Backup selected, ROP breached -> valid backup line)
        calc_s1_curd = _make_calc_result(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_CURD_400G",
            on_hand_stock=10.0,
            average_daily_demand=40.0,
            reorder_point=82.162,
            target_stock=126.85,
            suggested_replenishment_qty=116.85,
            reorder_triggered=True,
        )
        risk_s1_curd = _make_risk_assessment(calc_s1_curd, stock_risk_level=StockRiskLevel.CRITICAL_LOW)

        # Item 3: Store 2, Milk, on_hand = 20.0 (Primary selected -> valid primary line)
        calc_s2_milk = _make_calc_result(
            store_id="DS_BLR_INDIRANAGAR",
            sku_id="SKU_MILK_1L",
            on_hand_stock=20.0,
            average_daily_demand=20.0,
            reorder_point=40.0,
            target_stock=60.0,
            suggested_replenishment_qty=40.0,
            reorder_triggered=True,
        )
        risk_s2_milk = _make_risk_assessment(calc_s2_milk, stock_risk_level=StockRiskLevel.LOW_STOCK_REORDER)

        pos, recs, audit = self.engine.generate_purchase_orders(
            [calc_s1_milk, calc_s1_curd, calc_s2_milk],
            risk_assessments={
                (calc_s1_milk.store_id, calc_s1_milk.sku_id, calc_s1_milk.date): risk_s1_milk,
                (calc_s1_curd.store_id, calc_s1_curd.sku_id, calc_s1_curd.date): risk_s1_curd,
                (calc_s2_milk.store_id, calc_s2_milk.sku_id, calc_s2_milk.date): risk_s2_milk,
            }
        )

        # Exactly 2 POs: Store 1 Backup (only Curd, NOT Milk) and Store 2 Primary
        self.assertEqual(len(pos), 2)
        s1_backup_po = next(p for p in pos if p.store_id == "DS_BLR_KORAMANGALA")
        self.assertEqual(s1_backup_po.supplier_type, SupplierType.BACKUP)
        self.assertEqual(len(s1_backup_po.items), 1)
        self.assertEqual(s1_backup_po.items[0].sku_id, "SKU_CURD_400G")

        s2_primary_po = next(p for p in pos if p.store_id == "DS_BLR_INDIRANAGAR")
        self.assertEqual(s2_primary_po.supplier_type, SupplierType.PRIMARY)
        self.assertEqual(len(s2_primary_po.items), 1)
        self.assertEqual(s2_primary_po.items[0].sku_id, "SKU_MILK_1L")


if __name__ == "__main__":
    unittest.main()
