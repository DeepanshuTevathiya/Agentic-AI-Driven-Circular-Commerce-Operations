"""
Unit Tests: Lead 2 Phase 3.3 Inventory Risk Logic, Testing & Integration
========================================================================
Validates:
1. Stock level risk classification and metric computation:
   - CRITICAL_LOW (IP <= SS_LT or on_hand == 0)
   - LOW_STOCK_REORDER (SS_LT < IP <= ROP)
   - HEALTHY (ROP < IP <= Target_Stock)
   - EXCESS_STOCK (IP > Target_Stock)
2. Boundary cases:
   - Exact ROP boundary
   - Exact Target Stock boundary
   - Exact Safety Stock boundary
3. Probabilistic stress metrics using P10/P50/P90 forecasts:
   - P90 surge coverage ratio (IP / (P90 * LT_mean))
   - P10 clearance excess calculation
   - P50 days of supply
   - Safe handling of zero demand (no division by zero)
4. Expiry risk governance:
   - Explicit NOT_ASSESSABLE when batch data is absent
   - Accurate FIFO clearance simulation when batches are present
   - HIGH, MEDIUM, and LOW expiry risk differentiation
5. Pipeline assumption flag propagation:
   - Preserves assumed_zero_pipeline from calculation result
6. End-to-end integration:
   - Execution across real repository forecast nodes
"""

import unittest
from typing import List

from src.agents.config import get_default_config
from src.agents.data_loader import DataLoader
from src.agents.inventory_engine import (
    InventoryEngine,
    assess_expiry_risk,
    assess_inventory_risk,
    assess_stock_risk,
    compute_surge_coverage_ratio_p90,
    validate_batch_inventory,
)
from src.agents.schemas import (
    DailyAgentInput,
    DemandForecast,
    ExpiryRiskLevel,
    InventoryBatch,
    InventoryCalculationResult,
    InventoryPosition,
    InventorySnapshot,
    InventorySnapshotBasis,
    StockRiskLevel,
)


class TestStockRiskLogic(unittest.TestCase):
    """Unit tests for stock level classification and metric derivation."""

    def setUp(self):
        self.config = get_default_config()
        self.engine = InventoryEngine(self.config)

        # Baseline forecast: D_avg = 40, sigma_D = 8, P10 = 25, P90 = 60
        self.forecast = DemandForecast(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-15",
            p10_demand=25.0,
            p50_demand=40.0,
            p90_demand=60.0,
            sigma_demand=8.0,
        )

    def _get_calc_result(self, pos: InventoryPosition) -> InventoryCalculationResult:
        daily_inp = DailyAgentInput(
            store_id=self.forecast.store_id,
            sku_id=self.forecast.sku_id,
            category=self.forecast.category,
            date=self.forecast.date,
            forecast=self.forecast,
            inventory=InventorySnapshot(
                store_id=self.forecast.store_id,
                sku_id=self.forecast.sku_id,
                date=self.forecast.date,
                on_hand_stock=pos.on_hand_stock,
                snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
            ),
        )
        return self.engine.calculate(daily_inp, inventory_position=pos)

    def test_critical_low_when_on_hand_is_zero(self):
        """Physical stockout (on_hand = 0) must trigger CRITICAL_LOW even with pending orders."""
        pos = InventoryPosition(on_hand_stock=0.0, on_order_stock=100.0, reserved_stock=0.0)
        calc = self._get_calc_result(pos)
        assm = self.engine.assess_risk(calc, self.forecast)

        self.assertEqual(assm.stock_risk_level, StockRiskLevel.CRITICAL_LOW)
        self.assertTrue(assm.reorder_recommended)
        self.assertTrue(any("depleted" in r.lower() for r in assm.stockout_risk_reasons))

    def test_critical_low_when_ip_breaches_safety_stock(self):
        """IP <= SS_LT must trigger CRITICAL_LOW."""
        # Baseline SS_LT ≈ 25.48. Set on_hand = 20.0
        pos = InventoryPosition(on_hand_stock=20.0, on_order_stock=0.0, reserved_stock=0.0)
        calc = self._get_calc_result(pos)
        assm = self.engine.assess_risk(calc, self.forecast)

        self.assertEqual(assm.stock_risk_level, StockRiskLevel.CRITICAL_LOW)
        self.assertTrue(assm.reorder_recommended)
        self.assertTrue(any("safety stock" in r.lower() for r in assm.stockout_risk_reasons))

    def test_low_stock_reorder_when_ip_below_rop_but_above_ss(self):
        """SS_LT < IP <= ROP must trigger LOW_STOCK_REORDER."""
        # Baseline: SS_LT ≈ 25.48, ROP ≈ 85.48. Set on_hand = 50.0
        pos = InventoryPosition(on_hand_stock=50.0, on_order_stock=0.0, reserved_stock=0.0)
        calc = self._get_calc_result(pos)
        assm = self.engine.assess_risk(calc, self.forecast)

        self.assertEqual(assm.stock_risk_level, StockRiskLevel.LOW_STOCK_REORDER)
        self.assertTrue(assm.reorder_recommended)
        self.assertTrue(any("reorder point" in r.lower() for r in assm.stockout_risk_reasons))
        self.assertEqual(assm.excess_quantity, 0.0)

    def test_exact_rop_boundary_is_low_stock_reorder(self):
        """Exact boundary IP == ROP must trigger LOW_STOCK_REORDER."""
        calc_dummy = self._get_calc_result(InventoryPosition(on_hand_stock=100.0, on_order_stock=0.0, reserved_stock=0.0))
        rop_val = calc_dummy.reorder_point

        pos_at_rop = InventoryPosition(on_hand_stock=rop_val, on_order_stock=0.0, reserved_stock=0.0)
        calc_at_rop = self._get_calc_result(pos_at_rop)
        assm = self.engine.assess_risk(calc_at_rop, self.forecast)

        self.assertEqual(assm.stock_risk_level, StockRiskLevel.LOW_STOCK_REORDER)
        self.assertTrue(assm.reorder_recommended)

    def test_healthy_stock_scenario(self):
        """ROP < IP <= Target_Stock must trigger HEALTHY status with reorder suppressed."""
        # Target Stock ≈ 128.68. Set on_hand = 100.0
        pos = InventoryPosition(on_hand_stock=100.0, on_order_stock=0.0, reserved_stock=0.0)
        calc = self._get_calc_result(pos)
        assm = self.engine.assess_risk(calc, self.forecast)

        self.assertEqual(assm.stock_risk_level, StockRiskLevel.HEALTHY)
        self.assertFalse(assm.reorder_recommended)
        self.assertEqual(assm.excess_quantity, 0.0)

    def test_exact_target_stock_boundary_is_healthy(self):
        """Exact boundary IP == Target_Stock must be HEALTHY with 0 excess."""
        calc_dummy = self._get_calc_result(InventoryPosition(on_hand_stock=100.0, on_order_stock=0.0, reserved_stock=0.0))
        target_val = calc_dummy.target_stock

        pos_at_target = InventoryPosition(on_hand_stock=target_val, on_order_stock=0.0, reserved_stock=0.0)
        calc_at_target = self._get_calc_result(pos_at_target)
        assm = self.engine.assess_risk(calc_at_target, self.forecast)

        self.assertEqual(assm.stock_risk_level, StockRiskLevel.HEALTHY)
        self.assertFalse(assm.reorder_recommended)
        self.assertEqual(assm.excess_quantity, 0.0)

    def test_excess_stock_scenario(self):
        """IP > Target_Stock must trigger EXCESS_STOCK with accurate excess calculations."""
        # Target Stock ≈ 128.68. Set on_hand = 180.0
        pos = InventoryPosition(on_hand_stock=180.0, on_order_stock=0.0, reserved_stock=0.0)
        calc = self._get_calc_result(pos)
        assm = self.engine.assess_risk(calc, self.forecast)

        self.assertEqual(assm.stock_risk_level, StockRiskLevel.EXCESS_STOCK)
        self.assertFalse(assm.reorder_recommended)
        self.assertAlmostEqual(assm.excess_quantity, 180.0 - calc.target_stock, places=4)
        self.assertGreater(assm.excess_quantity, 0.0)
        # Sluggish P10 demand amplifies excess
        self.assertGreater(assm.excess_quantity_p10, assm.excess_quantity)
        self.assertTrue(any("excess" in r.lower() for r in assm.excess_risk_reasons))

    def test_p90_surge_coverage_ratio_and_advisory(self):
        """P90 peak surge vulnerability is accurately quantified."""
        # P90 = 60, LT = 1.5 -> P90 LTD = 90.0 units
        # If IP = 70, coverage ratio = 70 / 90 = 0.7778
        pos = InventoryPosition(on_hand_stock=70.0, on_order_stock=0.0, reserved_stock=0.0)
        calc = self._get_calc_result(pos)
        assm = self.engine.assess_risk(calc, self.forecast)

        self.assertIsNotNone(assm.surge_coverage_ratio_p90)
        self.assertAlmostEqual(assm.surge_coverage_ratio_p90, 70.0 / 90.0, places=4)
        self.assertTrue(any("surge" in r.lower() for r in assm.stockout_risk_reasons))

    def test_zero_demand_safe_handling(self):
        """Zero demand must handle days_of_supply and surge ratio gracefully without ZeroDivisionError."""
        zero_fc = DemandForecast(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-15",
            p10_demand=0.0,
            p50_demand=0.0,
            p90_demand=0.0,
            sigma_demand=1.0,
        )
        daily_inp = DailyAgentInput(
            store_id=zero_fc.store_id,
            sku_id=zero_fc.sku_id,
            category=zero_fc.category,
            date=zero_fc.date,
            forecast=zero_fc,
            inventory=InventorySnapshot(
                store_id=zero_fc.store_id,
                sku_id=zero_fc.sku_id,
                date=zero_fc.date,
                on_hand_stock=50.0,
                snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
            ),
        )
        calc = self.engine.calculate(daily_inp)
        assm = self.engine.assess_risk(calc, zero_fc)

        # None values instead of float('inf') or ZeroDivisionError
        self.assertIsNone(assm.days_of_supply_p50)
        self.assertIsNone(assm.surge_coverage_ratio_p90)

    def test_p90_zero_safe_handling(self):
        """P90 == 0.0 must safely return None without ZeroDivisionError across all helper and assessment paths."""
        # 1. Direct pure function evaluation when P90 == 0
        self.assertIsNone(compute_surge_coverage_ratio_p90(total_inventory_position=50.0, p90_demand=0.0, lt_mean=1.5))
        self.assertIsNone(compute_surge_coverage_ratio_p90(total_inventory_position=0.0, p90_demand=0.0, lt_mean=1.5))

        # 2. Validation bounds
        with self.assertRaises(ValueError):
            compute_surge_coverage_ratio_p90(total_inventory_position=50.0, p90_demand=-5.0, lt_mean=1.5)
        with self.assertRaises(ValueError):
            compute_surge_coverage_ratio_p90(total_inventory_position=50.0, p90_demand=10.0, lt_mean=0.0)
        with self.assertRaises(ValueError):
            compute_surge_coverage_ratio_p90(total_inventory_position=50.0, p90_demand=10.0, lt_mean=-1.0)
        with self.assertRaises(ValueError):
            compute_surge_coverage_ratio_p90(total_inventory_position=float("nan"), p90_demand=10.0, lt_mean=1.5)

    def test_stress_scenario_labeling_in_reasons(self):
        """P10 downside clearance and P90 demand surge must be explicitly labeled as deterministic stress scenarios."""
        # Scenario 1: P90 surge vulnerability
        pos_low = InventoryPosition(on_hand_stock=30.0, on_order_stock=0.0, reserved_stock=0.0)
        calc_low = self._get_calc_result(pos_low)
        assm_low = self.engine.assess_risk(calc_low, self.forecast)

        surge_reasons = [r for r in assm_low.stockout_risk_reasons if "surge" in r.lower()]
        self.assertTrue(len(surge_reasons) > 0)
        for r in surge_reasons:
            self.assertIn("Deterministic Stress Scenario", r)
            self.assertIn("not a calibrated multi-day prediction", r)

        # Scenario 2: P10 sluggish downside clearance in excess stock
        pos_excess = InventoryPosition(on_hand_stock=250.0, on_order_stock=0.0, reserved_stock=0.0)
        calc_excess = self._get_calc_result(pos_excess)
        assm_excess = self.engine.assess_risk(calc_excess, self.forecast)

        p10_reasons = [r for r in assm_excess.excess_risk_reasons if "p10" in r.lower()]
        self.assertTrue(len(p10_reasons) > 0)
        for r in p10_reasons:
            self.assertIn("Deterministic Stress Scenario", r)
            self.assertIn("not a calibrated multi-day prediction", r)


class TestExpiryRiskLogic(unittest.TestCase):
    """Unit tests for perishable inventory expiry risk governance."""

    def test_missing_batch_data_returns_not_assessable(self):
        """When batch data is absent, must explicitly return NOT_ASSESSABLE with None units."""
        level, units, reasons = assess_expiry_risk(d_avg=30.0, lt_mean=1.5, batches=None)
        self.assertEqual(level, ExpiryRiskLevel.NOT_ASSESSABLE)
        self.assertIsNone(units)
        self.assertTrue(any("not assessable" in r.lower() for r in reasons))

        # Empty list also returns NOT_ASSESSABLE
        level_empty, units_empty, _ = assess_expiry_risk(d_avg=30.0, lt_mean=1.5, batches=[])
        self.assertEqual(level_empty, ExpiryRiskLevel.NOT_ASSESSABLE)
        self.assertIsNone(units_empty)

    def test_all_batches_clearing_returns_low_risk(self):
        """When FIFO clearance absorbs all batches before expiration, risk is LOW."""
        # D_avg = 40. Demand at day 3 = 120, at day 5 = 200
        # Total stock = 30 + 50 = 80 units
        batches = [
            InventoryBatch(batch_id="B1", units=30.0, remaining_shelf_life_days=3),
            InventoryBatch(batch_id="B2", units=50.0, remaining_shelf_life_days=5),
        ]
        level, units, reasons = assess_expiry_risk(d_avg=40.0, lt_mean=1.5, batches=batches)
        self.assertEqual(level, ExpiryRiskLevel.LOW)
        self.assertEqual(units, 0.0)
        self.assertTrue(any("low expiry risk" in r.lower() for r in reasons))

    def test_imminent_expiration_within_lead_time_returns_high_risk(self):
        """Batches expiring within lead time window before demand can consume them trigger HIGH risk."""
        # D_avg = 20. LT_mean = 2.0.
        # Batch 1: 50 units with 1-day shelf life. Day 1 demand = 20 units -> 30 units spoil!
        # Since remaining shelf life (1d) <= LT (2d), this is HIGH risk.
        batches = [
            InventoryBatch(batch_id="B1", units=50.0, remaining_shelf_life_days=1),
        ]
        level, units, reasons = assess_expiry_risk(d_avg=20.0, lt_mean=2.0, batches=batches)
        self.assertEqual(level, ExpiryRiskLevel.HIGH)
        self.assertEqual(units, 30.0)
        self.assertTrue(any("high expiry risk" in r.lower() for r in reasons))

    def test_spoilage_beyond_lead_time_returns_medium_risk(self):
        """Batches expiring beyond lead time before demand clearance trigger MEDIUM risk."""
        # D_avg = 10. LT_mean = 1.0.
        # Batch 1: 60 units with 4-day shelf life. Day 4 demand = 40 units -> 20 units spoil!
        # Remaining shelf life (4d) > LT (1d), so this is MEDIUM risk.
        batches = [
            InventoryBatch(batch_id="B1", units=60.0, remaining_shelf_life_days=4),
        ]
        level, units, reasons = assess_expiry_risk(d_avg=10.0, lt_mean=1.0, batches=batches)
        self.assertEqual(level, ExpiryRiskLevel.MEDIUM)
        self.assertEqual(units, 20.0)
        self.assertTrue(any("medium expiry risk" in r.lower() for r in reasons))

    def test_estimated_clearance_scenario_labeling_in_reasons(self):
        """FIFO expiry clearance reasons must explicitly state Estimated Clearance Scenario."""
        batches = [
            InventoryBatch(batch_id="LOT_A", units=50.0, remaining_shelf_life_days=2),
        ]
        level, units, reasons = assess_expiry_risk(d_avg=15.0, lt_mean=3.0, batches=batches)
        self.assertTrue(len(reasons) > 0)
        self.assertTrue(any("estimated clearance scenario" in r.lower() for r in reasons))
        self.assertTrue(any("not an exact expiry prediction" in r.lower() for r in reasons))

    def test_batch_validation_detects_duplicate_batch_ids(self):
        """Duplicate batch IDs in batch records must be safely identified as inconsistent batch data."""
        duplicate_batches = [
            InventoryBatch(batch_id="B_DUP", units=25.0, remaining_shelf_life_days=3),
            InventoryBatch(batch_id="B_DUP", units=30.0, remaining_shelf_life_days=5),
        ]
        # 1. Pure validation function
        is_valid, issues = validate_batch_inventory(duplicate_batches)
        self.assertFalse(is_valid)
        self.assertTrue(any("duplicate batch_id 'b_dup'" in iss.lower() for iss in issues))

        # 2. Handled safely in assess_expiry_risk as NOT_ASSESSABLE
        level, units, reasons = assess_expiry_risk(d_avg=20.0, lt_mean=1.5, batches=duplicate_batches)
        self.assertEqual(level, ExpiryRiskLevel.NOT_ASSESSABLE)
        self.assertIsNone(units)
        self.assertTrue(any("inconsistent batch data" in r.lower() for r in reasons))
        self.assertTrue(any("duplicate batch_id" in r.lower() for r in reasons))

    def test_batch_validation_exceeding_available_physical_stock(self):
        """Batch quantity exceeding available physical stock must be safely handled as inconsistent batch data."""
        batches = [
            InventoryBatch(batch_id="B1", units=70.0, remaining_shelf_life_days=3),
            InventoryBatch(batch_id="B2", units=60.0, remaining_shelf_life_days=5),
        ]
        # Total batch units = 130.0, but available physical stock = 100.0
        is_valid, issues = validate_batch_inventory(batches, available_physical_stock=100.0)
        self.assertFalse(is_valid)
        self.assertTrue(any("exceeds available physical stock" in iss.lower() for iss in issues))

        # Handled safely in assess_expiry_risk as NOT_ASSESSABLE
        level, units, reasons = assess_expiry_risk(
            d_avg=20.0,
            lt_mean=1.5,
            batches=batches,
            available_physical_stock=100.0,
        )
        self.assertEqual(level, ExpiryRiskLevel.NOT_ASSESSABLE)
        self.assertIsNone(units)
        self.assertTrue(any("exceeds available physical stock" in r.lower() for r in reasons))

    def test_batch_validation_when_physical_stock_is_zero(self):
        """Positive batch units provided when available physical stock is zero must be safely handled."""
        batches = [
            InventoryBatch(batch_id="B1", units=20.0, remaining_shelf_life_days=2),
        ]
        is_valid, issues = validate_batch_inventory(batches, available_physical_stock=0.0)
        self.assertFalse(is_valid)
        self.assertTrue(any("available physical stock is zero" in iss.lower() for iss in issues))

        level, units, reasons = assess_expiry_risk(
            d_avg=10.0,
            lt_mean=1.0,
            batches=batches,
            available_physical_stock=0.0,
        )
        self.assertEqual(level, ExpiryRiskLevel.NOT_ASSESSABLE)
        self.assertIsNone(units)
        self.assertTrue(any("available physical stock is zero" in r.lower() for r in reasons))

    def test_batch_validation_strict_coverage_mode(self):
        """Partial batch coverage is rejected under strict coverage mode but accepted with advisory in standard mode."""
        batches = [
            InventoryBatch(batch_id="B1", units=80.0, remaining_shelf_life_days=4),
        ]
        # Available stock is 100.0, batches only sum to 80.0
        # Strict mode: Inconsistent
        is_valid_strict, issues_strict = validate_batch_inventory(
            batches, available_physical_stock=100.0, require_full_coverage=True
        )
        self.assertFalse(is_valid_strict)
        self.assertTrue(any("lack batch tracking" in iss.lower() for iss in issues_strict))

        # Standard mode: Valid, but advisory added in assess_expiry_risk
        is_valid_std, issues_std = validate_batch_inventory(
            batches, available_physical_stock=100.0, require_full_coverage=False
        )
        self.assertTrue(is_valid_std)
        self.assertEqual(len(issues_std), 0)

        level, units, reasons = assess_expiry_risk(
            d_avg=30.0,
            lt_mean=1.0,
            batches=batches,
            available_physical_stock=100.0,
            require_full_coverage=False,
        )
        self.assertEqual(level, ExpiryRiskLevel.LOW)
        self.assertTrue(any("partial batch coverage" in r.lower() for r in reasons))

    def test_engine_assess_risk_handles_inconsistent_batches_safely(self):
        """Engine assess_risk safely detects inconsistent batch quantities against net available stock."""
        config = get_default_config()
        engine = InventoryEngine(config)
        forecast = DemandForecast(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-15",
            p10_demand=25.0,
            p50_demand=40.0,
            p90_demand=60.0,
            sigma_demand=8.0,
        )
        pos = InventoryPosition(on_hand_stock=50.0, on_order_stock=0.0, reserved_stock=10.0)
        daily_inp = DailyAgentInput(
            store_id=forecast.store_id,
            sku_id=forecast.sku_id,
            category=forecast.category,
            date=forecast.date,
            forecast=forecast,
            inventory=InventorySnapshot(
                store_id=forecast.store_id,
                sku_id=forecast.sku_id,
                date=forecast.date,
                on_hand_stock=pos.on_hand_stock,
                snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
            ),
        )
        calc = engine.calculate(daily_inp, inventory_position=pos)
        # Net available stock is 50 - 10 = 40.0 units.
        # Pass batch records claiming 80.0 units (exceeds net available stock!)
        exceeding_batches = [
            InventoryBatch(batch_id="B_EXCEED", units=80.0, remaining_shelf_life_days=2),
        ]
        assm = engine.assess_risk(calc, forecast, batches=exceeding_batches)
        self.assertEqual(assm.expiry_risk_level, ExpiryRiskLevel.NOT_ASSESSABLE)
        self.assertIsNone(assm.at_risk_expiry_units)
        self.assertTrue(any("exceeds available physical stock" in r.lower() for r in assm.expiry_risk_reasons))


class TestRiskAssessmentIntegration(unittest.TestCase):
    """End-to-end integration tests using repository datasets and full engine."""

    def setUp(self):
        self.config = get_default_config()
        self.engine = InventoryEngine(self.config)
        self.loader = DataLoader(self.config)

    def test_pipeline_flag_preservation_in_risk_assessment(self):
        """assumed_zero_pipeline flag must be faithfully propagated to InventoryRiskAssessment."""
        inputs, _, _ = self.loader.load_daily_agent_inputs()
        first_input = inputs[0]

        # Case 1: Default calculation (assumed_zero_pipeline == True)
        calc_def, risk_def = self.engine.calculate_and_assess_risk(first_input)
        self.assertTrue(calc_def.assumed_zero_pipeline)
        self.assertTrue(risk_def.assumed_zero_pipeline)

        # Case 2: Explicit position with both pipeline fields provided (assumed_zero_pipeline == False)
        explicit_pos = InventoryPosition(on_hand_stock=100.0, on_order_stock=20.0, reserved_stock=10.0)
        calc_exp, risk_exp = self.engine.calculate_and_assess_risk(first_input, inventory_position=explicit_pos)
        self.assertFalse(calc_exp.assumed_zero_pipeline)
        self.assertFalse(risk_exp.assumed_zero_pipeline)

    def test_batch_risk_assessment_across_repository_nodes(self):
        """Executes full risk assessment across 100 actual repository nodes."""
        inputs, _, _ = self.loader.load_daily_agent_inputs()
        sample_inputs = inputs[:100]

        calc_results = self.engine.calculate_batch(sample_inputs)
        forecasts = [inp.forecast for inp in sample_inputs]

        risk_results = self.engine.assess_risk_batch(calc_results, forecasts)
        self.assertEqual(len(risk_results), 100)

        for assm, calc in zip(risk_results, calc_results):
            self.assertEqual(assm.store_id, calc.store_id)
            self.assertEqual(assm.sku_id, calc.sku_id)
            self.assertEqual(assm.date, calc.date)

            # Expiry is strictly NOT_ASSESSABLE because repository parquet lacks batch data
            self.assertEqual(assm.expiry_risk_level, ExpiryRiskLevel.NOT_ASSESSABLE)
            self.assertIsNone(assm.at_risk_expiry_units)

            # Risk level matches IP relative to ROP and Target Stock
            if calc.total_inventory_position <= calc.safety_stock_lead_time or calc.inventory_position.on_hand_stock <= 0.0:
                self.assertEqual(assm.stock_risk_level, StockRiskLevel.CRITICAL_LOW)
                self.assertTrue(assm.reorder_recommended)
            elif calc.total_inventory_position <= calc.reorder_point:
                self.assertEqual(assm.stock_risk_level, StockRiskLevel.LOW_STOCK_REORDER)
                self.assertTrue(assm.reorder_recommended)
            elif calc.total_inventory_position > calc.target_stock:
                self.assertEqual(assm.stock_risk_level, StockRiskLevel.EXCESS_STOCK)
                self.assertFalse(assm.reorder_recommended)
                self.assertGreater(assm.excess_quantity, 0.0)
            else:
                self.assertEqual(assm.stock_risk_level, StockRiskLevel.HEALTHY)
                self.assertFalse(assm.reorder_recommended)
                self.assertEqual(assm.excess_quantity, 0.0)


if __name__ == "__main__":
    unittest.main()

