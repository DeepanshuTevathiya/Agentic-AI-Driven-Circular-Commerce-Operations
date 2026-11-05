"""
Unit Tests: Lead 2 Phase 3.2 Core Inventory Engine Calculations
===============================================================
Validates:
1. Pure calculation functions against exact analytical baselines:
   - compute_average_daily_demand
   - compute_safety_stock (Lead-time SS: Z * sqrt(LT_mean * sigma_D^2 + D_avg^2 * sigma_LT^2))
   - compute_protection_safety_stock (Protection-period SS: Z * sqrt((LT_mean + R) * sigma_D^2 + D_avg^2 * sigma_LT^2))
   - compute_reorder_point ((D_avg * LT_mean) + SS_LT)
   - compute_target_stock (D_avg * (R + LT_mean) + SS_P)
   - compute_suggested_replenishment_qty
2. Mathematical relationships:
   - When R == 0: SS_P == SS_LT and Target_Stock == ROP
   - When R > 0: SS_P > SS_LT and Target_Stock > ROP
3. Edge cases:
   - Zero average daily demand
   - Deterministic lead time (sigma_LT = 0)
   - Zero demand uncertainty (sigma_D = 0)
   - Service level escalation (Z=1.960 vs Z=1.645)
4. Input validation & error handling:
   - Rejection of negative demand, negative sigma_D, non-positive LT_mean
   - Rejection of negative sigma_LT, negative Z, negative review period
   - Rejection of NaN and Infinite values
5. InventoryEngine class & pipeline integration:
   - Default beginning-of-day stock mapping with explicit simulation flag (assumed_zero_pipeline)
   - Explicit InventoryPosition handling (on-hand, on-order, reserved)
   - Independence from Target_Demand and diagnostic household alerts
   - Batch calculation across real repository data
"""

import math
import unittest
import numpy as np
from pydantic import ValidationError

from src.agents.config import Lead2Config, SupplierSimulationParams, get_default_config
from src.agents.data_loader import DataLoader
from src.agents.inventory_engine import (
    InventoryEngine,
    compute_average_daily_demand,
    compute_protection_safety_stock,
    compute_reorder_point,
    compute_safety_stock,
    compute_suggested_replenishment_qty,
    compute_target_stock,
)
from src.agents.schemas import (
    DailyAgentInput,
    DemandForecast,
    InventoryCalculationResult,
    InventoryPosition,
    InventorySnapshot,
    InventorySnapshotBasis,
)


class TestInventoryMathFunctions(unittest.TestCase):
    """Isolated unit tests for mathematical inventory formulas."""

    def setUp(self):
        # Canonical baseline inputs
        self.d_avg = 40.0
        self.sigma_d = 8.0
        self.lt_mean = 1.5
        self.sigma_lt = 0.3
        self.z = 1.645
        self.review_period = 1.0

        # Exact analytical derivation:
        # 1. Lead-time safety stock (for ROP):
        # var_demand_lt = 1.5 * (8.0^2) = 1.5 * 64.0 = 96.0
        # var_lt = (40.0^2) * (0.3^2) = 1600.0 * 0.09 = 144.0
        # total_var_lt = 96.0 + 144.0 = 240.0
        # sqrt(240.0) = 15.491933384829668
        # SS_LT = 1.645 * 15.491933384829668 = 25.484230418044804
        # LTD = 40.0 * 1.5 = 60.0
        # ROP = 60.0 + 25.484230418044804 = 85.4842304180448
        #
        # 2. Protection-period safety stock (for Target Stock):
        # exposure_horizon = 1.5 + 1.0 = 2.5
        # var_demand_p = 2.5 * (8.0^2) = 2.5 * 64.0 = 160.0
        # var_lt = 144.0
        # total_var_p = 160.0 + 144.0 = 304.0
        # sqrt(304.0) = 17.435595774162696
        # SS_P = 1.645 * 17.435595774162696 = 28.681555048497635
        # Target Stock S = 40.0 * 2.5 + SS_P = 100.0 + 28.681555048497635 = 128.68155504849763
        self.expected_ss_lt = 25.484230418044804
        self.expected_ss_p = 28.681555048497635
        self.expected_rop = 85.4842304180448
        self.expected_target_stock = 128.68155504849763

    def test_lead_time_safety_stock_exact_calculation(self):
        ss = compute_safety_stock(
            d_avg=self.d_avg,
            sigma_d=self.sigma_d,
            lt_mean=self.lt_mean,
            sigma_lt=self.sigma_lt,
            z=self.z,
        )
        self.assertAlmostEqual(ss, self.expected_ss_lt, places=6)

    def test_protection_safety_stock_exact_calculation(self):
        ss_p = compute_protection_safety_stock(
            d_avg=self.d_avg,
            sigma_d=self.sigma_d,
            lt_mean=self.lt_mean,
            review_period=self.review_period,
            sigma_lt=self.sigma_lt,
            z=self.z,
        )
        self.assertAlmostEqual(ss_p, self.expected_ss_p, places=6)

    def test_protection_safety_stock_greater_than_lead_time_safety_stock_when_r_positive(self):
        """When R > 0 and sigma_D > 0, SS_P must be strictly greater than SS_LT."""
        self.assertGreater(self.expected_ss_p, self.expected_ss_lt)

    def test_safety_stock_equality_when_sigma_d_zero_regardless_of_review_period(self):
        """
        Boundary condition: When sigma_D == 0, SS_P must be identically equal to SS_LT,
        even when review period R > 0 (e.g. R = 1.0, R = 7.0).
        """
        for r_val in [0.0, 1.0, 3.5, 7.0, 14.0]:
            ss_lt = compute_safety_stock(
                d_avg=self.d_avg,
                sigma_d=0.0,  # Zero demand uncertainty
                lt_mean=self.lt_mean,
                sigma_lt=self.sigma_lt,
                z=self.z,
            )
            ss_p = compute_protection_safety_stock(
                d_avg=self.d_avg,
                sigma_d=0.0,  # Zero demand uncertainty
                lt_mean=self.lt_mean,
                review_period=r_val,
                sigma_lt=self.sigma_lt,
                z=self.z,
            )
            self.assertEqual(
                ss_p,
                ss_lt,
                msg=f"SS_P ({ss_p}) must equal SS_LT ({ss_lt}) when sigma_D=0 at R={r_val}"
            )

    def test_protection_safety_stock_equals_lead_time_safety_stock_when_r_zero(self):
        """When review period R == 0 (continuous review), SS_P reduces identically to SS_LT."""
        ss_p_zero = compute_protection_safety_stock(
            d_avg=self.d_avg,
            sigma_d=self.sigma_d,
            lt_mean=self.lt_mean,
            review_period=0.0,
            sigma_lt=self.sigma_lt,
            z=self.z,
        )
        self.assertAlmostEqual(ss_p_zero, self.expected_ss_lt, places=6)

    def test_reorder_point_uses_lead_time_safety_stock(self):
        rop = compute_reorder_point(
            d_avg=self.d_avg,
            lt_mean=self.lt_mean,
            safety_stock=self.expected_ss_lt,
        )
        self.assertAlmostEqual(rop, self.expected_rop, places=6)

    def test_target_stock_uses_protection_safety_stock(self):
        ts = compute_target_stock(
            d_avg=self.d_avg,
            lt_mean=self.lt_mean,
            review_period=self.review_period,
            safety_stock=self.expected_ss_p,
        )
        self.assertAlmostEqual(ts, self.expected_target_stock, places=6)

    def test_target_stock_equals_rop_when_r_zero(self):
        """When review period is 0, Target Stock equals ROP."""
        ts_zero_r = compute_target_stock(
            d_avg=self.d_avg,
            lt_mean=self.lt_mean,
            review_period=0.0,
            safety_stock=self.expected_ss_lt,
        )
        self.assertAlmostEqual(ts_zero_r, self.expected_rop, places=6)

    def test_replenishment_trigger_when_ip_below_rop(self):
        # IP = 80.0 <= ROP (85.484) -> reorder triggered
        triggered, qty = compute_suggested_replenishment_qty(
            total_inventory_position=80.0,
            rop=self.expected_rop,
            target_stock=self.expected_target_stock,
        )
        self.assertTrue(triggered)
        # Order up to target stock: 128.681555 - 80.0 = 48.681555
        self.assertAlmostEqual(qty, self.expected_target_stock - 80.0, places=6)

    def test_replenishment_suppressed_when_ip_above_rop(self):
        # IP = 95.0 > ROP (85.484) -> no reorder
        triggered, qty = compute_suggested_replenishment_qty(
            total_inventory_position=95.0,
            rop=self.expected_rop,
            target_stock=self.expected_target_stock,
        )
        self.assertFalse(triggered)
        self.assertEqual(qty, 0.0)

    def test_replenishment_boundary_exact_match(self):
        # IP == ROP -> reorder triggered
        triggered, qty = compute_suggested_replenishment_qty(
            total_inventory_position=self.expected_rop,
            rop=self.expected_rop,
            target_stock=self.expected_target_stock,
        )
        self.assertTrue(triggered)
        self.assertAlmostEqual(qty, self.expected_target_stock - self.expected_rop, places=6)


class TestInventoryEdgeCases(unittest.TestCase):
    """Validates mathematical edge cases and boundary conditions."""

    def test_zero_daily_demand(self):
        """When D_avg = 0, SS reflects pure demand variability during exposure period."""
        ss_lt = compute_safety_stock(
            d_avg=0.0,
            sigma_d=5.0,
            lt_mean=2.0,
            sigma_lt=0.5,
            z=1.645,
        )
        # var_lt = 2.0 * (5^2) + 0 = 50.0
        expected_lt = 1.645 * math.sqrt(50.0)
        self.assertAlmostEqual(ss_lt, expected_lt, places=6)

        ss_p = compute_protection_safety_stock(
            d_avg=0.0,
            sigma_d=5.0,
            lt_mean=2.0,
            review_period=1.0,
            sigma_lt=0.5,
            z=1.645,
        )
        # var_p = (2.0 + 1.0) * (5^2) + 0 = 75.0
        expected_p = 1.645 * math.sqrt(75.0)
        self.assertAlmostEqual(ss_p, expected_p, places=6)
        self.assertGreater(ss_p, ss_lt)

        rop = compute_reorder_point(d_avg=0.0, lt_mean=2.0, safety_stock=ss_lt)
        self.assertAlmostEqual(rop, ss_lt, places=6)

        ts = compute_target_stock(d_avg=0.0, lt_mean=2.0, review_period=1.0, safety_stock=ss_p)
        self.assertAlmostEqual(ts, ss_p, places=6)

    def test_zero_uncertainty_deterministic(self):
        """When sigma_D = 0 and sigma_LT = 0, both SS values are 0."""
        ss_lt = compute_safety_stock(d_avg=50.0, sigma_d=0.0, lt_mean=2.0, sigma_lt=0.0, z=1.645)
        ss_p = compute_protection_safety_stock(d_avg=50.0, sigma_d=0.0, lt_mean=2.0, review_period=1.0, sigma_lt=0.0, z=1.645)
        self.assertEqual(ss_lt, 0.0)
        self.assertEqual(ss_p, 0.0)

        rop = compute_reorder_point(d_avg=50.0, lt_mean=2.0, safety_stock=ss_lt)
        self.assertEqual(rop, 100.0)

        ts = compute_target_stock(d_avg=50.0, lt_mean=2.0, review_period=1.0, safety_stock=ss_p)
        self.assertEqual(ts, 150.0)

    def test_deterministic_lead_time_variability(self):
        """When sigma_LT = 0 (fixed supplier lead time), formula reduces to Z * sigma_D * sqrt(T)."""
        ss_lt = compute_safety_stock(d_avg=40.0, sigma_d=6.0, lt_mean=4.0, sigma_lt=0.0, z=2.0)
        # SS = 2.0 * sqrt(4.0 * 36.0 + 0) = 2.0 * 12.0 = 24.0
        self.assertEqual(ss_lt, 24.0)

        ss_p = compute_protection_safety_stock(d_avg=40.0, sigma_d=6.0, lt_mean=4.0, review_period=5.0, sigma_lt=0.0, z=2.0)
        # SS_P = 2.0 * sqrt((4 + 5) * 36.0) = 2.0 * sqrt(324) = 2.0 * 18.0 = 36.0
        self.assertEqual(ss_p, 36.0)

    def test_service_level_escalation(self):
        """Higher Z-score (97.5% vs 95%) strictly increases safety stock."""
        ss_95 = compute_safety_stock(d_avg=30.0, sigma_d=5.0, lt_mean=1.5, sigma_lt=0.3, z=1.645)
        ss_975 = compute_safety_stock(d_avg=30.0, sigma_d=5.0, lt_mean=1.5, sigma_lt=0.3, z=1.960)
        self.assertGreater(ss_975, ss_95)


class TestInventoryErrorHandling(unittest.TestCase):
    """Validates parameter bounds, non-negativity, and finite value constraints."""

    def test_negative_parameters_raise_value_error(self):
        with self.assertRaises(ValueError):
            compute_safety_stock(d_avg=-1.0, sigma_d=5.0, lt_mean=1.5, sigma_lt=0.3, z=1.645)
        with self.assertRaises(ValueError):
            compute_protection_safety_stock(d_avg=10.0, sigma_d=-0.5, lt_mean=1.5, review_period=1.0, sigma_lt=0.3, z=1.645)
        with self.assertRaises(ValueError):
            compute_protection_safety_stock(d_avg=10.0, sigma_d=5.0, lt_mean=1.5, review_period=-1.0, sigma_lt=0.3, z=1.645)
        with self.assertRaises(ValueError):
            compute_safety_stock(d_avg=10.0, sigma_d=5.0, lt_mean=0.0, sigma_lt=0.3, z=1.645)
        with self.assertRaises(ValueError):
            compute_safety_stock(d_avg=10.0, sigma_d=5.0, lt_mean=-1.5, sigma_lt=0.3, z=1.645)
        with self.assertRaises(ValueError):
            compute_safety_stock(d_avg=10.0, sigma_d=5.0, lt_mean=1.5, sigma_lt=-0.1, z=1.645)
        with self.assertRaises(ValueError):
            compute_safety_stock(d_avg=10.0, sigma_d=5.0, lt_mean=1.5, sigma_lt=0.3, z=-1.645)
        with self.assertRaises(ValueError):
            compute_target_stock(d_avg=10.0, lt_mean=1.5, review_period=-1.0, safety_stock=10.0)

    def test_non_finite_values_raise_value_error(self):
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                compute_safety_stock(d_avg=bad_val, sigma_d=5.0, lt_mean=1.5, sigma_lt=0.3, z=1.645)
            with self.assertRaises(ValueError):
                compute_protection_safety_stock(d_avg=10.0, sigma_d=bad_val, lt_mean=1.5, review_period=1.0, sigma_lt=0.3, z=1.645)
            with self.assertRaises(ValueError):
                compute_reorder_point(d_avg=bad_val, lt_mean=1.5, safety_stock=10.0)
            with self.assertRaises(ValueError):
                compute_target_stock(d_avg=10.0, lt_mean=1.5, review_period=bad_val, safety_stock=10.0)
            with self.assertRaises(ValueError):
                compute_suggested_replenishment_qty(total_inventory_position=bad_val, rop=50.0, target_stock=100.0)


class TestInventoryEngineIntegration(unittest.TestCase):
    """Validates InventoryEngine with DailyAgentInput and repository data."""

    def setUp(self):
        self.config = get_default_config()
        self.engine = InventoryEngine(self.config)

        self.forecast = DemandForecast(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-15",
            p10_demand=30.0,
            p50_demand=45.0,
            p90_demand=65.0,
            sigma_demand=13.65,
        )
        self.snapshot = InventorySnapshot(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            date="2024-11-15",
            on_hand_stock=100.0,
            snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
        )
        self.daily_input = DailyAgentInput(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-15",
            forecast=self.forecast,
            inventory=self.snapshot,
            unconfirmed_alert_count=5,
        )

    def test_default_inventory_position_handling(self):
        """When inventory_position is not provided, defaults on-order/reserved to 0 and flags simulation assumption."""
        result = self.engine.calculate(self.daily_input)

        self.assertIsInstance(result, InventoryCalculationResult)
        self.assertEqual(result.store_id, "DS_BLR_KORAMANGALA")
        self.assertEqual(result.sku_id, "SKU_MILK_1L")
        self.assertEqual(result.date, "2024-11-15")
        self.assertEqual(result.average_daily_demand, 45.0)

        # Simulation assumption flag verified
        self.assertTrue(result.assumed_zero_pipeline)

        # Trusted physical stock used
        self.assertEqual(result.inventory_position.on_hand_stock, 100.0)
        self.assertEqual(result.inventory_position.reserved_stock, 0.0)
        self.assertEqual(result.inventory_position.on_order_stock, 0.0)
        self.assertEqual(result.total_inventory_position, 100.0)

        # Dual safety stock verification
        self.assertGreater(result.safety_stock_protection, result.safety_stock_lead_time)
        self.assertEqual(result.safety_stock, result.safety_stock_lead_time)

        # Unconfirmed alerts must NOT be reserved
        self.assertNotEqual(result.inventory_position.reserved_stock, 5.0)

    def test_explicit_inventory_position_handling(self):
        """When both incoming stock and reservations are explicitly provided, assumed_zero_pipeline is False."""
        custom_pos = InventoryPosition(
            on_hand_stock=100.0,
            reserved_stock=20.0,
            on_order_stock=40.0,
        )
        result = self.engine.calculate(self.daily_input, inventory_position=custom_pos)

        # Simulation flag is False when caller supplies explicit position with both fields
        self.assertFalse(result.assumed_zero_pipeline)
        # Total IP = 100 + 40 - 20 = 120.0
        self.assertEqual(result.total_inventory_position, 120.0)
        self.assertEqual(result.net_available_stock, 80.0)

    def test_pipeline_flag_object_existence_alone_does_not_determine_flag(self):
        """
        Verify that assumed_zero_pipeline=False ONLY when BOTH incoming stock and reservations
        are explicitly provided and valid. Mere object existence must NOT clear the flag.
        """
        # Case 1: Object created with on_hand only -> on_order and reserved were defaulted
        pos_on_hand_only = InventoryPosition(on_hand_stock=100.0)
        res_1 = self.engine.calculate(self.daily_input, inventory_position=pos_on_hand_only)
        self.assertTrue(
            res_1.assumed_zero_pipeline,
            "Object existence alone must NOT clear the assumption flag if pipeline fields were defaulted."
        )

        # Case 2: Object with on_order provided, but reserved_stock omitted/defaulted
        pos_on_order_only = InventoryPosition(on_hand_stock=100.0, on_order_stock=30.0)
        res_2 = self.engine.calculate(self.daily_input, inventory_position=pos_on_order_only)
        self.assertTrue(
            res_2.assumed_zero_pipeline,
            "Flag must remain True if reserved_stock was not explicitly provided."
        )

        # Case 3: Object with reserved provided, but on_order_stock omitted/defaulted
        pos_reserved_only = InventoryPosition(on_hand_stock=100.0, reserved_stock=15.0)
        res_3 = self.engine.calculate(self.daily_input, inventory_position=pos_reserved_only)
        self.assertTrue(
            res_3.assumed_zero_pipeline,
            "Flag must remain True if on_order_stock was not explicitly provided."
        )

        # Case 4: Object with BOTH on_order and reserved explicitly provided
        pos_both = InventoryPosition(on_hand_stock=100.0, on_order_stock=30.0, reserved_stock=15.0)
        res_4 = self.engine.calculate(self.daily_input, inventory_position=pos_both)
        self.assertFalse(
            res_4.assumed_zero_pipeline,
            "Flag must be False when both incoming stock and reservations are explicitly provided."
        )

        # Case 5: Passing explicit kwargs without an InventoryPosition object
        res_kwargs_both = self.engine.calculate(self.daily_input, on_order_stock=30.0, reserved_stock=15.0)
        self.assertFalse(res_kwargs_both.assumed_zero_pipeline)

        # Case 6: Passing only on_order_stock via kwargs
        res_kwargs_partial = self.engine.calculate(self.daily_input, on_order_stock=30.0)
        self.assertTrue(res_kwargs_partial.assumed_zero_pipeline)

    def test_category_supplier_params_applied(self):
        """Dairy category simulation parameters (LT=1.5, sigma_LT=0.30) are correctly mapped."""
        result = self.engine.calculate(self.daily_input)
        self.assertEqual(result.lead_time_mean_days, 1.5)
        self.assertEqual(result.lead_time_std_days, 0.30)
        self.assertEqual(result.review_period_days, 1.0)
        self.assertEqual(result.z_score, 1.645)

    def test_custom_z_score_applied(self):
        """Supplying contingency Z=1.960 escalates both safety stock values."""
        res_normal = self.engine.calculate(self.daily_input, z_score=1.645)
        res_contingency = self.engine.calculate(self.daily_input, z_score=1.960)

        self.assertGreater(res_contingency.safety_stock_lead_time, res_normal.safety_stock_lead_time)
        self.assertGreater(res_contingency.safety_stock_protection, res_normal.safety_stock_protection)
        self.assertGreater(res_contingency.reorder_point, res_normal.reorder_point)
        self.assertGreater(res_contingency.target_stock, res_normal.target_stock)

    def test_batch_calculation_across_repository_dataset(self):
        """Executes calculate_batch over the first 100 real repository inputs."""
        loader = DataLoader(self.config)
        inputs, _, _ = loader.load_daily_agent_inputs()

        sample_inputs = inputs[:100]
        results = self.engine.calculate_batch(sample_inputs)

        self.assertEqual(len(results), 100)
        for res, inp in zip(results, sample_inputs):
            self.assertEqual(res.store_id, inp.store_id)
            self.assertEqual(res.sku_id, inp.sku_id)
            self.assertEqual(res.date, inp.date)
            self.assertEqual(res.average_daily_demand, inp.forecast.p50_demand)
            self.assertGreater(res.safety_stock_lead_time, 0.0)
            self.assertGreater(res.safety_stock_protection, res.safety_stock_lead_time)
            self.assertGreater(res.reorder_point, res.safety_stock_lead_time)
            self.assertGreater(res.target_stock, res.reorder_point)
            self.assertGreaterEqual(res.suggested_replenishment_qty, 0.0)


if __name__ == "__main__":
    unittest.main()
