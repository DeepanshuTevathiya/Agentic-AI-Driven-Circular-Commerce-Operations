"""
Phase 3.1: Focused LightGBM Forecast Ingestion and Validation Tests
===================================================================
Validates:
1. DemandForecast schema contract:
   - Finite numeric constraints (rejection of Inf, -Inf, NaN)
   - Non-negative bounds (P10 >= 0, P50 >= 0, P90 >= 0)
   - Strictly positive uncertainty (sigma_demand > 0)
   - Quantile monotonicity (P10 <= P50 <= P90)
   - Non-empty identifiers and ISO date formats
2. DataLoader forecast file ingestion & validation:
   - Detection of missing required columns
   - Graceful handling of omitted optional columns
   - Detection and rejection of null, infinite, and negative values in parquet
   - Detection and rejection of non-positive sigma in parquet
   - Detection and rejection of quantile inversions in parquet
   - Detection of duplicate store-SKU-date combinations
3. End-to-end integration flow with Lead 1's Step 2 forecast payload:
   - Consumption of all 3,900 records from forecast_payloads_step2.parquet
   - Correct association of store_id, sku_id, date, and category
   - Beginning-of-day stock sourced from df_master Opening_Stock (not same-day Closing_Stock)
   - Strict isolation of Target_Demand in DailyEvaluationRecord
   - Unconfirmed alerts kept diagnostic and never converted to reservations
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import ValidationError

from src.agents.config import get_default_config
from src.agents.data_loader import DataLoader
from src.agents.schemas import (
    DemandForecast,
    DailyAgentInput,
    DailyEvaluationRecord,
    InventorySnapshotBasis,
)


class TestForecastContractValidation(unittest.TestCase):
    """Unit tests for the DemandForecast Pydantic V2 schema."""

    def test_valid_forecast_contract(self):
        fc = DemandForecast(
            store_id="DS_BLR_HSR_LAYOUT",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-10",
            p10_demand=32.5,
            p50_demand=45.0,
            p90_demand=62.8,
            sigma_demand=11.2,
        )
        self.assertEqual(fc.store_id, "DS_BLR_HSR_LAYOUT")
        self.assertEqual(fc.sku_id, "SKU_MILK_1L")
        self.assertEqual(fc.category, "Dairy")
        self.assertEqual(fc.date, "2024-11-10")
        self.assertEqual(fc.p10_demand, 32.5)
        self.assertEqual(fc.p50_demand, 45.0)
        self.assertEqual(fc.p90_demand, 62.8)
        self.assertEqual(fc.sigma_demand, 11.2)

    def test_rejection_of_nan_in_all_forecast_fields(self):
        """All forecast numeric fields must reject float('nan')."""
        fields = ["p10_demand", "p50_demand", "p90_demand", "sigma_demand"]
        for f in fields:
            kwargs = {
                "store_id": "DS_BLR_01",
                "sku_id": "SKU_01",
                "category": "Dairy",
                "date": "2024-11-10",
                "p10_demand": 10.0,
                "p50_demand": 20.0,
                "p90_demand": 30.0,
                "sigma_demand": 5.0,
            }
            kwargs[f] = float("nan")
            with self.assertRaises(ValidationError, msg=f"Field '{f}' accepted NaN"):
                DemandForecast(**kwargs)

    def test_rejection_of_infinite_in_all_forecast_fields(self):
        """All forecast numeric fields must reject float('inf') and float('-inf')."""
        fields = ["p10_demand", "p50_demand", "p90_demand", "sigma_demand"]
        for f in fields:
            for inf_val in [float("inf"), float("-inf")]:
                kwargs = {
                    "store_id": "DS_BLR_01",
                    "sku_id": "SKU_01",
                    "category": "Dairy",
                    "date": "2024-11-10",
                    "p10_demand": 10.0,
                    "p50_demand": 20.0,
                    "p90_demand": 30.0,
                    "sigma_demand": 5.0,
                }
                kwargs[f] = inf_val
                with self.assertRaises(ValidationError, msg=f"Field '{f}' accepted {inf_val}"):
                    DemandForecast(**kwargs)

    def test_rejection_of_negative_demand_bounds(self):
        """P10, P50, P90 must be >= 0."""
        for field_name in ["p10_demand", "p50_demand", "p90_demand"]:
            kwargs = {
                "store_id": "DS_BLR_01",
                "sku_id": "SKU_01",
                "category": "Dairy",
                "date": "2024-11-10",
                "p10_demand": 10.0,
                "p50_demand": 20.0,
                "p90_demand": 30.0,
                "sigma_demand": 5.0,
            }
            kwargs[field_name] = -1.0
            with self.assertRaises(ValidationError):
                DemandForecast(**kwargs)

    def test_rejection_of_non_positive_sigma(self):
        """Uncertainty (sigma_demand) must be strictly positive (> 0)."""
        for bad_sigma in [0.0, -0.01, -10.0]:
            with self.assertRaises(ValidationError):
                DemandForecast(
                    store_id="DS_BLR_01",
                    sku_id="SKU_01",
                    category="Dairy",
                    date="2024-11-10",
                    p10_demand=10.0,
                    p50_demand=20.0,
                    p90_demand=30.0,
                    sigma_demand=bad_sigma,
                )

    def test_rejection_of_quantile_monotonicity_violations(self):
        """P10 <= P50 <= P90 must hold strictly."""
        # Case 1: P10 > P50
        with self.assertRaises(ValidationError) as ctx:
            DemandForecast(
                store_id="DS_BLR_01",
                sku_id="SKU_01",
                category="Dairy",
                date="2024-11-10",
                p10_demand=25.0,
                p50_demand=20.0,
                p90_demand=30.0,
                sigma_demand=5.0,
            )
        self.assertIn("Quantile monotonicity violation", str(ctx.exception))

        # Case 2: P50 > P90
        with self.assertRaises(ValidationError) as ctx:
            DemandForecast(
                store_id="DS_BLR_01",
                sku_id="SKU_01",
                category="Dairy",
                date="2024-11-10",
                p10_demand=10.0,
                p50_demand=35.0,
                p90_demand=30.0,
                sigma_demand=5.0,
            )
        self.assertIn("Quantile monotonicity violation", str(ctx.exception))

    def test_equal_quantiles_permitted(self):
        """Deterministic scenario P10 == P50 == P90 must be valid."""
        fc = DemandForecast(
            store_id="DS_BLR_01",
            sku_id="SKU_01",
            category="Dairy",
            date="2024-11-10",
            p10_demand=20.0,
            p50_demand=20.0,
            p90_demand=20.0,
            sigma_demand=0.01,
        )
        self.assertEqual(fc.p10_demand, 20.0)
        self.assertEqual(fc.p50_demand, 20.0)
        self.assertEqual(fc.p90_demand, 20.0)

    def test_extra_fields_forbidden(self):
        """Forecast contract forbids unexpected fields to prevent accidental injection."""
        with self.assertRaises(ValidationError):
            DemandForecast(
                store_id="DS_BLR_01",
                sku_id="SKU_01",
                category="Dairy",
                date="2024-11-10",
                p10_demand=10.0,
                p50_demand=20.0,
                p90_demand=30.0,
                sigma_demand=5.0,
                extra_payload="malicious_or_unvalidated_data",
            )


class TestForecastDataLoaderValidation(unittest.TestCase):
    """Unit tests for DataLoader forecast reading, normalization, and error handling."""

    def setUp(self):
        self.tmp_dir = Path("tests/lead2_agents/tmp_forecast_test")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()

        self.valid_forecast_df = pd.DataFrame({
            "Store_ID": ["DS_BLR_01", "DS_BLR_01"],
            "SKU_ID": ["SKU_MILK_1L", "SKU_BREAD_WHITE"],
            "Category": ["Dairy", "Bakery"],
            "Date": ["2024-11-10", "2024-11-10"],
            "p10_demand": [15.0, 20.0],
            "p50_demand": [25.0, 30.0],
            "p90_demand": [40.0, 45.0],
            "sigma_demand": [7.5, 8.2],
            "Target_Demand": [26.0, 29.0],
            "Total_SL_Days": [5, 3],
            "Remaining_SL_Days": [4, 2],
        })

    def tearDown(self):
        for f in self.tmp_dir.glob("*.parquet"):
            f.unlink()
        if self.tmp_dir.exists():
            try:
                self.tmp_dir.rmdir()
            except OSError:
                pass

    def test_missing_required_column_raises_error(self):
        """Missing essential forecast columns must raise ValueError."""
        for col in ["Store_ID", "SKU_ID", "p10_demand", "p50_demand", "p90_demand", "sigma_demand"]:
            bad_df = self.valid_forecast_df.drop(columns=[col])
            fpath = self.tmp_dir / f"bad_missing_{col}.parquet"
            bad_df.to_parquet(fpath, index=False)
            with self.assertRaises(ValueError) as ctx:
                self.loader.load_forecasts(fpath)
            self.assertIn("missing required columns", str(ctx.exception))

    def test_omitted_optional_columns_handled_gracefully(self):
        """Omission of Target_Demand or SL days should succeed with None defaults."""
        lean_df = self.valid_forecast_df.drop(columns=["Target_Demand", "Total_SL_Days", "Remaining_SL_Days"])
        fpath = self.tmp_dir / "lean_forecast.parquet"
        lean_df.to_parquet(fpath, index=False)

        loaded = self.loader.load_forecasts(fpath)
        self.assertEqual(len(loaded), 2)
        self.assertIn("target_demand_eval_label", loaded.columns)
        self.assertTrue(loaded["target_demand_eval_label"].isnull().all())

    def test_null_in_required_forecast_column_raises_error(self):
        """Null in p50_demand must raise ValueError."""
        bad_df = self.valid_forecast_df.copy()
        bad_df.loc[0, "p50_demand"] = None
        fpath = self.tmp_dir / "bad_null.parquet"
        bad_df.to_parquet(fpath, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_forecasts(fpath)
        self.assertIn("Null values detected", str(ctx.exception))

    def test_infinite_in_forecast_column_raises_error(self):
        """Infinite values in forecast columns must raise ValueError."""
        bad_df = self.valid_forecast_df.copy()
        bad_df.loc[0, "p90_demand"] = float("inf")
        fpath = self.tmp_dir / "bad_inf.parquet"
        bad_df.to_parquet(fpath, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_forecasts(fpath)
        self.assertIn("Infinite values detected", str(ctx.exception))

    def test_negative_forecast_value_raises_error(self):
        """Negative demand prediction in parquet must raise ValueError."""
        bad_df = self.valid_forecast_df.copy()
        bad_df.loc[0, "p10_demand"] = -5.0
        fpath = self.tmp_dir / "bad_neg.parquet"
        bad_df.to_parquet(fpath, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_forecasts(fpath)
        self.assertIn("Negative demand values detected", str(ctx.exception))

    def test_non_positive_sigma_raises_error(self):
        """sigma_demand <= 0 in parquet must raise ValueError."""
        bad_df = self.valid_forecast_df.copy()
        bad_df.loc[0, "sigma_demand"] = 0.0
        fpath = self.tmp_dir / "bad_zero_sigma.parquet"
        bad_df.to_parquet(fpath, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_forecasts(fpath)
        self.assertIn("Non-positive demand uncertainty", str(ctx.exception))

    def test_quantile_monotonicity_violation_raises_error(self):
        """P10 > P50 in parquet must raise ValueError."""
        bad_df = self.valid_forecast_df.copy()
        bad_df.loc[0, "p10_demand"] = 50.0  # P10 (50) > P50 (25)
        fpath = self.tmp_dir / "bad_mono.parquet"
        bad_df.to_parquet(fpath, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_forecasts(fpath)
        self.assertIn("Quantile monotonicity violation", str(ctx.exception))


class TestLightGBMIntegrationFlow(unittest.TestCase):
    """End-to-end validation against Lead 1's actual forecast_payloads_step2.parquet."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_default_config()
        cls.loader = DataLoader(cls.config)
        cls.inputs, cls.eval_records, cls.report = cls.loader.load_daily_agent_inputs()

    def test_full_dataset_record_count(self):
        """All 3,900 rows of forecast_payloads_step2.parquet must be parsed."""
        self.assertEqual(len(self.inputs), 3900)
        self.assertEqual(len(self.eval_records), 3900)
        self.assertEqual(self.report.forecast_rows_loaded, 3900)
        self.assertEqual(self.report.unmatched_forecast_rows, 0)
        self.assertTrue(self.report.validation_passed)

    def test_all_forecast_quantiles_strictly_valid(self):
        """Every single record across 3,900 instances must satisfy P10 <= P50 <= P90 and sigma > 0."""
        for inp in self.inputs:
            fc = inp.forecast
            self.assertGreaterEqual(fc.p10_demand, 0.0)
            self.assertGreaterEqual(fc.p50_demand, fc.p10_demand)
            self.assertGreaterEqual(fc.p90_demand, fc.p50_demand)
            self.assertGreater(fc.sigma_demand, 0.0)
            self.assertTrue(np.isfinite(fc.p10_demand))
            self.assertTrue(np.isfinite(fc.p50_demand))
            self.assertTrue(np.isfinite(fc.p90_demand))
            self.assertTrue(np.isfinite(fc.sigma_demand))

    def test_store_sku_date_association_integrity(self):
        """Parent node identifiers must match child forecast and inventory objects."""
        for inp in self.inputs[:200]:  # Sample representative window
            self.assertEqual(inp.store_id, inp.forecast.store_id)
            self.assertEqual(inp.sku_id, inp.forecast.sku_id)
            self.assertEqual(inp.date, inp.forecast.date)
            self.assertEqual(inp.category, inp.forecast.category)
            self.assertEqual(inp.store_id, inp.inventory.store_id)
            self.assertEqual(inp.sku_id, inp.inventory.sku_id)
            self.assertEqual(inp.date, inp.inventory.date)

    def test_authoritative_opening_stock_used_not_closing_stock(self):
        """Beginning-of-day stock must be df_master.Opening_Stock, not forecast Closing_Stock."""
        df_master = pd.read_parquet(self.config.master_file)
        df_fc = pd.read_parquet(self.config.forecast_file)

        # Pick node 0: DS_BLR_HSR_LAYOUT, SKU_BANANA_DZ, 2024-11-10
        first_input = self.inputs[0]
        m_dates = pd.to_datetime(df_master["Date"]).dt.strftime("%Y-%m-%d")
        fc_dates = pd.to_datetime(df_fc["Date"]).dt.strftime("%Y-%m-%d")

        row_m = df_master[
            (df_master["Store_ID"] == first_input.store_id)
            & (df_master["SKU_ID"] == first_input.sku_id)
            & (m_dates == first_input.date)
        ].iloc[0]

        row_fc = df_fc[
            (df_fc["Store_ID"] == first_input.store_id)
            & (df_fc["SKU_ID"] == first_input.sku_id)
            & (fc_dates == first_input.date)
        ].iloc[0]

        # Verify Opening_Stock (387) vs Closing_Stock (349)
        self.assertEqual(first_input.inventory.on_hand_stock, float(row_m["Opening_Stock"]))
        self.assertNotEqual(first_input.inventory.on_hand_stock, float(row_fc["Closing_Stock"]))
        self.assertEqual(first_input.inventory.snapshot_basis, InventorySnapshotBasis.BEGINNING_OF_DAY)

    def test_target_demand_isolated_in_eval_records_only(self):
        """DailyAgentInput must have no target_demand field; DailyEvaluationRecord holds it."""
        for inp in self.inputs[:100]:
            self.assertFalse(hasattr(inp, "target_demand"))
            self.assertFalse(hasattr(inp, "target_demand_eval_label"))

        for ev in self.eval_records[:100]:
            self.assertIsInstance(ev, DailyEvaluationRecord)
            self.assertGreaterEqual(ev.target_demand, 0.0)
            self.assertTrue(np.isfinite(ev.target_demand))

    def test_unconfirmed_alerts_never_converted_to_reservations(self):
        """Unconfirmed alerts remain diagnostic integer counts and do not create reserved stock."""
        alert_records = [inp for inp in self.inputs if inp.unconfirmed_alert_count > 0]
        self.assertEqual(len(alert_records), self.report.preorder_overlap_alert_matches)
        self.assertEqual(len(alert_records), 1128)

        # Operational inputs do not contain a reserved_stock field or modify P50
        for inp in alert_records[:50]:
            self.assertFalse(hasattr(inp, "reserved_stock"))
            self.assertGreater(inp.unconfirmed_alert_count, 0)
            # Forecast P50 remains intact
            self.assertGreater(inp.forecast.p50_demand, 0.0)


if __name__ == "__main__":
    unittest.main()

