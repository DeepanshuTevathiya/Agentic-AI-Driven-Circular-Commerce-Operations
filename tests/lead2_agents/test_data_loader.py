"""
Unit & Integration Tests: Lead 2 Data Ingestion & DataLoader
============================================================
Tests:
- Duplicate key detection in forecast and master data
- Missing inventory match detection
- Correct Opening_Stock mapping (rejection of same-day Closing_Stock)
- Verification that unconfirmed alerts do NOT alter P50 or create reservations
- Date and identifier join consistency
- Separation of evaluation records from operational inputs
- End-to-end integration test against actual repository Parquet files
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.agents.config import Lead2Config, get_default_config
from src.agents.data_loader import DataLoader, IngestionAuditReport
from src.agents.schemas import (
    DailyEvaluationRecord,
    InventorySnapshotBasis,
)


class TestDataLoaderSynthetic(unittest.TestCase):
    """Synthetic unit tests isolating data loader edge cases and validation rules."""

    def setUp(self):
        self.tmp_dir = Path("tests/lead2_agents/tmp_test_data")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Baseline synthetic forecast (2 records)
        self.fc_df = pd.DataFrame({
            "Store_ID": ["DS_BLR_01", "DS_BLR_01"],
            "SKU_ID": ["SKU_MILK_1L", "SKU_MILK_1L"],
            "Category": ["Dairy", "Dairy"],
            "Date": ["2024-11-10", "2024-11-11"],
            "Closing_Stock": [80, 75],
            "Remaining_SL_Days": [4, 3],
            "Total_SL_Days": [5, 5],
            "Target_Demand": [42.0, 39.0],
            "p10_demand": [30.0, 28.0],
            "p50_demand": [40.0, 38.0],
            "p90_demand": [55.0, 50.0],
            "sigma_demand": [9.75, 8.58],
        })

        # Baseline synthetic master inventory (2 records, authoritative Opening_Stock)
        self.inv_df = pd.DataFrame({
            "Store_ID": ["DS_BLR_01", "DS_BLR_01"],
            "SKU_ID": ["SKU_MILK_1L", "SKU_MILK_1L"],
            "Date": ["2024-11-10", "2024-11-11"],
            "Opening_Stock": [120, 80],
            "Closing_Stock": [80, 75],
        })

        # Baseline diagnostic alerts
        self.alerts_df = pd.DataFrame({
            "store_id": ["DS_BLR_01"],
            "sku_id": ["SKU_MILK_1L"],
            "current_date": ["2024-11-10"],
            "committed_preorder_units": [3],
        })

        self.fc_path = self.tmp_dir / "test_forecast.parquet"
        self.inv_path = self.tmp_dir / "test_master.parquet"
        self.alerts_path = self.tmp_dir / "test_alerts.parquet"

        self.fc_df.to_parquet(self.fc_path, index=False)
        self.inv_df.to_parquet(self.inv_path, index=False)
        self.alerts_df.to_parquet(self.alerts_path, index=False)

        self.loader = DataLoader()

    def tearDown(self):
        for f in [self.fc_path, self.inv_path, self.alerts_path]:
            if f.exists():
                f.unlink()
        if self.tmp_dir.exists():
            try:
                self.tmp_dir.rmdir()
            except OSError:
                pass

    def test_correct_opening_stock_mapping(self):
        """Verifies on_hand_stock strictly matches Opening_Stock, NOT Closing_Stock."""
        joined, report = self.loader.load_harmonized_dataframe(
            forecast_path=self.fc_path,
            master_path=self.inv_path,
            preorders_path=self.alerts_path,
        )
        self.assertTrue(report.validation_passed)
        self.assertEqual(joined.loc[0, "on_hand_stock"], 120.0)
        self.assertNotEqual(joined.loc[0, "on_hand_stock"], 80.0)
        self.assertEqual(joined.loc[1, "on_hand_stock"], 80.0)
        self.assertNotEqual(joined.loc[1, "on_hand_stock"], 75.0)

    def test_duplicate_key_detection_in_forecast(self):
        """DataLoader must raise ValueError if duplicate keys exist in forecast data."""
        bad_fc = pd.concat([self.fc_df, self.fc_df.iloc[[0]]], ignore_index=True)
        bad_fc_path = self.tmp_dir / "bad_dup_fc.parquet"
        bad_fc.to_parquet(bad_fc_path, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_harmonized_dataframe(
                forecast_path=bad_fc_path,
                master_path=self.inv_path,
                preorders_path=self.alerts_path,
            )
        self.assertIn("duplicate keys in forecast dataset", str(ctx.exception))
        bad_fc_path.unlink()

    def test_missing_inventory_match_detection(self):
        """DataLoader must raise ValueError if forecast record has no corresponding Opening_Stock."""
        bad_inv = self.inv_df.iloc[[0]].copy()
        bad_inv_path = self.tmp_dir / "bad_missing_inv.parquet"
        bad_inv.to_parquet(bad_inv_path, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_harmonized_dataframe(
                forecast_path=self.fc_path,
                master_path=bad_inv_path,
                preorders_path=self.alerts_path,
            )
        self.assertIn("Missing inventory match", str(ctx.exception))
        bad_inv_path.unlink()

    def test_unconfirmed_alerts_isolation(self):
        """
        Verifies that Step 3 alerts:
        1. Are loaded under 'unconfirmed_alert_count'
        2. Do NOT alter P50 demand
        3. Default to 0 when no alert occurred for a date
        """
        joined, report = self.loader.load_harmonized_dataframe(
            forecast_path=self.fc_path,
            master_path=self.inv_path,
            preorders_path=self.alerts_path,
        )
        row_0 = joined.iloc[0]
        self.assertEqual(row_0["unconfirmed_alert_count"], 3)
        self.assertEqual(row_0["p50_demand"], 40.0)

        row_1 = joined.iloc[1]
        self.assertEqual(row_1["unconfirmed_alert_count"], 0)
        self.assertEqual(row_1["p50_demand"], 38.0)

    def test_evaluation_records_separated(self):
        """Evaluation records must be returned separately from operational inputs."""
        inputs, eval_records, report = self.loader.load_daily_agent_inputs(
            forecast_path=self.fc_path,
            master_path=self.inv_path,
            preorders_path=self.alerts_path,
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(eval_records), 2)
        self.assertTrue(report.validation_passed)

        # Operational input has no target_demand field
        self.assertFalse(hasattr(inputs[0], "target_demand_eval_label"))
        self.assertFalse(hasattr(inputs[0], "target_demand"))

        # Evaluation record carries the ground-truth label
        self.assertIsInstance(eval_records[0], DailyEvaluationRecord)
        self.assertEqual(eval_records[0].target_demand, 42.0)
        self.assertEqual(eval_records[1].target_demand, 39.0)


class TestDataLoaderIntegration(unittest.TestCase):
    """End-to-end integration test against the real repository Parquet files."""

    def setUp(self):
        self.config = get_default_config()
        self.loader = DataLoader(self.config)

    def test_full_repository_data_load(self):
        """Validates ingestion across all 3,900 rows of forecast_payloads_step2_corrected.parquet."""
        joined, report = self.loader.load_harmonized_dataframe()

        self.assertEqual(len(joined), 3900)
        self.assertEqual(report.forecast_rows_loaded, 3900)
        self.assertEqual(report.master_rows_loaded, 27450)
        self.assertEqual(report.preorders_rows_loaded, 8397)
        self.assertEqual(report.unmatched_forecast_rows, 0)
        self.assertEqual(report.opening_stock_null_count, 0)
        self.assertTrue(report.validation_passed)

        self.assertEqual(report.preorder_overlap_alert_matches, 1128)

        self.assertEqual(joined["date"].nunique(), 52)
        self.assertEqual(joined["date"].min(), "2024-11-10")
        self.assertEqual(joined["date"].max(), "2024-12-31")

        store_sku_combos = joined.groupby(["store_id", "sku_id"]).ngroups
        self.assertEqual(store_sku_combos, 75)

    def test_daily_agent_inputs_conversion(self):
        """Verifies conversion into validated DailyAgentInput and DailyEvaluationRecord models."""
        inputs, eval_records, report = self.loader.load_daily_agent_inputs()

        self.assertEqual(len(inputs), 3900)
        self.assertTrue(report.validation_passed)

        # Verify operational attributes
        first = inputs[0]
        self.assertEqual(first.inventory.snapshot_basis, InventorySnapshotBasis.BEGINNING_OF_DAY)
        self.assertGreaterEqual(first.inventory.on_hand_stock, 0.0)
        self.assertTrue(first.forecast.p10_demand <= first.forecast.p50_demand <= first.forecast.p90_demand)
        self.assertGreater(first.forecast.sigma_demand, 0.0)

        # Verify evaluation records exist and are separate
        self.assertGreater(len(eval_records), 0)
        self.assertIsInstance(eval_records[0], DailyEvaluationRecord)
        self.assertGreaterEqual(eval_records[0].target_demand, 0.0)

        # Verify no leakage into operational inputs
        self.assertFalse(hasattr(first, "target_demand_eval_label"))
        self.assertFalse(hasattr(first, "target_demand"))


if __name__ == "__main__":
    unittest.main()
