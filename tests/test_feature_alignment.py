"""
Unit & Integration Tests for Feature Alignment and Leakage Prevention
=====================================================================
Validates that Category_Daily_Orders is shifted strictly to D-1 under Scenario A.
Verifies no same-day or future data leaks into the prediction features.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.step2_retrain_corrected import (
    correct_category_daily_orders,
    engineer_features_corrected,
    split_temporal
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"


class TestFeatureAlignmentSynthetic(unittest.TestCase):
    """Synthetic unit tests for Category_Daily_Orders alignment under Scenario A."""

    def setUp(self):
        """Builds a deterministic 4-day multi-store multi-category synthetic panel."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="Asia/Kolkata")
        stores = ["STORE_A", "STORE_B"]
        skus = [
            ("SKU_MILK", "Dairy"),
            ("SKU_BREAD", "Bakery")
        ]

        rows = []
        for d_idx, d in enumerate(dates):
            # Deterministic distinct macro orders per day per category:
            # Dairy: Day 0 -> 100, Day 1 -> 150, Day 2 -> 200, Day 3 -> 250
            # Bakery: Day 0 -> 40,  Day 1 -> 60,  Day 2 -> 80,  Day 3 -> 100
            dairy_orders = 100.0 + (d_idx * 50.0)
            bakery_orders = 40.0 + (d_idx * 20.0)

            for s in stores:
                for sku, cat in skus:
                    cdo = dairy_orders if cat == "Dairy" else bakery_orders
                    rows.append({
                        "Store_ID": s,
                        "SKU_ID": sku,
                        "Category": cat,
                        "Date": d,
                        "Category_Daily_Orders": cdo,
                        "Daily_Sales_Qty": 10.0,
                        "Target_Demand": 10.0,
                        "Closing_Stock": 50.0,
                        "Remaining_SL_Days": 5,
                        "Total_SL_Days": 10
                    })

        self.df_synthetic = pd.DataFrame(rows)

    def test_prediction_date_receives_previous_day_orders(self):
        """Prediction for date D must strictly receive orders finalized on date D-1."""
        df_corr = correct_category_daily_orders(self.df_synthetic)

        # Day 1 (idx 1 = 2024-01-02) should receive Day 0 orders (Dairy=100, Bakery=40)
        day1 = df_corr[df_corr["Date"].dt.strftime("%Y-%m-%d") == "2024-01-02"]
        dairy_day1 = day1[day1["Category"] == "Dairy"]["Category_Daily_Orders"].unique()
        bakery_day1 = day1[day1["Category"] == "Bakery"]["Category_Daily_Orders"].unique()

        self.assertEqual(len(dairy_day1), 1)
        self.assertEqual(dairy_day1[0], 100.0)
        self.assertEqual(len(bakery_day1), 1)
        self.assertEqual(bakery_day1[0], 40.0)

        # Day 2 (idx 2 = 2024-01-03) should receive Day 1 orders (Dairy=150, Bakery=60)
        day2 = df_corr[df_corr["Date"].dt.strftime("%Y-%m-%d") == "2024-01-03"]
        dairy_day2 = day2[day2["Category"] == "Dairy"]["Category_Daily_Orders"].unique()
        bakery_day2 = day2[day2["Category"] == "Bakery"]["Category_Daily_Orders"].unique()

        self.assertEqual(dairy_day2[0], 150.0)
        self.assertEqual(bakery_day2[0], 60.0)

    def test_no_same_day_orders_leak(self):
        """Date D must never receive date D's actual finalized orders."""
        df_corr = correct_category_daily_orders(self.df_synthetic)

        for d_str, same_day_dairy, same_day_bakery in [
            ("2024-01-02", 150.0, 60.0),
            ("2024-01-03", 200.0, 80.0),
            ("2024-01-04", 250.0, 100.0)
        ]:
            day_data = df_corr[df_corr["Date"].dt.strftime("%Y-%m-%d") == d_str]
            dairy_val = day_data[day_data["Category"] == "Dairy"]["Category_Daily_Orders"].iloc[0]
            bakery_val = day_data[day_data["Category"] == "Bakery"]["Category_Daily_Orders"].iloc[0]

            self.assertNotEqual(dairy_val, same_day_dairy, f"Same-day leak detected on {d_str} for Dairy")
            self.assertNotEqual(bakery_val, same_day_bakery, f"Same-day leak detected on {d_str} for Bakery")

    def test_no_future_orders_leak(self):
        """Date D must never receive orders from dates > D."""
        df_corr = correct_category_daily_orders(self.df_synthetic)

        # On 2024-01-02, value must not be 200.0 (Day 2) or 250.0 (Day 3)
        day1 = df_corr[df_corr["Date"].dt.strftime("%Y-%m-%d") == "2024-01-02"]
        dairy_val = day1[day1["Category"] == "Dairy"]["Category_Daily_Orders"].iloc[0]
        self.assertNotIn(dairy_val, [200.0, 250.0])

    def test_category_level_shift_across_stores(self):
        """All stores must receive identical category-level macro values on date D."""
        df_corr = correct_category_daily_orders(self.df_synthetic)

        for d_str in ["2024-01-02", "2024-01-03", "2024-01-04"]:
            day_df = df_corr[df_corr["Date"].dt.strftime("%Y-%m-%d") == d_str]
            store_a_dairy = day_df[(day_df["Store_ID"] == "STORE_A") & (day_df["Category"] == "Dairy")]["Category_Daily_Orders"].iloc[0]
            store_b_dairy = day_df[(day_df["Store_ID"] == "STORE_B") & (day_df["Category"] == "Dairy")]["Category_Daily_Orders"].iloc[0]
            self.assertEqual(store_a_dairy, store_b_dairy)

    def test_first_date_gets_nan_not_zero(self):
        """The initial date in panel has no D-1; it must strictly be NaN, not silently zero-filled."""
        df_corr = correct_category_daily_orders(self.df_synthetic)

        day0 = df_corr[df_corr["Date"].dt.strftime("%Y-%m-%d") == "2024-01-01"]
        for val in day0["Category_Daily_Orders"]:
            self.assertTrue(np.isnan(val), "Day 1 must be NaN because D-1 does not exist in dataset")
            self.assertNotEqual(val, 0.0, "Day 1 must not be falsely imputed as 0.0")

    def test_missing_intermediate_date_produces_nan(self):
        """If date D-1 is missing from history, date D must produce NaN rather than taking D-2 or zero."""
        # Create dataset with Day 1 and Day 3, but Day 2 is missing
        dates = [
            pd.Timestamp("2024-01-01", tz="Asia/Kolkata"),
            pd.Timestamp("2024-01-03", tz="Asia/Kolkata")
        ]
        rows = [
            {"Store_ID": "STORE_A", "SKU_ID": "SKU_MILK", "Category": "Dairy", "Date": dates[0], "Category_Daily_Orders": 100.0},
            {"Store_ID": "STORE_A", "SKU_ID": "SKU_MILK", "Category": "Dairy", "Date": dates[1], "Category_Daily_Orders": 300.0}
        ]
        df_gap = pd.DataFrame(rows)
        df_corr = correct_category_daily_orders(df_gap)

        # On 2024-01-03, D-1 (2024-01-02) does not exist in records.
        day3_val = df_corr[df_corr["Date"] == dates[1]]["Category_Daily_Orders"].iloc[0]
        self.assertTrue(np.isnan(day3_val), "Date following a temporal gap must produce NaN, not borrow older data")

    def test_panel_row_count_and_columns_preserved(self):
        """Feature correction must not alter row count or drop any existing columns."""
        initial_len = len(self.df_synthetic)
        initial_cols = set(self.df_synthetic.columns)

        df_corr = correct_category_daily_orders(self.df_synthetic)
        self.assertEqual(len(df_corr), initial_len)
        self.assertEqual(set(df_corr.columns), initial_cols)


class TestFeatureAlignmentRealData(unittest.TestCase):
    """Integration test against actual repository dataset data/processed/df_master.parquet."""

    @classmethod
    def setUpClass(cls):
        parquet_path = PROC_DIR / "df_master.parquet"
        if not parquet_path.exists():
            raise unittest.SkipTest("df_master.parquet not present; skipping real data integration test.")
        cls.df_raw = pd.read_parquet(parquet_path)
        cls.df_raw["Date"] = pd.to_datetime(cls.df_raw["Date"])

    def test_df_master_exact_d_minus_1_match(self):
        """Every sampled date D in df_master must exactly match raw Category_Daily_Orders on D-1."""
        df_corr = correct_category_daily_orders(self.df_raw)
        self.assertEqual(len(df_corr), len(self.df_raw))

        # Sample dates across different quarters
        test_dates = [
            "2024-02-15",
            "2024-05-20",
            "2024-08-10",
            "2024-11-25",
            "2024-12-30"
        ]

        for d_str in test_dates:
            d_curr = pd.Timestamp(d_str, tz="Asia/Kolkata")
            d_prev = d_curr - pd.Timedelta(days=1)

            for cat in ["Produce", "Dairy", "Bakery", "Meat"]:
                corr_val = df_corr[(df_corr["Date"] == d_curr) & (df_corr["Category"] == cat)]["Category_Daily_Orders"].iloc[0]
                raw_prev_val = self.df_raw[(self.df_raw["Date"] == d_prev) & (self.df_raw["Category"] == cat)]["Category_Daily_Orders"].iloc[0]
                raw_curr_val = self.df_raw[(self.df_raw["Date"] == d_curr) & (self.df_raw["Category"] == cat)]["Category_Daily_Orders"].iloc[0]

                self.assertAlmostEqual(corr_val, raw_prev_val, places=4,
                    msg=f"Corrected value on {d_str} for {cat} must match raw orders on {d_prev.date()}")
                
                # Check that raw curr is different (ensuring test is discriminative)
                if not np.isclose(raw_prev_val, raw_curr_val):
                    self.assertNotEqual(corr_val, raw_curr_val,
                        msg=f"Potential leakage! Corrected value on {d_str} equal to same-day raw value for {cat}")

    def test_train_val_test_split_continuity(self):
        """Temporal splitting maintains strict continuity across partition boundaries."""
        df_feat = engineer_features_corrected(self.df_raw)
        train_df, val_df, test_df = split_temporal(df_feat)

        # Boundary 1: Train -> Val
        last_train_date = train_df["Date"].max()
        first_val_date = val_df["Date"].min()
        self.assertEqual(first_val_date - last_train_date, pd.Timedelta(days=1))

        # Boundary 2: Val -> Test
        last_val_date = val_df["Date"].max()
        first_test_date = test_df["Date"].min()
        self.assertEqual(first_test_date - last_val_date, pd.Timedelta(days=1))

        # The first test date must use Category_Daily_Orders from last val date
        for cat in ["Dairy", "Produce"]:
            test_cdo = test_df[(test_df["Date"] == first_test_date) & (test_df["Category"] == cat)]["Category_Daily_Orders"].iloc[0]
            raw_last_val_cdo = self.df_raw[(self.df_raw["Date"] == last_val_date) & (self.df_raw["Category"] == cat)]["Category_Daily_Orders"].iloc[0]
            self.assertAlmostEqual(test_cdo, raw_last_val_cdo, places=4)


if __name__ == "__main__":
    unittest.main()

