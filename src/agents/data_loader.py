"""
Lead 2 Data Loader & Ingestion Pipeline
=======================================
Implements robust, audited data ingestion for Lead 2 agents by joining:
1. forecast_payloads_step2.parquet (probabilistic forecasts)
2. df_master.parquet (authoritative beginning-of-day Opening_Stock)
3. committed_preorders_step3.parquet (diagnostic customer runout alerts)

CRITICAL INGESTION POLICIES:
- Uses Opening_Stock from df_master as beginning-of-day on_hand_stock.
- Same-day Closing_Stock from forecast_payloads_step2 is strictly rejected as starting inventory.
- Target_Demand is preserved solely as an offline evaluation label (never as an operational input).
- Step 3 alert counts are labeled 'unconfirmed_alert_count' and kept strictly diagnostic
  (they are NOT treated as confirmed orders, NOT reserved, and NOT subtracted/added from P50).
- Normalizes column casing to snake_case.
- Enforces strict key uniqueness and detects missing records or duplicate rows without silent drops.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from .config import Lead2Config, get_default_config
from .schemas import (
    DailyAgentInput,
    DailyEvaluationRecord,
    DemandForecast,
    InventorySnapshot,
    InventorySnapshotBasis,
)


@dataclass
class IngestionAuditReport:
    """Detailed audit metrics produced during data ingestion and validation."""
    forecast_rows_loaded: int = 0
    master_rows_loaded: int = 0
    preorders_rows_loaded: int = 0
    forecast_duplicates_detected: int = 0
    master_duplicates_detected: int = 0
    preorder_duplicates_detected: int = 0
    joined_rows_count: int = 0
    unmatched_forecast_rows: int = 0
    preorder_overlap_alert_matches: int = 0
    opening_stock_null_count: int = 0
    validation_passed: bool = False
    notes: List[str] = field(default_factory=list)


class DataLoader:
    """
    Manages loading, normalization, joining, and validation of multi-agent input data.
    """
    def __init__(self, config: Optional[Lead2Config] = None):
        self.config = config or get_default_config()

    @staticmethod
    def _canonicalize_date(series: pd.Series) -> pd.Series:
        """Converts datetime, timezone-aware, or string dates into YYYY-MM-DD string series."""
        return pd.to_datetime(series).dt.strftime("%Y-%m-%d")

    def load_forecasts(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Loads and normalizes forecast_payloads_step2.parquet.
        Extracts probabilistic demand quantiles and uncertainty metrics.
        """
        fpath = path or self.config.forecast_file
        if not fpath.exists():
            raise FileNotFoundError(f"Missing forecast file: {fpath}")

        df = pd.read_parquet(fpath)
        
        # Explicit column mapping and normalization
        rename_map = {
            "Store_ID": "store_id",
            "SKU_ID": "sku_id",
            "Category": "category",
            "Date": "raw_date",
            "p10_demand": "p10_demand",
            "p50_demand": "p50_demand",
            "p90_demand": "p90_demand",
            "sigma_demand": "sigma_demand",
            "Target_Demand": "target_demand_eval_label",
            "Total_SL_Days": "total_shelf_life_days",
            "Remaining_SL_Days": "remaining_shelf_life_days",
        }
        
        # Validate that expected raw columns exist
        missing = [c for c in ["Store_ID", "SKU_ID", "Category", "Date", "p10_demand", "p50_demand", "p90_demand", "sigma_demand"] if c not in df.columns]
        if missing:
            raise ValueError(f"Forecast file {fpath} is missing required columns: {missing}")

        norm_df = df[list(rename_map.keys())].rename(columns=rename_map).copy()
        norm_df["date"] = self._canonicalize_date(norm_df["raw_date"])
        norm_df = norm_df.drop(columns=["raw_date"])
        
        return norm_df

    def load_master_inventory(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Loads and normalizes df_master.parquet.
        Extracts authoritative beginning-of-day Opening_Stock.
        """
        mpath = path or self.config.master_file
        if not mpath.exists():
            raise FileNotFoundError(f"Missing master inventory file: {mpath}")

        df = pd.read_parquet(mpath)
        
        required = ["Store_ID", "SKU_ID", "Date", "Opening_Stock"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Master file {mpath} is missing required columns: {missing}")

        norm_df = pd.DataFrame({
            "store_id": df["Store_ID"].astype(str),
            "sku_id": df["SKU_ID"].astype(str),
            "date": self._canonicalize_date(df["Date"]),
            "on_hand_stock": df["Opening_Stock"].astype(float),
        })
        
        return norm_df

    def load_diagnostic_alerts(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Loads committed_preorders_step3.parquet for diagnostic tracking only.
        Labels the quantity 'unconfirmed_alert_units'.
        """
        ppath = path or self.config.preorders_file
        if not ppath.exists():
            # Alerts are optional/diagnostic; return an empty structured DataFrame if missing
            return pd.DataFrame(columns=["store_id", "sku_id", "date", "unconfirmed_alert_count"])

        df = pd.read_parquet(ppath)
        
        # Verify required columns
        for col in ["store_id", "sku_id", "current_date", "committed_preorder_units"]:
            if col not in df.columns:
                raise ValueError(f"Preorders file {ppath} missing expected column '{col}'")

        norm_df = pd.DataFrame({
            "store_id": df["store_id"].astype(str),
            "sku_id": df["sku_id"].astype(str),
            "date": self._canonicalize_date(df["current_date"]),
            "unconfirmed_alert_count": df["committed_preorder_units"].astype(int),
        })
        
        return norm_df

    def load_harmonized_dataframe(
        self,
        forecast_path: Optional[Path] = None,
        master_path: Optional[Path] = None,
        preorders_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, IngestionAuditReport]:
        """
        Loads, audits, and joins forecasts with authoritative beginning-of-day stock
        and diagnostic alerts.
        
        Returns:
            Tuple of (harmonized_dataframe, IngestionAuditReport)
        """
        report = IngestionAuditReport()

        # 1. Load component datasets
        fc_df = self.load_forecasts(forecast_path)
        report.forecast_rows_loaded = len(fc_df)

        inv_df = self.load_master_inventory(master_path)
        report.master_rows_loaded = len(inv_df)

        alerts_df = self.load_diagnostic_alerts(preorders_path)
        report.preorders_rows_loaded = len(alerts_df)

        # 2. Check duplicate primary keys (store_id, sku_id, date) before merging
        key_cols = ["store_id", "sku_id", "date"]
        
        fc_dup = fc_df.duplicated(subset=key_cols).sum()
        report.forecast_duplicates_detected = int(fc_dup)
        if fc_dup > 0:
            raise ValueError(f"Integrity Error: Found {fc_dup} duplicate keys in forecast dataset.")

        inv_dup = inv_df.duplicated(subset=key_cols).sum()
        report.master_duplicates_detected = int(inv_dup)
        if inv_dup > 0:
            raise ValueError(f"Integrity Error: Found {inv_dup} duplicate keys in master inventory dataset.")

        if len(alerts_df) > 0:
            al_dup = alerts_df.duplicated(subset=key_cols).sum()
            report.preorder_duplicates_detected = int(al_dup)
            if al_dup > 0:
                report.notes.append(f"Warning: {al_dup} duplicate keys in diagnostic alerts. Grouping by sum.")
                alerts_df = alerts_df.groupby(key_cols, as_index=False)["unconfirmed_alert_count"].sum()

        # 3. Perform left join of forecast onto master inventory (ensuring beginning-of-day stock is matched)
        joined = pd.merge(fc_df, inv_df, on=key_cols, how="left")
        report.joined_rows_count = len(joined)

        # Audit missing inventory matches
        missing_inv = joined["on_hand_stock"].isnull().sum()
        report.opening_stock_null_count = int(missing_inv)
        report.unmatched_forecast_rows = int(missing_inv)
        if missing_inv > 0:
            report.notes.append(f"Critical: {missing_inv} forecast records could not be matched with Opening_Stock.")
            raise ValueError(f"Missing inventory match for {missing_inv} forecast rows. Ingestion aborted.")

        # 4. Left join diagnostic customer alerts
        if len(alerts_df) > 0:
            joined = pd.merge(joined, alerts_df, on=key_cols, how="left")
            overlap_matches = joined["unconfirmed_alert_count"].notnull().sum()
            report.preorder_overlap_alert_matches = int(overlap_matches)
            joined["unconfirmed_alert_count"] = joined["unconfirmed_alert_count"].fillna(0).astype(int)
        else:
            joined["unconfirmed_alert_count"] = 0
            report.preorder_overlap_alert_matches = 0

        # 5. Final integrity checks on joined panel
        post_dup = joined.duplicated(subset=key_cols).sum()
        if post_dup > 0:
            raise ValueError(f"Post-join duplicate records detected: {post_dup}")

        # Check for unexpected nulls in mandatory fields
        mandatory_fields = [
            "store_id", "sku_id", "category", "date",
            "p10_demand", "p50_demand", "p90_demand", "sigma_demand",
            "on_hand_stock", "unconfirmed_alert_count"
        ]
        null_counts = joined[mandatory_fields].isnull().sum()
        if null_counts.any():
            bad_cols = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Integrity Error: Null values detected in mandatory fields: {bad_cols}")

        report.validation_passed = True
        report.notes.append(
            f"Successfully harmonized {len(joined)} records. "
            f"Opening_Stock verified as on_hand_stock. "
            f"{report.preorder_overlap_alert_matches} unconfirmed alert events merged (diagnostic only)."
        )

        return joined, report

    def load_daily_agent_inputs(
        self,
        forecast_path: Optional[Path] = None,
        master_path: Optional[Path] = None,
        preorders_path: Optional[Path] = None,
    ) -> Tuple[List[DailyAgentInput], List[DailyEvaluationRecord], IngestionAuditReport]:
        """
        Loads, harmonizes, and parses all records into:
        - List[DailyAgentInput]: Operational inputs (no ground-truth labels).
        - List[DailyEvaluationRecord]: Evaluation-only records with ground-truth demand.

        Returns:
            Tuple of (operational_inputs, evaluation_records, audit_report)
        """
        df, report = self.load_harmonized_dataframe(forecast_path, master_path, preorders_path)

        inputs: List[DailyAgentInput] = []
        eval_records: List[DailyEvaluationRecord] = []

        for row in df.itertuples(index=False):
            fc = DemandForecast(
                store_id=row.store_id,
                sku_id=row.sku_id,
                category=row.category,
                date=row.date,
                p10_demand=float(row.p10_demand),
                p50_demand=float(row.p50_demand),
                p90_demand=float(row.p90_demand),
                sigma_demand=float(row.sigma_demand),
            )

            inv = InventorySnapshot(
                store_id=row.store_id,
                sku_id=row.sku_id,
                date=row.date,
                on_hand_stock=float(row.on_hand_stock),
                remaining_shelf_life_days=int(row.remaining_shelf_life_days) if pd.notnull(row.remaining_shelf_life_days) else None,
                total_shelf_life_days=int(row.total_shelf_life_days) if pd.notnull(row.total_shelf_life_days) else None,
                snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
            )

            inp = DailyAgentInput(
                store_id=row.store_id,
                sku_id=row.sku_id,
                category=row.category,
                date=row.date,
                forecast=fc,
                inventory=inv,
                unconfirmed_alert_count=int(row.unconfirmed_alert_count),
            )
            inputs.append(inp)

            # Build evaluation record if ground-truth label is available
            if pd.notnull(row.target_demand_eval_label):
                eval_rec = DailyEvaluationRecord(
                    store_id=row.store_id,
                    sku_id=row.sku_id,
                    date=row.date,
                    target_demand=float(row.target_demand_eval_label),
                )
                eval_records.append(eval_rec)

        return inputs, eval_records, report


def load_daily_agent_inputs(
    forecast_path: Optional[Path] = None,
    master_path: Optional[Path] = None,
    preorders_path: Optional[Path] = None,
) -> Tuple[List[DailyAgentInput], List[DailyEvaluationRecord], IngestionAuditReport]:
    """Shortcut module function to load validated DailyAgentInput list and evaluation records."""
    loader = DataLoader()
    return loader.load_daily_agent_inputs(forecast_path, master_path, preorders_path)
