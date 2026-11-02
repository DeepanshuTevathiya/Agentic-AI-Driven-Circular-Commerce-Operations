"""
STEP 2: Macro Dark-Store Probabilistic Forecasting Engine (Optimized)
======================================================================
Role: Arunesh Sharma (Role #5) — AI/ML Modelling Lead: Demand & Consumption Agents
Project: Agentic AI-Driven Circular Commerce Operations (Quick-Commerce Setting)
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR  = BASE_DIR / "outputs"
PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

test_log = []

def log(msg: str, level: str = "INFO"):
    prefix = {"INFO": "  ✓", "WARN": "  ⚠", "FAIL": "  ✗", "HEAD": "\n──"}
    line = f"{prefix.get(level, '  ')} {msg}"
    print(line)
    test_log.append(line)


def load_data():
    log("Loading master panel (data/processed/df_master.parquet)...", "HEAD")
    parquet_path = PROC_DIR / "df_master.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}. Please execute Step 1 first.")
    
    df = pd.read_parquet(parquet_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store_ID", "SKU_ID", "Date"]).reset_index(drop=True)
    log(f"Loaded master panel: {len(df):,} rows × {df.shape[1]} cols")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Engineering temporal lags, multi-scale rolling statistics, and degradation metrics...", "HEAD")
    df = df.copy()
    target_col = "Target_Demand" if "Target_Demand" in df.columns else "Daily_Sales_Qty"
    group = df.groupby(["Store_ID", "SKU_ID"])[target_col]
    
    # 1. Calendar & Bangalore Dynamic Surge Features (Created first for group profiles)
    dt_series = df["Date"].dt
    df["day_of_week"] = dt_series.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = dt_series.day
    df["month"]        = dt_series.month
    df["is_salary_surge"] = df["day_of_month"].apply(lambda d: 1 if d <= 5 or d >= 28 else 0)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # 2. Autoregressive Lag Features
    for lag in [1, 2, 3, 7, 14]:
        df[f"demand_lag_{lag}"] = group.shift(lag)
        
    # 3. Multi-Scale Rolling Statistics (Shifted by 1 to prevent leak)
    df["rolling_mean_3"]  = group.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["rolling_mean_7"]  = group.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df["rolling_mean_14"] = group.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df["rolling_mean_21"] = group.transform(lambda x: x.shift(1).rolling(21, min_periods=1).mean())
    
    df["rolling_std_7"]  = group.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std()).fillna(0)
    df["rolling_min_7"]  = group.transform(lambda x: x.shift(1).rolling(7, min_periods=1).min())
    df["rolling_max_7"]  = group.transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
    df["ewma_7"]         = group.transform(lambda x: x.shift(1).ewm(alpha=0.30, min_periods=1).mean())
    df["demand_momentum_7"] = (df["demand_lag_1"] / (df["rolling_mean_7"] + 1e-3)).clip(0.1, 5.0)
    
    # 4. Perishability & Shelf-Life Degradation Dynamics
    if "Remaining_SL_Days" in df.columns and "Total_SL_Days" in df.columns:
        df["shelf_life_ratio"] = (df["Remaining_SL_Days"] / (df["Total_SL_Days"] + 1e-4)).clip(0.0, 1.0)
        df["expiry_urgency_flag"] = (df["Remaining_SL_Days"] <= 2).astype(int)
    else:
        df["shelf_life_ratio"] = 1.0
        df["expiry_urgency_flag"] = 0
        
    # 5. Macro Signal
    if "Category_Daily_Orders" not in df.columns:
        df["Category_Daily_Orders"] = 0.0
    else:
        df["Category_Daily_Orders"] = df["Category_Daily_Orders"].fillna(0.0)
        
    df_clean = df.dropna(subset=["demand_lag_14"]).reset_index(drop=True)
    log(f"Features created successfully: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols")
    return df_clean


def split_temporal(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    log("Executing temporal train/validation/test split...", "HEAD")
    unique_dates = np.sort(df["Date"].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx   = int(n_dates * (train_ratio + val_ratio))
    
    train_end_date = unique_dates[train_end_idx]
    val_end_date   = unique_dates[val_end_idx]
    
    train_df = df[df["Date"] <= train_end_date].copy()
    val_df   = df[(df["Date"] > train_end_date) & (df["Date"] <= val_end_date)].copy()
    test_df  = df[df["Date"] > val_end_date].copy()
    
    log(f"Train split : {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df):,} rows)")
    log(f"Val split   : {val_df['Date'].min().date()} to {val_df['Date'].max().date()} ({len(val_df):,} rows)")
    log(f"Test split  : {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df):,} rows)")
    return train_df, val_df, test_df


class MultiQuantileDemandEngine:
    def __init__(self, quantiles=[0.10, 0.50, 0.90]):
        self.quantiles = quantiles
        self.models = {}
        self.feature_cols = [
            "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_7", "demand_lag_14",
            "rolling_mean_3", "rolling_mean_7", "rolling_mean_14", "rolling_mean_21",
            "rolling_std_7", "rolling_min_7", "rolling_max_7",
            "ewma_7", "demand_momentum_7",
            "day_of_week", "is_weekend", "day_of_month", "month",
            "is_salary_surge", "is_monsoon",
            "shelf_life_ratio", "expiry_urgency_flag", "Category_Daily_Orders",
            "Store_ID", "SKU_ID", "Category"
        ]
        self.cat_cols = ["Store_ID", "SKU_ID", "Category"]

    def _prepare_xy(self, df: pd.DataFrame, target_col="Target_Demand"):
        X = df[self.feature_cols].copy()
        for col in self.cat_cols:
            X[col] = X[col].astype("category")
        y = df[target_col].values if target_col in df.columns else None
        return X, y

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, target_col="Target_Demand"):
        log("Training Multi-Quantile LightGBM models (P10, P50, P90)...", "HEAD")
        X_train, y_train = self._prepare_xy(train_df, target_col)
        X_val, y_val     = self._prepare_xy(val_df, target_col)
        
        for q in self.quantiles:
            t0 = time.time()
            params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "boosting_type": "gbdt",
                "n_estimators": 500,
                "learning_rate": 0.035,
                "num_leaves": 35,
                "min_child_samples": 12,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)]
            )
            elapsed = time.time() - t0
            self.models[q] = model
            log(f"Model P{int(q*100):02d} trained in {elapsed:.2f}s | Best Iteration: {model.best_iteration_}")

    def predict_quantiles(self, df: pd.DataFrame) -> dict:
        X, _ = self._prepare_xy(df)
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X)
            
        p10 = np.maximum(0.0, preds[0.10])
        p50 = np.maximum(p10, preds[0.50])
        p90 = np.maximum(p50, preds[0.90])
        sigma_d = np.maximum(0.1, (p90 - p10) / (2 * 1.2816))
        
        return {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "sigma_d": sigma_d,
            "raw_p10": preds[0.10],
            "raw_p50": preds[0.50],
            "raw_p90": preds[0.90]
        }


def pinball_loss(y_true, y_pred, alpha):
    err = y_true - y_pred
    return np.mean(np.maximum(alpha * err, (alpha - 1) * err))

def compute_wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return (np.sum(np.abs(y_true - y_pred)) / (denom + 1e-6)) * 100


def run_step2_tests(engine: MultiQuantileDemandEngine, test_df: pd.DataFrame) -> dict:
    log("Running Step 2 Verification Suite...", "HEAD")
    target_col = "Target_Demand" if "Target_Demand" in test_df.columns else "Daily_Sales_Qty"
    y_test = test_df[target_col].values
    
    # Warm-up call
    sample_batch = test_df.iloc[:100].copy()
    _ = engine.predict_quantiles(sample_batch)
    
    # TEST 2.3: Inference Latency Benchmark
    log("TEST 2.3 — Inference Latency Benchmark (100 SKUs batch)", "HEAD")
    t_start = time.perf_counter()
    _ = engine.predict_quantiles(sample_batch)
    latency_ms = (time.perf_counter() - t_start) * 1000
    status_lat = "INFO" if latency_ms <= 25.0 else "WARN"
    log(f"Inference latency (100 SKUs batch): {latency_ms:.2f} ms (Target: <= 25 ms)", status_lat)
    
    # Full Test Predictions
    preds = engine.predict_quantiles(test_df)
    p10, p50, p90 = preds["p10"], preds["p50"], preds["p90"]
    
    # TEST 2.2: Quantile Monotonicity Test
    log("TEST 2.2 — Quantile Monotonicity Verification", "HEAD")
    raw_crossings = np.sum(preds["raw_p10"] > preds["raw_p50"]) + np.sum(preds["raw_p50"] > preds["raw_p90"])
    final_crossings = np.sum((p10 > p50) | (p50 > p90))
    log(f"Raw quantile crossings: {raw_crossings} ({raw_crossings / (2 * len(test_df)) * 100:.2f}%)")
    log(f"Post-processed crossings: {final_crossings} (Target: 0.0%)", "INFO" if final_crossings == 0 else "FAIL")
    
    # TEST 2.1: WAPE & Pinball Loss Evaluation
    log("TEST 2.1 — Predictive Accuracy Benchmarks (WAPE & Pinball Loss)", "HEAD")
    overall_wape = compute_wape(y_test, p50)
    
    test_eval_df = test_df.copy()
    test_eval_df["y_true"] = y_test
    test_eval_df["p50"]    = p50
    
    category_wape = {}
    for cat, grp in test_eval_df.groupby("Category"):
        w = compute_wape(grp["y_true"].values, grp["p50"].values)
        category_wape[cat] = round(w, 2)
        log(f"Category [{cat:8s}] WAPE: {w:5.2f}%")
        
    dairy_wape = category_wape.get("Dairy", 0.0)
    produce_wape = category_wape.get("Produce", 0.0)
    
    naive_pred = test_df["demand_lag_7"].values
    naive_wape = compute_wape(y_test, naive_pred)
    improvement = naive_wape - overall_wape
    
    pb_10 = pinball_loss(y_test, p10, 0.10)
    pb_50 = pinball_loss(y_test, p50, 0.50)
    pb_90 = pinball_loss(y_test, p90, 0.90)
    
    log(f"Model Overall WAPE      : {overall_wape:.2f}%  (Target: <= 18.0%)", "INFO" if overall_wape <= 18.0 else "WARN")
    log(f"Dairy Perishables WAPE  : {dairy_wape:.2f}%  (Target: <= 22.0%)", "INFO" if dairy_wape <= 22.0 else "WARN")
    log(f"Naive 7-Day Baseline WAPE: {naive_wape:.2f}%")
    log(f"Model Improvement       : +{improvement:.2f}% absolute WAPE reduction", "INFO")
    log(f"Pinball Losses -> P10: {pb_10:.3f} | P50: {pb_50:.3f} | P90: {pb_90:.3f}")
    
    passed_2_1 = (overall_wape <= 18.0) and (dairy_wape <= 22.0)
    passed_2_2 = (final_crossings == 0)
    passed_2_3 = (latency_ms <= 25.0)
    
    return {
        "overall_wape": round(overall_wape, 2),
        "dairy_wape": dairy_wape,
        "produce_wape": produce_wape,
        "naive_wape": round(naive_wape, 2),
        "wape_improvement": round(improvement, 2),
        "pinball_p10": round(pb_10, 3),
        "pinball_p50": round(pb_50, 3),
        "pinball_p90": round(pb_90, 3),
        "latency_ms": round(latency_ms, 2),
        "monotonicity_violations": int(final_crossings),
        "passed_2_1": bool(passed_2_1),
        "passed_2_2": bool(passed_2_2),
        "passed_2_3": bool(passed_2_3),
        "all_passed": bool(passed_2_1 and passed_2_2 and passed_2_3),
        "test_eval_df": test_eval_df,
        "preds": preds
    }


def export_feature_importance(engine: MultiQuantileDemandEngine):
    p50_model = engine.models[0.50]
    imp_df = pd.DataFrame({
        "Feature": engine.feature_cols,
        "Importance_Gain": p50_model.booster_.feature_importance(importance_type="gain"),
        "Importance_Split": p50_model.booster_.feature_importance(importance_type="split")
    }).sort_values("Importance_Gain", ascending=False).reset_index(drop=True)
    
    out_csv = PROC_DIR / "step2_feature_importance.csv"
    imp_df.to_csv(out_csv, index=False)
    log(f"Feature importance exported -> {out_csv}")
    log("Top 5 Driving Features for Dark-Store Demand:")
    for idx, row in imp_df.head(5).iterrows():
        log(f"   {idx+1}. {row['Feature']:20s} (Gain: {row['Importance_Gain']:,.1f})")


def generate_downstream_payload(test_df: pd.DataFrame, preds: dict):
    log("Generating Downstream Handoff Payload for Deepanshu & Rajat...", "HEAD")
    payload_df = test_df[[
        "Store_ID", "SKU_ID", "Date", "Category", "Closing_Stock", 
        "Remaining_SL_Days", "Total_SL_Days", "Target_Demand"
    ]].copy()
    
    payload_df["p10_demand"]   = np.round(preds["p10"], 2)
    payload_df["p50_demand"]   = np.round(preds["p50"], 2)
    payload_df["p90_demand"]   = np.round(preds["p90"], 2)
    payload_df["sigma_demand"] = np.round(preds["sigma_d"], 3)
    
    expected_shelf_life_demand = payload_df["p50_demand"] * payload_df["Remaining_SL_Days"]
    payload_df["overstock_risk_units"] = np.maximum(
        0.0, 
        payload_df["Closing_Stock"] - expected_shelf_life_demand
    ).round(2)
    
    out_parquet = PROC_DIR / "forecast_payloads_step2.parquet"
    payload_df.to_parquet(out_parquet, index=False, compression="snappy")
    log(f"Saved downstream payload -> {out_parquet} ({len(payload_df):,} rows)")


def write_step2_report(results: dict):
    report_path = OUT_DIR / "step2_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("STEP 2 — MACRO DEMAND FORECASTING ENGINE TEST REPORT\n")
        f.write("Role: Arunesh Sharma (Role #5) — AI/ML Demand & Consumption Lead\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("1. QUANTITATIVE SUCCESS METRIC BENCHMARKS:\n")
        f.write(f"  • Overall WAPE            : {results['overall_wape']}% (Target: <= 18.0%) -> {'PASSED [OK]' if results['passed_2_1'] else 'CONTINGENCY TRIGGERED'}\n")
        f.write(f"  • Dairy Perishables WAPE  : {results['dairy_wape']}% (Target: <= 22.0%)\n")
        f.write(f"  • Produce Perishables WAPE: {results['produce_wape']}%\n")
        f.write(f"  • Naive Baseline WAPE     : {results['naive_wape']}%\n")
        f.write(f"  • Accuracy Gain vs Naive  : +{results['wape_improvement']}% absolute WAPE reduction\n")
        f.write(f"  • Quantile Pinball Losses : P10={results['pinball_p10']} | P50={results['pinball_p50']} | P90={results['pinball_p90']}\n\n")
        
        f.write("2. QUANTILE MONOTONICITY & CONSISTENCY (TEST 2.2):\n")
        f.write(f"  • Post-Processed Monotonicity Violations: {results['monotonicity_violations']} (0.00%)\n")
        f.write(f"  • Non-Crossing Constraint P10 <= P50 <= P90: PASSED [OK]\n\n")
        
        f.write("3. SYSTEM INFERENCE LATENCY (TEST 2.3):\n")
        f.write(f"  • Batch Latency (100 SKUs): {results['latency_ms']} ms (Target: <= 25 ms) -> {'PASSED [OK]' if results['passed_2_3'] else 'FAILED'}\n\n")
        
        f.write("4. INTEGRATION HANDSHAKE WITH DOWNSTREAM AGENTS:\n")
        f.write("  • Deepanshu (Role #6 - Inventory Lead): Receives p50_demand (D_avg), p90_demand, sigma_demand for ROP\n")
        f.write("  • Rajat (Role #7 - Circularity Lead): Receives overstock_risk_units (Omega_t) and p10_demand\n\n")
        
        f.write("5. OVERALL STEP 2 STATUS:\n")
        f.write(f"  STATUS: {'PASSED [OK]' if results['all_passed'] else 'REQUIRES REVIEW'}\n\n")
        f.write("-" * 65 + "\n")
        f.write("EXECUTION TRACE LOG:\n")
        f.write("\n".join(test_log))
    log(f"Test report written -> {report_path}")


if __name__ == "__main__":
    print("=" * 65)
    print("STEP 2 — MACRO DARK-STORE PROBABILISTIC FORECASTING ENGINE")
    print("=" * 65)
    df_raw = load_data()
    df_feat = engineer_features(df_raw)
    train_df, val_df, test_df = split_temporal(df_feat)
    
    engine = MultiQuantileDemandEngine(quantiles=[0.10, 0.50, 0.90])
    engine.train(train_df, val_df)
    
    model_path = OUT_DIR / "lgb_p10_p50_p90.joblib"
    joblib.dump(engine, model_path)
    log(f"Model engine saved -> {model_path}")
    
    results = run_step2_tests(engine, test_df)
    export_feature_importance(engine)
    generate_downstream_payload(test_df, results["preds"])
    write_step2_report(results)
    
    print("\n" + "=" * 65)
    print("STEP 2 EXECUTION SUMMARY")
    print("=" * 65)
    print(f"  Test 2.1 WAPE Benchmark     : {'PASSED [OK]' if results['passed_2_1'] else 'FAILED'}")
    print(f"  Test 2.2 Quantile Non-Cross : {'PASSED [OK]' if results['passed_2_2'] else 'FAILED'}")
    print(f"  Test 2.3 Inference Latency  : {'PASSED [OK]' if results['passed_2_3'] else 'FAILED'}")
    print(f"\n  Overall Step 2 Status: {'ALL SUCCESS METRICS ACHIEVED [OK]' if results['all_passed'] else 'CONTINGENCY PROTOCOL ENGAGED'}")
    print(f"  Overall Test WAPE    : {results['overall_wape']}% (Target: <= 18.0%)")
    print(f"  Dairy Category WAPE  : {results['dairy_wape']}% (Target: <= 22.0%)")
    print(f"  Inference Latency    : {results['latency_ms']} ms (Target: <= 25.0 ms)")
    print("=" * 65)
