"""
STEP 4: Multi-Agent Protocol, Pydantic Schemas & Downstream Handshake
=====================================================================
Role: Arunesh Sharma (Role #5) — AI/ML Modelling Lead: Demand & Consumption Agents
Project: Agentic AI-Driven Circular Commerce Operations (Quick-Commerce Setting)

PURPOSE:
  1. Define production-grade Pydantic V2 data contracts matching all requirements
     requested by Deepanshu (Role #6 - Inventory/Procurement Lead) and Rajat (Role #7 - Circularity Lead):
       - Store-level demand forecast (per item, store, day)
       - Low (P10) / Likely (P50) / High (P90) range + sigma_D
       - Short horizon (2-day for dark-store fulfillment) & Long horizon (14-day for procurement)
       - Customer-level depletion alerts (who needs what, when, confidence, reason)
       - Historical item accuracy (WAPE per category & trust factor)
       - Confidence score (0 to 1) & explainable reason string
       - Overstock risk units (Omega_t) for dynamic markdown / food rescue
  2. Implement autonomous Agent Interfaces:
       - DarkStoreDemandAgent
       - ConsumerConsumptionAgent
       - MultiAgentCoordinator (Dispatch bus)
  3. Formulate Section 4 Academic Dossier mathematical handoffs:
       - Safety Stock: SS = Z * sqrt(LT * sigma_D^2 + D_avg^2 * sigma_LT^2)
       - Reorder Point: ROP = D_avg * LT + SS
       - Overstock Risk: Omega_t = max(0, Closing_Stock - D_avg * Remaining_SL)
  4. Execute Verification Suite:
       - Test 4.1: Pydantic Schema Validation (1,000 payloads, 100% pass)
       - Test 4.2: Boundary Value Stress Test (Zero/negative/infinite values, 0 NaN/Inf)
       - Test 4.3: End-to-End Handshake Latency (100 SKUs batch <= 50 ms)
  5. Export artifacts & JSON contracts for Deepanshu, Rajat, and Subhajit.

OUTPUTS:
  src/step4_multi_agent_protocols.py
  outputs/step4_test_report.txt
  data/processed/agent_handoff_payloads_step4.parquet
  data/processed/sample_deepanshu_handoff.json
  data/processed/sample_rajat_handoff.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
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


# ===========================================================================
# 1. PRODUCTION PYDANTIC V2 DATA SCHEMAS (CONTRACTS)
# ===========================================================================

class StoreDemandForecastPayload(BaseModel):
    """
    Contract delivered from Arunesh's DarkStoreDemandAgent to Deepanshu's
    Inventory & Procurement Agent and Rajat's Circularity Agent.
    """
    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Unique product SKU identifier")
    category: str = Field(..., description="Product category (Dairy, Bakery, Produce, Meat)")
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)")
    
    # 1. Low / Likely / High Probabilistic Bounds
    p10_demand: float = Field(..., ge=0.0, description="10th percentile conservative clearance demand")
    p50_demand: float = Field(..., ge=0.0, description="50th percentile median expected demand (D_avg)")
    p90_demand: float = Field(..., ge=0.0, description="90th percentile peak surge demand")
    sigma_demand: float = Field(..., ge=0.01, description="Estimated daily demand standard deviation (sigma_D)")
    
    # 2. Multi-Horizon Demand Horizons (Requested by Deepanshu)
    short_horizon_demand_2d: float = Field(..., ge=0.0, description="2-day cumulative demand for immediate dark-store dispatch")
    long_horizon_demand_14d: float = Field(..., ge=0.0, description="14-day cumulative demand for supplier purchase orders")
    
    # 3. Committed Consumer Demand (From Step 3 Household Agent)
    committed_preorders_qty: int = Field(default=0, ge=0, description="Guaranteed units pre-ordered by households in this store catchment")
    
    # 4. Past Accuracy & Trust Calibration (Requested by Deepanshu)
    historical_item_wape: float = Field(..., ge=0.0, le=100.0, description="Historical WAPE for this item/category (%)")
    forecast_confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0 to 1) for automated PO approval")
    forecast_explanation: str = Field(..., description="Plain-English explanation for dark-store operator auditability")
    
    # 5. Circularity & Waste Spoilage Bounds (For Rajat)
    closing_stock: float = Field(..., ge=0.0, description="Current stock on hand in dark store")
    remaining_shelf_life_days: int = Field(..., ge=0, description="Remaining shelf life in days before expiry")
    overstock_risk_units: float = Field(..., ge=0.0, description="Omega_t: Unsold units projected to exceed shelf life demand")
    recommended_action: Literal["STANDARD_REPLENISHMENT", "URGENT_RESTOCK", "TRIGGER_MARKDOWN_DISCOUNT", "MAINTAIN_STOCK"] = Field(...)

    @model_validator(mode="after")
    def verify_monotonicity_and_bounds(self):
        # Monotonicity rule: P10 <= P50 <= P90
        if not (self.p10_demand <= self.p50_demand <= self.p90_demand):
            raise ValueError(f"Quantile monotonicity violation: P10 ({self.p10_demand}) <= P50 ({self.p50_demand}) <= P90 ({self.p90_demand}) failed")
        return self


class HouseholdDepletionAlertPayload(BaseModel):
    """
    Contract delivered from Arunesh's ConsumerConsumptionAgent to Fulfilment & App Services.
    """
    customer_id: str = Field(...)
    store_id: str = Field(...)
    sku_id: str = Field(...)
    current_date: str = Field(...)
    estimated_cycle_days: float = Field(..., gt=0.0)
    predicted_remaining_days: float = Field(..., ge=0.0)
    advance_alert_window_days: float = Field(..., gt=0.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    recommended_action: Literal["PUSH_REORDER_CONFIRMATION", "URGENT_DEPLETION_WARNING", "STANDBY"] = Field(...)
    explanation: str = Field(...)


class DeepanshuInventoryDecisionPayload(BaseModel):
    """
    Downstream payload generated by Deepanshu's Inventory Agent using Arunesh's inputs.
    """
    store_id: str
    sku_id: str
    date: str
    service_level_target: float = Field(default=0.95, description="95% cycle service level")
    supplier_lead_time_days: float = Field(default=1.5, description="Mean supplier delivery lead time (LT_avg)")
    supplier_lead_time_std: float = Field(default=0.3, description="Lead time variance (sigma_LT)")
    safety_stock_units: float = Field(..., ge=0.0, description="SS = Z * sqrt(LT*sigma_D^2 + D^2*sigma_LT^2)")
    reorder_point_units: float = Field(..., ge=0.0, description="ROP = D_avg * LT + SS")
    recommended_po_quantity: float = Field(..., ge=0.0, description="Automated purchase order quantity")
    po_urgency: Literal["NORMAL", "EXPEDITE", "HOLD"] = Field(...)


class RajatCircularityDecisionPayload(BaseModel):
    """
    Downstream payload generated by Rajat's Circularity Agent using Arunesh's inputs.
    """
    store_id: str
    sku_id: str
    date: str
    overstock_risk_units: float = Field(..., ge=0.0)
    markdown_discount_pct: float = Field(..., ge=0.0, le=50.0, description="Automated discount (e.g. 15-40% off)")
    food_rescue_eligible: bool = Field(...)
    circularity_action: Literal["STANDARD_PRICING", "DYNAMIC_MARKDOWN_20", "DYNAMIC_MARKDOWN_40", "DIVERT_TO_FOOD_BANK"] = Field(...)


# ===========================================================================
# 2. AUTONOMOUS AGENT INTERFACES
# ===========================================================================

class DarkStoreDemandAgent:
    """
    Upstream Intelligence Agent that assembles probabilistic macro forecasts,
    multi-horizon horizons, item accuracy, and circularity risk.
    """
    def __init__(self):
        # Empirical historical accuracy established in Step 2
        self.category_wape_lookup = {
            "Produce": 14.90,
            "Dairy"  : 16.58,
            "Bakery" : 16.76,
            "Meat"   : 19.49
        }

    def generate_forecast_contract(
        self,
        store_id: str,
        sku_id: str,
        category: str,
        date: str,
        p10: float,
        p50: float,
        p90: float,
        closing_stock: float,
        remaining_shelf_life: int,
        committed_preorders: int = 0
    ) -> StoreDemandForecastPayload:
        # Enforce mathematical lower clamping to avoid degenerate division/NaN/negatives
        p10_c = max(0.0, float(p10))
        p50_c = max(p10_c, float(p50))
        p90_c = max(p50_c, float(p90))
        closing_stock_c = max(0.0, float(closing_stock))
        remaining_sl_c  = max(0, int(remaining_shelf_life))
        
        # Avoid zero division: minimum daily demand rate
        p50_safe = max(0.01, p50_c)
        sigma_d = max(0.10, (p90_c - p10_c) / (2.0 * 1.2816))
        
        # Multi-horizon cumulative calculations:
        # 2-day short horizon for dark-store fulfillment buffer
        short_2d = round(p50_safe * 2.0, 2)
        # 14-day long horizon for supplier purchase planning
        long_14d = round(p50_safe * 14.0, 2)
        
        # Item accuracy & trust
        cat_wape = self.category_wape_lookup.get(category, 16.20)
        trust_factor = max(0.50, min(0.98, 1.0 - (cat_wape / 100.0)))
        
        # Overstock Spoilage Risk for Rajat:
        # Unsold units risking expiry = max(0, Closing_Stock - p50 * Remaining_SL)
        expected_sl_clearance = p50_safe * max(1, remaining_sl_c)
        overstock_units = max(0.0, closing_stock_c - expected_sl_clearance)
        
        # Operational recommendation logic
        if overstock_units > (p50_safe * 2.0) and remaining_sl_c <= 2:
            action = "TRIGGER_MARKDOWN_DISCOUNT"
            explanation = (
                f"Circularity Alert: Current stock ({closing_stock_c:.0f}) exceeds projected clearance "
                f"({expected_sl_clearance:.0f}) before expiry in {remaining_sl_c} days. "
                f"Dynamic markdown recommended to prevent landfill waste."
            )
        elif closing_stock_c < (p50_safe * 1.5):
            action = "URGENT_RESTOCK"
            explanation = (
                f"Stockout Alert: Current stock ({closing_stock_c:.0f}) covers less than 1.5 days "
                f"of median demand ({p50_safe:.1f}/day). Reorder triggered."
            )
        else:
            action = "STANDARD_REPLENISHMENT"
            explanation = (
                f"Stable operational status. Daily median forecast: {p50_safe:.1f} units "
                f"(Range: {p10_c:.1f} to {p90_c:.1f}). Stock levels adequate."
            )
            
        return StoreDemandForecastPayload(
            store_id=store_id,
            sku_id=sku_id,
            category=category,
            date=date,
            p10_demand=round(p10_c, 2),
            p50_demand=round(p50_c, 2),
            p90_demand=round(p90_c, 2),
            sigma_demand=round(sigma_d, 3),
            short_horizon_demand_2d=short_2d,
            long_horizon_demand_14d=long_14d,
            committed_preorders_qty=int(committed_preorders),
            historical_item_wape=round(cat_wape, 2),
            forecast_confidence_score=round(trust_factor, 3),
            forecast_explanation=explanation,
            closing_stock=round(closing_stock_c, 2),
            remaining_shelf_life_days=remaining_sl_c,
            overstock_risk_units=round(overstock_units, 2),
            recommended_action=action
        )


# ===========================================================================
# 3. DOWNSTREAM RECEIVER HANDSHAKE ENGINES (Deepanshu & Rajat)
# ===========================================================================

class MockDeepanshuInventoryAgent:
    """
    Executes Section 4 Dossier equations using StoreDemandForecastPayload.
    """
    def __init__(self, z_service_level: float = 1.645):
        self.z = z_service_level  # 95% cycle service level (Z = 1.645)

    def compute_inventory_decisions(self, payload: StoreDemandForecastPayload) -> DeepanshuInventoryDecisionPayload:
        lt_avg = 1.5   # Mean supplier lead time = 1.5 days in Bangalore
        sigma_lt = 0.3 # Lead time std = 0.3 days
        
        d_avg = payload.p50_demand
        sigma_d = payload.sigma_demand
        
        # Safety Stock formula from Academic Dossier Section 4:
        # SS = Z * sqrt( (LT_avg * sigma_D^2) + (D_avg^2 * sigma_LT^2) )
        term_d  = lt_avg * (sigma_d ** 2)
        term_lt = (d_avg ** 2) * (sigma_lt ** 2)
        ss = self.z * np.sqrt(term_d + term_lt)
        
        # Reorder Point formula:
        # ROP = (D_avg * LT_avg) + SS
        rop = (d_avg * lt_avg) + ss
        
        # Purchase Order quantity calculation:
        # If Current Stock <= ROP, order up to 3 days of demand + Safety Stock - Committed Preorders
        if payload.closing_stock <= rop:
            recommended_po = max(0.0, (d_avg * 3.0 + ss) - payload.closing_stock + payload.committed_preorders_qty)
            urgency = "EXPEDITE" if payload.closing_stock < (d_avg * 1.0) else "NORMAL"
        else:
            recommended_po = 0.0
            urgency = "HOLD"
            
        return DeepanshuInventoryDecisionPayload(
            store_id=payload.store_id,
            sku_id=payload.sku_id,
            date=payload.date,
            service_level_target=0.95,
            supplier_lead_time_days=lt_avg,
            supplier_lead_time_std=sigma_lt,
            safety_stock_units=round(float(ss), 2),
            reorder_point_units=round(float(rop), 2),
            recommended_po_quantity=round(float(recommended_po), 2),
            po_urgency=urgency
        )


class MockRajatCircularityAgent:
    """
    Executes Waste Prevention & Dynamic Markdown Routing using StoreDemandForecastPayload.
    """
    def compute_circularity_decisions(self, payload: StoreDemandForecastPayload) -> RajatCircularityDecisionPayload:
        overstock = payload.overstock_risk_units
        remaining_sl = payload.remaining_shelf_life_days
        
        if overstock > 0 and remaining_sl <= 1:
            discount = 40.0
            action = "DYNAMIC_MARKDOWN_40"
            rescue = True
        elif overstock > 0 and remaining_sl <= 2:
            discount = 20.0
            action = "DYNAMIC_MARKDOWN_20"
            rescue = False
        elif overstock > (payload.p50_demand * 3.0) and remaining_sl <= 1:
            discount = 50.0
            action = "DIVERT_TO_FOOD_BANK"
            rescue = True
        else:
            discount = 0.0
            action = "STANDARD_PRICING"
            rescue = False
            
        return RajatCircularityDecisionPayload(
            store_id=payload.store_id,
            sku_id=payload.sku_id,
            date=payload.date,
            overstock_risk_units=overstock,
            markdown_discount_pct=discount,
            food_rescue_eligible=rescue,
            circularity_action=action
        )


# ===========================================================================
# 4. STEP 4 VERIFICATION SUITE
# ===========================================================================

def run_step4_tests() -> dict:
    log("Running Step 4 Verification Suite...", "HEAD")
    
    agent = DarkStoreDemandAgent()
    deepanshu_engine = MockDeepanshuInventoryAgent()
    rajat_engine = MockRajatCircularityAgent()
    
    # ── TEST 4.1: Pydantic Contract & Schema Validation ────────────────────
    log("TEST 4.1 — Pydantic Schema Validation (1,000 Real & Synthetic Payloads)", "HEAD")
    
    # Load actual predictions from Step 2 to validate against real models
    p2_path = PROC_DIR / "forecast_payloads_step2.parquet"
    if p2_path.exists():
        df_p2 = pd.read_parquet(p2_path)
    else:
        raise FileNotFoundError("Missing forecast_payloads_step2.parquet from Step 2.")
        
    validation_passes = 0
    validation_failures = 0
    validated_payloads: List[StoreDemandForecastPayload] = []
    
    # Sample 1,000 payloads
    sample_df = df_p2.sample(min(1000, len(df_p2)), random_state=42)
    
    for _, row in sample_df.iterrows():
        try:
            payload = agent.generate_forecast_contract(
                store_id=row["Store_ID"],
                sku_id=row["SKU_ID"],
                category=row["Category"],
                date=str(row["Date"])[:10],
                p10=row["p10_demand"],
                p50=row["p50_demand"],
                p90=row["p90_demand"],
                closing_stock=row["Closing_Stock"],
                remaining_shelf_life=int(row["Remaining_SL_Days"]),
                committed_preorders=np.random.randint(0, 15)
            )
            # Serialize to JSON and parse back to verify round-trip contract
            json_str = payload.model_dump_json()
            _ = StoreDemandForecastPayload.model_validate_json(json_str)
            validated_payloads.append(payload)
            validation_passes += 1
        except Exception as e:
            validation_failures += 1
            log(f"Validation failure: {e}", "FAIL")
            
    pass_rate_schema = (validation_passes / len(sample_df)) * 100
    status_schema = "INFO" if pass_rate_schema == 100.0 else "FAIL"
    log(f"Schema Validation Pass Rate : {pass_rate_schema:.2f}%  (Target: 100.0%)", status_schema)
    log(f"Total Validated Contracts   : {validation_passes:,} / {len(sample_df):,}")
    
    # ── TEST 4.2: Boundary Value Stress Test ──────────────────────────────
    log("TEST 4.2 — Boundary Value Stress Test (Zero / Inverted / Extreme Surges)", "HEAD")
    boundary_anomalies = 0
    stress_cases = [
        # Zero stock
        {"p10": 0.0, "p50": 0.0, "p90": 0.0, "stock": 0.0, "sl": 0},
        # Extreme surge (100x demand)
        {"p10": 500.0, "p50": 1500.0, "p90": 3000.0, "stock": 50.0, "sl": 1},
        # High stock, 0 remaining shelf life (Perishability cliff)
        {"p10": 10.0, "p50": 20.0, "p90": 35.0, "stock": 500.0, "sl": 0},
        # Negative incoming raw predictions (model noise resilience)
        {"p10": -5.0, "p50": -1.0, "p90": 12.0, "stock": -10.0, "sl": -1},
    ]
    
    for idx, sc in enumerate(stress_cases):
        try:
            p = agent.generate_forecast_contract(
                store_id="DS_BLR_STRESS",
                sku_id="SKU_STRESS_TEST",
                category="Dairy",
                date="2024-12-31",
                p10=sc["p10"],
                p50=sc["p50"],
                p90=sc["p90"],
                closing_stock=sc["stock"],
                remaining_shelf_life=sc["sl"],
                committed_preorders=0
            )
            # Test downstream receivers with stress payload
            inv_res = deepanshu_engine.compute_inventory_decisions(p)
            circ_res = rajat_engine.compute_circularity_decisions(p)
            
            # Check for NaN / Inf / Negative outputs
            for val in [p.p10_demand, p.p50_demand, p.p90_demand, p.sigma_demand, inv_res.safety_stock_units, inv_res.reorder_point_units]:
                if np.isnan(val) or np.isinf(val) or val < 0:
                    boundary_anomalies += 1
                    log(f"Stress Case {idx+1} produced anomaly: {val}", "FAIL")
                    
        except Exception as e:
            boundary_anomalies += 1
            log(f"Stress Case {idx+1} crashed: {e}", "FAIL")
            
    status_stress = "INFO" if boundary_anomalies == 0 else "FAIL"
    log(f"Boundary Anomalies Detected : {boundary_anomalies}  (Target: 0)", status_stress)
    log(f"Mathematical Bounds Resilient: Clamping enforced D_avg >= 0.01, sigma_D >= 0.10, SS >= 0.0")
    
    # ── TEST 4.3: End-to-End Handshake Latency Benchmark ───────────────────
    log("TEST 4.3 — End-to-End Handshake Latency Benchmark (100 SKUs batch)", "HEAD")
    
    batch_sample = validated_payloads[:100]
    t_start = time.perf_counter()
    for p in batch_sample:
        _ = deepanshu_engine.compute_inventory_decisions(p)
        _ = rajat_engine.compute_circularity_decisions(p)
        _ = p.model_dump_json()
    total_latency_ms = (time.perf_counter() - t_start) * 1000
    avg_latency_per_sku_ms = total_latency_ms / len(batch_sample)
    
    status_lat = "INFO" if total_latency_ms <= 50.0 else "WARN"
    log(f"Batch Handshake Latency (100 SKUs): {total_latency_ms:.2f} ms  (Target: <= 50.0 ms)", status_lat)
    log(f"Average Handshake Latency / SKU   : {avg_latency_per_sku_ms:.3f} ms")
    
    # ── Generate Full Downstream Handshake Dataset ────────────────────────
    log("Generating Full Downstream Handshake Dataset for Team Leads...", "HEAD")
    records = []
    deepanshu_samples = []
    rajat_samples = []
    
    for p in validated_payloads:
        inv = deepanshu_engine.compute_inventory_decisions(p)
        circ = rajat_engine.compute_circularity_decisions(p)
        
        row_dict = p.model_dump()
        row_dict.update({
            "deepanshu_safety_stock": inv.safety_stock_units,
            "deepanshu_reorder_point": inv.reorder_point_units,
            "deepanshu_recommended_po_qty": inv.recommended_po_quantity,
            "deepanshu_po_urgency": inv.po_urgency,
            "rajat_markdown_discount_pct": circ.markdown_discount_pct,
            "rajat_food_rescue_eligible": circ.food_rescue_eligible,
            "rajat_circularity_action": circ.circularity_action
        })
        records.append(row_dict)
        
        if len(deepanshu_samples) < 3:
            deepanshu_samples.append(inv.model_dump())
            rajat_samples.append(circ.model_dump())
            
    df_step4 = pd.DataFrame(records)
    out_parquet = PROC_DIR / "agent_handoff_payloads_step4.parquet"
    df_step4.to_parquet(out_parquet, index=False, compression="snappy")
    log(f"Saved consolidated multi-agent payloads ({len(df_step4):,} rows) -> {out_parquet}")
    
    # Save sample JSONs for Deepanshu & Rajat
    json_deepanshu_path = PROC_DIR / "sample_deepanshu_handoff.json"
    json_rajat_path     = PROC_DIR / "sample_rajat_handoff.json"
    with open(json_deepanshu_path, "w", encoding="utf-8") as f:
        json.dump(deepanshu_samples, f, indent=2)
    with open(json_rajat_path, "w", encoding="utf-8") as f:
        json.dump(rajat_samples, f, indent=2)
        
    log(f"Sample JSON contract for Deepanshu -> {json_deepanshu_path}")
    log(f"Sample JSON contract for Rajat     -> {json_rajat_path}")
    
    passed_4_1 = (pass_rate_schema == 100.0)
    passed_4_2 = (boundary_anomalies == 0)
    passed_4_3 = (total_latency_ms <= 50.0)
    all_passed = passed_4_1 and passed_4_2 and passed_4_3
    
    return {
        "pass_rate_schema": pass_rate_schema,
        "boundary_anomalies": boundary_anomalies,
        "total_latency_ms": round(total_latency_ms, 2),
        "avg_latency_per_sku_ms": round(avg_latency_per_sku_ms, 3),
        "passed_4_1": bool(passed_4_1),
        "passed_4_2": bool(passed_4_2),
        "passed_4_3": bool(passed_4_3),
        "all_passed": bool(all_passed)
    }


# ===========================================================================
# 5. TEST REPORT WRITER
# ===========================================================================
def write_step4_report(results: dict):
    report_path = OUT_DIR / "step4_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("STEP 4 — MULTI-AGENT PROTOCOL & DOWNSTREAM HANDSHAKE REPORT\n")
        f.write("Role: Arunesh Sharma (Role #5) — AI/ML Demand & Consumption Lead\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("1. DOWNSTREAM CONTRACTS DELIVERED TO DEEPANSHU (ROLE #6):\n")
        f.write("  • Store-level forecast    : [Store_ID, SKU_ID, Date] daily stream\n")
        f.write("  • Low/Likely/High Range   : p10_demand, p50_demand (D_avg), p90_demand, sigma_demand\n")
        f.write("  • Multi-Horizon Demand    : 2-day short horizon (fulfilment) & 14-day long horizon (procurement)\n")
        f.write("  • Committed Preorders     : committed_preorders_qty aggregated from Step 3\n")
        f.write("  • Past Accuracy & Trust   : historical_item_wape (14.9-19.5%) & forecast_confidence_score\n")
        f.write("  • Explainable Rationale   : forecast_explanation plain-English string\n\n")
        
        f.write("2. DOWNSTREAM CONTRACTS DELIVERED TO RAJAT (ROLE #7):\n")
        f.write("  • Overstock Risk (Omega_t): max(0, Closing_Stock - D_avg * Remaining_SL)\n")
        f.write("  • Dynamic Markdown Flags  : TRIGGER_MARKDOWN_DISCOUNT (20% or 40% off)\n")
        f.write("  • Food Rescue Diversion   : DIVERT_TO_FOOD_BANK for high overstock near expiry\n\n")
        
        f.write("3. QUANTITATIVE VERIFICATION BENCHMARKS:\n")
        f.write(f"  • Test 4.1: Pydantic Schema Validation : {results['pass_rate_schema']}% (Target: 100.0%) -> {'PASSED [OK]' if results['passed_4_1'] else 'FAILED'}\n")
        f.write(f"  • Test 4.2: Boundary Anomalies (0 NaN) : {results['boundary_anomalies']} (Target: 0) -> {'PASSED [OK]' if results['passed_4_2'] else 'FAILED'}\n")
        f.write(f"  • Test 4.3: Batch Handshake Latency    : {results['total_latency_ms']} ms (Target: <= 50.0 ms) -> {'PASSED [OK]' if results['passed_4_3'] else 'FAILED'}\n")
        f.write(f"  • Test 4.3: Per-SKU Latency            : {results['avg_latency_per_sku_ms']} ms/SKU\n\n")
        
        f.write("4. OVERALL STEP 4 STATUS:\n")
        f.write(f"  STATUS: {'PASSED [OK]' if results['all_passed'] else 'REQUIRES REVIEW'}\n\n")
        f.write("-" * 65 + "\n")
        f.write("EXECUTION TRACE LOG:\n")
        f.write("\n".join(test_log))
        
    log(f"Audit report written -> {report_path}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("STEP 4 — MULTI-AGENT PROTOCOL & DOWNSTREAM HANDSHAKE")
    print("=" * 65)
    
    results = run_step4_tests()
    write_step4_report(results)
    
    print("\n" + "=" * 65)
    print("STEP 4 EXECUTION SUMMARY")
    print("=" * 65)
    print(f"  Test 4.1 Schema Validation  : {'PASSED [OK]' if results['passed_4_1'] else 'FAILED'}")
    print(f"  Test 4.2 Boundary Value     : {'PASSED [OK]' if results['passed_4_2'] else 'FAILED'}")
    print(f"  Test 4.3 Handshake Latency  : {'PASSED [OK]' if results['passed_4_3'] else 'FAILED'}")
    print(f"\n  Overall Step 4 Status       : {'ALL SUCCESS METRICS ACHIEVED [OK]' if results['all_passed'] else 'CONTINGENCY PROTOCOL ENGAGED'}")
    print(f"  Schema Pass Rate            : {results['pass_rate_schema']}% (Target: 100.0%)")
    print(f"  Boundary Anomalies (0 NaN)  : {results['boundary_anomalies']} (Target: 0)")
    print(f"  Batch Latency (100 SKUs)    : {results['total_latency_ms']} ms (Target: <= 50.0 ms)")
    print("=" * 65)
