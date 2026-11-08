"""
Lead 2 Core Inventory Calculation Engine
========================================
Implements deterministic, mathematically rigorous inventory policy calculations:
1. Average Daily Demand (D_avg from forecast P50)
2. Dynamic Safety Stock (SS):
     SS = Z * sqrt(LT_mean * sigma_D^2 + D_avg^2 * sigma_LT^2)
   Accounts simultaneously for daily demand uncertainty (sigma_D) and supplier
   delivery lead-time variability (sigma_LT).
3. Reorder Point (ROP):
     ROP = (D_avg * LT_mean) + SS
4. Target Stock (Order-Up-To Level, S):
     S = D_avg * (R + LT_mean) + SS
   Where R is the replenishment review period in days.
5. Inventory Position & Replenishment Decision:
     IP = on_hand_stock + on_order_stock - reserved_stock
     reorder_triggered = (IP <= ROP)
     suggested_replenishment_qty = max(0.0, S - IP) if reorder_triggered else 0.0

OPERATIONAL SAFETY & DATA INTEGRITY CONSTRAINTS:
- Relies exclusively on validated forecast P50 (never consumes Target_Demand).
- Treats diagnostic household alerts (unconfirmed_alert_count) as unconfirmed signals;
  never treats them as firm orders, reservations, or deductions from demand.
- Requires explicit, trusted beginning-of-day stock (Opening_Stock).
- Does not invent pending POs or reservations; defaults explicitly to 0.0 if not provided.
- Purely deterministic, fully vectorized/testable, and free of side effects.
- Leaves MOQ, case pack, supplier selection, and perishability markdown to downstream agents.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..common.config import Lead2Config, SKU_TO_CATEGORY, SupplierSimulationParams, get_default_config
from ..common.schemas import (
    DailyAgentInput,
    DemandForecast,
    ExpiryRiskLevel,
    InventoryBatch,
    InventoryCalculationResult,
    InventoryPosition,
    InventoryRiskAssessment,
    StockRiskLevel,
)


def compute_average_daily_demand(forecast: DemandForecast) -> float:
    """
    Extracts the median expected daily demand (D_avg) from forecast P50.
    
    Raises:
        ValueError: If forecast demand is negative or non-finite.
    """
    d_avg = float(forecast.p50_demand)
    if not math.isfinite(d_avg) or d_avg < 0.0:
        raise ValueError(f"Invalid average daily demand {d_avg}. Must be non-negative and finite.")
    return d_avg


def compute_safety_stock(
    d_avg: float,
    sigma_d: float,
    lt_mean: float,
    sigma_lt: float,
    z: float,
) -> float:
    """
    Computes dynamic lead-time safety stock under demand uncertainty and lead-time variability:
        SS_LT = Z * sqrt(LT_mean * sigma_D^2 + D_avg^2 * sigma_LT^2)

    Used specifically for calculating the continuous/lead-time Reorder Point (ROP).

    Args:
        d_avg: Expected daily demand (units/day), >= 0.
        sigma_d: Daily demand uncertainty standard deviation (units/day), >= 0.
        lt_mean: Mean supplier delivery lead time (days), > 0.
        sigma_lt: Supplier lead time standard deviation (days), >= 0.
        z: Service level factor (Z-score, e.g. 1.645 for 95% CSL), >= 0.

    Returns:
        Lead-time safety stock in units (float >= 0.0).

    Raises:
        ValueError: If any parameter is negative, non-finite, or if lt_mean <= 0.
    """
    for name, val in [("d_avg", d_avg), ("sigma_d", sigma_d), ("sigma_lt", sigma_lt), ("z", z)]:
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"Parameter '{name}' must be non-negative and finite, got {val}.")

    if not math.isfinite(lt_mean) or lt_mean <= 0.0:
        raise ValueError(f"Parameter 'lt_mean' must be strictly positive (> 0), got {lt_mean}.")

    # Component variances:
    # 1. Variance in demand during average lead time: LT_mean * sigma_D^2
    var_demand = lt_mean * (sigma_d ** 2)
    # 2. Variance in lead time under average demand: D_avg^2 * sigma_LT^2
    var_lead_time = (d_avg ** 2) * (sigma_lt ** 2)

    total_var = var_demand + var_lead_time
    ss = z * math.sqrt(total_var)
    return float(ss)


def compute_protection_safety_stock(
    d_avg: float,
    sigma_d: float,
    lt_mean: float,
    review_period: float,
    sigma_lt: float,
    z: float,
) -> float:
    """
    Computes dynamic safety stock over the full exposure horizon (review period R + lead time LT_mean):
        SS_P = Z * sqrt((LT_mean + R) * sigma_D^2 + D_avg^2 * sigma_LT^2)

    Under periodic review (R > 0), stockouts can occur anytime between order placement and the
    arrival of the NEXT order, spanning (R + LT_mean) days. Assuming deterministic review period R,
    this buffers against demand variance across the full protection interval.

    Used specifically for calculating Target Stock (Order-Up-To Level, S).
    When review_period == 0.0, SS_P reduces identically to SS_LT.

    Args:
        d_avg: Expected daily demand (units/day), >= 0.
        sigma_d: Daily demand uncertainty standard deviation (units/day), >= 0.
        lt_mean: Mean supplier delivery lead time (days), > 0.
        review_period: Review interval (days), >= 0.
        sigma_lt: Supplier lead time standard deviation (days), >= 0.
        z: Service level factor (Z-score), >= 0.

    Returns:
        Protection safety stock in units (float >= 0.0).

    Raises:
        ValueError: If any parameter is negative, non-finite, or if lt_mean <= 0.
    """
    for name, val in [
        ("d_avg", d_avg),
        ("sigma_d", sigma_d),
        ("review_period", review_period),
        ("sigma_lt", sigma_lt),
        ("z", z),
    ]:
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"Parameter '{name}' must be non-negative and finite, got {val}.")

    if not math.isfinite(lt_mean) or lt_mean <= 0.0:
        raise ValueError(f"Parameter 'lt_mean' must be strictly positive (> 0), got {lt_mean}.")

    # Exposure horizon across review period and lead time
    exposure_horizon = lt_mean + review_period
    var_demand = exposure_horizon * (sigma_d ** 2)
    var_lead_time = (d_avg ** 2) * (sigma_lt ** 2)

    total_var = var_demand + var_lead_time
    ss_p = z * math.sqrt(total_var)
    return float(ss_p)


def compute_reorder_point(
    d_avg: float,
    lt_mean: float,
    safety_stock: float,
) -> float:
    """
    Computes continuous/periodic review reorder point (ROP):
        ROP = (D_avg * LT_mean) + SS

    Args:
        d_avg: Expected daily demand (units/day), >= 0.
        lt_mean: Mean supplier delivery lead time (days), > 0.
        safety_stock: Safety stock buffer (units), >= 0.

    Returns:
        Reorder point in units (float >= 0.0).
    """
    for name, val in [("d_avg", d_avg), ("safety_stock", safety_stock)]:
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"Parameter '{name}' must be non-negative and finite, got {val}.")

    if not math.isfinite(lt_mean) or lt_mean <= 0.0:
        raise ValueError(f"Parameter 'lt_mean' must be strictly positive (> 0), got {lt_mean}.")

    rop = (d_avg * lt_mean) + safety_stock
    return float(rop)


def compute_target_stock(
    d_avg: float,
    lt_mean: float,
    review_period: float,
    safety_stock: float,
) -> float:
    """
    Computes order-up-to target stock level (S):
        S = D_avg * (R + LT_mean) + SS

    Covers expected demand across the entire exposure period (review period R + lead time LT_mean)
    plus safety stock buffer.

    Args:
        d_avg: Expected daily demand (units/day), >= 0.
        lt_mean: Mean supplier delivery lead time (days), > 0.
        review_period: Inventory review interval (days), >= 0.
        safety_stock: Safety stock buffer (units), >= 0.

    Returns:
        Target stock in units (float >= 0.0).
    """
    for name, val in [("d_avg", d_avg), ("review_period", review_period), ("safety_stock", safety_stock)]:
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"Parameter '{name}' must be non-negative and finite, got {val}.")

    if not math.isfinite(lt_mean) or lt_mean <= 0.0:
        raise ValueError(f"Parameter 'lt_mean' must be strictly positive (> 0), got {lt_mean}.")

    target_stock = (d_avg * (review_period + lt_mean)) + safety_stock
    return float(target_stock)


def compute_suggested_replenishment_qty(
    total_inventory_position: float,
    rop: float,
    target_stock: float,
) -> Tuple[bool, float]:
    """
    Determines whether a reorder is triggered and calculates unconstrained replenishment quantity:
        reorder_triggered = (total_inventory_position <= ROP)
        suggested_qty = max(0.0, target_stock - total_inventory_position) if reorder_triggered else 0.0

    Args:
        total_inventory_position: On-hand + On-order - Reserved stock (units).
        rop: Reorder point (units).
        target_stock: Order-up-to level (units).

    Returns:
        Tuple of (reorder_triggered: bool, suggested_qty: float).
    """
    for name, val in [
        ("total_inventory_position", total_inventory_position),
        ("rop", rop),
        ("target_stock", target_stock),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}.")

    if total_inventory_position <= rop:
        reorder_triggered = True
        suggested_qty = max(0.0, target_stock - total_inventory_position)
    else:
        reorder_triggered = False
        suggested_qty = 0.0

    return reorder_triggered, float(suggested_qty)


def compute_surge_coverage_ratio_p90(
    total_inventory_position: float,
    p90_demand: float,
    lt_mean: float,
) -> Optional[float]:
    """
    Computes the surge coverage ratio under a deterministic P90 peak demand stress scenario:
        surge_coverage_ratio_p90 = IP / (P90 * LT_mean)

    STATISTICAL & SAFETY GOVERNANCE:
    - Deterministic Stress Scenario: This is a stress evaluation metric, NOT a calibrated multi-day prediction.
    - Zero/Non-positive handling: Safely returns None if p90_demand <= 0.0 or lt_mean <= 0.0,
      preventing ZeroDivisionError, infinite values, or division by non-positive rates.
    - Requires finite total_inventory_position, p90_demand >= 0.0, and lt_mean > 0.0.

    Args:
        total_inventory_position: Total inventory position (units).
        p90_demand: 90th percentile daily demand forecast (units/day), >= 0.
        lt_mean: Supplier lead time mean (days), > 0.

    Returns:
        Surge coverage ratio (float >= 0.0) or None if p90_demand <= 0.0 or lt_mean <= 0.0.
    """
    if not math.isfinite(total_inventory_position):
        raise ValueError(f"total_inventory_position must be finite, got {total_inventory_position}.")
    if not math.isfinite(p90_demand) or p90_demand < 0.0:
        raise ValueError(f"p90_demand must be non-negative and finite, got {p90_demand}.")
    if not math.isfinite(lt_mean) or lt_mean <= 0.0:
        raise ValueError(f"lt_mean must be strictly positive (> 0), got {lt_mean}.")

    p90_ltd = p90_demand * lt_mean
    if p90_ltd <= 0.0:
        return None

    return float(total_inventory_position / p90_ltd)


def validate_batch_inventory(
    batches: Optional[List[InventoryBatch]],
    available_physical_stock: Optional[float] = None,
    require_full_coverage: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validates batch records against available physical stock and internal consistency rules.

    Integrity rules:
    1. If batches is None or empty: returns (True, []) - treated as unprovided data (NOT_ASSESSABLE).
    2. Batch fields must be finite and valid (units > 0, remaining_shelf_life_days >= 0).
    3. Batch IDs must be unique across the collection (no duplicate batch/lot IDs).
    4. If available_physical_stock is specified:
       - available_physical_stock must be finite and >= 0.
       - If available_physical_stock <= 0 and batches is non-empty: inconsistent.
       - Total batch units must not exceed available physical stock.
       - If require_full_coverage=True and total batch units < available_physical_stock: inconsistent.

    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
    """
    if not batches:
        return True, []

    issues: List[str] = []
    seen_ids = set()
    total_units = 0.0

    for b in batches:
        if not math.isfinite(b.units) or b.units <= 0.0:
            issues.append(f"Batch '{b.batch_id}' has invalid non-positive or non-finite units: {b.units}.")
        else:
            total_units += b.units

        if b.remaining_shelf_life_days < 0:
            issues.append(f"Batch '{b.batch_id}' has negative shelf life days: {b.remaining_shelf_life_days}.")

        if b.batch_id in seen_ids:
            issues.append(f"Duplicate batch_id '{b.batch_id}' detected in batch records.")
        seen_ids.add(b.batch_id)

    if available_physical_stock is not None:
        if not math.isfinite(available_physical_stock) or available_physical_stock < 0.0:
            issues.append(
                f"Available physical stock must be non-negative and finite, got {available_physical_stock}."
            )
        elif available_physical_stock <= 0.0 and total_units > 0.0:
            issues.append(
                f"Batches provided ({total_units:.1f} units) but available physical stock is zero ({available_physical_stock:.1f})."
            )
        elif total_units > available_physical_stock + 1e-4:
            issues.append(
                f"Total batch quantity ({total_units:.1f} units) exceeds available physical stock ({available_physical_stock:.1f} units)."
            )
        elif require_full_coverage and total_units < available_physical_stock - 1e-4:
            issues.append(
                f"Total batch quantity ({total_units:.1f} units) does not cover available physical stock ({available_physical_stock:.1f} units); {available_physical_stock - total_units:.1f} units lack batch tracking."
            )

    is_valid = len(issues) == 0
    return is_valid, issues


def assess_stock_risk(
    calc: InventoryCalculationResult,
    forecast: DemandForecast,
) -> Tuple[StockRiskLevel, bool, List[str], List[str], float, float, Optional[float], Optional[float]]:
    """
    Evaluates stock level risks (Critical Low, Low Reorder, Healthy, Excess)
    and computes deterministic stress scenarios (P10 downside clearance, P90 demand surge;
    not calibrated multi-day predictions).

    Returns:
        Tuple of:
        - StockRiskLevel
        - reorder_recommended (bool)
        - stockout_risk_reasons (List[str])
        - excess_risk_reasons (List[str])
        - excess_quantity (float)
        - excess_quantity_p10 (float)
        - surge_coverage_ratio_p90 (Optional[float])
        - days_of_supply_p50 (Optional[float])
    """
    ip = calc.total_inventory_position
    on_hand = calc.inventory_position.on_hand_stock
    rop = calc.reorder_point
    target_stock = calc.target_stock
    ss_lt = calc.safety_stock_lead_time
    lt_mean = calc.lead_time_mean_days
    review_period = calc.review_period_days
    ss_p = calc.safety_stock_protection

    stockout_reasons: List[str] = []
    excess_reasons: List[str] = []

    # 1. P90 surge coverage metric under deterministic stress scenario: IP / (P90 * LT_mean)
    # Safely handles P90 == 0.0 without division by zero.
    p90_ltd = forecast.p90_demand * lt_mean
    surge_ratio = compute_surge_coverage_ratio_p90(
        total_inventory_position=ip,
        p90_demand=forecast.p90_demand,
        lt_mean=lt_mean,
    )

    # 2. P50 Days of supply: IP / P50
    if calc.average_daily_demand > 0.0:
        dos_p50 = ip / calc.average_daily_demand
    else:
        dos_p50 = None

    # 3. Excess calculations:
    excess_qty = max(0.0, ip - target_stock)

    # P10 clearance demand stress scenario over exposure period (not a calibrated multi-day prediction)
    p10_exposure_target = (forecast.p10_demand * (review_period + lt_mean)) + ss_p
    excess_qty_p10 = max(0.0, ip - p10_exposure_target)

    # 4. Stock risk level determination
    if on_hand <= 0.0 or ip <= ss_lt:
        level = StockRiskLevel.CRITICAL_LOW
        reorder_recommended = True
        if on_hand <= 0.0:
            stockout_reasons.append("Critical Stockout: Physical on-hand inventory is completely depleted (0 units).")
        if ip <= ss_lt:
            stockout_reasons.append(
                f"Critical Low Stock: Inventory position ({ip:.2f}) has breached lead-time safety stock ({ss_lt:.2f})."
            )
        if surge_ratio is not None and surge_ratio < 1.0:
            stockout_reasons.append(
                f"Surge Vulnerability (Deterministic Stress Scenario): Current inventory covers only {surge_ratio:.1%} of deterministic P90 peak demand stress over lead time ({p90_ltd:.2f} units; not a calibrated multi-day prediction)."
            )

    elif ip <= rop:
        level = StockRiskLevel.LOW_STOCK_REORDER
        reorder_recommended = True
        stockout_reasons.append(
            f"Reorder Triggered: Inventory position ({ip:.2f}) is at or below the reorder point ({rop:.2f})."
        )
        if surge_ratio is not None and surge_ratio < 1.0:
            stockout_reasons.append(
                f"Surge Vulnerability (Deterministic Stress Scenario): Current inventory covers only {surge_ratio:.1%} of deterministic P90 peak demand stress over lead time ({p90_ltd:.2f} units; not a calibrated multi-day prediction)."
            )

    elif ip > target_stock:
        level = StockRiskLevel.EXCESS_STOCK
        reorder_recommended = False
        excess_reasons.append(
            f"Excess Stock: Inventory position ({ip:.2f}) exceeds target stock ({target_stock:.2f}) by {excess_qty:.2f} units."
        )
        if excess_qty_p10 > excess_qty:
            excess_reasons.append(
                f"Sluggish Clearance Vulnerability (Deterministic Stress Scenario): Under downside P10 demand stress scenario ({forecast.p10_demand:.2f} units/day), excess rises to {excess_qty_p10:.2f} units (not a calibrated multi-day prediction)."
            )
        if dos_p50 is not None:
            excess_reasons.append(f"Holding approximately {dos_p50:.1f} days of supply at median demand velocity.")

    else:
        level = StockRiskLevel.HEALTHY
        reorder_recommended = False
        if surge_ratio is not None and surge_ratio < 1.0:
            stockout_reasons.append(
                f"Surge Advisory (Deterministic Stress Scenario): Inventory position ({ip:.2f}) is within normal operating band but would deplete under a deterministic P90 demand surge stress scenario ({p90_ltd:.2f} units; not a calibrated multi-day prediction)."
            )

    return (
        level,
        reorder_recommended,
        stockout_reasons,
        excess_reasons,
        float(excess_qty),
        float(excess_qty_p10),
        float(surge_ratio) if surge_ratio is not None else None,
        float(dos_p50) if dos_p50 is not None else None,
    )


def assess_expiry_risk(
    d_avg: float,
    lt_mean: float,
    batches: Optional[List[InventoryBatch]] = None,
    available_physical_stock: Optional[float] = None,
    require_full_coverage: bool = False,
) -> Tuple[ExpiryRiskLevel, Optional[float], List[str]]:
    """
    Evaluates perishable expiry risk using an estimated clearance scenario under repeated daily P50 demand
    (not an exact expiry prediction).

    STRICT GOVERNANCE RULES:
    1. Only evaluates expiry risk when reliable batch-level inventory and expiry information exists.
    2. If batches is None or empty, explicitly returns ExpiryRiskLevel.NOT_ASSESSABLE with at_risk_expiry_units = None.
       Never infers expiry quantities from aggregate stock or diagnostic scalar shelf-life fields.
    3. Validates batch quantities against available physical stock. Inconsistent batch data (e.g. batch units
       exceeding available physical stock, zero physical stock with positive batches, duplicate batch IDs,
       or non-finite units) is safely handled by returning ExpiryRiskLevel.NOT_ASSESSABLE with at_risk_expiry_units = None
       and explanatory inconsistency reasons.
    4. When reliable batches are provided, performs deterministic FIFO clearance simulation under repeated daily P50 demand
       as an estimated clearance scenario (not an exact expiry prediction).

    Returns:
        Tuple of (ExpiryRiskLevel, Optional[at_risk_expiry_units], List[reasons]).
    """
    if not batches:
        return (
            ExpiryRiskLevel.NOT_ASSESSABLE,
            None,
            ["Batch-level inventory and expiration dates unavailable in source data. Expiry risk is not assessable without discrete batch tracking."]
        )

    # Validate batch data against available physical stock and internal consistency
    is_valid, issues = validate_batch_inventory(
        batches=batches,
        available_physical_stock=available_physical_stock,
        require_full_coverage=require_full_coverage,
    )
    if not is_valid:
        return (
            ExpiryRiskLevel.NOT_ASSESSABLE,
            None,
            [f"Inconsistent batch data: {'; '.join(issues)}. Expiry risk is not assessable with unreliable batch records."]
        )

    # Sort batches ascending by remaining shelf life (FIFO order)
    sorted_batches = sorted(batches, key=lambda b: b.remaining_shelf_life_days)

    total_at_risk = 0.0
    imminent_spoilage_within_lt = False
    reasons: List[str] = []

    # If partial batch coverage is permitted, note unbatched stock
    total_batch_units = sum(b.units for b in sorted_batches)
    if available_physical_stock is not None and total_batch_units < available_physical_stock - 1e-4:
        reasons.append(
            f"Partial Batch Coverage: Evaluated {total_batch_units:.1f} of {available_physical_stock:.1f} available physical units ({available_physical_stock - total_batch_units:.1f} units unbatched)."
        )

    # Simulate FIFO clearance scenario under repeated daily P50 demand:
    cum_batch_stock = 0.0
    for b in sorted_batches:
        cum_batch_stock += b.units
        t_expiry = b.remaining_shelf_life_days
        cum_demand_at_expiry = t_expiry * d_avg

        unclearable = max(0.0, cum_batch_stock - cum_demand_at_expiry)
        if unclearable > total_at_risk:
            new_risk = unclearable - total_at_risk
            total_at_risk = unclearable
            reasons.append(
                f"Batch '{b.batch_id}' ({b.units:.1f} units, {t_expiry}d shelf life): {new_risk:.1f} units estimated at risk under repeated P50 clearance scenario (not an exact expiry prediction)."
            )
            if t_expiry <= lt_mean:
                imminent_spoilage_within_lt = True

    if total_at_risk > 0.0:
        if imminent_spoilage_within_lt:
            level = ExpiryRiskLevel.HIGH
            reasons.insert(
                0,
                f"High Expiry Risk (Estimated Clearance Scenario): {total_at_risk:.1f} units estimated at risk of expiring within supplier lead time window ({lt_mean:.1f} days) under repeated P50 demand (not an exact expiry prediction)."
            )
        else:
            level = ExpiryRiskLevel.MEDIUM
            reasons.insert(
                0,
                f"Medium Expiry Risk (Estimated Clearance Scenario): {total_at_risk:.1f} units estimated at risk of expiring beyond lead time window before repeated P50 demand clears them (not an exact expiry prediction)."
            )
    else:
        level = ExpiryRiskLevel.LOW
        reasons.append(
            f"Low Expiry Risk (Estimated Clearance Scenario): All {len(batches)} batches ({total_batch_units:.1f} total units) estimated to clear under repeated P50 demand."
        )

    return level, float(total_at_risk), reasons


def assess_inventory_risk(
    calc_result: InventoryCalculationResult,
    forecast: DemandForecast,
    batches: Optional[List[InventoryBatch]] = None,
    require_full_coverage: bool = False,
) -> InventoryRiskAssessment:
    """
    Integrates stock level risks and perishable expiry risks into a validated InventoryRiskAssessment.
    Preserves the assumed_zero_pipeline indicator.
    Validates batch quantities against available physical stock (net_available_stock).
    """
    (
        stock_level,
        reorder_rec,
        stockout_reasons,
        excess_reasons,
        excess_qty,
        excess_qty_p10,
        surge_ratio,
        dos_p50,
    ) = assess_stock_risk(calc_result, forecast)

    expiry_level, at_risk_units, expiry_reasons = assess_expiry_risk(
        d_avg=calc_result.average_daily_demand,
        lt_mean=calc_result.lead_time_mean_days,
        batches=batches,
        available_physical_stock=calc_result.net_available_stock,
        require_full_coverage=require_full_coverage,
    )

    return InventoryRiskAssessment(
        store_id=calc_result.store_id,
        sku_id=calc_result.sku_id,
        date=calc_result.date,
        stock_risk_level=stock_level,
        reorder_recommended=reorder_rec,
        stockout_risk_reasons=stockout_reasons,
        excess_risk_reasons=excess_reasons,
        excess_quantity=excess_qty,
        excess_quantity_p10=excess_qty_p10,
        surge_coverage_ratio_p90=surge_ratio,
        days_of_supply_p50=dos_p50,
        expiry_risk_level=expiry_level,
        at_risk_expiry_units=at_risk_units,
        expiry_risk_reasons=expiry_reasons,
        assumed_zero_pipeline=calc_result.assumed_zero_pipeline,
    )


class InventoryEngine:
    """
    Deterministic Core Inventory Engine for Lead 2 Quick-Commerce Operations.
    
    Transforms operational daily agent inputs (forecasts + opening inventory snapshots)
    into validated inventory policy recommendations (SS, ROP, Target Stock, and Replenishment Qty).
    """

    def __init__(self, config: Optional[Lead2Config] = None):
        self.config = config or get_default_config()

    def get_supplier_params(
        self,
        category: str,
        supplier_type: Optional[Any] = None,
    ) -> SupplierSimulationParams:
        """
        Resolves simulation parameters for a given product category and supplier tier.

        Args:
            category: Product category (Bakery, Dairy, Meat, Produce).
            supplier_type: Optional SupplierType or string ('PRIMARY' or 'BACKUP'). Defaults to PRIMARY.

        Returns:
            SupplierSimulationParams with category and supplier specific lead time mean and standard deviation.
        """
        is_backup = False
        if supplier_type is not None:
            val = supplier_type.value if hasattr(supplier_type, "value") else str(supplier_type)
            is_backup = (val.upper() == "BACKUP")

        primary_params = self.config.category_supplier_params.get(
            category, self.config.default_supplier_params
        )

        if is_backup:
            proc_params = self.config.category_procurement_params.get(category)
            if proc_params is not None:
                return SupplierSimulationParams(
                    lead_time_mean_days=proc_params.backup_lead_time_days,       # 0.8 days
                    lead_time_std_days=proc_params.backup_lead_time_std_days,    # 0.15 days
                    review_period_days=primary_params.review_period_days,         # 1.0 day
                    min_order_qty=proc_params.min_order_qty,
                    cogs_price_ratio=proc_params.backup_cogs_ratio,              # 0.75
                    holding_cost_per_unit_day=primary_params.holding_cost_per_unit_day,
                    stockout_penalty_per_unit=primary_params.stockout_penalty_per_unit,
                    disposal_cost_per_unit=primary_params.disposal_cost_per_unit,
                )

        return primary_params

    def calculate(
        self,
        daily_input: DailyAgentInput,
        inventory_position: Optional[InventoryPosition] = None,
        on_order_stock: Optional[float] = None,
        reserved_stock: Optional[float] = None,
        z_score: Optional[float] = None,
        supplier_params: Optional[Any] = None,
        supplier_type: Optional[Any] = None,
    ) -> InventoryCalculationResult:
        """
        Executes deterministic inventory policy calculations for a single operational node.

        Args:
            daily_input: Validated DailyAgentInput containing forecast and opening inventory.
            inventory_position: Optional trusted InventoryPosition. Object existence alone does
                                NOT clear the simulation assumption flag; both incoming stock
                                (on_order_stock) and reservations (reserved_stock) must be
                                explicitly provided and validated.
            on_order_stock: Optional explicit incoming purchase order stock quantity.
            reserved_stock: Optional explicit confirmed customer order reservation quantity.
            z_score: Optional custom service level Z-score. Defaults to config.default_z_score (1.645).
            supplier_params: Optional custom supplier parameters (SupplierSimulationParams or SupplierProfile).
            supplier_type: Optional supplier tier (SupplierType or 'PRIMARY'/'BACKUP'). Defaults to PRIMARY.

        Returns:
            Validated InventoryCalculationResult.
        """
        # 1. Resolve inventory position and pipeline assumption flag
        # EXPLICIT SIMULATION ASSUMPTION:
        # assumed_zero_pipeline is False ONLY when both incoming stock (on_order_stock)
        # and reservations (reserved_stock) are explicitly provided and valid.
        # Mere existence of an InventoryPosition object with defaulted attributes does NOT
        # clear the assumption flag.
        if inventory_position is not None:
            # Check which fields were explicitly set during construction of the object
            fields_set = getattr(inventory_position, "model_fields_set", set())
            has_on_order = ("on_order_stock" in fields_set) or (on_order_stock is not None)
            has_reserved = ("reserved_stock" in fields_set) or (reserved_stock is not None)

            eff_on_hand = inventory_position.on_hand_stock
            eff_on_order = on_order_stock if on_order_stock is not None else inventory_position.on_order_stock
            eff_reserved = reserved_stock if reserved_stock is not None else inventory_position.reserved_stock

            pos = InventoryPosition(
                on_hand_stock=eff_on_hand,
                on_order_stock=eff_on_order,
                reserved_stock=eff_reserved,
            )
            assumed_zero_pipeline = not (has_on_order and has_reserved)
        else:
            has_on_order = on_order_stock is not None
            has_reserved = reserved_stock is not None

            pos = InventoryPosition(
                on_hand_stock=daily_input.inventory.on_hand_stock,
                on_order_stock=on_order_stock if has_on_order else 0.0,
                reserved_stock=reserved_stock if has_reserved else 0.0,
            )
            assumed_zero_pipeline = not (has_on_order and has_reserved)

        # 2. Resolve parameters (supports SupplierSimulationParams, SupplierProfile, and SupplierType)
        if supplier_params is not None:
            if hasattr(supplier_params, "lead_time_mean_days") and hasattr(supplier_params, "lead_time_std_days"):
                lt_mean = float(supplier_params.lead_time_mean_days)
                sigma_lt = float(supplier_params.lead_time_std_days)
                review_period = float(
                    getattr(supplier_params, "review_period_days", self.config.default_supplier_params.review_period_days)
                )
            else:
                raise TypeError(f"Unsupported supplier_params type: {type(supplier_params)}")
        else:
            params = self.get_supplier_params(daily_input.category, supplier_type=supplier_type)
            lt_mean = float(params.lead_time_mean_days)
            sigma_lt = float(params.lead_time_std_days)
            review_period = float(params.review_period_days)

        z = z_score if z_score is not None else self.config.default_z_score

        # 3. Compute demand and lead-time terms
        d_avg = compute_average_daily_demand(daily_input.forecast)
        sigma_d = float(daily_input.forecast.sigma_demand)

        # 4. Core inventory equations
        # SS_LT: Lead-time safety stock covering transit uncertainty (used for ROP)
        ss_lt = compute_safety_stock(
            d_avg=d_avg,
            sigma_d=sigma_d,
            lt_mean=lt_mean,
            sigma_lt=sigma_lt,
            z=z,
        )

        # SS_P: Protection-period safety stock covering (LT_mean + review_period) uncertainty (used for Target Stock)
        ss_protection = compute_protection_safety_stock(
            d_avg=d_avg,
            sigma_d=sigma_d,
            lt_mean=lt_mean,
            review_period=review_period,
            sigma_lt=sigma_lt,
            z=z,
        )

        rop = compute_reorder_point(
            d_avg=d_avg,
            lt_mean=lt_mean,
            safety_stock=ss_lt,
        )

        target_stock = compute_target_stock(
            d_avg=d_avg,
            lt_mean=lt_mean,
            review_period=review_period,
            safety_stock=ss_protection,
        )

        # 5. Review inventory position and evaluate replenishment
        reorder_triggered, suggested_qty = compute_suggested_replenishment_qty(
            total_inventory_position=pos.total_inventory_position,
            rop=rop,
            target_stock=target_stock,
        )

        return InventoryCalculationResult(
            store_id=daily_input.store_id,
            sku_id=daily_input.sku_id,
            date=daily_input.date,
            average_daily_demand=d_avg,
            demand_uncertainty_sigma=sigma_d,
            lead_time_mean_days=lt_mean,
            lead_time_std_days=sigma_lt,
            review_period_days=review_period,
            z_score=z,
            safety_stock_lead_time=ss_lt,
            safety_stock_protection=ss_protection,
            safety_stock=ss_lt,
            reorder_point=rop,
            target_stock=target_stock,
            inventory_position=pos,
            reorder_triggered=reorder_triggered,
            suggested_replenishment_qty=suggested_qty,
            assumed_zero_pipeline=assumed_zero_pipeline,
        )

    def calculate_for_supplier(
        self,
        daily_input: DailyAgentInput,
        supplier: Any,
        inventory_position: Optional[InventoryPosition] = None,
        on_order_stock: Optional[float] = None,
        reserved_stock: Optional[float] = None,
        z_score: Optional[float] = None,
    ) -> InventoryCalculationResult:
        """
        Calculates inventory requirements specifically tailored to a designated supplier profile or tier.

        Args:
            daily_input: Operational DailyAgentInput.
            supplier: SupplierProfile, SupplierSimulationParams, SupplierType, or string ('PRIMARY', 'BACKUP').
            inventory_position: Optional inventory position override.
            on_order_stock: Optional on-order quantity.
            reserved_stock: Optional reserved quantity.
            z_score: Optional service level Z-score.

        Returns:
            InventoryCalculationResult tailored to the specified supplier's lead time and variability.
        """
        if isinstance(supplier, str) or hasattr(supplier, "value"):
            return self.calculate(
                daily_input=daily_input,
                inventory_position=inventory_position,
                on_order_stock=on_order_stock,
                reserved_stock=reserved_stock,
                z_score=z_score,
                supplier_type=supplier,
            )
        else:
            return self.calculate(
                daily_input=daily_input,
                inventory_position=inventory_position,
                on_order_stock=on_order_stock,
                reserved_stock=reserved_stock,
                z_score=z_score,
                supplier_params=supplier,
            )

    def recalculate_for_supplier(
        self,
        calc_result: InventoryCalculationResult,
        supplier: Any,
        category: Optional[str] = None,
        z_score: Optional[float] = None,
    ) -> InventoryCalculationResult:
        """
        Re-evaluates an existing InventoryCalculationResult using a specified supplier's
        lead time mean and standard deviation. Preserves demand forecast and inventory position.

        Args:
            calc_result: Output from a prior InventoryEngine calculation.
            supplier: SupplierProfile, SupplierSimulationParams, SupplierType, or string ('PRIMARY', 'BACKUP').
            category: Optional category override (defaults to SKU_TO_CATEGORY lookup).
            z_score: Optional service level Z-score override.

        Returns:
            New InventoryCalculationResult with recalculated supplier-specific metrics.
        """
        if hasattr(supplier, "lead_time_mean_days") and hasattr(supplier, "lead_time_std_days"):
            lt_mean = float(supplier.lead_time_mean_days)
            sigma_lt = float(supplier.lead_time_std_days)
            review_period = float(
                getattr(supplier, "review_period_days", calc_result.review_period_days)
            )
        else:
            cat = category or SKU_TO_CATEGORY.get(calc_result.sku_id, "Produce")
            params = self.get_supplier_params(cat, supplier_type=supplier)
            lt_mean = float(params.lead_time_mean_days)
            sigma_lt = float(params.lead_time_std_days)
            review_period = float(params.review_period_days)

        d_avg = calc_result.average_daily_demand
        sigma_d = calc_result.demand_uncertainty_sigma
        z = z_score if z_score is not None else calc_result.z_score
        pos = calc_result.inventory_position

        ss_lt = compute_safety_stock(
            d_avg=d_avg,
            sigma_d=sigma_d,
            lt_mean=lt_mean,
            sigma_lt=sigma_lt,
            z=z,
        )
        ss_protection = compute_protection_safety_stock(
            d_avg=d_avg,
            sigma_d=sigma_d,
            lt_mean=lt_mean,
            review_period=review_period,
            sigma_lt=sigma_lt,
            z=z,
        )
        rop = compute_reorder_point(
            d_avg=d_avg,
            lt_mean=lt_mean,
            safety_stock=ss_lt,
        )
        target_stock = compute_target_stock(
            d_avg=d_avg,
            lt_mean=lt_mean,
            review_period=review_period,
            safety_stock=ss_protection,
        )
        reorder_triggered, suggested_qty = compute_suggested_replenishment_qty(
            total_inventory_position=pos.total_inventory_position,
            rop=rop,
            target_stock=target_stock,
        )

        return InventoryCalculationResult(
            store_id=calc_result.store_id,
            sku_id=calc_result.sku_id,
            date=calc_result.date,
            average_daily_demand=d_avg,
            demand_uncertainty_sigma=sigma_d,
            lead_time_mean_days=lt_mean,
            lead_time_std_days=sigma_lt,
            review_period_days=review_period,
            z_score=z,
            safety_stock_lead_time=ss_lt,
            safety_stock_protection=ss_protection,
            safety_stock=ss_lt,
            reorder_point=rop,
            target_stock=target_stock,
            inventory_position=pos,
            reorder_triggered=reorder_triggered,
            suggested_replenishment_qty=suggested_qty,
            assumed_zero_pipeline=calc_result.assumed_zero_pipeline,
        )

    def calculate_batch(
        self,
        daily_inputs: List[DailyAgentInput],
        inventory_positions: Optional[Dict[Tuple[str, str, str], InventoryPosition]] = None,
        z_score: Optional[float] = None,
    ) -> List[InventoryCalculationResult]:
        """
        Executes inventory calculations over a sequence of DailyAgentInputs.

        Args:
            daily_inputs: List of DailyAgentInput records.
            inventory_positions: Optional map of (store_id, sku_id, date) -> InventoryPosition.
            z_score: Optional custom service level Z-score.

        Returns:
            List of InventoryCalculationResult models matching input order.
        """
        results: List[InventoryCalculationResult] = []
        pos_map = inventory_positions or {}

        for inp in daily_inputs:
            pos_key = (inp.store_id, inp.sku_id, inp.date)
            pos = pos_map.get(pos_key, None)
            res = self.calculate(
                daily_input=inp,
                inventory_position=pos,
                z_score=z_score,
            )
            results.append(res)

        return results

    def assess_risk(
        self,
        calc_result: InventoryCalculationResult,
        forecast: DemandForecast,
        batches: Optional[List[InventoryBatch]] = None,
        require_full_coverage: bool = False,
    ) -> InventoryRiskAssessment:
        """
        Assesses inventory risk (stockout, excess, expiry) for an already calculated node.

        Args:
            calc_result: Output from calculate().
            forecast: Validated DemandForecast containing P10, P50, P90 bounds.
            batches: Optional list of physical InventoryBatch objects for expiry evaluation.
            require_full_coverage: If True, requires total batch units to cover available physical stock.

        Returns:
            Validated InventoryRiskAssessment.
        """
        return assess_inventory_risk(
            calc_result=calc_result,
            forecast=forecast,
            batches=batches,
            require_full_coverage=require_full_coverage,
        )

    def calculate_and_assess_risk(
        self,
        daily_input: DailyAgentInput,
        inventory_position: Optional[InventoryPosition] = None,
        on_order_stock: Optional[float] = None,
        reserved_stock: Optional[float] = None,
        batches: Optional[List[InventoryBatch]] = None,
        require_full_coverage: bool = False,
        z_score: Optional[float] = None,
        supplier_params: Optional[SupplierSimulationParams] = None,
    ) -> Tuple[InventoryCalculationResult, InventoryRiskAssessment]:
        """
        Calculates inventory metrics and performs full risk assessment in a single call.

        Returns:
            Tuple of (InventoryCalculationResult, InventoryRiskAssessment).
        """
        calc_res = self.calculate(
            daily_input=daily_input,
            inventory_position=inventory_position,
            on_order_stock=on_order_stock,
            reserved_stock=reserved_stock,
            z_score=z_score,
            supplier_params=supplier_params,
        )
        risk_res = self.assess_risk(
            calc_result=calc_res,
            forecast=daily_input.forecast,
            batches=batches,
            require_full_coverage=require_full_coverage,
        )
        return calc_res, risk_res

    def assess_risk_batch(
        self,
        calc_results: List[InventoryCalculationResult],
        forecasts: List[DemandForecast],
        batch_map: Optional[Dict[Tuple[str, str, str], List[InventoryBatch]]] = None,
        require_full_coverage: bool = False,
    ) -> List[InventoryRiskAssessment]:
        """
        Performs batch risk assessment across a sequence of calculation results and forecasts.
        """
        if len(calc_results) != len(forecasts):
            raise ValueError(f"Mismatch between calc_results ({len(calc_results)}) and forecasts ({len(forecasts)})")

        bmap = batch_map or {}
        assessments: List[InventoryRiskAssessment] = []
        for calc, fc in zip(calc_results, forecasts):
            key = (calc.store_id, calc.sku_id, calc.date)
            batches = bmap.get(key, None)
            assm = self.assess_risk(
                calc_result=calc,
                forecast=fc,
                batches=batches,
                require_full_coverage=require_full_coverage,
            )
            assessments.append(assm)

        return assessments


