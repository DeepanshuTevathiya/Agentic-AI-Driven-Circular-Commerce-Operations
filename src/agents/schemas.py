"""
Lead 2 Data Contracts & Pydantic V2 Schemas
============================================
Defines validated, immutable data schemas for:
- DemandForecast (probabilistic multi-quantile demand inputs)
- InventorySnapshot (beginning-of-day physical stock and optional shelf life)
- InventoryPosition (on-hand, reserved, and on-order stock)
- DailyAgentInput (unified validated daily contract per store-SKU-date)
"""

from datetime import date as dt_date, datetime as dt_datetime
from enum import Enum
from typing import List, Optional, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class InventorySnapshotBasis(str, Enum):
    """
    Explicitly tracks the operational timing basis of on-hand inventory stock.
    Beginning-of-day stock (Opening_Stock) is required for operational decisioning.
    Closing_Stock of the same day represents post-sales inventory and must not
    be used as available starting inventory.
    """
    BEGINNING_OF_DAY = "BEGINNING_OF_DAY"  # True opening stock (valid for decisions)
    END_OF_DAY = "END_OF_DAY"              # Post-consumption closing stock (invalid as starting inventory)


def _validate_non_empty_str(v: str, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


def _normalize_date_str(v: Union[str, dt_date, dt_datetime]) -> str:
    """Helper to canonicalize dates to YYYY-MM-DD string format."""
    if isinstance(v, (dt_datetime, dt_date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, str):
        cleaned = v.strip()
        # Accept YYYY-MM-DD or parse full ISO string
        try:
            parsed = dt_datetime.strptime(cleaned[:10], "%Y-%m-%d").date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError as err:
            raise ValueError(f"Invalid date string '{v}'. Expected YYYY-MM-DD format.") from err
    raise ValueError(f"Unsupported date type: {type(v)}")


class DemandForecast(BaseModel):
    """
    Validated probabilistic macro demand forecast from Step 2.
    Enforces quantile monotonicity (P10 <= P50 <= P90), finite numbers, and strictly positive uncertainty.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Unique product SKU identifier")
    category: str = Field(..., description="Product category (Dairy, Bakery, Produce, Meat)")
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)")

    p10_demand: float = Field(..., ge=0.0, description="10th percentile clearance/lower demand bound")
    p50_demand: float = Field(..., ge=0.0, description="50th percentile median expected demand (D_avg)")
    p90_demand: float = Field(..., ge=0.0, description="90th percentile peak/upper demand bound")
    sigma_demand: float = Field(..., gt=0.0, description="Standard deviation of daily demand uncertainty (sigma_D > 0)")

    @field_validator("store_id", "sku_id", "category")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)

    @model_validator(mode="after")
    def verify_monotonicity(self) -> "DemandForecast":
        """Enforces P10 <= P50 <= P90 quantile ordering."""
        if not (self.p10_demand <= self.p50_demand <= self.p90_demand):
            raise ValueError(
                f"Quantile monotonicity violation: P10 ({self.p10_demand}) <= "
                f"P50 ({self.p50_demand}) <= P90 ({self.p90_demand}) check failed."
            )
        return self


class InventorySnapshot(BaseModel):
    """
    Validated snapshot of physical inventory available at the beginning of the operating day.
    
    CRITICAL INVENTORY SEMANTICS:
    - on_hand_stock MUST represent the starting/opening stock available before daily sales.
    - Same-day Closing_Stock must NOT be used as on_hand_stock.
    - remaining_shelf_life_days and total_shelf_life_days are optional until timing
      semantics across inbound batches are fully audited.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Unique product SKU identifier")
    date: str = Field(..., description="Snapshot date (YYYY-MM-DD)")
    on_hand_stock: float = Field(..., ge=0.0, description="Beginning-of-day available physical inventory (Opening_Stock)")
    
    remaining_shelf_life_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Scalar remaining shelf-life days (optional pending batch-level verification)"
    )
    total_shelf_life_days: Optional[int] = Field(
        default=None,
        gt=0,
        description="Nominal factory total shelf-life days (optional)"
    )
    snapshot_basis: InventorySnapshotBasis = Field(
        default=InventorySnapshotBasis.BEGINNING_OF_DAY,
        description="Explicit metadata certifying this snapshot represents beginning-of-day stock"
    )

    @field_validator("store_id", "sku_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)

    @model_validator(mode="after")
    def verify_snapshot_basis(self) -> "InventorySnapshot":
        if self.snapshot_basis != InventorySnapshotBasis.BEGINNING_OF_DAY:
            raise ValueError(
                f"Invalid snapshot_basis '{self.snapshot_basis}'. "
                "Operational inventory snapshots must be BEGINNING_OF_DAY. "
                "Same-day closing stock cannot be used as starting inventory."
            )
        return self


class InventoryPosition(BaseModel):
    """
    Comprehensive inventory position representation for order replenishment calculations.
    Tracks physical on-hand stock, active reservations, and in-transit purchase orders.

    VALIDATION:
    reserved_stock must not exceed on_hand_stock. If a state where reserved > on_hand
    is encountered, it indicates an upstream data integrity problem that must be surfaced,
    not silently clamped.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    on_hand_stock: float = Field(..., ge=0.0, description="Physical beginning-of-day stock")
    reserved_stock: float = Field(default=0.0, ge=0.0, description="Inventory committed to confirmed customer orders")
    on_order_stock: float = Field(default=0.0, ge=0.0, description="Inventory currently ordered from supplier but not yet received")

    @model_validator(mode="after")
    def validate_reservation_feasibility(self) -> "InventoryPosition":
        """Reject reserved_stock > on_hand_stock instead of silently clamping."""
        if self.reserved_stock > self.on_hand_stock:
            raise ValueError(
                f"reserved_stock ({self.reserved_stock}) exceeds on_hand_stock ({self.on_hand_stock}). "
                "Physical reservations cannot exceed physical inventory. "
                "Fix the upstream data before constructing InventoryPosition."
            )
        return self

    @property
    def net_available_stock(self) -> float:
        """Physical stock free for immediate allocation: on_hand - reserved."""
        return self.on_hand_stock - self.reserved_stock

    @property
    def total_inventory_position(self) -> float:
        """Standard inventory position for (s, S) or ROP review: on_hand + on_order - reserved."""
        return self.on_hand_stock + self.on_order_stock - self.reserved_stock


class DailyAgentInput(BaseModel):
    """
    Unified OPERATIONAL input contract delivered to Lead 2 Inventory, Procurement,
    and Fulfilment agents. Combines verified DemandForecast and beginning-of-day
    InventorySnapshot for a single (store_id, sku_id, date) operational node.

    SAFETY & INTEGRITY RULES:
    1. Forecast and inventory identifiers must match exactly.
    2. unconfirmed_alert_count preserves diagnostic Step 3 customer runout alert counts.
       It is strictly unconfirmed and must NOT be automatically added to reserved_stock
       or subtracted from P50.
    3. This schema intentionally excludes ground-truth demand labels. Evaluation data
       lives in DailyEvaluationRecord to prevent accidental operational leakage.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Product SKU identifier")
    category: str = Field(..., description="Product category")
    date: str = Field(..., description="Operating date (YYYY-MM-DD)")

    forecast: DemandForecast = Field(..., description="Verified multi-quantile demand forecast")
    inventory: InventorySnapshot = Field(..., description="Beginning-of-day physical inventory snapshot")

    unconfirmed_alert_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Diagnostic count of unconfirmed household replenishment alert events (Step 3). "
            "NOT order quantities. NOT reserved stock."
        ),
    )

    @model_validator(mode="after")
    def verify_consistency(self) -> "DailyAgentInput":
        """Ensures identifiers and dates across parent, forecast, and inventory match perfectly."""
        if self.store_id != self.forecast.store_id or self.store_id != self.inventory.store_id:
            raise ValueError(
                f"Store ID mismatch: root='{self.store_id}', "
                f"forecast='{self.forecast.store_id}', inventory='{self.inventory.store_id}'"
            )
        if self.sku_id != self.forecast.sku_id or self.sku_id != self.inventory.sku_id:
            raise ValueError(
                f"SKU ID mismatch: root='{self.sku_id}', "
                f"forecast='{self.forecast.sku_id}', inventory='{self.inventory.sku_id}'"
            )
        if self.date != self.forecast.date or self.date != self.inventory.date:
            raise ValueError(
                f"Date mismatch: root='{self.date}', "
                f"forecast='{self.forecast.date}', inventory='{self.inventory.date}'"
            )
        if self.category != self.forecast.category:
            raise ValueError(
                f"Category mismatch: root='{self.category}', forecast='{self.forecast.category}'"
            )
        return self


class DailyEvaluationRecord(BaseModel):
    """
    Evaluation-only record pairing a DailyAgentInput key with the ground-truth demand label.
    Kept strictly separate from operational inputs so agents cannot accidentally consume
    actual demand during decision-making.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Product SKU identifier")
    date: str = Field(..., description="Evaluation date (YYYY-MM-DD)")
    target_demand: float = Field(..., ge=0.0, description="Ground-truth / latent demand label for offline evaluation")

    @field_validator("store_id", "sku_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)


class InventoryCalculationResult(BaseModel):
    """
    Deterministic output from the Core Inventory Engine for a single operational node.
    Contains safety stock, reorder point, target stock, and suggested replenishment quantity.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Product SKU identifier")
    date: str = Field(..., description="Operating calculation date (YYYY-MM-DD)")

    average_daily_demand: float = Field(..., ge=0.0, description="D_avg from forecast P50")
    demand_uncertainty_sigma: float = Field(..., ge=0.0, description="sigma_D from forecast")

    lead_time_mean_days: float = Field(..., gt=0.0, description="Supplier lead time mean in days (LT_mean)")
    lead_time_std_days: float = Field(..., ge=0.0, description="Supplier lead time std in days (sigma_LT)")
    review_period_days: float = Field(..., ge=0.0, description="Review period in days (R)")
    z_score: float = Field(..., ge=0.0, description="Service level Z-score factor")

    safety_stock_lead_time: float = Field(
        ...,
        ge=0.0,
        description="Lead-time safety stock SS_LT = Z * sqrt(LT_mean * sigma_D^2 + D_avg^2 * sigma_LT^2), used for ROP",
    )
    safety_stock_protection: float = Field(
        ...,
        ge=0.0,
        description="Protection-period safety stock SS_P = Z * sqrt((LT_mean + R) * sigma_D^2 + D_avg^2 * sigma_LT^2), used for Target Stock",
    )
    safety_stock: float = Field(
        ...,
        ge=0.0,
        description="Lead-time safety stock (alias to safety_stock_lead_time for backward compatibility)",
    )
    reorder_point: float = Field(..., ge=0.0, description="Reorder point ROP = (D_avg * LT_mean) + SS_LT")
    target_stock: float = Field(..., ge=0.0, description="Target stock S = D_avg * (R + LT_mean) + SS_P")

    inventory_position: InventoryPosition = Field(..., description="Inventory position (on_hand, reserved, on_order)")
    reorder_triggered: bool = Field(..., description="True if total_inventory_position <= ROP")
    suggested_replenishment_qty: float = Field(
        ...,
        ge=0.0,
        description="Suggested unconstrained order qty: max(0, S - IP) if reorder_triggered else 0.0"
    )
    assumed_zero_pipeline: bool = Field(
        default=False,
        description="Explicit simulation assumption flag: True if on_order and reserved were defaulted to 0.0"
    )

    @field_validator("store_id", "sku_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)

    @property
    def total_inventory_position(self) -> float:
        return self.inventory_position.total_inventory_position

    @property
    def net_available_stock(self) -> float:
        return self.inventory_position.net_available_stock


class StockRiskLevel(str, Enum):
    """Operational inventory stock risk classification."""
    CRITICAL_LOW = "CRITICAL_LOW"            # At or below safety stock or physical stockout
    LOW_STOCK_REORDER = "LOW_STOCK_REORDER"  # At or below ROP; replenishment triggered
    HEALTHY = "HEALTHY"                      # Optimal operating band (ROP < IP <= Target Stock)
    EXCESS_STOCK = "EXCESS_STOCK"            # Above Target Stock; holding excess inventory


class ExpiryRiskLevel(str, Enum):
    """
    Perishable inventory expiration risk classification based on an estimated
    clearance scenario under repeated daily P50 demand (not an exact expiry prediction).
    """
    NOT_ASSESSABLE = "NOT_ASSESSABLE"        # Discrete batch data absent, inconsistent, or unverified
    LOW = "LOW"                              # Batches present and estimated to clear before expiry
    MEDIUM = "MEDIUM"                        # Batches present with units estimated to expire beyond lead time
    HIGH = "HIGH"                            # Batches present with units estimated to expire within lead time window


class InventoryBatch(BaseModel):
    """
    Physical perishable inventory batch with discrete expiration tracking.
    Used for reliable batch-level perishability and expiry risk evaluation.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    batch_id: str = Field(..., description="Unique batch/lot identifier")
    units: float = Field(..., gt=0.0, description="Available batch physical quantity")
    remaining_shelf_life_days: int = Field(..., ge=0, description="Remaining shelf life days until expiration")
    creation_date: Optional[str] = Field(default=None, description="Batch manufacture or intake date (YYYY-MM-DD)")

    @field_validator("batch_id")
    @classmethod
    def validate_batch_id(cls, v: str) -> str:
        return _validate_non_empty_str(v, "batch_id")


class InventoryRiskAssessment(BaseModel):
    """
    Comprehensive risk assessment output combining stock-level and perishable expiry risks.
    Evaluates deterministic stress scenarios (P10 downside clearance, P90 demand surge;
    not calibrated multi-day predictions) alongside deterministic inventory metrics.
    Perishable expiry uses an estimated clearance scenario under repeated daily P50 demand
    (not an exact expiry prediction).
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Product SKU identifier")
    date: str = Field(..., description="Operating calculation date (YYYY-MM-DD)")

    stock_risk_level: StockRiskLevel = Field(..., description="Stock level operational status")
    reorder_recommended: bool = Field(..., description="Whether a replenishment order is recommended")
    stockout_risk_reasons: List[str] = Field(default_factory=list, description="Detailed explanatory reasons for stockout/low-stock risks")
    excess_risk_reasons: List[str] = Field(default_factory=list, description="Detailed explanatory reasons for excess stock risks")

    excess_quantity: float = Field(default=0.0, ge=0.0, description="Quantity exceeding Target Stock (max(0, IP - S))")
    excess_quantity_p10: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Quantity exceeding Target Stock under deterministic downside (P10) demand stress scenario "
            "(deterministic stress scenario, not a calibrated multi-day prediction)"
        ),
    )
    surge_coverage_ratio_p90: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Ratio of IP to deterministic P90 peak demand stress scenario over lead time: IP / (P90 * LT_mean) "
            "(deterministic stress scenario, not a calibrated multi-day prediction; None if P90 <= 0)"
        ),
    )
    days_of_supply_p50: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated days of supply under repeated median P50 demand velocity: IP / P50",
    )

    expiry_risk_level: ExpiryRiskLevel = Field(
        default=ExpiryRiskLevel.NOT_ASSESSABLE,
        description=(
            "Perishable expiry risk level under estimated clearance scenario. "
            "Explicitly NOT_ASSESSABLE if batch data is unavailable or inconsistent."
        ),
    )
    at_risk_expiry_units: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Estimated units at risk of expiring under repeated P50 demand clearance scenario "
            "(estimated clearance scenario, not an exact expiry prediction; None when NOT_ASSESSABLE)"
        ),
    )
    expiry_risk_reasons: List[str] = Field(default_factory=list, description="Detailed explanatory reasons for expiry risks")

    assumed_zero_pipeline: bool = Field(..., description="Propagated flag indicating whether on_order/reserved were defaulted to 0.0")

    @field_validator("store_id", "sku_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)


