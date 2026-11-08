"""
Lead 2 Inventory Module
=======================
Implements core deterministic inventory calculation and risk assessment.
"""

from .engine import (
    InventoryEngine,
    compute_average_daily_demand,
    compute_safety_stock,
    compute_protection_safety_stock,
    compute_reorder_point,
    compute_target_stock,
    compute_suggested_replenishment_qty,
    compute_surge_coverage_ratio_p90,
    validate_batch_inventory,
    assess_stock_risk,
    assess_expiry_risk,
    assess_inventory_risk,
)

__all__ = [
    "InventoryEngine",
    "compute_average_daily_demand",
    "compute_safety_stock",
    "compute_protection_safety_stock",
    "compute_reorder_point",
    "compute_target_stock",
    "compute_suggested_replenishment_qty",
    "compute_surge_coverage_ratio_p90",
    "validate_batch_inventory",
    "assess_stock_risk",
    "assess_expiry_risk",
    "assess_inventory_risk",
]
