"""
Lead 2 Configuration Module
===========================
Maintains path configuration, environment constants, and simulation-only parameters.
Keeps operational parameters separate from data contracts and execution logic.

IMPORTANT:
All supplier lead-time and cost parameters defined here are simulation-only defaults
calibrated against Step 5 simulation constants. No real supplier catalogs or vendor
contracts currently exist in the repository.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Root repository directories
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# Known canonical dark-store hubs in Bangalore
CANONICAL_STORES: List[str] = [
    "DS_BLR_HSR_LAYOUT",
    "DS_BLR_INDIRANAGAR",
    "DS_BLR_JAYANAGAR",
    "DS_BLR_KORAMANGALA",
    "DS_BLR_WHITEFIELD",
]

# Known canonical 15 quick-commerce SKUs
CANONICAL_SKUS: List[str] = [
    "SKU_BANANA_DZ",
    "SKU_BREAD_WHITE",
    "SKU_BREAD_WW",
    "SKU_BUNS_6PK",
    "SKU_BUTTER_100G",
    "SKU_CHICKEN_500G",
    "SKU_CURD_400G",
    "SKU_EGGS_12PK",
    "SKU_MILK_1L",
    "SKU_MUTTON_500G",
    "SKU_ONION_1KG",
    "SKU_PANEER_200G",
    "SKU_POTATO_1KG",
    "SKU_SPINACH_500G",
    "SKU_TOMATO_1KG",
]

CANONICAL_CATEGORIES: List[str] = ["Bakery", "Dairy", "Meat", "Produce"]


@dataclass(frozen=True)
class SupplierSimulationParams:
    """
    Simulation-only assumptions for supplier lead times and economics.
    These are placeholder values derived from Step 5 simulation code.
    No empirical supplier dataset exists in the repository.
    """
    lead_time_mean_days: float = 1.5
    lead_time_std_days: float = 0.30
    review_period_days: float = 1.0
    min_order_qty: float = 0.0
    cogs_price_ratio: float = 0.60  # Default procurement cost as 60% of retail price
    holding_cost_per_unit_day: float = 2.0
    stockout_penalty_per_unit: float = 45.0
    disposal_cost_per_unit: float = 5.0


# Category-specific simulation assumptions (derived from Step 5 simulation constants)
CATEGORY_SUPPLIER_PARAMS: Dict[str, SupplierSimulationParams] = {
    "Meat": SupplierSimulationParams(
        lead_time_mean_days=1.0,
        lead_time_std_days=0.25,
        review_period_days=1.0,
        cogs_price_ratio=0.60,
    ),
    "Dairy": SupplierSimulationParams(
        lead_time_mean_days=1.5,
        lead_time_std_days=0.30,
        review_period_days=1.0,
        cogs_price_ratio=0.60,
    ),
    "Produce": SupplierSimulationParams(
        lead_time_mean_days=1.5,
        lead_time_std_days=0.30,
        review_period_days=1.0,
        cogs_price_ratio=0.60,
    ),
    "Bakery": SupplierSimulationParams(
        lead_time_mean_days=1.5,
        lead_time_std_days=0.30,
        review_period_days=1.0,
        cogs_price_ratio=0.60,
    ),
}


@dataclass
class Lead2Config:
    """
    Master configuration object for Lead 2 agents and data loaders.
    """
    # File paths
    forecast_file: Path = PROCESSED_DATA_DIR / "forecast_payloads_step2.parquet"
    master_file: Path = PROCESSED_DATA_DIR / "df_master.parquet"
    preorders_file: Path = PROCESSED_DATA_DIR / "committed_preorders_step3.parquet"
    
    # Inventory service level configuration
    default_service_level: float = 0.95
    default_z_score: float = 1.645  # Standard normal inverse CDF for 95% cycle service level
    contingency_z_score: float = 1.960  # Contingency Z for 97.5% cycle service level
    
    # Supplier simulation settings
    default_supplier_params: SupplierSimulationParams = field(default_factory=SupplierSimulationParams)
    category_supplier_params: Dict[str, SupplierSimulationParams] = field(
        default_factory=lambda: dict(CATEGORY_SUPPLIER_PARAMS)
    )


def get_default_config() -> Lead2Config:
    """Returns a default Lead2Config instance."""
    return Lead2Config()

