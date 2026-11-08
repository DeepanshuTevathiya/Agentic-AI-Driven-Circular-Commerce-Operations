"""
Lead 2 Configuration Module (Common)
====================================
Maintains path configuration, environment constants, and simulation-only parameters.
Keeps operational parameters separate from data contracts and execution logic.

IMPORTANT:
All supplier lead-time, packaging, and cost parameters defined here are simulation-only defaults
calibrated against Step 5 simulation constants. No real supplier catalogs or vendor
contracts currently exist in the repository.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Root repository directories (adjusted for src/agents/common nesting)
COMMON_DIR = Path(__file__).resolve().parent
AGENTS_DIR = COMMON_DIR.parent
SRC_DIR = AGENTS_DIR.parent
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

# Canonical SKU-to-Category mapping
SKU_TO_CATEGORY: Dict[str, str] = {
    "SKU_BREAD_WHITE": "Bakery",
    "SKU_BREAD_WW": "Bakery",
    "SKU_BUNS_6PK": "Bakery",
    "SKU_BUTTER_100G": "Dairy",
    "SKU_CURD_400G": "Dairy",
    "SKU_EGGS_12PK": "Dairy",
    "SKU_MILK_1L": "Dairy",
    "SKU_PANEER_200G": "Dairy",
    "SKU_CHICKEN_500G": "Meat",
    "SKU_MUTTON_500G": "Meat",
    "SKU_BANANA_DZ": "Produce",
    "SKU_ONION_1KG": "Produce",
    "SKU_POTATO_1KG": "Produce",
    "SKU_SPINACH_500G": "Produce",
    "SKU_TOMATO_1KG": "Produce",
}


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
        min_order_qty=5.0,
        cogs_price_ratio=0.60,
    ),
    "Dairy": SupplierSimulationParams(
        lead_time_mean_days=1.5,
        lead_time_std_days=0.30,
        review_period_days=1.0,
        min_order_qty=10.0,
        cogs_price_ratio=0.60,
    ),
    "Produce": SupplierSimulationParams(
        lead_time_mean_days=1.5,
        lead_time_std_days=0.30,
        review_period_days=1.0,
        min_order_qty=10.0,
        cogs_price_ratio=0.60,
    ),
    "Bakery": SupplierSimulationParams(
        lead_time_mean_days=1.5,
        lead_time_std_days=0.30,
        review_period_days=1.0,
        min_order_qty=15.0,
        cogs_price_ratio=0.60,
    ),
}


@dataclass(frozen=True)
class CategoryProcurementParams:
    """
    Simulation-only procurement packaging, minimum order quantities, and supplier terms per category.
    All parameters are simulation assumptions and explicitly labeled as such.
    """
    min_order_qty: float                 # MOQ in base units
    case_pack_size: int                  # Discrete pack size for bulk shipment rounding
    primary_cogs_ratio: float = 0.60     # Primary vendor wholesale cost = 60% retail
    backup_cogs_ratio: float = 0.75      # Backup expedited vendor wholesale cost = 75% retail
    backup_lead_time_days: float = 0.8   # Backup vendor expedited lead time in days
    backup_lead_time_std_days: float = 0.15 # Backup vendor lead time variance


# Approved category-specific procurement parameters (Phase 4.2 specification)
CATEGORY_PROCUREMENT_PARAMS: Dict[str, CategoryProcurementParams] = {
    "Bakery": CategoryProcurementParams(min_order_qty=15.0, case_pack_size=10),
    "Dairy": CategoryProcurementParams(min_order_qty=10.0, case_pack_size=12),
    "Meat": CategoryProcurementParams(min_order_qty=5.0, case_pack_size=6),
    "Produce": CategoryProcurementParams(min_order_qty=10.0, case_pack_size=10),
}


@dataclass(frozen=True)
class ProcurementApprovalThresholds:
    """
    Escalation and review thresholds for Purchase Orders.

    IMPORTANT:
    These are review and escalation triggers, NOT permissions for automatic approval.
    Every purchase order requires explicit human approval before issuance or dispatch.
    """
    max_line_item_cost_inr: float = 5000.0   # ₹5,000 per PO line triggers line-item escalation
    max_total_po_cost_inr: float = 25000.0    # ₹25,000 per store order triggers order-level escalation
    require_human_approval_for_dispatch: bool = True  # Strict invariant: no PO may be dispatched without human signoff
    flag_assumed_zero_pipeline_reorders: bool = True  # Flag LOW_STOCK_REORDER under assumed-zero pipeline


@dataclass
class Lead2Config:
    """
    Master configuration object for Lead 2 agents, data loaders, and procurement engine.
    """
    # File paths
    forecast_file: Path = PROCESSED_DATA_DIR / "forecast_payloads_step2_corrected.parquet"
    master_file: Path = PROCESSED_DATA_DIR / "df_master.parquet"
    preorders_file: Path = PROCESSED_DATA_DIR / "committed_preorders_step3.parquet"
    
    # Inventory service level configuration
    default_service_level: float = 0.95
    default_z_score: float = 1.645      # Standard normal inverse CDF for 95% cycle service level
    contingency_z_score: float = 1.960  # Contingency Z for 97.5% cycle service level
    
    # Supplier simulation settings (Inventory Engine)
    default_supplier_params: SupplierSimulationParams = field(default_factory=SupplierSimulationParams)
    category_supplier_params: Dict[str, SupplierSimulationParams] = field(
        default_factory=lambda: dict(CATEGORY_SUPPLIER_PARAMS)
    )

    # Procurement simulation settings (Procurement Engine)
    category_procurement_params: Dict[str, CategoryProcurementParams] = field(
        default_factory=lambda: dict(CATEGORY_PROCUREMENT_PARAMS)
    )
    procurement_thresholds: ProcurementApprovalThresholds = field(
        default_factory=ProcurementApprovalThresholds
    )


def get_default_config() -> Lead2Config:
    """Returns a default Lead2Config instance."""
    return Lead2Config()

