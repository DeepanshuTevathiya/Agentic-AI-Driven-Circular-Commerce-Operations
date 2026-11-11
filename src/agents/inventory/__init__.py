"""
Lead 2 Inventory Module
=======================
Implements core deterministic inventory calculation, risk assessment,
operational inventory ledger, batch tracking, and durable persistence.
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
from .repository import OperationalInventoryRepository
from .schemas import (
    BatchBalance,
    DuplicateBatchPostingError,
    GrnLinePostingReconciliation,
    GrnPostingReconciliation,
    IdempotencyConflictError,
    InventoryBatchRecord,
    InventoryGovernanceError,
    InventoryLedgerEntry,
    InventoryPersistenceError,
    InventoryPostingReason,
    InventoryTransactionType,
    StoreSkuBalance,
    compute_payload_fingerprint,
)

__all__ = [
    # Engine & calculations
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
    # Operational Ledger Schemas (Phase 5.7.1)
    "BatchBalance",
    "DuplicateBatchPostingError",
    "GrnLinePostingReconciliation",
    "GrnPostingReconciliation",
    "IdempotencyConflictError",
    "InventoryBatchRecord",
    "InventoryGovernanceError",
    "InventoryLedgerEntry",
    "InventoryPersistenceError",
    "InventoryPostingReason",
    "InventoryTransactionType",
    "StoreSkuBalance",
    "compute_payload_fingerprint",
    # Repository (Phase 5.7.1)
    "OperationalInventoryRepository",
]
