"""
Operational Inventory Ledger & Batch Data Contracts (Phase 5.7.1)
=================================================================
Defines validated, immutable Pydantic V2 schemas for:
- InventoryTransactionType (extensible movement classifications)
- InventoryPostingReason (standard business rationale)
- InventoryLedgerEntry (immutable append-only physical stock movement)
- InventoryBatchRecord (durable inventory batch model with shelf-life tracking)
- StoreSkuBalance (derived on-hand stock position per store and SKU)
- BatchBalance (derived on-hand stock and perishability status per batch)
- GrnPostingReconciliation & GrnLinePostingReconciliation (audit reconciliation)
"""

from datetime import date as dt_date, datetime as dt_datetime
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..common.config import CANONICAL_SKUS, CANONICAL_STORES
from ..common.schemas import InventoryBatch
from ..fulfilment.schemas import validate_and_normalize_iso_timestamp


class InventoryGovernanceError(Exception):
    """Raised when an inventory operation violates governance or integrity invariants."""
    pass


class DuplicateBatchPostingError(InventoryGovernanceError):
    """Raised when attempting to post a batch that has already been posted from a GRN."""
    pass


class InventoryPersistenceError(Exception):
    """Raised on database integrity or persistence failures."""
    pass


class IdempotencyConflictError(InventoryPersistenceError):
    """Raised when an idempotency key is reused with a conflicting payload."""
    pass


class InventoryTransactionType(str, Enum):
    """Classification of immutable stock movements in the operational inventory ledger."""
    INBOUND_RECEIPT = "INBOUND_RECEIPT"          # Inbound receipt from accepted GRN (Phase 5.7.1)
    OUTBOUND_DISPATCH = "OUTBOUND_DISPATCH"      # Outbound store dispatch / customer fulfillment
    RETURN_TO_SUPPLIER = "RETURN_TO_SUPPLIER"    # Physical return of stock to vendor
    DAMAGE_WRITE_OFF = "DAMAGE_WRITE_OFF"        # Disposal/destruction of damaged store inventory
    EXPIRED_WRITE_OFF = "EXPIRED_WRITE_OFF"      # Disposal of expired perishable stock
    CYCLE_COUNT_ADJUSTMENT = "CYCLE_COUNT_ADJUSTMENT"  # Physical inventory audit adjustment
    INTERNAL_TRANSFER = "INTERNAL_TRANSFER"      # Inter-store or intra-facility movement


class InventoryPostingReason(str, Enum):
    """Business rationale for inventory ledger postings."""
    GOODS_RECEIPT = "GOODS_RECEIPT"
    RETURN_TO_ORIGIN = "RETURN_TO_ORIGIN"
    CERTIFIED_DISPOSAL = "CERTIFIED_DISPOSAL"
    CUSTOMER_ORDER_FULFILLED = "CUSTOMER_ORDER_FULFILLED"
    INVENTORY_AUDIT = "INVENTORY_AUDIT"
    QUALITY_EXPIRY = "QUALITY_EXPIRY"


def _validate_non_empty_str(v: str, field_name: str) -> str:
    """Helper to validate non-empty stripped strings."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


def compute_payload_fingerprint(payload: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint for a canonicalized payload dictionary.
    Keys are sorted, timestamps are normalized strings, and separators are standardized.
    Does NOT use Python's built-in hash().
    """
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


class InventoryLedgerEntry(BaseModel):
    """
    Immutable operational inventory ledger record.
    Represents an atomic, append-only physical stock movement.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    posting_id: str = Field(..., description="Unique posting identifier (e.g. POST-...)")
    transaction_id: str = Field(..., description="Unique logical transaction identifier (e.g. TXN-...)")
    idempotency_key: str = Field(..., description="Caller idempotency key for durable deduplication")
    store_id: str = Field(..., description="Target dark store identifier")
    sku_id: str = Field(..., description="Product SKU identifier")
    batch_id: str = Field(..., description="Inventory batch identifier")
    transaction_type: InventoryTransactionType = Field(..., description="Type of stock movement")
    quantity: float = Field(..., description="Signed quantity (+ for additions, - for depletions)")
    unit_cost: float = Field(..., ge=0.0, description="Unit cost in INR")
    total_cost: float = Field(..., description="Total cost in INR: round(quantity * unit_cost, 2)")
    remaining_shelf_life_days: Optional[int] = Field(default=None, ge=0, description="Remaining shelf-life days at posting")
    expiry_date: Optional[str] = Field(default=None, description="Calculated expiration date YYYY-MM-DD")

    # Source provenance references
    fo_id: Optional[str] = Field(default=None, description="Originating Fulfilment Order identifier")
    shipment_id: Optional[str] = Field(default=None, description="Originating shipment identifier")
    grn_id: Optional[str] = Field(default=None, description="Originating Goods Receipt Note identifier")
    fulfilment_batch_id: Optional[str] = Field(default=None, description="Originating FulfilmentBatch identifier")

    # Audit metadata
    actor_type: str = Field(default="STORE_RECEIVER", description="Actor role initiating posting")
    actor_id: str = Field(..., description="Actor identifier")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of movement")
    reason: str = Field(..., description="Business rationale for posting")
    payload_fingerprint: str = Field(..., description="Deterministic SHA-256 fingerprint of posting payload")
    notes: Optional[str] = Field(default=None, description="Operational remarks")
    created_at: str = Field(..., description="Record creation timestamp")

    @field_validator("posting_id", "transaction_id", "idempotency_key", "batch_id", "actor_id", "reason", "payload_fingerprint")
    @classmethod
    def validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("store_id")
    @classmethod
    def validate_store(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "store_id")
        if clean not in CANONICAL_STORES:
            raise ValueError(f"store_id '{clean}' must be one of {CANONICAL_STORES}")
        return clean

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("timestamp", "created_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v: Any, info) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

    @field_validator("expiry_date")
    @classmethod
    def validate_expiry_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = _validate_non_empty_str(v, "expiry_date")
        try:
            dt_date.fromisoformat(clean)
        except ValueError as err:
            raise ValueError(f"expiry_date '{clean}' must be in YYYY-MM-DD format: {err}") from err
        if len(clean) != 10 or clean[4] != '-' or clean[7] != '-':
            raise ValueError(f"expiry_date '{clean}' must be in YYYY-MM-DD format.")
        return clean

    @model_validator(mode="after")
    def validate_ledger_entry_invariants(self) -> "InventoryLedgerEntry":
        # Non-zero movement quantity check
        if abs(self.quantity) < 1e-4:
            raise ValueError("Ledger movement quantity must be non-zero.")

        # Inbound receipt must have positive quantity
        if self.transaction_type == InventoryTransactionType.INBOUND_RECEIPT and self.quantity <= 0.0:
            raise ValueError(f"INBOUND_RECEIPT quantity must be strictly positive; got {self.quantity}.")

        # Financial valuation check
        expected_total = round(self.quantity * self.unit_cost, 2)
        if abs(self.total_cost - expected_total) > 1e-2:
            raise ValueError(
                f"total_cost mismatch: expected {expected_total} ({self.quantity} * {self.unit_cost}), got {self.total_cost}."
            )

        return self


class InventoryBatchRecord(BaseModel):
    """
    Durable representation of a physical inventory batch tracked in the store.
    Directly traceable to an accepted FulfilmentBatch and GRN.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    batch_id: str = Field(..., description="Unique batch identifier")
    store_id: str = Field(..., description="Store holding this batch")
    sku_id: str = Field(..., description="Product SKU identifier")
    initial_units: float = Field(..., gt=0.0, description="Initial accepted units received into this batch")
    on_hand_units: float = Field(..., ge=0.0, description="Current derived on-hand units for this batch")
    receipt_date: str = Field(..., description="Intake date YYYY-MM-DD")
    expiry_date: str = Field(..., description="Calculated expiration date YYYY-MM-DD")
    remaining_shelf_life_days_at_receipt: int = Field(..., ge=0, description="Verified remaining shelf-life days at receipt")
    unit_cost: float = Field(..., ge=0.0, description="Unit cost in INR")
    fo_id: str = Field(..., description="Originating Fulfilment Order identifier")
    grn_id: str = Field(..., description="Originating Goods Receipt Note identifier")
    fulfilment_batch_id: str = Field(..., description="Source FulfilmentBatch identifier")
    created_at: str = Field(..., description="Intake record creation timestamp")

    @field_validator("batch_id", "fo_id", "grn_id", "fulfilment_batch_id")
    @classmethod
    def validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("store_id")
    @classmethod
    def validate_store(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "store_id")
        if clean not in CANONICAL_STORES:
            raise ValueError(f"store_id '{clean}' must be one of {CANONICAL_STORES}")
        return clean

    @field_validator("sku_id")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        clean = _validate_non_empty_str(v, "sku_id")
        if clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{clean}' must be one of {CANONICAL_SKUS}")
        return clean

    @field_validator("receipt_date", "expiry_date")
    @classmethod
    def validate_dates(cls, v: str, info) -> str:
        clean = _validate_non_empty_str(v, info.field_name)
        try:
            dt_date.fromisoformat(clean)
        except ValueError as err:
            raise ValueError(f"{info.field_name} '{clean}' must be YYYY-MM-DD: {err}") from err
        return clean

    def to_inventory_batch(self, as_of_date: Optional[str] = None) -> InventoryBatch:
        """
        Converts to core InventoryBatch representation used by Inventory Engine.
        Calculates remaining shelf-life days as of as_of_date (or intake date).
        """
        ref_date = dt_date.fromisoformat(as_of_date) if as_of_date else dt_date.fromisoformat(self.receipt_date)
        exp_date = dt_date.fromisoformat(self.expiry_date)
        remaining_days = max(0, (exp_date - ref_date).days)
        return InventoryBatch(
            batch_id=self.batch_id,
            units=self.on_hand_units,
            remaining_shelf_life_days=remaining_days,
            creation_date=self.receipt_date,
        )


class StoreSkuBalance(BaseModel):
    """
    Aggregated on-hand inventory position for a single SKU at a dark store.
    Derived dynamically from the immutable ledger.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str
    sku_id: str
    on_hand_stock: float = Field(..., ge=0.0, description="Authoritative on-hand quantity derived from ledger entries")
    total_cost_inr: float = Field(default=0.0, ge=0.0, description="Total valuation in INR of current on-hand stock")
    active_batch_count: int = Field(default=0, ge=0, description="Number of distinct active batches with positive stock")
    last_movement_at: Optional[str] = Field(default=None, description="Timestamp of most recent ledger movement")


class BatchBalance(BaseModel):
    """
    Per-batch on-hand balance and expiration status derived dynamically from the ledger.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str
    sku_id: str
    batch_id: str
    on_hand_quantity: float = Field(..., ge=0.0, description="Current on-hand quantity in this batch")
    receipt_date: str
    expiry_date: str
    remaining_shelf_life_days_at_receipt: int
    current_remaining_shelf_life_days: Optional[int] = None
    is_expired: bool = False
    unit_cost: float = Field(default=0.0, ge=0.0)
    fo_id: Optional[str] = None
    grn_id: Optional[str] = None
    fulfilment_batch_id: Optional[str] = None


class GrnLinePostingReconciliation(BaseModel):
    """Reconciliation of posted quantities against a single GRN line."""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sku_id: str
    accepted_quantity: float = Field(..., ge=0.0)
    rejected_quantity: float = Field(..., ge=0.0)
    posted_quantity: float = Field(..., ge=0.0)
    unposted_accepted_quantity: float = Field(..., ge=0.0)
    batch_references: Tuple[str, ...] = Field(default_factory=tuple)
    posted_batch_ids: Tuple[str, ...] = Field(default_factory=tuple)
    is_fully_posted: bool = False


class GrnPostingReconciliation(BaseModel):
    """Full GRN-to-ledger posting audit reconciliation."""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    grn_id: str
    fo_id: str
    store_id: str
    supplier_id: str
    receipt_at: str
    total_accepted_quantity: float = Field(..., ge=0.0)
    total_posted_quantity: float = Field(..., ge=0.0)
    is_fully_posted: bool = False
    lines: Tuple[GrnLinePostingReconciliation, ...] = Field(default_factory=tuple)
