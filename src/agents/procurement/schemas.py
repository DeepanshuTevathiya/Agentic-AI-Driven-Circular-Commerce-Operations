"""
Lead 2 Procurement Data Contracts & Pydantic V2 Schemas
=======================================================
Defines validated, immutable data schemas for:
- SupplierProfile (simulated vendor attributes, lead times, MOQ, case packs)
- PurchaseOrderLineItem (SKU-level procurement line items with case-pack and MOQ constraints)
- PurchaseOrder (store-supplier level purchase order with escalation and approval gates)

CRITICAL PROCUREMENT POLICIES & CONSTRAINTS:
1. Purely simulated vendor and contract data — explicitly labeled as simulation.
2. Case-pack rounding: final order quantities must be integer multiples of the case pack.
3. Minimum Order Quantity (MOQ): non-zero order quantities must meet or exceed the MOQ.
4. Escalation thresholds (₹5,000 per line, ₹25,000 per store PO) flag orders for human review;
   they do NOT authorize automated approval.
5. MANDATORY HUMAN APPROVAL: No purchase order may transition to APPROVED or ISSUED
   without an explicit human operator signature/identifier ('approved_by').
"""

import math
from datetime import date as dt_date, datetime as dt_datetime
from enum import Enum
from typing import List, Optional, Tuple, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..common.config import (
    CANONICAL_CATEGORIES,
    CANONICAL_SKUS,
    CANONICAL_STORES,
)


class SupplierType(str, Enum):
    """Classification of simulated suppliers."""
    PRIMARY = "PRIMARY"  # Default contracted vendor with standard lead time and cost
    BACKUP = "BACKUP"    # Secondary expedited vendor with shorter lead time and price premium


class PurchaseOrderUrgency(str, Enum):
    """Operational urgency classification for purchase orders."""
    STANDARD = "STANDARD"    # Routine cycle replenishment
    EXPEDITE = "EXPEDITE"    # Critical reorder due to imminent or actual stockout risk
    EMERGENCY = "EMERGENCY"  # Immediate priority dispatch under severe stockout shock


class PurchaseOrderStatus(str, Enum):
    """
    Lifecycle status of a Purchase Order.

    IMPORTANT:
    Transitions to APPROVED or ISSUED strictly require explicit human signoff ('approved_by').
    """
    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ISSUED = "ISSUED"
    CANCELLED = "CANCELLED"


class ApprovalAction(str, Enum):
    """Classification of human review decisions."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"


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
        try:
            parsed = dt_datetime.strptime(cleaned[:10], "%Y-%m-%d").date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError as err:
            raise ValueError(f"Invalid date string '{v}'. Expected YYYY-MM-DD format.") from err
    raise ValueError(f"Unsupported date type: {type(v)}")


class SupplierProfile(BaseModel):
    """
    Validated profile for a simulated quick-commerce supplier.
    Contains delivery lead time, MOQ, packaging sizing, and pricing terms.
    Explicitly labeled as a simulation entity.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    supplier_id: str = Field(..., description="Unique simulated supplier identifier")
    supplier_name: str = Field(..., description="Simulated vendor trade name")
    supplier_type: SupplierType = Field(..., description="PRIMARY or BACKUP supplier tier")
    category: str = Field(..., description="Product category handled by this vendor")

    lead_time_mean_days: float = Field(..., gt=0.0, description="Mean delivery lead time in days")
    lead_time_std_days: float = Field(..., ge=0.0, description="Lead time standard deviation in days")
    cogs_ratio: float = Field(..., gt=0.0, le=1.0, description="Wholesale procurement cost ratio relative to retail price")
    min_order_qty: float = Field(..., ge=0.0, description="Minimum order quantity (MOQ) in base units")
    case_pack_size: int = Field(..., ge=1, description="Number of base units per shipping case pack")

    is_active: bool = Field(default=True, description="Whether the vendor is available for active dispatch")
    is_simulation_vendor: bool = Field(default=True, description="Certifies this profile is a simulation assumption")

    @field_validator("supplier_id", "supplier_name")
    @classmethod
    def validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v_clean = _validate_non_empty_str(v, "category")
        if v_clean not in CANONICAL_CATEGORIES:
            raise ValueError(f"Category '{v_clean}' must be one of {CANONICAL_CATEGORIES}")
        return v_clean


class PurchaseOrderLineItem(BaseModel):
    """
    Individual SKU line item within a Purchase Order.
    Encapsulates raw inventory replenishment deficit, discrete case-pack rounding,
    minimum order quantity adherence, and financial valuation.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    store_id: str = Field(..., description="Bangalore dark store identifier")
    sku_id: str = Field(..., description="Unique product SKU identifier")
    category: str = Field(..., description="Product category")

    raw_suggested_qty: float = Field(..., ge=0.0, description="Unrounded net replenishment deficit from InventoryEngine")
    case_pack_rounded_qty: float = Field(..., ge=0.0, description="Quantity rounded up to the nearest integer case pack")
    final_order_qty: float = Field(..., ge=0.0, description="Final order quantity after applying both case-pack and MOQ constraints")

    case_pack_size: int = Field(..., ge=1, description="Units per case pack")
    moq: float = Field(..., ge=0.0, description="Supplier minimum order quantity")

    unit_retail_price: float = Field(..., gt=0.0, description="Unit retail price from df_master in INR")
    unit_procurement_cost: float = Field(..., gt=0.0, description="Unit wholesale cost in INR (e.g. 0.60 * retail)")
    total_line_cost: float = Field(..., ge=0.0, description="Total procurement cost = round(final_order_qty * unit_procurement_cost, 2)")

    requires_escalation: bool = Field(default=False, description="Flagged for human operator review (e.g. line cost > ₹5,000)")
    escalation_reasons: Tuple[str, ...] = Field(default_factory=tuple, description="Specific triggers requiring human review")

    @field_validator("store_id")
    @classmethod
    def validate_store_id(cls, v: str) -> str:
        v_clean = _validate_non_empty_str(v, "store_id")
        if v_clean not in CANONICAL_STORES:
            raise ValueError(f"store_id '{v_clean}' must be one of {CANONICAL_STORES}")
        return v_clean

    @field_validator("sku_id")
    @classmethod
    def validate_sku_id(cls, v: str) -> str:
        v_clean = _validate_non_empty_str(v, "sku_id")
        if v_clean not in CANONICAL_SKUS:
            raise ValueError(f"sku_id '{v_clean}' must be one of {CANONICAL_SKUS}")
        return v_clean

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v_clean = _validate_non_empty_str(v, "category")
        if v_clean not in CANONICAL_CATEGORIES:
            raise ValueError(f"category '{v_clean}' must be one of {CANONICAL_CATEGORIES}")
        return v_clean

    @model_validator(mode="after")
    def validate_line_item_math(self) -> "PurchaseOrderLineItem":
        # Wholesale cost must not exceed retail price
        if self.unit_procurement_cost > self.unit_retail_price + 1e-4:
            raise ValueError(
                f"Wholesale procurement cost ({self.unit_procurement_cost}) cannot exceed "
                f"retail price ({self.unit_retail_price}) for SKU '{self.sku_id}'."
            )

        # Case-pack multiple check: if final_order_qty > 0, it must be an integer multiple of case_pack_size
        if self.final_order_qty > 0.0:
            remainder = abs(self.final_order_qty % self.case_pack_size)
            if remainder > 1e-4 and abs(remainder - self.case_pack_size) > 1e-4:
                raise ValueError(
                    f"final_order_qty ({self.final_order_qty}) is not an exact integer multiple of "
                    f"case_pack_size ({self.case_pack_size}) for SKU '{self.sku_id}'."
                )

            # MOQ check: if ordering, must be >= MOQ
            if self.final_order_qty < (self.moq - 1e-4):
                raise ValueError(
                    f"final_order_qty ({self.final_order_qty}) is below supplier MOQ ({self.moq}) "
                    f"for SKU '{self.sku_id}'."
                )

        # Total line cost mathematical consistency check
        expected_cost = round(self.final_order_qty * self.unit_procurement_cost, 2)
        if abs(self.total_line_cost - expected_cost) > 0.05:
            raise ValueError(
                f"total_line_cost mismatch for SKU '{self.sku_id}': reported={self.total_line_cost}, "
                f"expected={expected_cost} (final_qty={self.final_order_qty} * unit_cost={self.unit_procurement_cost})."
            )

        # Automatic escalation flagging for line cost > ₹5,000
        if self.total_line_cost > 5000.0 and not self.requires_escalation:
            raise ValueError(
                f"Line item cost ₹{self.total_line_cost:.2f} exceeds threshold ₹5,000.00; "
                "requires_escalation must be set to True with an escalation reason."
            )

        return self


class PurchaseOrderAuditRecord(BaseModel):
    """
    Immutable audit record documenting a single human review action on a Purchase Order.
    Preserves full audit trail for regulatory and operational governance.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    po_id: str = Field(..., description="Unique Purchase Order identifier")
    previous_status: PurchaseOrderStatus = Field(..., description="Order status before action")
    new_status: PurchaseOrderStatus = Field(..., description="Order status after action")
    reviewer_id: str = Field(..., description="Non-empty identifier of human reviewer")
    timestamp: str = Field(..., description="ISO 8601 timestamp of review action")
    action: ApprovalAction = Field(..., description="APPROVE or REJECT action")
    reason: str = Field(..., description="Review rationale or mandatory rejection reason")
    requires_escalation: bool = Field(..., description="Whether order had active escalation flags")
    escalation_reasons: Tuple[str, ...] = Field(default_factory=tuple, description="Escalation reasons active at review")

    @field_validator("po_id", "reviewer_id", "reason", "timestamp")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)


class PurchaseOrder(BaseModel):
    """
    Validated multi-item Purchase Order dispatched to a specific supplier for a dark store.

    SAFETY & GOVERNANCE:
    - Encapsulates line items, total units, total expenditure, and escalation reasons.
    - MANDATORY HUMAN APPROVAL: status cannot be APPROVED or ISSUED unless 'approved_by'
      is explicitly populated with a human operator identifier and timestamp.
    """
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    po_id: str = Field(..., description="Unique Purchase Order identifier (e.g. PO-20241110-DS_BLR_HSR_LAYOUT-001)")
    created_date: str = Field(..., description="Creation date (YYYY-MM-DD)")
    store_id: str = Field(..., description="Destination Bangalore dark store identifier")
    supplier_id: str = Field(..., description="Target vendor identifier")
    supplier_type: SupplierType = Field(..., description="PRIMARY or BACKUP supplier")

    items: Tuple[PurchaseOrderLineItem, ...] = Field(..., min_length=1, description="List of SKU line items in this order")

    total_order_qty: float = Field(..., ge=0.0, description="Sum of final_order_qty across all line items")
    total_order_cost: float = Field(..., ge=0.0, description="Sum of total_line_cost across all line items in INR")

    urgency: PurchaseOrderUrgency = Field(default=PurchaseOrderUrgency.STANDARD, description="Operational order urgency")
    status: PurchaseOrderStatus = Field(default=PurchaseOrderStatus.PENDING_APPROVAL, description="Order approval status")

    requires_escalation: bool = Field(default=False, description="Flagged for mandatory operator escalation/review")
    escalation_reasons: Tuple[str, ...] = Field(default_factory=tuple, description="Consolidated escalation triggers")

    approved_by: Optional[str] = Field(default=None, description="Human operator signature / ID required for approval/issuance")
    approval_timestamp: Optional[str] = Field(default=None, description="Timestamp when human approval was registered (ISO format)")

    rejected_by: Optional[str] = Field(default=None, description="Human reviewer identifier who rejected the order")
    rejection_timestamp: Optional[str] = Field(default=None, description="Timestamp when rejection was registered (ISO format)")
    rejection_reason: Optional[str] = Field(default=None, description="Mandatory explanatory rationale for rejection")

    approval_reason: Optional[str] = Field(default=None, description="Recorded approval rationale or notes")
    audit_trail: Tuple[PurchaseOrderAuditRecord, ...] = Field(default_factory=tuple, description="Immutable audit history of all review events")

    assumed_zero_pipeline_flag: bool = Field(
        default=False,
        description="Propagated flag indicating whether inventory position relied on assumed zero pipeline"
    )
    is_simulation_order: bool = Field(default=True, description="Certifies this purchase order is a simulation output")

    @field_validator("po_id", "supplier_id")
    @classmethod
    def validate_identifiers(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("store_id")
    @classmethod
    def validate_store_id(cls, v: str) -> str:
        v_clean = _validate_non_empty_str(v, "store_id")
        if v_clean not in CANONICAL_STORES:
            raise ValueError(f"store_id '{v_clean}' must be one of {CANONICAL_STORES}")
        return v_clean

    @field_validator("created_date")
    @classmethod
    def validate_date(cls, v: Union[str, dt_date]) -> str:
        return _normalize_date_str(v)

    @model_validator(mode="after")
    def validate_purchase_order_integrity(self) -> "PurchaseOrder":
        # 1. Store consistency across all line items
        for item in self.items:
            if item.store_id != self.store_id:
                raise ValueError(
                    f"Line item store_id '{item.store_id}' does not match PurchaseOrder store_id '{self.store_id}'."
                )

        # 2. Total quantity sum consistency
        expected_qty = sum(item.final_order_qty for item in self.items)
        if abs(self.total_order_qty - expected_qty) > 1e-4:
            raise ValueError(
                f"total_order_qty mismatch: reported={self.total_order_qty}, expected sum of items={expected_qty}."
            )

        # 3. Total cost sum consistency
        expected_cost = round(sum(item.total_line_cost for item in self.items), 2)
        if abs(self.total_order_cost - expected_cost) > 0.05:
            raise ValueError(
                f"total_order_cost mismatch: reported={self.total_order_cost}, expected sum of items={expected_cost}."
            )

        # 4. Escalation threshold consistency
        exceeds_order_threshold = self.total_order_cost > 25000.0
        any_item_escalated = any(item.requires_escalation for item in self.items)
        if (exceeds_order_threshold or any_item_escalated) and not self.requires_escalation:
            raise ValueError(
                f"PurchaseOrder total cost ₹{self.total_order_cost:.2f} or item escalation triggers review threshold; "
                "requires_escalation must be True."
            )

        # 5. MANDATORY HUMAN APPROVAL ENFORCEMENT
        # A Purchase Order cannot be marked APPROVED or ISSUED without an explicit human signoff and timestamp
        if self.status in [PurchaseOrderStatus.APPROVED, PurchaseOrderStatus.ISSUED]:
            if not self.approved_by or not self.approved_by.strip():
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' cannot have status '{self.status.value}' "
                    "without explicit human operator approval ('approved_by'). Automated issuance is strictly prohibited."
                )
            if not self.approval_timestamp or not self.approval_timestamp.strip():
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' cannot have status '{self.status.value}' "
                    "without explicit approval timestamp ('approval_timestamp')."
                )

        # 6. MANDATORY REJECTION INTEGRITY ENFORCEMENT
        # A Purchase Order cannot be marked REJECTED without reviewer, timestamp, and rejection reason
        if self.status == PurchaseOrderStatus.REJECTED:
            if not self.rejected_by or not self.rejected_by.strip():
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' cannot have status 'REJECTED' "
                    "without explicit human reviewer identifier ('rejected_by')."
                )
            if not self.rejection_timestamp or not self.rejection_timestamp.strip():
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' cannot have status 'REJECTED' "
                    "without explicit rejection timestamp ('rejection_timestamp')."
                )
            if not self.rejection_reason or not self.rejection_reason.strip():
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' cannot have status 'REJECTED' "
                    "without explicit rejection reason ('rejection_reason')."
                )

        return self
