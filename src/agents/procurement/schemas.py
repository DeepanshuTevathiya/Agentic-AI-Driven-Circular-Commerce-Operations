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
import re
from datetime import date as dt_date, datetime as dt_datetime, timezone
from enum import Enum
from typing import Any, List, Optional, Tuple, Union
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


# Strict ISO 8601 regex pattern: requires YYYY-MM-DDTHH:MM:SS with explicit timezone (Z or +/-HH:MM)
ISO8601_STRICT_PATTERN = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"[Tt](?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?"
    r"(?:[Zz]|[+-](?:(?:0\d|1[0-3]):[0-5]\d|14:00))$"
)


def validate_and_normalize_iso_timestamp(
    v: Any,
    field_name: str = "timestamp",
    error_cls: type = ValueError,
) -> str:
    """
    Validates that the provided input is a valid, timezone-aware ISO 8601 timestamp,
    and returns a normalized UTC ISO 8601 string representation ending in 'Z'.

    Strict Validation Rules:
    - Must be a non-empty string or timezone-aware datetime instance.
    - Must match strict ISO 8601 format with explicit timezone (Z or +/-HH:MM offset).
    - Timezone-naive timestamps, missing timezones, and malformed strings are rejected.
    - Invalid calendar dates and times (e.g. Feb 30, hour 25, minute 60) are rejected.
    - Invalid timezone offsets (e.g. +25:00, +05:60) are rejected.
    - Successfully validated timestamps are normalized to UTC ending with 'Z'.
    - Never uses machine local timezone or assumes UTC for naive inputs.

    Args:
        v: The input timestamp value to validate.
        field_name: Name of the field for error reporting.
        error_cls: Exception class to raise on error (default ValueError; can be ApprovalGovernanceError).

    Returns:
        Normalized UTC ISO 8601 string (e.g. '2026-11-08T12:30:00Z').
    """
    if v is None:
        raise error_cls(f"Field '{field_name}' cannot be None.")

    if isinstance(v, dt_datetime):
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise error_cls(
                f"Governance Violation: Field '{field_name}' must be timezone-aware. Naive datetime is prohibited."
            )
        dt_utc = v.astimezone(timezone.utc)
        if dt_utc.microsecond == 0:
            return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    if not isinstance(v, str):
        raise error_cls(
            f"Governance Violation: Field '{field_name}' must be a string or datetime, got {type(v).__name__}."
        )

    v_clean = v.strip()
    if not v_clean:
        raise error_cls(
            f"Governance Violation: Field '{field_name}' cannot be empty or whitespace-only."
        )

    if not ISO8601_STRICT_PATTERN.match(v_clean):
        raise error_cls(
            f"Governance Violation: Field '{field_name}' must be a valid timezone-aware ISO 8601 timestamp "
            f"(e.g. 'YYYY-MM-DDTHH:MM:SSZ' or '+HH:MM'). Got '{v}'."
        )

    try:
        dt = dt_datetime.fromisoformat(v_clean)
    except Exception as err:
        raise error_cls(
            f"Governance Violation: Field '{field_name}' has invalid date or time values: '{v}'."
        ) from err

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise error_cls(
            f"Governance Violation: Field '{field_name}' must have an explicit timezone offset or UTC indicator: '{v}'."
        )

    dt_utc = dt.astimezone(timezone.utc)
    if dt_utc.microsecond == 0:
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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

    @field_validator("po_id", "reviewer_id", "reason")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_audit_timestamp(cls, v: Any) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name="timestamp", error_cls=ValueError)


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

    @field_validator("approval_timestamp", "rejection_timestamp", mode="before")
    @classmethod
    def validate_po_timestamps(cls, v: Any, info) -> Optional[str]:
        if v is None:
            return None
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=ValueError)

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

        # 5. MANDATORY TERMINAL DECISION AUDIT TRAIL ENFORCEMENT
        terminal_records = [
            rec for rec in self.audit_trail
            if rec.action in [ApprovalAction.APPROVE, ApprovalAction.REJECT]
        ]

        # 5a. MANDATORY HUMAN APPROVAL ENFORCEMENT (APPROVED or ISSUED)
        # A Purchase Order cannot be marked APPROVED or ISSUED without explicit human signoff,
        # normalized timestamp, and an exact matching APPROVE audit record in audit_trail.
        # NOTE: Reviewer IDs are validated for consistency against the audit trail; authenticating
        # human identity and role-based permissions remains an application-layer responsibility.
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

            # Audit record presence and uniqueness
            if len(terminal_records) == 0:
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' in status '{self.status.value}' "
                    "must contain a matching APPROVE audit record in 'audit_trail'."
                )
            if len(terminal_records) > 1:
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' in status '{self.status.value}' "
                    f"contains duplicate or conflicting terminal decision audit records ({len(terminal_records)} found)."
                )

            audit = terminal_records[0]
            if audit.action != ApprovalAction.APPROVE:
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' in status '{self.status.value}' "
                    f"has conflicting terminal decision action '{audit.action.value}'. Expected APPROVE."
                )
            if audit.po_id != self.po_id:
                raise ValueError(
                    f"Mandatory Governance Violation: Approval audit record po_id '{audit.po_id}' "
                    f"does not match Purchase Order po_id '{self.po_id}'."
                )
            if audit.previous_status != PurchaseOrderStatus.PENDING_APPROVAL:
                raise ValueError(
                    f"Mandatory Governance Violation: Approval audit record previous_status '{audit.previous_status.value}' "
                    "must be PENDING_APPROVAL."
                )
            if audit.new_status != PurchaseOrderStatus.APPROVED:
                raise ValueError(
                    f"Mandatory Governance Violation: Approval audit record new_status '{audit.new_status.value}' "
                    "must be APPROVED."
                )
            if audit.reviewer_id != self.approved_by:
                raise ValueError(
                    f"Mandatory Governance Violation: Approval audit record reviewer_id '{audit.reviewer_id}' "
                    f"does not match Purchase Order approved_by '{self.approved_by}'."
                )
            if audit.timestamp != self.approval_timestamp:
                raise ValueError(
                    f"Mandatory Governance Violation: Approval audit record timestamp '{audit.timestamp}' "
                    f"does not match Purchase Order approval_timestamp '{self.approval_timestamp}'."
                )

        # 5b. MANDATORY REJECTION INTEGRITY ENFORCEMENT (REJECTED)
        # A Purchase Order cannot be marked REJECTED without reviewer, timestamp, rejection reason,
        # and an exact matching REJECT audit record in audit_trail.
        elif self.status == PurchaseOrderStatus.REJECTED:
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

            # Audit record presence and uniqueness
            if len(terminal_records) == 0:
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' in status 'REJECTED' "
                    "must contain a matching REJECT audit record in 'audit_trail'."
                )
            if len(terminal_records) > 1:
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' in status 'REJECTED' "
                    f"contains duplicate or conflicting terminal decision audit records ({len(terminal_records)} found)."
                )

            audit = terminal_records[0]
            if audit.action != ApprovalAction.REJECT:
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' in status 'REJECTED' "
                    f"has conflicting terminal decision action '{audit.action.value}'. Expected REJECT."
                )
            if audit.po_id != self.po_id:
                raise ValueError(
                    f"Mandatory Governance Violation: Rejection audit record po_id '{audit.po_id}' "
                    f"does not match Purchase Order po_id '{self.po_id}'."
                )
            if audit.previous_status != PurchaseOrderStatus.PENDING_APPROVAL:
                raise ValueError(
                    f"Mandatory Governance Violation: Rejection audit record previous_status '{audit.previous_status.value}' "
                    "must be PENDING_APPROVAL."
                )
            if audit.new_status != PurchaseOrderStatus.REJECTED:
                raise ValueError(
                    f"Mandatory Governance Violation: Rejection audit record new_status '{audit.new_status.value}' "
                    "must be REJECTED."
                )
            if audit.reviewer_id != self.rejected_by:
                raise ValueError(
                    f"Mandatory Governance Violation: Rejection audit record reviewer_id '{audit.reviewer_id}' "
                    f"does not match Purchase Order rejected_by '{self.rejected_by}'."
                )
            if audit.timestamp != self.rejection_timestamp:
                raise ValueError(
                    f"Mandatory Governance Violation: Rejection audit record timestamp '{audit.timestamp}' "
                    f"does not match Purchase Order rejection_timestamp '{self.rejection_timestamp}'."
                )
            if audit.reason != self.rejection_reason:
                raise ValueError(
                    f"Mandatory Governance Violation: Rejection audit record reason '{audit.reason}' "
                    f"does not match Purchase Order rejection_reason '{self.rejection_reason}'."
                )

        # 5c. NON-TERMINAL STATUS AUDIT RECORD PROHIBITION (DRAFT, PENDING_APPROVAL)
        elif self.status in [PurchaseOrderStatus.DRAFT, PurchaseOrderStatus.PENDING_APPROVAL]:
            if len(terminal_records) > 0:
                raise ValueError(
                    f"Mandatory Governance Violation: Purchase Order '{self.po_id}' in status '{self.status.value}' "
                    f"cannot contain terminal decision audit records ({len(terminal_records)} found)."
                )

        # 5d. PROHIBITION OF CANCELLED STATUS IN PHASE 4
        # PurchaseOrderStatus.CANCELLED is reserved for a future explicit cancellation workflow.
        # Direct construction or transition to CANCELLED is strictly prohibited in Phase 4 governance.
        elif self.status == PurchaseOrderStatus.CANCELLED:
            raise ValueError(
                f"Mandatory Governance Violation: Purchase Order '{self.po_id}' cannot have status 'CANCELLED'. "
                "Order cancellation is not supported in Phase 4 governance."
            )
        else:
            raise ValueError(
                f"Mandatory Governance Violation: Unsupported Purchase Order status '{self.status}'."
            )

        return self
