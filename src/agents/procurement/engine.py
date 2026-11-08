"""
Lead 2 Core Procurement Engine
==============================
Converts deterministic Inventory Engine replenishment recommendations into
actionable, validated procurement line items and grouped Purchase Orders.

KEY PROCUREMENT POLICIES & CONSTRAINTS:
1. Purely simulated vendor and contract terms — explicitly labeled as simulation.
2. Supplier Selection:
   - Primary Supplier: 0.60 COGS price ratio, category-specific standard lead time (1.0-1.5 days).
   - Backup Supplier: 0.75 COGS price ratio, expedited lead time (0.8 days).
   - Selection Rule: Backup supplier is selected ONLY when the inventory situation is
     critical (StockRiskLevel.CRITICAL_LOW or on_hand <= 0 / IP <= SS_LT) AND remaining stock
     coverage is less than the primary supplier's delivery lead time. Otherwise, prefer primary.
3. Quantity Calculation:
   - Starts with unconstrained suggested_replenishment_qty from InventoryEngine.
   - If suggested_replenishment_qty <= 0, no order line is generated.
   - Rounds upward to category-specific case-pack size.
   - Enforces category-specific Minimum Order Quantity (MOQ).
   - In critical / stockout situations: quantity is raised to MOQ (and case pack multiple).
   - In non-critical situations: if raising to MOQ pushes projected inventory above target stock,
     flags the recommendation for manual review rather than silently accepting excess.
4. Financial Valuation:
   - Wholesale unit cost = retail unit price * supplier COGS ratio.
   - Line cost = final order quantity * wholesale unit cost.
   - Store PO total = sum of line costs.
5. Purchase Order Generation:
   - Groups line items by destination store and target supplier.
   - Generates draft Purchase Orders in non-issued status (PENDING_APPROVAL).
   - Escalation review thresholds (₹5,000 per line, ₹25,000 per store PO) flag orders
     for mandatory operator review; they NEVER authorize automated approval.
   - MANDATORY HUMAN APPROVAL: strictly leaves approved_by=None; no PO can be issued
     or dispatched without human operator signoff.
6. Data Integrity Restrictions:
   - Never consumes Target_Demand as operational input.
   - Preserves and propagates assumed_zero_pipeline flag.
   - Expiry risk remains NOT_ASSESSABLE without validated batch records.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..common.config import (
    CANONICAL_CATEGORIES,
    CANONICAL_SKUS,
    CANONICAL_STORES,
    PROJECT_ROOT,
    SKU_TO_CATEGORY,
    Lead2Config,
    get_default_config,
)
from ..common.schemas import (
    DailyAgentInput,
    ExpiryRiskLevel,
    InventoryCalculationResult,
    InventoryPosition,
    InventoryRiskAssessment,
    StockRiskLevel,
)
from ..inventory.engine import (
    compute_protection_safety_stock,
    compute_target_stock,
)
from .schemas import (
    PurchaseOrder,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierProfile,
    SupplierType,
)


# Default retail benchmark prices (derived from median Unit_Price in df_master.parquet)
DEFAULT_SKU_RETAIL_PRICES: Dict[str, float] = {
    # Bakery (~₹35)
    "SKU_BREAD_WHITE": 35.01,
    "SKU_BREAD_WW": 34.99,
    "SKU_BUNS_6PK": 35.02,
    # Dairy (~₹55)
    "SKU_BUTTER_100G": 54.97,
    "SKU_CURD_400G": 55.01,
    "SKU_EGGS_12PK": 55.01,
    "SKU_MILK_1L": 55.00,
    "SKU_PANEER_200G": 55.02,
    # Meat (~₹160)
    "SKU_CHICKEN_500G": 160.07,
    "SKU_MUTTON_500G": 159.84,
    # Produce (~₹42)
    "SKU_BANANA_DZ": 41.96,
    "SKU_ONION_1KG": 42.04,
    "SKU_POTATO_1KG": 42.00,
    "SKU_SPINACH_500G": 42.05,
    "SKU_TOMATO_1KG": 42.01,
}

DEFAULT_CATEGORY_RETAIL_PRICES: Dict[str, float] = {
    "Bakery": 35.0,
    "Dairy": 55.0,
    "Meat": 160.0,
    "Produce": 42.0,
}


def load_retail_price_catalog(path: Optional[Path] = None) -> Dict[str, float]:
    """
    Loads SKU-level unit retail prices from df_master.parquet.
    Falls back to DEFAULT_SKU_RETAIL_PRICES if the file is absent or incomplete.
    """
    master_path = path or (PROJECT_ROOT / "data" / "processed" / "df_master.parquet")
    if not master_path.exists():
        return dict(DEFAULT_SKU_RETAIL_PRICES)

    try:
        df = pd.read_parquet(master_path, columns=["SKU_ID", "Unit_Price"])
        if "SKU_ID" in df.columns and "Unit_Price" in df.columns:
            prices = df.groupby("SKU_ID")["Unit_Price"].median().round(2).to_dict()
            # Ensure all canonical SKUs exist in returned catalog
            for sku, default_p in DEFAULT_SKU_RETAIL_PRICES.items():
                if sku not in prices:
                    prices[sku] = default_p
            return prices
    except Exception:
        pass

    return dict(DEFAULT_SKU_RETAIL_PRICES)


def get_simulated_supplier_profile(
    category: str,
    supplier_type: SupplierType = SupplierType.PRIMARY,
    config: Optional[Lead2Config] = None,
) -> SupplierProfile:
    """
    Constructs a validated SupplierProfile for a specific category and supplier tier.
    All supplier entities are simulated placeholders for operations modeling.
    """
    cfg = config or get_default_config()
    if category not in CANONICAL_CATEGORIES:
        raise ValueError(f"Category '{category}' must be one of {CANONICAL_CATEGORIES}")

    proc_params = cfg.category_procurement_params.get(category)
    if proc_params is None:
        raise ValueError(f"Missing procurement parameters for category '{category}'.")

    supp_params = cfg.category_supplier_params.get(category, cfg.default_supplier_params)

    if supplier_type == SupplierType.PRIMARY:
        return SupplierProfile(
            supplier_id=f"SIM_SUPP_{category.upper()}_PRIMARY",
            supplier_name=f"Simulated {category} Primary Vendor",
            supplier_type=SupplierType.PRIMARY,
            category=category,
            lead_time_mean_days=supp_params.lead_time_mean_days,
            lead_time_std_days=supp_params.lead_time_std_days,
            cogs_ratio=proc_params.primary_cogs_ratio,
            min_order_qty=proc_params.min_order_qty,
            case_pack_size=proc_params.case_pack_size,
            is_active=True,
            is_simulation_vendor=True,
        )
    elif supplier_type == SupplierType.BACKUP:
        return SupplierProfile(
            supplier_id=f"SIM_SUPP_{category.upper()}_BACKUP",
            supplier_name=f"Simulated {category} Backup Vendor (Expedited)",
            supplier_type=SupplierType.BACKUP,
            category=category,
            lead_time_mean_days=proc_params.backup_lead_time_days,
            lead_time_std_days=proc_params.backup_lead_time_std_days,
            cogs_ratio=proc_params.backup_cogs_ratio,
            min_order_qty=proc_params.min_order_qty,
            case_pack_size=proc_params.case_pack_size,
            is_active=True,
            is_simulation_vendor=True,
        )
    else:
        raise ValueError(f"Unsupported supplier type: {supplier_type}")


def select_supplier(
    calc_result: InventoryCalculationResult,
    category: str,
    risk_assessment: Optional[InventoryRiskAssessment] = None,
    config: Optional[Lead2Config] = None,
) -> Tuple[SupplierProfile, str]:
    """
    Selects the optimal supplier profile (PRIMARY vs BACKUP) based on inventory criticality
    and stock coverage relative to primary lead time.

    BUSINESS RULE:
    Select the backup supplier when the inventory situation is critical and the primary
    supplier's lead time exceeds remaining stock coverage. Otherwise, prefer the primary supplier.

    Args:
        calc_result: Output from InventoryEngine.calculate().
        category: Product category (Bakery, Dairy, Meat, Produce).
        risk_assessment: Optional output from InventoryEngine.assess_risk().
        config: Optional Lead2Config instance.

    Returns:
        Tuple of (selected SupplierProfile, selection rationale string).
    """
    cfg = config or get_default_config()
    primary_profile = get_simulated_supplier_profile(category, SupplierType.PRIMARY, cfg)
    backup_profile = get_simulated_supplier_profile(category, SupplierType.BACKUP, cfg)

    # 1. Evaluate whether the inventory situation is critical
    if risk_assessment is not None:
        is_critical = (risk_assessment.stock_risk_level == StockRiskLevel.CRITICAL_LOW)
        risk_label = risk_assessment.stock_risk_level.value
    else:
        # Fallback using calc_result attributes directly (matches assess_stock_risk logic)
        on_hand = calc_result.inventory_position.on_hand_stock
        ip = calc_result.total_inventory_position
        ss_lt = calc_result.safety_stock_lead_time
        is_critical = (on_hand <= 0.0 or ip <= ss_lt)
        risk_label = "CRITICAL_LOW" if is_critical else "NON_CRITICAL"

    # 2. Evaluate remaining stock coverage in days
    # Stock coverage = net_available_stock / d_avg
    d_avg = calc_result.average_daily_demand
    available_stock = calc_result.net_available_stock
    primary_lt = primary_profile.lead_time_mean_days

    if d_avg > 0.0:
        stock_coverage_days = max(0.0, available_stock) / d_avg
    else:
        stock_coverage_days = 999.0 if available_stock > 0.0 else 0.0

    # 3. Decision rule:
    # Select backup supplier when situation is critical AND primary lead time exceeds remaining stock coverage.
    if is_critical and (stock_coverage_days < primary_lt):
        reason = (
            f"Backup vendor selected: critical stock risk ({risk_label}) with remaining stock coverage "
            f"({stock_coverage_days:.2f} days) below primary delivery lead time ({primary_lt:.2f} days). "
            f"Expedited dispatch ({backup_profile.lead_time_mean_days:.1f} days) mitigates stockout risk "
            f"despite higher wholesale cost ratio ({backup_profile.cogs_ratio:.2f})."
        )
        return backup_profile, reason
    else:
        if not is_critical:
            reason = (
                f"Primary vendor selected: standard replenishment under non-critical stock risk ({risk_label}); "
                f"cost-optimal procurement at {primary_profile.cogs_ratio:.2f} wholesale COGS ratio."
            )
        else:
            reason = (
                f"Primary vendor selected: remaining stock coverage ({stock_coverage_days:.2f} days) "
                f"sufficiently buffers primary supplier delivery lead time ({primary_lt:.2f} days) "
                f"without requiring expedited backup cost premium."
            )
        return primary_profile, reason


def calculate_procurement_quantity(
    raw_suggested_qty: float,
    case_pack_size: int,
    moq: float,
    total_inventory_position: float,
    target_stock: float,
    is_critical: bool,
) -> Tuple[float, float, float, bool, List[str]]:
    """
    Computes case-pack rounded quantity, applies MOQ constraints, and evaluates
    non-critical target stock excess review flags.

    PROCUREMENT QUANTITY RULES:
    1. If raw_suggested_qty <= 0, returns all zeros (no order line).
    2. Round upward to case_pack_size.
    3. If case_pack_rounded_qty < moq:
       - Compute smallest multiple of case_pack_size that meets or exceeds MOQ.
       - For critical / stockout situations: allow quantity to be raised to MOQ without excess flag.
       - For non-critical situations: if MOQ adjustment would push projected inventory above target stock,
         flag recommendation for manual review (requires_escalation=True) rather than silently accepting excess.

    Returns:
        Tuple of:
        - case_pack_rounded_qty (float)
        - final_order_qty (float)
        - moq_adjustment_qty (float)
        - requires_escalation (bool)
        - escalation_reasons (List[str])
    """
    if raw_suggested_qty <= 0.0:
        return 0.0, 0.0, 0.0, False, []

    # Step 3: Round upward to integer multiple of case-pack size
    case_pack_rounded_qty = float(math.ceil(raw_suggested_qty / case_pack_size) * case_pack_size)

    # Step 4: Evaluate MOQ constraint
    requires_escalation = False
    escalation_reasons: List[str] = []

    if case_pack_rounded_qty < moq:
        # Determine smallest multiple of case_pack_size that meets or exceeds MOQ
        moq_multiple = float(math.ceil(moq / case_pack_size) * case_pack_size)
        moq_adjustment = moq_multiple - case_pack_rounded_qty
        final_order_qty = moq_multiple

        projected_inventory = total_inventory_position + final_order_qty

        if is_critical:
            # Step 5: For critical/out-of-stock situations, allow quantity to be raised to MOQ
            pass
        else:
            # Step 6: For non-critical situations, if MOQ adjustment pushes projected inventory
            # above target stock, flag for manual review rather than silently accepting excess
            if projected_inventory > target_stock:
                excess_units = projected_inventory - target_stock
                requires_escalation = True
                escalation_reasons.append(
                    f"MOQ Adjustment Excess: Non-critical stock position raised to MOQ ({final_order_qty:.0f} units; "
                    f"raw suggested={raw_suggested_qty:.1f}), pushing projected inventory ({projected_inventory:.1f}) "
                    f"{excess_units:.1f} units above target stock ({target_stock:.1f}). Flagged for manual review."
                )
    else:
        # Case pack rounding already met or exceeded MOQ
        final_order_qty = case_pack_rounded_qty
        moq_adjustment = 0.0

    return case_pack_rounded_qty, final_order_qty, moq_adjustment, requires_escalation, escalation_reasons


@dataclass
class ProcurementRecommendation:
    """
    Full SKU-level procurement audit and decision record produced by the Procurement Engine.
    Preserves all intermediate stages: raw deficit, case pack rounding, MOQ adjustment,
    supplier selection reasoning, financial valuation, and escalation triggers.
    """
    store_id: str
    sku_id: str
    category: str
    date: str
    reorder_triggered: bool

    selected_supplier: SupplierProfile
    supplier_selection_reason: str

    raw_suggested_qty: float
    case_pack_rounded_qty: float
    moq_adjustment_qty: float
    final_order_qty: float

    unit_retail_price: float
    unit_procurement_cost: float
    total_line_cost: float

    requires_escalation: bool
    escalation_reasons: List[str] = field(default_factory=list)

    line_item: Optional[PurchaseOrderLineItem] = None
    assumed_zero_pipeline: bool = False


class ProcurementEngine:
    """
    Core Procurement Engine for Lead 2 Quick-Commerce Operations.

    Converts deterministic InventoryEngine replenishment outputs into discrete,
    packaged, and MOQ-constrained Purchase Orders grouped by dark store and supplier.
    Enforces strict human governance, review thresholds, and simulation boundaries.
    """

    def __init__(
        self,
        config: Optional[Lead2Config] = None,
        price_catalog: Optional[Dict[str, float]] = None,
    ):
        self.config = config or get_default_config()
        self.price_catalog = price_catalog or load_retail_price_catalog()

    def get_retail_price(
        self,
        sku_id: str,
        category: Optional[str] = None,
        explicit_price: Optional[float] = None,
    ) -> float:
        """
        Resolves unit retail price in INR for a given SKU.
        Hierarchy: explicit price > price catalog > SKU default > Category default.
        """
        if explicit_price is not None and explicit_price > 0.0:
            return float(explicit_price)

        if sku_id in self.price_catalog and self.price_catalog[sku_id] > 0.0:
            return float(self.price_catalog[sku_id])

        if sku_id in DEFAULT_SKU_RETAIL_PRICES:
            return float(DEFAULT_SKU_RETAIL_PRICES[sku_id])

        resolved_cat = category or SKU_TO_CATEGORY.get(sku_id, "Produce")
        return float(DEFAULT_CATEGORY_RETAIL_PRICES.get(resolved_cat, 50.0))

    def generate_recommendation(
        self,
        calc_result: InventoryCalculationResult,
        unit_retail_price: Optional[float] = None,
        category: Optional[str] = None,
        risk_assessment: Optional[InventoryRiskAssessment] = None,
        supplier: Optional[SupplierProfile] = None,
    ) -> ProcurementRecommendation:
        """
        Evaluates a single InventoryCalculationResult and builds a ProcurementRecommendation.
        If suggested_replenishment_qty > 0, generates a validated PurchaseOrderLineItem.

        Args:
            calc_result: Output from InventoryEngine.calculate().
            unit_retail_price: Optional explicit unit retail price.
            category: Optional category string (defaults to SKU_TO_CATEGORY resolution).
            risk_assessment: Optional Output from InventoryEngine.assess_risk().
            supplier: Optional forced SupplierProfile override.

        Returns:
            ProcurementRecommendation record.
        """
        sku_id = calc_result.sku_id
        store_id = calc_result.store_id
        date = calc_result.date
        cat = category or SKU_TO_CATEGORY.get(sku_id, "Produce")

        # 1. Resolve unit retail price
        retail_price = self.get_retail_price(sku_id, cat, unit_retail_price)

        # 2. Supplier Selection
        if supplier is not None:
            sel_supplier = supplier
            sel_reason = f"Explicit supplier override: {supplier.supplier_id} ({supplier.supplier_type.value})"
        else:
            sel_supplier, sel_reason = select_supplier(
                calc_result=calc_result,
                category=cat,
                risk_assessment=risk_assessment,
                config=self.config,
            )

        # 3. If no replenishment is suggested, return zero recommendation
        if calc_result.suggested_replenishment_qty <= 0.0:
            return ProcurementRecommendation(
                store_id=store_id,
                sku_id=sku_id,
                category=cat,
                date=date,
                reorder_triggered=calc_result.reorder_triggered,
                selected_supplier=sel_supplier,
                supplier_selection_reason=sel_reason,
                raw_suggested_qty=0.0,
                case_pack_rounded_qty=0.0,
                moq_adjustment_qty=0.0,
                final_order_qty=0.0,
                unit_retail_price=retail_price,
                unit_procurement_cost=round(retail_price * sel_supplier.cogs_ratio, 4),
                total_line_cost=0.0,
                requires_escalation=False,
                escalation_reasons=[],
                line_item=None,
                assumed_zero_pipeline=calc_result.assumed_zero_pipeline,
            )

        # Use supplier-specific lead-time mean and variability (0.8d / 0.15d for backup)
        if sel_supplier.supplier_type == SupplierType.BACKUP:
            d_avg = calc_result.average_daily_demand
            sigma_d = calc_result.demand_uncertainty_sigma
            z = calc_result.z_score
            review_period = calc_result.review_period_days
            backup_lt = float(sel_supplier.lead_time_mean_days)
            backup_sigma_lt = float(sel_supplier.lead_time_std_days)

            # Recalculate protection safety stock & target stock using backup supplier's parameters
            ss_protection_backup = compute_protection_safety_stock(
                d_avg=d_avg,
                sigma_d=sigma_d,
                lt_mean=backup_lt,
                review_period=review_period,
                sigma_lt=backup_sigma_lt,
                z=z,
            )
            target_stock_backup = compute_target_stock(
                d_avg=d_avg,
                lt_mean=backup_lt,
                review_period=review_period,
                safety_stock=ss_protection_backup,
            )
            raw_qty = max(0.0, target_stock_backup - calc_result.total_inventory_position)
            effective_target_stock = target_stock_backup
        else:
            raw_qty = float(calc_result.suggested_replenishment_qty)
            effective_target_stock = calc_result.target_stock

        # 4. Determine criticality status for quantity calculation
        if risk_assessment is not None:
            is_critical = (risk_assessment.stock_risk_level == StockRiskLevel.CRITICAL_LOW)
        else:
            on_hand = calc_result.inventory_position.on_hand_stock
            ip = calc_result.total_inventory_position
            ss_lt = calc_result.safety_stock_lead_time
            is_critical = (on_hand <= 0.0 or ip <= ss_lt)

        # 5. Quantity calculation (case pack rounding + MOQ constraints)
        (
            case_pack_qty,
            final_qty,
            moq_adj,
            qty_escalation,
            qty_reasons,
        ) = calculate_procurement_quantity(
            raw_suggested_qty=raw_qty,
            case_pack_size=sel_supplier.case_pack_size,
            moq=sel_supplier.min_order_qty,
            total_inventory_position=calc_result.total_inventory_position,
            target_stock=effective_target_stock,
            is_critical=is_critical,
        )

        # 6. Cost calculation
        unit_cost = round(retail_price * sel_supplier.cogs_ratio, 4)
        total_cost = round(final_qty * unit_cost, 2)

        # 7. Escalation evaluation (line-item value threshold ₹5,000)
        line_requires_escalation = qty_escalation
        line_escalation_reasons = list(qty_reasons)

        max_line_limit = self.config.procurement_thresholds.max_line_item_cost_inr
        if total_cost > max_line_limit:
            line_requires_escalation = True
            reason_str = (
                f"High Line Value: Total line cost ₹{total_cost:,.2f} exceeds "
                f"single-line review limit ₹{max_line_limit:,.2f}."
            )
            if reason_str not in line_escalation_reasons:
                line_escalation_reasons.append(reason_str)

        # Check escalation policy: Assumed-zero pipeline with LOW_STOCK_REORDER
        if self.config.procurement_thresholds.flag_assumed_zero_pipeline_reorders:
            is_low_stock = False
            if risk_assessment is not None:
                is_low_stock = (risk_assessment.stock_risk_level == StockRiskLevel.LOW_STOCK_REORDER)
            else:
                is_low_stock = calc_result.reorder_triggered and not is_critical

            if calc_result.assumed_zero_pipeline and is_low_stock:
                line_requires_escalation = True
                pipeline_reason = (
                    f"Assumed-Zero Pipeline with LOW_STOCK_REORDER: Reorder triggered for SKU '{sku_id}' "
                    f"under assumed-zero pipeline. Verification of physical incoming stock "
                    f"recommended prior to approval."
                )
                if pipeline_reason not in line_escalation_reasons:
                    line_escalation_reasons.append(pipeline_reason)

        # 8. Construct validated PurchaseOrderLineItem
        line_item = PurchaseOrderLineItem(
            store_id=store_id,
            sku_id=sku_id,
            category=cat,
            raw_suggested_qty=raw_qty,
            case_pack_rounded_qty=case_pack_qty,
            final_order_qty=final_qty,
            case_pack_size=sel_supplier.case_pack_size,
            moq=sel_supplier.min_order_qty,
            unit_retail_price=retail_price,
            unit_procurement_cost=unit_cost,
            total_line_cost=total_cost,
            requires_escalation=line_requires_escalation,
            escalation_reasons=line_escalation_reasons,
        )

        return ProcurementRecommendation(
            store_id=store_id,
            sku_id=sku_id,
            category=cat,
            date=date,
            reorder_triggered=calc_result.reorder_triggered,
            selected_supplier=sel_supplier,
            supplier_selection_reason=sel_reason,
            raw_suggested_qty=raw_qty,
            case_pack_rounded_qty=case_pack_qty,
            moq_adjustment_qty=moq_adj,
            final_order_qty=final_qty,
            unit_retail_price=retail_price,
            unit_procurement_cost=unit_cost,
            total_line_cost=total_cost,
            requires_escalation=line_requires_escalation,
            escalation_reasons=line_escalation_reasons,
            line_item=line_item,
            assumed_zero_pipeline=calc_result.assumed_zero_pipeline,
        )

    def generate_purchase_orders(
        self,
        calculation_results: List[InventoryCalculationResult],
        price_catalog: Optional[Dict[str, float]] = None,
        risk_assessments: Optional[Dict[Tuple[str, str, str], InventoryRiskAssessment]] = None,
        date: Optional[str] = None,
    ) -> Tuple[List[PurchaseOrder], List[ProcurementRecommendation], Dict[str, Any]]:
        """
        Converts a batch of InventoryCalculationResults into draft Purchase Orders
        grouped by destination store and selected supplier.

        SAFETY & GOVERNANCE:
        - Generated POs strictly remain in status PENDING_APPROVAL with approved_by=None.
        - Escalation flags are set when store orders exceed ₹25,000 or contain escalated items.
        - Propagates assumed_zero_pipeline_flag to order headers.

        Args:
            calculation_results: List of InventoryCalculationResult models.
            price_catalog: Optional custom pricing dictionary.
            risk_assessments: Optional map of (store_id, sku_id, date) -> InventoryRiskAssessment.
            date: Optional operating date override.

        Returns:
            Tuple of:
            - purchase_orders: List of validated draft PurchaseOrder models.
            - recommendations: List of all SKU-level ProcurementRecommendation objects.
            - audit_summary: Comprehensive dictionary of execution metrics and statistics.
        """
        if price_catalog:
            self.price_catalog.update(price_catalog)

        risk_map = risk_assessments or {}
        recommendations: List[ProcurementRecommendation] = []
        active_lines: List[PurchaseOrderLineItem] = []

        # Map to hold (recommendation, calc_result) paired for grouping
        rec_calc_pairs: List[Tuple[ProcurementRecommendation, InventoryCalculationResult]] = []

        for calc in calculation_results:
            key = (calc.store_id, calc.sku_id, calc.date)
            risk = risk_map.get(key, None)

            rec = self.generate_recommendation(
                calc_result=calc,
                risk_assessment=risk,
            )
            recommendations.append(rec)

            if rec.line_item is not None and rec.final_order_qty > 0.0:
                active_lines.append(rec.line_item)
                rec_calc_pairs.append((rec, calc))

        # Group active line items by (store_id, supplier_id)
        groups: Dict[Tuple[str, str], List[Tuple[ProcurementRecommendation, InventoryCalculationResult]]] = {}
        for rec, calc in rec_calc_pairs:
            grp_key = (rec.store_id, rec.selected_supplier.supplier_id)
            if grp_key not in groups:
                groups[grp_key] = []
            groups[grp_key].append((rec, calc))

        purchase_orders: List[PurchaseOrder] = []
        max_po_limit = self.config.procurement_thresholds.max_total_po_cost_inr

        for (grp_store_id, grp_supplier_id), items_pairs in sorted(groups.items()):
            first_rec = items_pairs[0][0]
            first_supplier = first_rec.selected_supplier
            op_date = date or first_rec.date

            grp_items = [pair[0].line_item for pair in items_pairs]
            total_qty = round(sum(item.final_order_qty for item in grp_items), 4)
            total_cost = round(sum(item.total_line_cost for item in grp_items), 2)

            # Determine urgency: EXPEDITE if backup supplier is utilized, else STANDARD
            if first_supplier.supplier_type == SupplierType.BACKUP:
                order_urgency = PurchaseOrderUrgency.EXPEDITE
            else:
                order_urgency = PurchaseOrderUrgency.STANDARD

            # Escalation evaluation
            exceeds_store_limit = total_cost > max_po_limit
            any_item_escalated = any(item.requires_escalation for item in grp_items)
            order_requires_escalation = exceeds_store_limit or any_item_escalated

            order_escalation_reasons: List[str] = []
            if exceeds_store_limit:
                order_escalation_reasons.append(
                    f"High Order Value: Total store order cost ₹{total_cost:,.2f} exceeds "
                    f"store-level review threshold ₹{max_po_limit:,.2f}."
                )

            # Consolidate distinct line-item reasons
            for item in grp_items:
                if item.requires_escalation:
                    for r in item.escalation_reasons:
                        annotated = f"[{item.sku_id}] {r}"
                        if annotated not in order_escalation_reasons:
                            order_escalation_reasons.append(annotated)

            # Propagate assumed zero pipeline flag if any item relied on it
            order_zero_pipeline = any(pair[1].assumed_zero_pipeline for pair in items_pairs)

            clean_date = op_date.replace("-", "")
            po_id = f"PO-{clean_date}-{grp_store_id}-{grp_supplier_id}"

            # Instantiate draft Purchase Order
            po = PurchaseOrder(
                po_id=po_id,
                created_date=op_date,
                store_id=grp_store_id,
                supplier_id=grp_supplier_id,
                supplier_type=first_supplier.supplier_type,
                items=grp_items,
                total_order_qty=total_qty,
                total_order_cost=total_cost,
                urgency=order_urgency,
                status=PurchaseOrderStatus.PENDING_APPROVAL,
                requires_escalation=order_requires_escalation,
                escalation_reasons=order_escalation_reasons,
                approved_by=None,
                approval_timestamp=None,
                assumed_zero_pipeline_flag=order_zero_pipeline,
                is_simulation_order=True,
            )
            purchase_orders.append(po)

        # Compile audit summary
        total_primary = sum(1 for po in purchase_orders if po.supplier_type == SupplierType.PRIMARY)
        total_backup = sum(1 for po in purchase_orders if po.supplier_type == SupplierType.BACKUP)
        escalated_pos = sum(1 for po in purchase_orders if po.requires_escalation)
        escalated_lines = sum(1 for rec in recommendations if rec.requires_escalation and rec.line_item is not None)

        audit_summary = {
            "total_nodes_evaluated": len(calculation_results),
            "reorder_triggered_nodes": sum(1 for r in recommendations if r.reorder_triggered),
            "active_line_items_count": len(active_lines),
            "draft_purchase_orders_count": len(purchase_orders),
            "primary_orders_count": total_primary,
            "backup_orders_count": total_backup,
            "escalated_purchase_orders_count": escalated_pos,
            "escalated_line_items_count": escalated_lines,
            "total_procurement_units": round(sum(po.total_order_qty for po in purchase_orders), 4),
            "total_procurement_cost_inr": round(sum(po.total_order_cost for po in purchase_orders), 2),
            "assumed_zero_pipeline_orders_count": sum(1 for po in purchase_orders if po.assumed_zero_pipeline_flag),
            "mandatory_human_approval_enforced": True,
        }

        return purchase_orders, recommendations, audit_summary
