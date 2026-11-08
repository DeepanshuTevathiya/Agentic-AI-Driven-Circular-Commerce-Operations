"""
Unit Tests: Lead 2 Procurement Schemas & Data Contracts
========================================================
Tests:
- SupplierProfile validation, canonical categories, immutability, extra-field prohibition
- PurchaseOrderLineItem case-pack integer multiples, MOQ checks, cost checks
- PurchaseOrderLineItem escalation requirements (> ₹5,000)
- PurchaseOrder store consistency, quantity/cost aggregation checks
- PurchaseOrder order-level escalation requirements (> ₹25,000 or item-level escalation)
- MANDATORY HUMAN APPROVAL enforcement on APPROVED and ISSUED statuses
- Configuration defaults and CategoryProcurementParams integration
"""

import unittest
from pydantic import ValidationError

from src.agents.procurement.schemas import (
    ApprovalAction,
    SupplierProfile,
    SupplierType,
    PurchaseOrderLineItem,
    PurchaseOrder,
    PurchaseOrderAuditRecord,
    PurchaseOrderUrgency,
    PurchaseOrderStatus,
)
from src.agents.common.config import (
    get_default_config,
    CATEGORY_PROCUREMENT_PARAMS,
    ProcurementApprovalThresholds,
)


class TestSupplierProfileSchema(unittest.TestCase):
    def test_valid_supplier_profile(self):
        sp = SupplierProfile(
            supplier_id="SUPP_DAIRY_PRIMARY_01",
            supplier_name="Bengaluru Dairy Union (Simulation)",
            supplier_type=SupplierType.PRIMARY,
            category="Dairy",
            lead_time_mean_days=1.5,
            lead_time_std_days=0.30,
            cogs_ratio=0.60,
            min_order_qty=10.0,
            case_pack_size=12,
        )
        self.assertEqual(sp.supplier_id, "SUPP_DAIRY_PRIMARY_01")
        self.assertEqual(sp.supplier_type, SupplierType.PRIMARY)
        self.assertEqual(sp.category, "Dairy")
        self.assertEqual(sp.case_pack_size, 12)
        self.assertTrue(sp.is_simulation_vendor)

    def test_invalid_category_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            SupplierProfile(
                supplier_id="SUPP_INVALID",
                supplier_name="Invalid Vendor",
                supplier_type=SupplierType.PRIMARY,
                category="Electronics",
                lead_time_mean_days=1.0,
                lead_time_std_days=0.2,
                cogs_ratio=0.60,
                min_order_qty=10.0,
                case_pack_size=10,
            )
        self.assertIn("Category 'Electronics' must be one of", str(ctx.exception))

    def test_invalid_lead_time_rejected(self):
        with self.assertRaises(ValidationError):
            SupplierProfile(
                supplier_id="SUPP_ZERO_LT",
                supplier_name="Zero LT Vendor",
                supplier_type=SupplierType.PRIMARY,
                category="Dairy",
                lead_time_mean_days=0.0,  # Must be > 0.0
                lead_time_std_days=0.1,
                cogs_ratio=0.60,
                min_order_qty=10.0,
                case_pack_size=12,
            )

    def test_invalid_cogs_ratio_rejected(self):
        with self.assertRaises(ValidationError):
            SupplierProfile(
                supplier_id="SUPP_BAD_COGS",
                supplier_name="Overcharging Vendor",
                supplier_type=SupplierType.PRIMARY,
                category="Dairy",
                lead_time_mean_days=1.0,
                lead_time_std_days=0.1,
                cogs_ratio=1.05,  # Must be <= 1.0
                min_order_qty=10.0,
                case_pack_size=12,
            )

    def test_immutability_and_extra_fields(self):
        sp = SupplierProfile(
            supplier_id="SUPP_MEAT_PRIMARY_01",
            supplier_name="Bengaluru Meats (Simulation)",
            supplier_type=SupplierType.PRIMARY,
            category="Meat",
            lead_time_mean_days=1.0,
            lead_time_std_days=0.25,
            cogs_ratio=0.60,
            min_order_qty=5.0,
            case_pack_size=6,
        )
        with self.assertRaises(ValidationError):
            sp.min_order_qty = 10.0  # Frozen model cannot be mutated

        with self.assertRaises(ValidationError):
            SupplierProfile(
                supplier_id="SUPP_EXTRA",
                supplier_name="Vendor with extra fields",
                supplier_type=SupplierType.PRIMARY,
                category="Meat",
                lead_time_mean_days=1.0,
                lead_time_std_days=0.25,
                cogs_ratio=0.60,
                min_order_qty=5.0,
                case_pack_size=6,
                arbitrary_field="invalid",
            )


class TestPurchaseOrderLineItemSchema(unittest.TestCase):
    def test_valid_line_item(self):
        # Dairy: case_pack=12, moq=10. Order 24 units at ₹30 wholesale (₹50 retail) = ₹720 total cost
        item = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            raw_suggested_qty=17.5,
            case_pack_rounded_qty=24.0,
            final_order_qty=24.0,
            case_pack_size=12,
            moq=10.0,
            unit_retail_price=50.0,
            unit_procurement_cost=30.0,
            total_line_cost=720.0,
            requires_escalation=False,
        )
        self.assertEqual(item.final_order_qty, 24.0)
        self.assertEqual(item.total_line_cost, 720.0)
        self.assertFalse(item.requires_escalation)

    def test_zero_order_allowed(self):
        item = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            raw_suggested_qty=0.0,
            case_pack_rounded_qty=0.0,
            final_order_qty=0.0,
            case_pack_size=12,
            moq=10.0,
            unit_retail_price=50.0,
            unit_procurement_cost=30.0,
            total_line_cost=0.0,
            requires_escalation=False,
        )
        self.assertEqual(item.final_order_qty, 0.0)
        self.assertEqual(item.total_line_cost, 0.0)

    def test_non_multiple_case_pack_rejected(self):
        # Case pack is 12, trying to order 15 units
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderLineItem(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                raw_suggested_qty=15.0,
                case_pack_rounded_qty=15.0,
                final_order_qty=15.0,
                case_pack_size=12,
                moq=10.0,
                unit_retail_price=50.0,
                unit_procurement_cost=30.0,
                total_line_cost=450.0,
            )
        self.assertIn("not an exact integer multiple of case_pack_size", str(ctx.exception))

    def test_below_moq_rejected(self):
        # MOQ is 15 (Bakery), but case pack is 5 and ordering 10 units
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderLineItem(
                store_id="DS_BLR_INDIRANAGAR",
                sku_id="SKU_BREAD_WW",
                category="Bakery",
                raw_suggested_qty=10.0,
                case_pack_rounded_qty=10.0,
                final_order_qty=10.0,
                case_pack_size=5,
                moq=15.0,
                unit_retail_price=40.0,
                unit_procurement_cost=24.0,
                total_line_cost=240.0,
            )
        self.assertIn("is below supplier MOQ", str(ctx.exception))

    def test_cost_greater_than_retail_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderLineItem(
                store_id="DS_BLR_WHITEFIELD",
                sku_id="SKU_EGGS_12PK",
                category="Dairy",
                raw_suggested_qty=12.0,
                case_pack_rounded_qty=12.0,
                final_order_qty=12.0,
                case_pack_size=12,
                moq=10.0,
                unit_retail_price=40.0,
                unit_procurement_cost=45.0,  # Wholesale > retail!
                total_line_cost=540.0,
            )
        self.assertIn("Wholesale procurement cost (45.0) cannot exceed retail price (40.0)", str(ctx.exception))

    def test_total_cost_mismatch_rejected(self):
        # 12 units * ₹30 = 360, but reporting 400
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderLineItem(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                raw_suggested_qty=12.0,
                case_pack_rounded_qty=12.0,
                final_order_qty=12.0,
                case_pack_size=12,
                moq=10.0,
                unit_retail_price=50.0,
                unit_procurement_cost=30.0,
                total_line_cost=400.0,  # Wrong cost
            )
        self.assertIn("total_line_cost mismatch", str(ctx.exception))

    def test_line_cost_above_threshold_requires_escalation(self):
        # 180 units * ₹30 = ₹5,400 (> ₹5,000)
        # Missing requires_escalation=True should raise ValidationError
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderLineItem(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                raw_suggested_qty=180.0,
                case_pack_rounded_qty=180.0,
                final_order_qty=180.0,
                case_pack_size=12,
                moq=10.0,
                unit_retail_price=50.0,
                unit_procurement_cost=30.0,
                total_line_cost=5400.0,
                requires_escalation=False,  # Should fail
            )
        self.assertIn("exceeds threshold ₹5,000.00; requires_escalation must be set to True", str(ctx.exception))

        # With requires_escalation=True and escalation reasons, should pass
        item_ok = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            raw_suggested_qty=180.0,
            case_pack_rounded_qty=180.0,
            final_order_qty=180.0,
            case_pack_size=12,
            moq=10.0,
            unit_retail_price=50.0,
            unit_procurement_cost=30.0,
            total_line_cost=5400.0,
            requires_escalation=True,
            escalation_reasons=["Line item value ₹5,400 exceeds single-line review limit ₹5,000"],
        )
        self.assertTrue(item_ok.requires_escalation)

    def test_line_cost_exact_5000_boundary_does_not_escalate(self):
        """Line cost of exactly ₹5,000.00 must NOT require escalation (strictly > ₹5,000)."""
        # 200 units * ₹25.0 = ₹5,000.00 exactly
        item_5000 = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_POTATO_1KG",
            category="Produce",
            raw_suggested_qty=200.0,
            case_pack_rounded_qty=200.0,
            final_order_qty=200.0,
            case_pack_size=10,
            moq=10.0,
            unit_retail_price=50.0,
            unit_procurement_cost=25.0,
            total_line_cost=5000.0,
            requires_escalation=False,
        )
        self.assertFalse(item_5000.requires_escalation)
        self.assertEqual(item_5000.total_line_cost, 5000.0)

        # However, line cost strictly greater than ₹5,000 (e.g. ₹5,002.00) MUST raise ValidationError if not escalated
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrderLineItem(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_POTATO_1KG",
                category="Produce",
                raw_suggested_qty=200.0,
                case_pack_rounded_qty=200.0,
                final_order_qty=200.0,
                case_pack_size=10,
                moq=10.0,
                unit_retail_price=50.0,
                unit_procurement_cost=25.01,
                total_line_cost=5002.0,
                requires_escalation=False,
            )
        self.assertIn("exceeds threshold ₹5,000.00; requires_escalation must be set to True", str(ctx.exception))


class TestPurchaseOrderSchema(unittest.TestCase):
    def setUp(self):
        self.item1 = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            raw_suggested_qty=24.0,
            case_pack_rounded_qty=24.0,
            final_order_qty=24.0,
            case_pack_size=12,
            moq=10.0,
            unit_retail_price=50.0,
            unit_procurement_cost=30.0,
            total_line_cost=720.0,
            requires_escalation=False,
        )
        self.item2 = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_CURD_400G",
            category="Dairy",
            raw_suggested_qty=12.0,
            case_pack_rounded_qty=12.0,
            final_order_qty=12.0,
            case_pack_size=12,
            moq=10.0,
            unit_retail_price=35.0,
            unit_procurement_cost=21.0,
            total_line_cost=252.0,
            requires_escalation=False,
        )

    def test_valid_purchase_order(self):
        po = PurchaseOrder(
            po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SUPP_DAIRY_PRIMARY_01",
            supplier_type=SupplierType.PRIMARY,
            items=[self.item1, self.item2],
            total_order_qty=36.0,
            total_order_cost=972.0,
            urgency=PurchaseOrderUrgency.STANDARD,
            status=PurchaseOrderStatus.PENDING_APPROVAL,
            requires_escalation=False,
        )
        self.assertEqual(po.po_id, "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01")
        self.assertEqual(po.total_order_qty, 36.0)
        self.assertEqual(po.total_order_cost, 972.0)
        self.assertEqual(po.status, PurchaseOrderStatus.PENDING_APPROVAL)
        self.assertIsNone(po.approved_by)

    def test_store_id_mismatch_rejected(self):
        mismatched_item = PurchaseOrderLineItem(
            store_id="DS_BLR_INDIRANAGAR",  # Different store!
            sku_id="SKU_MILK_1L",
            category="Dairy",
            raw_suggested_qty=24.0,
            case_pack_rounded_qty=24.0,
            final_order_qty=24.0,
            case_pack_size=12,
            moq=10.0,
            unit_retail_price=50.0,
            unit_procurement_cost=30.0,
            total_line_cost=720.0,
            requires_escalation=False,
        )
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SUPP_DAIRY_PRIMARY_01",
                supplier_type=SupplierType.PRIMARY,
                items=[mismatched_item],
                total_order_qty=24.0,
                total_order_cost=720.0,
            )
        self.assertIn("does not match PurchaseOrder store_id", str(ctx.exception))

    def test_total_quantity_mismatch_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SUPP_DAIRY_PRIMARY_01",
                supplier_type=SupplierType.PRIMARY,
                items=[self.item1, self.item2],
                total_order_qty=40.0,  # Expected 36.0
                total_order_cost=972.0,
            )
        self.assertIn("total_order_qty mismatch", str(ctx.exception))

    def test_total_cost_mismatch_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SUPP_DAIRY_PRIMARY_01",
                supplier_type=SupplierType.PRIMARY,
                items=[self.item1, self.item2],
                total_order_qty=36.0,
                total_order_cost=1000.0,  # Expected 972.0
            )
        self.assertIn("total_order_cost mismatch", str(ctx.exception))

    def test_order_cost_above_threshold_requires_escalation(self):
        # Create an order exceeding ₹25,000
        # Line item: 960 units * ₹30 = ₹28,800
        big_item = PurchaseOrderLineItem(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            raw_suggested_qty=960.0,
            case_pack_rounded_qty=960.0,
            final_order_qty=960.0,
            case_pack_size=12,
            moq=10.0,
            unit_retail_price=50.0,
            unit_procurement_cost=30.0,
            total_line_cost=28800.0,
            requires_escalation=True,
            escalation_reasons=["Line item value exceeds ₹5,000"],
        )
        # If PO has requires_escalation=False, it should fail
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-BIG-ORDER",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SUPP_DAIRY_PRIMARY_01",
                supplier_type=SupplierType.PRIMARY,
                items=[big_item],
                total_order_qty=960.0,
                total_order_cost=28800.0,
                requires_escalation=False,  # Must be True
            )
        self.assertIn("requires_escalation must be True", str(ctx.exception))

        # With requires_escalation=True, passes
        po_ok = PurchaseOrder(
            po_id="PO-BIG-ORDER",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SUPP_DAIRY_PRIMARY_01",
            supplier_type=SupplierType.PRIMARY,
            items=[big_item],
            total_order_qty=960.0,
            total_order_cost=28800.0,
            requires_escalation=True,
            escalation_reasons=["Order total ₹28,800.00 exceeds store PO threshold ₹25,000.00"],
        )
        self.assertTrue(po_ok.requires_escalation)

    def test_order_cost_exact_25000_boundary_does_not_escalate(self):
        """PurchaseOrder total cost of exactly ₹25,000.00 must NOT trigger escalation (strictly > ₹25,000)."""
        produce_skus = ["SKU_BANANA_DZ", "SKU_ONION_1KG", "SKU_POTATO_1KG", "SKU_SPINACH_500G", "SKU_TOMATO_1KG"]
        items_5000 = [
            PurchaseOrderLineItem(
                store_id="DS_BLR_KORAMANGALA",
                sku_id=sku,
                category="Produce",
                raw_suggested_qty=200.0,
                case_pack_rounded_qty=200.0,
                final_order_qty=200.0,
                case_pack_size=10,
                moq=10.0,
                unit_retail_price=50.0,
                unit_procurement_cost=25.0,
                total_line_cost=5000.0,
                requires_escalation=False,
            )
            for sku in produce_skus
        ]

        po_25k = PurchaseOrder(
            po_id="PO-EXACT-25K",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_PRODUCE_PRIMARY",
            supplier_type=SupplierType.PRIMARY,
            items=items_5000,
            total_order_qty=1000.0,
            total_order_cost=25000.0,
            requires_escalation=False,
        )
        self.assertFalse(po_25k.requires_escalation)
        self.assertEqual(po_25k.total_order_cost, 25000.0)

        # However, total order cost strictly greater than ₹25,000 (e.g. ₹25,002.00) MUST fail if requires_escalation=False
        items_25002 = items_5000[:4] + [
            PurchaseOrderLineItem(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_TOMATO_1KG",
                category="Produce",
                raw_suggested_qty=200.0,
                case_pack_rounded_qty=200.0,
                final_order_qty=200.0,
                case_pack_size=10,
                moq=10.0,
                unit_retail_price=50.0,
                unit_procurement_cost=25.01,
                total_line_cost=5002.0,
                requires_escalation=True,
                escalation_reasons=["Line item value exceeds ₹5,000"],
            )
        ]
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-EXACT-25K-FAIL",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_PRODUCE_PRIMARY",
                supplier_type=SupplierType.PRIMARY,
                items=items_25002,
                total_order_qty=1000.0,
                total_order_cost=25002.0,
                requires_escalation=False,
            )
        self.assertIn("triggers review threshold; requires_escalation must be True", str(ctx.exception))

    def test_mandatory_human_approval_enforcement(self):
        # Status APPROVED without approved_by must fail
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-APPROVAL-TEST",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SUPP_DAIRY_PRIMARY_01",
                supplier_type=SupplierType.PRIMARY,
                items=[self.item1],
                total_order_qty=24.0,
                total_order_cost=720.0,
                status=PurchaseOrderStatus.APPROVED,
                approved_by=None,
            )
        self.assertIn("Mandatory Governance Violation", str(ctx.exception))
        self.assertIn("Automated issuance is strictly prohibited", str(ctx.exception))

        # Status ISSUED without approved_by must fail
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-APPROVAL-TEST",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SUPP_DAIRY_PRIMARY_01",
                supplier_type=SupplierType.PRIMARY,
                items=[self.item1],
                total_order_qty=24.0,
                total_order_cost=720.0,
                status=PurchaseOrderStatus.ISSUED,
                approved_by="   ",  # Whitespace-only string must also fail
            )
        self.assertIn("Mandatory Governance Violation", str(ctx.exception))

        # Status APPROVED without audit record must fail
        with self.assertRaises(ValidationError) as ctx:
            PurchaseOrder(
                po_id="PO-APPROVAL-TEST",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SUPP_DAIRY_PRIMARY_01",
                supplier_type=SupplierType.PRIMARY,
                items=[self.item1],
                total_order_qty=24.0,
                total_order_cost=720.0,
                status=PurchaseOrderStatus.APPROVED,
                approved_by="Operator_Lead_Ops_BLR_01",
                approval_timestamp="2024-11-15T10:30:00Z",
            )
        self.assertIn("Mandatory Governance Violation", str(ctx.exception))
        self.assertIn("must contain a matching APPROVE audit record", str(ctx.exception))

        # Matching APPROVE audit record
        audit_record_approved = PurchaseOrderAuditRecord(
            po_id="PO-APPROVAL-TEST",
            previous_status=PurchaseOrderStatus.PENDING_APPROVAL,
            new_status=PurchaseOrderStatus.APPROVED,
            reviewer_id="Operator_Lead_Ops_BLR_01",
            timestamp="2024-11-15T10:30:00Z",
            action=ApprovalAction.APPROVE,
            reason="Approved by authorized human operator.",
            requires_escalation=False,
            escalation_reasons=(),
        )

        # Status APPROVED with explicit human signoff and matching audit record succeeds
        po_approved = PurchaseOrder(
            po_id="PO-APPROVAL-TEST",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SUPP_DAIRY_PRIMARY_01",
            supplier_type=SupplierType.PRIMARY,
            items=[self.item1],
            total_order_qty=24.0,
            total_order_cost=720.0,
            status=PurchaseOrderStatus.APPROVED,
            approved_by="Operator_Lead_Ops_BLR_01",
            approval_timestamp="2024-11-15T10:30:00Z",
            audit_trail=(audit_record_approved,),
        )
        self.assertEqual(po_approved.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(po_approved.approved_by, "Operator_Lead_Ops_BLR_01")
        self.assertEqual(len(po_approved.audit_trail), 1)

        # Status ISSUED with explicit human signoff and matching audit record succeeds
        po_issued = PurchaseOrder(
            po_id="PO-APPROVAL-TEST",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SUPP_DAIRY_PRIMARY_01",
            supplier_type=SupplierType.PRIMARY,
            items=[self.item1],
            total_order_qty=24.0,
            total_order_cost=720.0,
            status=PurchaseOrderStatus.ISSUED,
            approved_by="Operator_Lead_Ops_BLR_01",
            approval_timestamp="2024-11-15T10:30:00Z",
            audit_trail=(audit_record_approved,),
        )
        self.assertEqual(po_issued.status, PurchaseOrderStatus.ISSUED)
        self.assertEqual(len(po_issued.audit_trail), 1)


class TestProcurementConfigIntegration(unittest.TestCase):
    def test_category_procurement_params(self):
        # Validate specific approved simulation parameters
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Bakery"].min_order_qty, 15.0)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Bakery"].case_pack_size, 10)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Bakery"].backup_lead_time_days, 0.8)

        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Dairy"].min_order_qty, 10.0)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Dairy"].case_pack_size, 12)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Dairy"].backup_lead_time_days, 0.8)

        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Meat"].min_order_qty, 5.0)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Meat"].case_pack_size, 6)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Meat"].backup_lead_time_days, 0.8)

        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Produce"].min_order_qty, 10.0)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Produce"].case_pack_size, 10)
        self.assertEqual(CATEGORY_PROCUREMENT_PARAMS["Produce"].backup_lead_time_days, 0.8)

    def test_procurement_thresholds(self):
        thresholds = ProcurementApprovalThresholds()
        self.assertEqual(thresholds.max_line_item_cost_inr, 5000.0)
        self.assertEqual(thresholds.max_total_po_cost_inr, 25000.0)
        self.assertTrue(thresholds.require_human_approval_for_dispatch)

    def test_lead2_config_procurement_integration(self):
        cfg = get_default_config()
        self.assertIn("Bakery", cfg.category_procurement_params)
        self.assertIn("Dairy", cfg.category_procurement_params)
        self.assertIn("Meat", cfg.category_procurement_params)
        self.assertIn("Produce", cfg.category_procurement_params)
        self.assertEqual(cfg.procurement_thresholds.max_line_item_cost_inr, 5000.0)
        self.assertEqual(cfg.procurement_thresholds.max_total_po_cost_inr, 25000.0)


if __name__ == "__main__":
    unittest.main()
