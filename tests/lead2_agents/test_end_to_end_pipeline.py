"""
End-to-End Integration Tests: Lead 2 Complete Operational Pipeline
==================================================================
Comprehensive verification of the full multi-agent pipeline:
DataLoader -> Inventory Engine -> Procurement Engine -> PO Approval Workflow

TEST SCENARIOS (Phase 4.5):
1. Real historical dataset ingestion & execution across all 3,900 test partition rows.
2. Healthy inventory scenario: suppresses reorders, generates 0 unnecessary POs.
3. Low inventory scenario: triggers replenishment and generates valid procurement line.
4. Critical inventory scenario: selects expedited backup supplier when coverage < lead time.
5. Case-pack rounding and MOQ constraints: preserved end-to-end into PO line items.
6. Multi-SKU grouping: correctly clusters items by (store_id, supplier_id).
7. Financial valuation & escalation flags: preserved on lines (> ₹5k) and orders (> ₹25k).
8. Assumed-zero-pipeline warnings: propagate through to PO headers and audit summary.
9. Human approval governance: approves draft PO with non-empty operator evidence.
10. Human rejection governance: rejects draft PO with mandatory reviewer reason.
11. Audit trail immutability: verified tuple protection against tampering.
12. Separation of approval from issuance: no automatic dispatch or issuance occurs.
"""

import unittest
from pathlib import Path

from src.agents import (
    DataLoader,
    InventoryEngine,
    ProcurementEngine,
    PurchaseOrderApprovalWorkflow,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    StockRiskLevel,
    SupplierType,
    get_default_config,
    load_daily_agent_inputs,
    CANONICAL_CATEGORIES,
    CANONICAL_SKUS,
    CANONICAL_STORES,
    SKU_TO_CATEGORY,
)
from src.agents.common.schemas import (
    DailyAgentInput,
    DemandForecast,
    InventoryPosition,
    InventorySnapshot,
    InventorySnapshotBasis,
)


def _make_agent_input(
    store_id: str = "DS_BLR_KORAMANGALA",
    sku_id: str = "SKU_MILK_1L",
    category: str = "Dairy",
    date: str = "2024-11-15",
    on_hand_stock: float = 20.0,
    p10_demand: float = None,
    p50_demand: float = 30.0,
    p90_demand: float = None,
    sigma_demand: float = 6.0,
) -> DailyAgentInput:
    """Helper to construct a valid DailyAgentInput with monotonic quantiles."""
    p10 = p10_demand if p10_demand is not None else round(p50_demand * 0.7, 2)
    p90 = p90_demand if p90_demand is not None else round(p50_demand * 1.4, 2)
    return DailyAgentInput(
        store_id=store_id,
        sku_id=sku_id,
        category=category,
        date=date,
        forecast=DemandForecast(
            store_id=store_id,
            sku_id=sku_id,
            category=category,
            date=date,
            p10_demand=p10,
            p50_demand=p50_demand,
            p90_demand=p90,
            sigma_demand=sigma_demand,
        ),
        inventory=InventorySnapshot(
            store_id=store_id,
            sku_id=sku_id,
            date=date,
            on_hand_stock=on_hand_stock,
            snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
        ),
    )


class TestRealDatasetEndToEndPipeline(unittest.TestCase):
    """Verifies end-to-end integration using the actual historical Parquet datasets."""

    def setUp(self):
        self.config = get_default_config()
        self.loader = DataLoader(self.config)
        self.inv_engine = InventoryEngine(self.config)
        self.proc_engine = ProcurementEngine(self.config)
        self.workflow = PurchaseOrderApprovalWorkflow(self.config)

    def test_real_dataset_loading_and_contract_integrity(self):
        """Verifies DataLoader ingestion across all 3,900 rows of forecast_payloads_step2_corrected.parquet."""
        self.assertTrue(self.config.forecast_file.exists(), f"Missing forecast file: {self.config.forecast_file}")
        self.assertTrue(self.config.master_file.exists(), f"Missing master file: {self.config.master_file}")
        self.assertIn("step2_corrected", str(self.config.forecast_file))

        inputs, eval_recs, report = self.loader.load_daily_agent_inputs()

        # Ingestion integrity metrics
        self.assertTrue(report.validation_passed)
        self.assertEqual(len(inputs), 3900)
        self.assertEqual(len(eval_recs), 3900)
        self.assertEqual(report.unmatched_forecast_rows, 0)
        self.assertEqual(report.opening_stock_null_count, 0)

        # Identifier validation
        stores_found = {inp.store_id for inp in inputs}
        skus_found = {inp.sku_id for inp in inputs}
        self.assertEqual(stores_found, set(CANONICAL_STORES))
        self.assertEqual(skus_found, set(CANONICAL_SKUS))

        # Check all SKUs resolve to valid canonical categories
        for sku in skus_found:
            self.assertIn(sku, SKU_TO_CATEGORY)
            self.assertIn(SKU_TO_CATEGORY[sku], CANONICAL_CATEGORIES)

    def test_real_dataset_single_day_end_to_end_flow(self):
        """
        Executes the full pipeline on a multi-store day (2024-11-10):
        DataLoader -> InventoryEngine -> ProcurementEngine -> PurchaseOrderApprovalWorkflow
        """
        inputs, eval_recs, _ = self.loader.load_daily_agent_inputs()
        day_inputs = [inp for inp in inputs if inp.date == "2024-11-10"]
        self.assertEqual(len(day_inputs), 75)  # 5 stores x 15 SKUs

        # Step 2: Inventory Engine Execution
        calc_results = self.inv_engine.calculate_batch(day_inputs)
        self.assertEqual(len(calc_results), 75)

        # Step 3: Procurement Engine Execution
        pos, recs, audit = self.proc_engine.generate_purchase_orders(calc_results, date="2024-11-10")
        self.assertEqual(len(recs), 75)
        self.assertGreater(len(pos), 0)
        self.assertTrue(audit["mandatory_human_approval_enforced"])

        # Step 4: PO Governance Checks
        for po in pos:
            # 1. Non-issued state
            self.assertEqual(po.status, PurchaseOrderStatus.PENDING_APPROVAL)
            self.assertIsNone(po.approved_by)
            self.assertIsNone(po.approval_timestamp)
            self.assertFalse(self.workflow.is_issued_or_dispatched(po))
            self.assertFalse(self.workflow.can_be_dispatched(po))

            # 2. Store consistency
            for item in po.items:
                self.assertEqual(item.store_id, po.store_id)
                self.assertGreater(item.final_order_qty, 0.0)
                # Case pack rounding check
                self.assertEqual(item.final_order_qty % item.case_pack_size, 0)
                # MOQ check
                self.assertGreaterEqual(item.final_order_qty, item.moq)

            # 3. Sum integrity
            self.assertAlmostEqual(po.total_order_qty, sum(i.final_order_qty for i in po.items), places=3)
            self.assertAlmostEqual(po.total_order_cost, round(sum(i.total_line_cost for i in po.items), 2), places=2)

        # Step 5: Test Approval of First Order
        first_po = pos[0]
        approved_po = self.workflow.approve(
            po=first_po,
            approved_by="Lead_Operator_BLR_01",
            approval_timestamp="2024-11-10T09:00:00Z",
            approval_reason="Verified against physical dark store shelf capacity.",
        )
        self.assertEqual(approved_po.status, PurchaseOrderStatus.APPROVED)
        self.assertEqual(approved_po.approved_by, "Lead_Operator_BLR_01")
        self.assertEqual(len(approved_po.audit_trail), 1)
        self.assertTrue(self.workflow.can_be_dispatched(approved_po))
        self.assertFalse(self.workflow.is_issued_or_dispatched(approved_po))

        # Step 6: Test Rejection of Second Order (if present)
        if len(pos) > 1:
            second_po = pos[1]
            rejected_po = self.workflow.reject(
                po=second_po,
                rejected_by="Ops_Director_BLR",
                rejection_timestamp="2024-11-10T09:15:00Z",
                rejection_reason="Excess inventory scheduled for inter-store transfer.",
            )
            self.assertEqual(rejected_po.status, PurchaseOrderStatus.REJECTED)
            self.assertEqual(rejected_po.rejected_by, "Ops_Director_BLR")
            self.assertEqual(len(rejected_po.audit_trail), 1)
            self.assertFalse(self.workflow.can_be_dispatched(rejected_po))


class TestEndToEndPipelineScenarios(unittest.TestCase):
    """Isolated scenario tests for specific operational conditions."""

    def setUp(self):
        self.config = get_default_config()
        self.inv_engine = InventoryEngine(self.config)
        self.proc_engine = ProcurementEngine(self.config)
        self.workflow = PurchaseOrderApprovalWorkflow(self.config)

    def test_scenario_1_healthy_inventory_produces_no_po(self):
        """Scenario 1: Healthy inventory (IP > ROP) produces zero reorders and zero POs."""
        # On hand 100 units > Target Stock 95 units
        healthy_input = _make_agent_input(on_hand_stock=100.0)
        calc = self.inv_engine.calculate(healthy_input)

        self.assertFalse(calc.reorder_triggered)
        self.assertEqual(calc.suggested_replenishment_qty, 0.0)

        pos, recs, audit = self.proc_engine.generate_purchase_orders([calc])
        self.assertEqual(len(pos), 0)
        self.assertEqual(len(recs), 1)
        self.assertEqual(audit["active_line_items_count"], 0)

    def test_scenario_2_low_inventory_produces_valid_recommendation(self):
        """Scenario 2: Low inventory triggers replenishment and generates a valid line item."""
        low_input = _make_agent_input(on_hand_stock=15.0)
        calc = self.inv_engine.calculate(low_input)

        self.assertTrue(calc.reorder_triggered)
        self.assertGreater(calc.suggested_replenishment_qty, 0.0)

        rec = self.proc_engine.generate_recommendation(calc)
        self.assertIsNotNone(rec.line_item)
        self.assertGreater(rec.final_order_qty, 0.0)
        self.assertEqual(rec.final_order_qty % rec.line_item.case_pack_size, 0)
        self.assertGreaterEqual(rec.final_order_qty, rec.line_item.moq)

    def test_scenario_3_critical_inventory_selects_backup_supplier(self):
        """Scenario 3: Critical stockout with coverage < primary LT selects expedited backup supplier."""
        # Complete stockout on Dairy (primary LT = 1.5d, backup LT = 0.8d)
        stockout_input = _make_agent_input(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            on_hand_stock=0.0,
            p50_demand=40.0,
        )
        calc = self.inv_engine.calculate(stockout_input)
        risk = self.inv_engine.assess_risk(calc, stockout_input.forecast)

        self.assertEqual(risk.stock_risk_level, StockRiskLevel.CRITICAL_LOW)

        rec = self.proc_engine.generate_recommendation(calc, risk_assessment=risk)
        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.BACKUP)
        self.assertEqual(rec.selected_supplier.cogs_ratio, 0.75)
        self.assertEqual(rec.selected_supplier.lead_time_mean_days, 0.8)
        self.assertIn("Backup vendor selected", rec.supplier_selection_reason)

        # PO header must have EXPEDITE urgency
        pos, _, _ = self.proc_engine.generate_purchase_orders([calc], risk_assessments={(calc.store_id, calc.sku_id, calc.date): risk})
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0].urgency, PurchaseOrderUrgency.EXPEDITE)
        self.assertEqual(pos[0].supplier_type, SupplierType.BACKUP)

    def test_scenario_3b_critical_inventory_backup_rop_not_breached_suppresses_backup_po(self):
        """Scenario 3b: When backup supplier is selected but backup ROP is not breached, no PO is generated."""
        # Dairy: primary LT = 1.5d, backup LT = 0.8d.
        # Demand = 40.0/day. On hand = 0.0 (physical stockout -> CRITICAL_LOW, coverage = 0.0 < 1.5d).
        # But incoming on_order_stock = 50.0 -> IP = 50.0.
        # Primary ROP = 85.48 (breached!).
        # But Backup ROP = 47.36 (with sigma_D=8.0) -> IP (50.0) > 47.36 (NOT breached!).
        agent_input = _make_agent_input(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            on_hand_stock=0.0,
            p50_demand=40.0,
        )
        calc = self.inv_engine.calculate(agent_input, on_order_stock=50.0)
        risk = self.inv_engine.assess_risk(calc, agent_input.forecast)

        self.assertEqual(risk.stock_risk_level, StockRiskLevel.CRITICAL_LOW)

        rec = self.proc_engine.generate_recommendation(calc, risk_assessment=risk)
        self.assertEqual(rec.selected_supplier.supplier_type, SupplierType.BACKUP)
        self.assertFalse(rec.reorder_triggered)
        self.assertEqual(rec.raw_suggested_qty, 0.0)
        self.assertEqual(rec.final_order_qty, 0.0)
        self.assertIsNone(rec.line_item)
        self.assertIn("Backup reorder threshold not breached", rec.supplier_selection_reason)

        pos, _, _ = self.proc_engine.generate_purchase_orders(
            [calc],
            risk_assessments={(calc.store_id, calc.sku_id, calc.date): risk}
        )
        self.assertEqual(len(pos), 0)

    def test_scenario_4_case_pack_and_moq_rules_preserved(self):
        """Scenario 4: Case-pack rounding and MOQ constraints are strictly enforced across categories."""
        # Bakery: case pack = 10, MOQ = 15
        bakery_input = _make_agent_input(
            sku_id="SKU_BREAD_WHITE",
            category="Bakery",
            on_hand_stock=5.0,
            p50_demand=10.0,
        )
        calc = self.inv_engine.calculate(bakery_input)
        rec = self.proc_engine.generate_recommendation(calc)

        self.assertEqual(rec.final_order_qty % 10, 0)
        self.assertGreaterEqual(rec.final_order_qty, 15.0)

    def test_scenario_5_multiple_skus_grouped_strictly_by_store_and_supplier(self):
        """Scenario 5: Multiple SKUs across stores and categories group strictly by (store_id, supplier_id)."""
        inputs = [
            _make_agent_input(store_id="DS_BLR_KORAMANGALA", sku_id="SKU_MILK_1L", category="Dairy", on_hand_stock=5.0),
            _make_agent_input(store_id="DS_BLR_KORAMANGALA", sku_id="SKU_CURD_400G", category="Dairy", on_hand_stock=5.0),
            _make_agent_input(store_id="DS_BLR_KORAMANGALA", sku_id="SKU_CHICKEN_500G", category="Meat", on_hand_stock=5.0),
            _make_agent_input(store_id="DS_BLR_INDIRANAGAR", sku_id="SKU_MILK_1L", category="Dairy", on_hand_stock=5.0),
        ]
        calcs = self.inv_engine.calculate_batch(inputs)
        pos, _, _ = self.proc_engine.generate_purchase_orders(calcs)

        # Expected POs:
        # 1. Koramangala + Dairy Primary (2 items: Milk, Curd)
        # 2. Koramangala + Meat Primary (1 item: Chicken)
        # 3. Indiranagar + Dairy Primary (1 item: Milk)
        self.assertEqual(len(pos), 3)

        koramangala_dairy = next(p for p in pos if p.store_id == "DS_BLR_KORAMANGALA" and "DAIRY" in p.supplier_id)
        self.assertEqual(len(koramangala_dairy.items), 2)

        koramangala_meat = next(p for p in pos if p.store_id == "DS_BLR_KORAMANGALA" and "MEAT" in p.supplier_id)
        self.assertEqual(len(koramangala_meat.items), 1)

        indiranagar_dairy = next(p for p in pos if p.store_id == "DS_BLR_INDIRANAGAR" and "DAIRY" in p.supplier_id)
        self.assertEqual(len(indiranagar_dairy.items), 1)

    def test_scenario_6_financial_valuation_and_escalation_preserved(self):
        """Scenario 6: Financial calculations and escalation thresholds are preserved into the PO."""
        # Meat SKU with high unit price (₹160 retail) and large deficit
        input_high = _make_agent_input(
            sku_id="SKU_CHICKEN_500G",
            category="Meat",
            on_hand_stock=5.0,
            p50_demand=60.0,
        )
        calc = self.inv_engine.calculate(input_high)
        pos, recs, audit = self.proc_engine.generate_purchase_orders([calc])

        self.assertEqual(len(pos), 1)
        po = pos[0]
        # Line cost exceeds ₹5,000 threshold
        self.assertGreater(po.total_order_cost, 5000.0)
        self.assertTrue(po.requires_escalation)
        self.assertTrue(any("High Line Value" in r for r in po.escalation_reasons))

    def test_scenario_7_assumed_zero_pipeline_propagated(self):
        """Scenario 7: Assumed-zero pipeline assumption propagates through to the PO header."""
        input_data = _make_agent_input(on_hand_stock=5.0)
        calc = self.inv_engine.calculate(input_data)  # defaults to assumed_zero_pipeline=True
        self.assertTrue(calc.assumed_zero_pipeline)

        pos, _, audit = self.proc_engine.generate_purchase_orders([calc])
        self.assertEqual(len(pos), 1)
        self.assertTrue(pos[0].assumed_zero_pipeline_flag)
        self.assertEqual(audit["assumed_zero_pipeline_orders_count"], 1)

    def test_scenario_8_po_approval_and_audit_trail_integrity(self):
        """Scenario 8 & 9: Full human approval and rejection flow preserves audit immutability."""
        input_data = _make_agent_input(on_hand_stock=5.0)
        calc = self.inv_engine.calculate(input_data)
        pos, _, _ = self.proc_engine.generate_purchase_orders([calc])
        po = pos[0]

        # 1. Approval
        approved_po = self.workflow.approve(
            po=po,
            approved_by="Auditor_Lead_01",
            approval_timestamp="2024-11-15T10:00:00Z",
            approval_reason="Stockout replenishment approved.",
        )
        self.assertEqual(approved_po.status, PurchaseOrderStatus.APPROVED)
        self.assertIsInstance(approved_po.audit_trail, tuple)
        self.assertEqual(len(approved_po.audit_trail), 1)
        self.assertEqual(approved_po.audit_trail[0].reviewer_id, "Auditor_Lead_01")

        # 2. Verify immutability of audit_trail
        with self.assertRaises(AttributeError):
            approved_po.audit_trail.append("fake_event")

        # 3. Rejection of original pending PO
        rejected_po = self.workflow.reject(
            po=po,
            rejected_by="Auditor_Lead_01",
            rejection_timestamp="2024-11-15T10:05:00Z",
            rejection_reason="Duplicate order cancelled.",
        )
        self.assertEqual(rejected_po.status, PurchaseOrderStatus.REJECTED)
        self.assertEqual(rejected_po.rejected_by, "Auditor_Lead_01")
        self.assertEqual(len(rejected_po.audit_trail), 1)

    def test_scenario_10_no_path_automatically_issues_po(self):
        """Scenario 10: Strict decoupling: no path automatically marks a PO as ISSUED or dispatched."""
        input_data = _make_agent_input(on_hand_stock=5.0)
        calc = self.inv_engine.calculate(input_data)
        pos, _, _ = self.proc_engine.generate_purchase_orders([calc])
        po = pos[0]

        # Initial draft is not issued
        self.assertFalse(self.workflow.is_issued_or_dispatched(po))
        self.assertFalse(self.workflow.can_be_dispatched(po))

        # Approved order is still NOT issued
        approved_po = self.workflow.approve(
            po=po,
            approved_by="Ops_Lead",
            approval_timestamp="2024-11-15T10:00:00Z",
        )
        self.assertEqual(approved_po.status, PurchaseOrderStatus.APPROVED)
        self.assertNotEqual(approved_po.status, PurchaseOrderStatus.ISSUED)
        self.assertFalse(self.workflow.is_issued_or_dispatched(approved_po))
        self.assertTrue(self.workflow.can_be_dispatched(approved_po))


class TestPackageFilenamesAndInterfaces(unittest.TestCase):
    """
    Verifies that all Lead 2 packages use canonical __init__.py files (not init.py)
    and export their interfaces properly.
    """

    def test_package_init_filenames_and_no_bare_init_files(self):
        """Verify that packages have __init__.py and NO files are named init.py."""
        from pathlib import Path
        import src.agents as agents_pkg
        import src.agents.common as common_pkg
        import src.agents.inventory as inventory_pkg
        import src.agents.procurement as procurement_pkg

        # Check that each package module has __file__ ending with __init__.py
        for pkg in [agents_pkg, common_pkg, inventory_pkg, procurement_pkg]:
            pkg_file = Path(pkg.__file__)
            self.assertEqual(
                pkg_file.name,
                "__init__.py",
                f"Package file for {pkg.__name__} must be '__init__.py', got '{pkg_file.name}'",
            )

        # Check agents root and subdirectories for any illegal 'init.py'
        agents_dir = Path(agents_pkg.__file__).parent
        bare_init_files = list(agents_dir.rglob("init.py"))
        self.assertEqual(bare_init_files, [], f"Found illegal bare 'init.py' files: {bare_init_files}")

    def test_package_interface_imports(self):
        """Verify clean imports from all package interfaces."""
        # Common
        from src.agents.common import (
            DataLoader,
            DailyAgentInput,
            InventoryPosition,
            Lead2Config,
        )
        self.assertIsNotNone(DataLoader)
        self.assertIsNotNone(DailyAgentInput)
        self.assertIsNotNone(InventoryPosition)
        self.assertIsNotNone(Lead2Config)

        # Inventory
        from src.agents.inventory import InventoryEngine, assess_inventory_risk
        self.assertIsNotNone(InventoryEngine)
        self.assertIsNotNone(assess_inventory_risk)

        # Procurement
        from src.agents.procurement import (
            ProcurementEngine,
            PurchaseOrder,
            PurchaseOrderStatus,
            SupplierType,
            approve_purchase_order,
        )
        self.assertIsNotNone(ProcurementEngine)
        self.assertIsNotNone(PurchaseOrder)
        self.assertIsNotNone(PurchaseOrderStatus)
        self.assertIsNotNone(SupplierType)
        self.assertIsNotNone(approve_purchase_order)

        # Top-level agents facade
        import src.agents as ag
        self.assertTrue(hasattr(ag, "DataLoader"))
        self.assertTrue(hasattr(ag, "InventoryEngine"))
        self.assertTrue(hasattr(ag, "ProcurementEngine"))
        self.assertTrue(hasattr(ag, "PurchaseOrder"))
        self.assertTrue(hasattr(ag, "approve_purchase_order"))


if __name__ == "__main__":
    unittest.main()
