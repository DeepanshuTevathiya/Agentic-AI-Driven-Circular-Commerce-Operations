"""
Unit Tests: Lead 2 Pydantic Schemas & Data Contracts
====================================================
Tests:
- Valid and invalid DemandForecast contracts
- Quantile monotonicity violations (P10 <= P50 <= P90)
- Uncertainty and identifier constraints
- Valid and invalid InventorySnapshot contracts
- Enforcement of BEGINNING_OF_DAY snapshot basis
- InventoryPosition net/total stock calculations and reservation rejection
- DailyAgentInput consistency and isolation of unconfirmed alerts
- DailyEvaluationRecord separation from operational inputs
"""

import unittest
from pydantic import ValidationError

from src.agents.schemas import (
    DemandForecast,
    InventorySnapshot,
    InventoryPosition,
    DailyAgentInput,
    DailyEvaluationRecord,
    InventorySnapshotBasis,
)


class TestDemandForecastSchema(unittest.TestCase):
    def test_valid_demand_forecast(self):
        fc = DemandForecast(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            category="Dairy",
            date="2024-11-15",
            p10_demand=30.0,
            p50_demand=45.0,
            p90_demand=65.0,
            sigma_demand=13.65,
        )
        self.assertEqual(fc.store_id, "DS_BLR_KORAMANGALA")
        self.assertEqual(fc.date, "2024-11-15")
        self.assertEqual(fc.p50_demand, 45.0)

    def test_monotonicity_violation_p10_greater_than_p50(self):
        with self.assertRaises(ValidationError) as ctx:
            DemandForecast(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                date="2024-11-15",
                p10_demand=50.0,
                p50_demand=40.0,
                p90_demand=60.0,
                sigma_demand=5.0,
            )
        self.assertIn("Quantile monotonicity violation", str(ctx.exception))

    def test_monotonicity_violation_p50_greater_than_p90(self):
        with self.assertRaises(ValidationError) as ctx:
            DemandForecast(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                date="2024-11-15",
                p10_demand=20.0,
                p50_demand=50.0,
                p90_demand=40.0,
                sigma_demand=5.0,
            )
        self.assertIn("Quantile monotonicity violation", str(ctx.exception))

    def test_negative_demand_rejected(self):
        with self.assertRaises(ValidationError):
            DemandForecast(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                date="2024-11-15",
                p10_demand=-5.0,
                p50_demand=10.0,
                p90_demand=20.0,
                sigma_demand=5.0,
            )

    def test_non_positive_sigma_rejected(self):
        with self.assertRaises(ValidationError):
            DemandForecast(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                date="2024-11-15",
                p10_demand=10.0,
                p50_demand=15.0,
                p90_demand=20.0,
                sigma_demand=0.0,
            )

    def test_empty_identifiers_rejected(self):
        with self.assertRaises(ValidationError):
            DemandForecast(
                store_id="   ",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                date="2024-11-15",
                p10_demand=10.0,
                p50_demand=15.0,
                p90_demand=20.0,
                sigma_demand=2.0,
            )

    def test_invalid_date_rejected(self):
        with self.assertRaises(ValidationError):
            DemandForecast(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                category="Dairy",
                date="invalid-date-string",
                p10_demand=10.0,
                p50_demand=15.0,
                p90_demand=20.0,
                sigma_demand=2.0,
            )


class TestInventorySnapshotSchema(unittest.TestCase):
    def test_valid_inventory_snapshot(self):
        inv = InventorySnapshot(
            store_id="DS_BLR_HSR_LAYOUT",
            sku_id="SKU_TOMATO_1KG",
            date="2024-11-10",
            on_hand_stock=120.0,
            remaining_shelf_life_days=4,
            total_shelf_life_days=5,
            snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
        )
        self.assertEqual(inv.on_hand_stock, 120.0)
        self.assertEqual(inv.snapshot_basis, InventorySnapshotBasis.BEGINNING_OF_DAY)

    def test_negative_on_hand_stock_rejected(self):
        with self.assertRaises(ValidationError):
            InventorySnapshot(
                store_id="DS_BLR_HSR_LAYOUT",
                sku_id="SKU_TOMATO_1KG",
                date="2024-11-10",
                on_hand_stock=-10.0,
            )

    def test_end_of_day_basis_rejected(self):
        """Operational inventory must reject END_OF_DAY closing stock."""
        with self.assertRaises(ValidationError) as ctx:
            InventorySnapshot(
                store_id="DS_BLR_HSR_LAYOUT",
                sku_id="SKU_TOMATO_1KG",
                date="2024-11-10",
                on_hand_stock=50.0,
                snapshot_basis=InventorySnapshotBasis.END_OF_DAY,
            )
        self.assertIn("must be BEGINNING_OF_DAY", str(ctx.exception))


class TestInventoryPositionSchema(unittest.TestCase):
    def test_inventory_position_calculations(self):
        pos = InventoryPosition(
            on_hand_stock=100.0,
            reserved_stock=15.0,
            on_order_stock=40.0,
        )
        self.assertEqual(pos.net_available_stock, 85.0)
        self.assertEqual(pos.total_inventory_position, 125.0)

    def test_reserved_exceeding_on_hand_rejected(self):
        """reserved_stock > on_hand_stock must raise ValueError, not silently clamp."""
        with self.assertRaises(ValidationError) as ctx:
            InventoryPosition(
                on_hand_stock=10.0,
                reserved_stock=25.0,
                on_order_stock=50.0,
            )
        self.assertIn("reserved_stock", str(ctx.exception))
        self.assertIn("exceeds on_hand_stock", str(ctx.exception))

    def test_reserved_equal_to_on_hand_accepted(self):
        """Edge case: reserved == on_hand should be valid (zero available)."""
        pos = InventoryPosition(
            on_hand_stock=50.0,
            reserved_stock=50.0,
            on_order_stock=20.0,
        )
        self.assertEqual(pos.net_available_stock, 0.0)
        self.assertEqual(pos.total_inventory_position, 20.0)

    def test_negative_quantities_rejected(self):
        with self.assertRaises(ValidationError):
            InventoryPosition(on_hand_stock=50.0, reserved_stock=-5.0)


class TestDailyAgentInputSchema(unittest.TestCase):
    def setUp(self):
        self.fc = DemandForecast(
            store_id="DS_BLR_INDIRANAGAR",
            sku_id="SKU_BREAD_WHITE",
            category="Bakery",
            date="2024-11-12",
            p10_demand=25.0,
            p50_demand=35.0,
            p90_demand=48.0,
            sigma_demand=8.97,
        )
        self.inv = InventorySnapshot(
            store_id="DS_BLR_INDIRANAGAR",
            sku_id="SKU_BREAD_WHITE",
            date="2024-11-12",
            on_hand_stock=80.0,
            remaining_shelf_life_days=3,
            total_shelf_life_days=3,
            snapshot_basis=InventorySnapshotBasis.BEGINNING_OF_DAY,
        )

    def test_valid_daily_agent_input(self):
        inp = DailyAgentInput(
            store_id="DS_BLR_INDIRANAGAR",
            sku_id="SKU_BREAD_WHITE",
            category="Bakery",
            date="2024-11-12",
            forecast=self.fc,
            inventory=self.inv,
            unconfirmed_alert_count=2,
        )
        self.assertEqual(inp.unconfirmed_alert_count, 2)

    def test_no_target_demand_on_operational_input(self):
        """DailyAgentInput must NOT have a target_demand_eval_label field."""
        inp = DailyAgentInput(
            store_id="DS_BLR_INDIRANAGAR",
            sku_id="SKU_BREAD_WHITE",
            category="Bakery",
            date="2024-11-12",
            forecast=self.fc,
            inventory=self.inv,
        )
        self.assertFalse(hasattr(inp, "target_demand_eval_label"))

    def test_extra_field_rejected(self):
        """DailyAgentInput forbids extra fields (including old target_demand_eval_label)."""
        with self.assertRaises(ValidationError):
            DailyAgentInput(
                store_id="DS_BLR_INDIRANAGAR",
                sku_id="SKU_BREAD_WHITE",
                category="Bakery",
                date="2024-11-12",
                forecast=self.fc,
                inventory=self.inv,
                target_demand_eval_label=36.0,
            )

    def test_store_id_mismatch_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            DailyAgentInput(
                store_id="DS_BLR_WHITEFIELD",
                sku_id="SKU_BREAD_WHITE",
                category="Bakery",
                date="2024-11-12",
                forecast=self.fc,
                inventory=self.inv,
            )
        self.assertIn("Store ID mismatch", str(ctx.exception))

    def test_sku_id_mismatch_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            DailyAgentInput(
                store_id="DS_BLR_INDIRANAGAR",
                sku_id="SKU_EGGS_12PK",
                category="Bakery",
                date="2024-11-12",
                forecast=self.fc,
                inventory=self.inv,
            )
        self.assertIn("SKU ID mismatch", str(ctx.exception))

    def test_date_mismatch_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            DailyAgentInput(
                store_id="DS_BLR_INDIRANAGAR",
                sku_id="SKU_BREAD_WHITE",
                category="Bakery",
                date="2024-11-13",
                forecast=self.fc,
                inventory=self.inv,
            )
        self.assertIn("Date mismatch", str(ctx.exception))


class TestDailyEvaluationRecordSchema(unittest.TestCase):
    def test_valid_evaluation_record(self):
        rec = DailyEvaluationRecord(
            store_id="DS_BLR_KORAMANGALA",
            sku_id="SKU_MILK_1L",
            date="2024-11-15",
            target_demand=42.5,
        )
        self.assertEqual(rec.target_demand, 42.5)
        self.assertEqual(rec.date, "2024-11-15")

    def test_negative_target_demand_rejected(self):
        with self.assertRaises(ValidationError):
            DailyEvaluationRecord(
                store_id="DS_BLR_KORAMANGALA",
                sku_id="SKU_MILK_1L",
                date="2024-11-15",
                target_demand=-1.0,
            )

    def test_empty_identifier_rejected(self):
        with self.assertRaises(ValidationError):
            DailyEvaluationRecord(
                store_id="   ",
                sku_id="SKU_MILK_1L",
                date="2024-11-15",
                target_demand=10.0,
            )


if __name__ == "__main__":
    unittest.main()
