"""
Unit & Integration Tests: Fulfilment Domain Schemas & Invariants
===============================================================
Phase 5.2 Verification Tests:
1. Valid fulfilment order construction from an approved PO.
2. Reject non-approved PO input (PENDING_APPROVAL, REJECTED, CANCELLED).
3. Reject PO with invalid or missing approval audit evidence.
4. PO snapshot commercial terms remain consistent.
5. Reject mismatched PO, store, supplier, or SKU identifiers.
6. Reject negative quantities.
7. Reject NaN and infinity.
8. Reject duplicate SKU lines.
9. Reject inconsistent quantity totals.
10. Reject malformed timestamps.
11. Normalize non-UTC offsets to UTC.
12. Reject timezone-naive timestamps.
13. JSON serialization and reconstruction.
14. Nested collection immutability.
15. Full supplier acceptance.
16. Partial supplier acceptance.
17. Full rejection with reason.
18. Change request without modifying the original PO.
19. Acknowledged quantities exceeding ordered quantities rejected.
20. Duplicate SKU acknowledgment lines rejected.
21. Supplier identifier mismatch rejected.
22. Full shipment notice.
23. Multiple partial shipments.
24. Shipment exceeding supplier-accepted quantity rejected.
25. Unknown SKU shipment rejected.
26. Dispatch before acknowledgment rejected.
27. Expected delivery before dispatch rejected.
28. Duplicate shipment ID rejected.
29. Full delivery with full acceptance in GRN.
30. Partial receipt in GRN.
31. Multiple partial receipts.
32. Delivered = accepted + rejected validation.
33. Damaged goods excluded from accepted quantity.
34. Excess delivery rejected.
35. Wrong SKU or store rejected.
36. Missing batch details rejected.
37. Invalid remaining shelf life rejected.
38. Supervisor sign-off requirements (> ₹5,000 threshold).
"""

import json
import unittest
from pydantic import ValidationError

from src.agents.fulfilment.schemas import (
    AdvanceShipmentNotice,
    CancellationReasonCategory,
    DiscrepancyReason,
    DockInspectionLine,
    FulfilmentActorType,
    FulfilmentBatch,
    FulfilmentEvent,
    FulfilmentEventType,
    FulfilmentOrder,
    FulfilmentOrderLine,
    FulfilmentOrderStatus,
    GoodsReceiptNote,
    ShipmentLineItem,
    ShipmentStatus,
    SupplierAcknowledgment,
    SupplierAcknowledgmentLine,
    SupplierAcknowledgmentStatus,
    validate_timestamp_sequence,
)
from src.agents.procurement.approval import approve_purchase_order, reject_purchase_order
from src.agents.procurement.schemas import (
    ApprovalAction,
    PurchaseOrder,
    PurchaseOrderAuditRecord,
    PurchaseOrderLineItem,
    PurchaseOrderStatus,
    PurchaseOrderUrgency,
    SupplierType,
)


def _make_test_approved_po(
    store_id: str = "DS_BLR_KORAMANGALA",
    sku_id: str = "SKU_MILK_1L",
    category: str = "Dairy",
    final_order_qty: float = 24.0,
    unit_procurement_cost: float = 30.0,
    po_id: str = "PO-20241115-DS_BLR_KORAMANGALA-DAIRY-01",
    approval_timestamp: str = "2024-11-15T10:30:00Z",
) -> PurchaseOrder:
    """Helper to construct a legally approved PurchaseOrder with verified audit trail."""
    item = PurchaseOrderLineItem(
        store_id=store_id,
        sku_id=sku_id,
        category=category,
        raw_suggested_qty=20.0,
        case_pack_rounded_qty=24.0,
        final_order_qty=final_order_qty,
        case_pack_size=12,
        moq=10.0,
        unit_retail_price=50.0,
        unit_procurement_cost=unit_procurement_cost,
        total_line_cost=round(final_order_qty * unit_procurement_cost, 2),
        requires_escalation=False,
        escalation_reasons=[],
    )
    pending_po = PurchaseOrder(
        po_id=po_id,
        created_date="2024-11-15",
        store_id=store_id,
        supplier_id="SIM_SUPP_DAIRY_PRIMARY",
        supplier_type=SupplierType.PRIMARY,
        items=[item],
        total_order_qty=final_order_qty,
        total_order_cost=round(final_order_qty * unit_procurement_cost, 2),
        urgency=PurchaseOrderUrgency.STANDARD,
        status=PurchaseOrderStatus.PENDING_APPROVAL,
        requires_escalation=False,
        escalation_reasons=[],
        audit_trail=(),
    )
    return approve_purchase_order(
        po=pending_po,
        approved_by="OPERATOR_DEEPANSHU",
        approval_timestamp=approval_timestamp,
        approval_reason="Verified replenishment need.",
    )


class TestFulfilmentSchemas(unittest.TestCase):
    def setUp(self):
        self.approved_po = _make_test_approved_po()
        self.created_at = "2024-11-15T11:00:00Z"
        self.fo_id = "FO-20241115-DS_BLR_KORAMANGALA-DAIRY-01"

    def _make_valid_fo(self) -> FulfilmentOrder:
        line = FulfilmentOrderLine(
            sku_id=self.approved_po.items[0].sku_id,
            category=self.approved_po.items[0].category,
            ordered_quantity=self.approved_po.items[0].final_order_qty,
            unit_cost=self.approved_po.items[0].unit_procurement_cost,
            supplier_accepted_quantity=0.0,
            supplier_declined_quantity=0.0,
            supplier_unconfirmed_quantity=self.approved_po.items[0].final_order_qty,
        )
        evt = FulfilmentEvent(
            event_id=f"EVT-{self.fo_id}-001",
            idempotency_key=f"CREATE:{self.fo_id}:1",
            fo_id=self.fo_id,
            event_type=FulfilmentEventType.CREATE_FULFILMENT_ORDER,
            actor_type=FulfilmentActorType.FULFILMENT_AGENT,
            actor_id="AGENT_FULFILMENT",
            timestamp=self.created_at,
            previous_status=None,
            new_status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
            sequence_number=1,
            related_reference_id=self.approved_po.po_id,
        )
        return FulfilmentOrder(
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            po_snapshot=self.approved_po,
            store_id=self.approved_po.store_id,
            supplier_id=self.approved_po.supplier_id,
            supplier_type=self.approved_po.supplier_type,
            status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
            lines=(line,),
            created_at=self.created_at,
            updated_at=self.created_at,
            event_history=(evt,),
            is_simulation=True,
        )

    # 1. Valid fulfilment order construction from an approved PO
    def test_01_valid_fulfilment_order_construction(self):
        fo = self._make_valid_fo()
        self.assertEqual(fo.fo_id, self.fo_id)
        self.assertEqual(fo.status, FulfilmentOrderStatus.READY_TO_TRANSMIT)
        self.assertEqual(fo.lines[0].ordered_quantity, 24.0)
        self.assertEqual(fo.lines[0].supplier_unconfirmed_quantity, 24.0)
        self.assertEqual(fo.total_confirmed_pipeline_qty, 0.0)  # Unacknowledged orders do not count as pipeline

    # 2. Reject non-approved PO input
    def test_02_reject_non_approved_po_input(self):
        # Pending PO
        pending_po = _make_test_approved_po()
        # Direct unvalidated mutation or creation with PENDING_APPROVAL
        raw_pending = PurchaseOrder(
            po_id="PO-TEST-PENDING",
            created_date="2024-11-15",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            supplier_type=SupplierType.PRIMARY,
            items=self.approved_po.items,
            total_order_qty=24.0,
            total_order_cost=720.0,
            status=PurchaseOrderStatus.PENDING_APPROVAL,
        )
        line = FulfilmentOrderLine(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            ordered_quantity=24.0,
            unit_cost=30.0,
            supplier_unconfirmed_quantity=24.0,
        )
        with self.assertRaises(ValidationError) as ctx:
            FulfilmentOrder(
                fo_id="FO-PENDING",
                po_id=raw_pending.po_id,
                po_snapshot=raw_pending,
                store_id=raw_pending.store_id,
                supplier_id=raw_pending.supplier_id,
                supplier_type=raw_pending.supplier_type,
                status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
                lines=(line,),
                created_at="2024-11-15T11:00:00Z",
                updated_at="2024-11-15T11:00:00Z",
            )
        self.assertIn("must have status 'APPROVED'", str(ctx.exception))

    # 3. Reject PO with invalid or missing approval audit evidence
    def test_03_reject_po_with_missing_audit_evidence(self):
        # Construct an APPROVED PO that lacks audit trail
        with self.assertRaises(ValidationError):
            PurchaseOrder(
                po_id="PO-TAMPERED",
                created_date="2024-11-15",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                supplier_type=SupplierType.PRIMARY,
                items=self.approved_po.items,
                total_order_qty=24.0,
                total_order_cost=720.0,
                status=PurchaseOrderStatus.APPROVED,
                approved_by="OPERATOR_FAKE",
                approval_timestamp="2024-11-15T10:30:00Z",
                audit_trail=(),  # Missing audit trail
            )

    # 4. PO snapshot commercial terms remain consistent
    def test_04_po_snapshot_commercial_terms_preserved(self):
        fo = self._make_valid_fo()
        self.assertEqual(fo.po_snapshot.total_order_cost, 720.0)
        self.assertEqual(fo.po_snapshot.items[0].unit_procurement_cost, 30.0)
        self.assertEqual(fo.po_snapshot.status, PurchaseOrderStatus.APPROVED)

    # 5. Reject mismatched PO, store, supplier, or SKU identifiers
    def test_05_reject_mismatched_identifiers(self):
        # Mismatched store_id
        with self.assertRaises(ValidationError) as ctx:
            FulfilmentOrder(
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                po_snapshot=self.approved_po,
                store_id="DS_BLR_HSR_LAYOUT",  # Mismatch with PO store
                supplier_id=self.approved_po.supplier_id,
                supplier_type=self.approved_po.supplier_type,
                status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
                lines=self._make_valid_fo().lines,
                created_at=self.created_at,
                updated_at=self.created_at,
            )
        self.assertIn("store_id mismatch", str(ctx.exception))

    # 6. Reject negative quantities
    def test_06_reject_negative_quantities(self):
        with self.assertRaises(ValidationError):
            FulfilmentOrderLine(
                sku_id="SKU_MILK_1L",
                category="Dairy",
                ordered_quantity=-10.0,  # Negative ordered qty
                unit_cost=30.0,
            )

    # 7. Reject NaN and infinity
    def test_07_reject_nan_and_infinity(self):
        with self.assertRaises(ValidationError):
            FulfilmentOrderLine(
                sku_id="SKU_MILK_1L",
                category="Dairy",
                ordered_quantity=float("nan"),
                unit_cost=30.0,
            )
        with self.assertRaises(ValidationError):
            FulfilmentOrderLine(
                sku_id="SKU_MILK_1L",
                category="Dairy",
                ordered_quantity=float("inf"),
                unit_cost=30.0,
            )

    # 8. Reject duplicate SKU lines
    def test_08_reject_duplicate_sku_lines(self):
        line1 = FulfilmentOrderLine(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            ordered_quantity=24.0,
            unit_cost=30.0,
            supplier_unconfirmed_quantity=24.0,
        )
        line2 = FulfilmentOrderLine(
            sku_id="SKU_MILK_1L",  # Duplicate
            category="Dairy",
            ordered_quantity=24.0,
            unit_cost=30.0,
            supplier_unconfirmed_quantity=24.0,
        )
        with self.assertRaises(ValidationError) as ctx:
            FulfilmentOrder(
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                po_snapshot=self.approved_po,
                store_id=self.approved_po.store_id,
                supplier_id=self.approved_po.supplier_id,
                supplier_type=self.approved_po.supplier_type,
                status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
                lines=(line1, line2),
                created_at=self.created_at,
                updated_at=self.created_at,
            )
        self.assertIn("Duplicate SKU", str(ctx.exception))

    # 9. Reject inconsistent quantity totals
    def test_09_reject_inconsistent_quantity_totals(self):
        with self.assertRaises(ValidationError) as ctx:
            FulfilmentOrderLine(
                sku_id="SKU_MILK_1L",
                category="Dairy",
                ordered_quantity=24.0,
                unit_cost=30.0,
                supplier_accepted_quantity=10.0,
                supplier_declined_quantity=5.0,
                supplier_unconfirmed_quantity=0.0,  # 10 + 5 + 0 = 15 != 24
            )
        self.assertIn("Confirmation mismatch", str(ctx.exception))

    # 10. Reject malformed timestamps
    def test_10_reject_malformed_timestamps(self):
        with self.assertRaises(ValidationError):
            SupplierAcknowledgment(
                acknowledgment_id="ACK-001",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                status=SupplierAcknowledgmentStatus.ACCEPTED,
                acknowledged_at="invalid-date",
                expected_delivery_at="2024-11-16T12:00:00Z",
            )

    # 11. Normalize non-UTC offsets to UTC
    def test_11_normalize_non_utc_offsets(self):
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-001",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T16:30:00+05:30",  # 11:00 UTC
            expected_delivery_at="2024-11-16T17:30:00+05:30",  # 12:00 UTC
        )
        self.assertEqual(ack.acknowledged_at, "2024-11-15T11:00:00Z")
        self.assertEqual(ack.expected_delivery_at, "2024-11-16T12:00:00Z")

    # 12. Reject timezone-naive timestamps
    def test_12_reject_timezone_naive_timestamps(self):
        with self.assertRaises(ValidationError) as ctx:
            SupplierAcknowledgment(
                acknowledgment_id="ACK-001",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                status=SupplierAcknowledgmentStatus.ACCEPTED,
                acknowledged_at="2024-11-15 11:00:00",  # naive
                expected_delivery_at="2024-11-16T12:00:00Z",
            )
        self.assertIn("timezone-aware", str(ctx.exception))

    # 13. JSON serialization and reconstruction
    def test_13_json_serialization_and_reconstruction(self):
        fo = self._make_valid_fo()
        dumped_json = fo.model_dump_json()
        reconstructed = FulfilmentOrder.model_validate_json(dumped_json)
        self.assertEqual(reconstructed.fo_id, fo.fo_id)
        self.assertEqual(reconstructed.status, fo.status)
        self.assertEqual(len(reconstructed.lines), len(fo.lines))

    # 14. Nested collection immutability
    def test_14_nested_collection_immutability(self):
        fo = self._make_valid_fo()
        self.assertIsInstance(fo.lines, tuple)
        self.assertIsInstance(fo.event_history, tuple)
        with self.assertRaises(TypeError):
            fo.lines[0] = None  # Tuple is immutable

    # 15. Full supplier acceptance
    def test_15_full_supplier_acceptance(self):
        ack_line = SupplierAcknowledgmentLine(sku_id="SKU_MILK_1L", accepted_quantity=24.0, declined_quantity=0.0)
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-001",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            status=SupplierAcknowledgmentStatus.ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T12:00:00Z",
            lines=(ack_line,),
        )
        self.assertEqual(ack.status, SupplierAcknowledgmentStatus.ACCEPTED)
        self.assertEqual(ack.lines[0].accepted_quantity, 24.0)

    # 16. Partial supplier acceptance
    def test_16_partial_supplier_acceptance(self):
        ack_line = SupplierAcknowledgmentLine(sku_id="SKU_MILK_1L", accepted_quantity=18.0, declined_quantity=6.0)
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-002",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            status=SupplierAcknowledgmentStatus.PARTIALLY_ACCEPTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            expected_delivery_at="2024-11-16T12:00:00Z",
            lines=(ack_line,),
        )
        self.assertEqual(ack.status, SupplierAcknowledgmentStatus.PARTIALLY_ACCEPTED)
        self.assertEqual(ack.lines[0].accepted_quantity, 18.0)
        self.assertEqual(ack.lines[0].declined_quantity, 6.0)

    # 17. Full rejection with reason
    def test_17_full_rejection_with_reason(self):
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-003",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            status=SupplierAcknowledgmentStatus.REJECTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            rejection_reason="Capacity shortage at dairy plant.",
        )
        self.assertEqual(ack.status, SupplierAcknowledgmentStatus.REJECTED)
        self.assertIsNotNone(ack.rejection_reason)

    def test_17b_rejection_without_reason_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            SupplierAcknowledgment(
                acknowledgment_id="ACK-003",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                status=SupplierAcknowledgmentStatus.REJECTED,
                acknowledged_at="2024-11-15T11:30:00Z",
                rejection_reason="",  # Empty
            )
        self.assertIn("rejection_reason is mandatory", str(ctx.exception))

    # 18. Change request without modifying the original PO
    def test_18_change_request_does_not_modify_po(self):
        ack = SupplierAcknowledgment(
            acknowledgment_id="ACK-004",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            status=SupplierAcknowledgmentStatus.CHANGE_REQUESTED,
            acknowledged_at="2024-11-15T11:30:00Z",
            change_request_details={"proposed_pack_size": 24, "proposed_delivery_window": "Evening"},
        )
        self.assertEqual(ack.status, SupplierAcknowledgmentStatus.CHANGE_REQUESTED)
        # Commercial PO remains unchanged
        self.assertEqual(self.approved_po.items[0].case_pack_size, 12)

    # 19. Acknowledged quantities exceeding ordered quantities rejected
    def test_19_acknowledged_exceeding_ordered_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            FulfilmentOrderLine(
                sku_id="SKU_MILK_1L",
                category="Dairy",
                ordered_quantity=24.0,
                unit_cost=30.0,
                supplier_accepted_quantity=30.0,  # Exceeds 24.0
                supplier_declined_quantity=0.0,
                supplier_unconfirmed_quantity=0.0,
            )
        self.assertTrue(
            "cannot exceed ordered_quantity" in str(ctx.exception)
            or "Confirmation mismatch" in str(ctx.exception)
        )

    # 20. Duplicate SKU acknowledgment lines rejected
    def test_20_duplicate_sku_acknowledgment_lines_rejected(self):
        line1 = SupplierAcknowledgmentLine(sku_id="SKU_MILK_1L", accepted_quantity=12.0)
        line2 = SupplierAcknowledgmentLine(sku_id="SKU_MILK_1L", accepted_quantity=12.0)
        with self.assertRaises(ValidationError) as ctx:
            SupplierAcknowledgment(
                acknowledgment_id="ACK-005",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                status=SupplierAcknowledgmentStatus.ACCEPTED,
                acknowledged_at="2024-11-15T11:30:00Z",
                expected_delivery_at="2024-11-16T12:00:00Z",
                lines=(line1, line2),
            )
        self.assertIn("Duplicate SKU", str(ctx.exception))

    # 21. Supplier identifier mismatch rejected
    def test_21_supplier_identifier_mismatch_rejected(self):
        line = self._make_valid_fo().lines[0]
        with self.assertRaises(ValidationError) as ctx:
            FulfilmentOrder(
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                po_snapshot=self.approved_po,
                store_id=self.approved_po.store_id,
                supplier_id="SIM_SUPP_WRONG_VENDOR",  # Mismatch
                supplier_type=self.approved_po.supplier_type,
                status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
                lines=(line,),
                created_at=self.created_at,
                updated_at=self.created_at,
            )
        self.assertIn("supplier_id mismatch", str(ctx.exception))

    # 22. Full shipment notice
    def test_22_full_shipment_notice(self):
        item = ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0)
        asn = AdvanceShipmentNotice(
            shipment_id="ASN-001",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            dispatch_at="2024-11-15T13:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(item,),
            carrier="ColdChainExpress",
            tracking_reference="TRK-98765",
        )
        self.assertEqual(asn.lines[0].shipped_quantity, 24.0)

    # 23. Multiple partial shipments
    def test_23_multiple_partial_shipments(self):
        asn1 = AdvanceShipmentNotice(
            shipment_id="ASN-001A",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            dispatch_at="2024-11-15T13:00:00Z",
            expected_arrival_at="2024-11-16T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=12.0),),
        )
        asn2 = AdvanceShipmentNotice(
            shipment_id="ASN-001B",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            dispatch_at="2024-11-16T13:00:00Z",
            expected_arrival_at="2024-11-17T08:00:00Z",
            lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=12.0),),
        )
        self.assertEqual(asn1.lines[0].shipped_quantity, 12.0)
        self.assertEqual(asn2.lines[0].shipped_quantity, 12.0)

    # 24. Shipment exceeding supplier-accepted quantity rejected
    def test_24_shipment_exceeding_accepted_rejected(self):
        with self.assertRaises(ValidationError):
            FulfilmentOrderLine(
                sku_id="SKU_MILK_1L",
                category="Dairy",
                ordered_quantity=24.0,
                unit_cost=30.0,
                supplier_accepted_quantity=12.0,
                supplier_declined_quantity=12.0,
                supplier_unconfirmed_quantity=0.0,
                shipped_quantity=18.0,  # Exceeds accepted 12.0
            )

    # 25. Unknown SKU shipment rejected
    def test_25_unknown_sku_shipment_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            ShipmentLineItem(sku_id="SKU_UNKNOWN_ITEM", shipped_quantity=10.0)
        self.assertIn("must be one of", str(ctx.exception))

    # 26. Dispatch before acknowledgment rejected
    def test_26_dispatch_before_acknowledgment_rejected(self):
        with self.assertRaises(ValueError):
            validate_timestamp_sequence("2024-11-15T12:00:00Z", "2024-11-15T10:00:00Z", "acknowledged_at", "dispatch_at")

    # 27. Expected delivery before dispatch rejected
    def test_27_expected_delivery_before_dispatch_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            AdvanceShipmentNotice(
                shipment_id="ASN-001",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                dispatch_at="2024-11-15T14:00:00Z",
                expected_arrival_at="2024-11-15T12:00:00Z",  # Arrival before dispatch!
                lines=(ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
            )
        self.assertIn("cannot precede", str(ctx.exception))

    # 28. Duplicate shipment ID rejected
    def test_28_duplicate_sku_in_single_asn_rejected(self):
        item1 = ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=12.0)
        item2 = ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=12.0)
        with self.assertRaises(ValidationError) as ctx:
            AdvanceShipmentNotice(
                shipment_id="ASN-DUP",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                dispatch_at="2024-11-15T12:00:00Z",
                expected_arrival_at="2024-11-16T12:00:00Z",
                lines=(item1, item2),
            )
        self.assertIn("Duplicate SKU", str(ctx.exception))

    # 29. Full delivery with full acceptance in GRN
    def test_29_full_delivery_full_acceptance_grn(self):
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            batch_references=("BATCH-001",),
        )
        grn = GoodsReceiptNote(
            grn_id="GRN-001",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            shipment_id="ASN-001",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
        )
        self.assertEqual(grn.total_accepted_quantity, 24.0)
        self.assertEqual(grn.total_rejected_quantity, 0.0)
        self.assertEqual(grn.total_accepted_cost, 720.0)

    # 30. Partial receipt in GRN
    def test_30_partial_receipt_grn(self):
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=12.0,
            accepted_quantity=12.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            batch_references=("BATCH-001A",),
        )
        grn = GoodsReceiptNote(
            grn_id="GRN-002",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            shipment_id="ASN-001A",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
        )
        self.assertEqual(grn.total_accepted_quantity, 12.0)

    # 31. Multiple partial receipts
    def test_31_multiple_partial_receipts_line_reconciliation(self):
        line = FulfilmentOrderLine(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            ordered_quantity=24.0,
            unit_cost=30.0,
            supplier_accepted_quantity=24.0,
            shipped_quantity=24.0,
            delivered_quantity=24.0,  # 12 + 12
            accepted_quantity=24.0,   # 12 + 12
            rejected_quantity=0.0,
        )
        self.assertEqual(line.remaining_unfulfilled_quantity, 0.0)
        self.assertEqual(line.outstanding_confirmed_quantity, 0.0)

    # 32. Delivered = accepted + rejected validation
    def test_32_delivered_reconciliation_validation(self):
        with self.assertRaises(ValidationError) as ctx:
            DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=20.0,
                rejected_quantity=2.0,  # 20 + 2 = 22 != 24
                unit_cost=30.0,
                remaining_shelf_life_days=5,
                batch_references=("BATCH-001",),
            )
        self.assertIn("Dock inspection quantity mismatch", str(ctx.exception))

    # 33. Damaged goods excluded from accepted quantity
    def test_33_damaged_goods_excluded_from_accepted(self):
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=20.0,
            rejected_quantity=4.0,
            damaged_quantity=4.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
            batch_references=("BATCH-001",),
        )
        self.assertEqual(dock_line.accepted_quantity, 20.0)
        self.assertEqual(dock_line.rejected_quantity, 4.0)
        self.assertEqual(dock_line.damaged_line_cost, 120.0)

    # 34. Excess delivery rejected
    def test_34_excess_delivery_rejected(self):
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=28.0,  # 24 ordered + 4 excess
            accepted_quantity=24.0,   # Only approved balance accepted
            rejected_quantity=4.0,
            excess_quantity=4.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            discrepancy_reasons=(DiscrepancyReason.EXCESS_DELIVERY,),
            batch_references=("BATCH-001",),
        )
        self.assertEqual(dock_line.accepted_quantity, 24.0)
        self.assertEqual(dock_line.rejected_quantity, 4.0)
        self.assertEqual(dock_line.excess_quantity, 4.0)

    # 35. Wrong SKU or store rejected
    def test_35_wrong_sku_or_store_rejected(self):
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            batch_references=("BATCH-001",),
        )
        with self.assertRaises(ValidationError) as ctx:
            GoodsReceiptNote(
                grn_id="GRN-001",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                shipment_id="ASN-001",
                store_id="STORE_NOT_REAL",  # Non-canonical store
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                receipt_at="2024-11-16T08:30:00Z",
                receiver_id="RECEIVER_ARUN",
                inspection_lines=(dock_line,),
            )
        self.assertIn("must be one of", str(ctx.exception))

    # 36. Missing batch details rejected
    def test_36_missing_batch_details_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=24.0,
                rejected_quantity=0.0,
                unit_cost=30.0,
                remaining_shelf_life_days=5,
                batch_references=(),  # Missing batch info for accepted units
            )
        self.assertIn("must be backed by at least one batch reference", str(ctx.exception))

    # 37. Invalid remaining shelf life rejected
    def test_37_missing_shelf_life_quality_hold(self):
        with self.assertRaises(ValidationError) as ctx:
            DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=24.0,
                rejected_quantity=0.0,
                unit_cost=30.0,
                remaining_shelf_life_days=None,  # Missing shelf-life info
                batch_references=("BATCH-001",),
            )
        self.assertIn("Quality Hold Violation", str(ctx.exception))

    # 38. Supervisor sign-off requirements (> ₹5,000 threshold)
    def test_38_supervisor_signoff_required_over_5000(self):
        # 200 units of Meat at ₹30/unit damaged = ₹6,000 (> ₹5,000)
        dock_line = DockInspectionLine(
            sku_id="SKU_CHICKEN_500G",
            delivered_quantity=200.0,
            accepted_quantity=0.0,
            rejected_quantity=200.0,
            damaged_quantity=200.0,
            unit_cost=30.0,
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        # Without supervisor signoff -> rejected
        with self.assertRaises(ValidationError) as ctx:
            GoodsReceiptNote(
                grn_id="GRN-HIGH-DAMAGE",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                shipment_id="ASN-001",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                receipt_at="2024-11-16T08:30:00Z",
                receiver_id="RECEIVER_ARUN",
                inspection_lines=(dock_line,),
            )
        self.assertIn("Mandatory supervisor sign-off", str(ctx.exception))

        # With supervisor signoff -> accepted
        grn = GoodsReceiptNote(
            grn_id="GRN-HIGH-DAMAGE",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            shipment_id="ASN-001",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
            supervisor_signoff_by="MGR_SUNIL",
            supervisor_signoff_at="2024-11-16T09:00:00Z",
        )
        self.assertEqual(grn.supervisor_signoff_by, "MGR_SUNIL")

    # FIX A: Physical over-delivery PO 100 / ASN 100 / delivered 110
    def test_fix_a_physical_over_delivery_po100_asn100_del110(self):
        line = FulfilmentOrderLine(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            ordered_quantity=100.0,
            unit_cost=30.0,
            supplier_accepted_quantity=100.0,
            shipped_quantity=100.0,
            delivered_quantity=110.0,  # 110 physical delivered
            accepted_quantity=100.0,   # Up to approved order quantity accepted
            rejected_quantity=10.0,    # 10 excess units rejected
        )
        self.assertEqual(line.delivered_quantity, 110.0)
        self.assertEqual(line.accepted_quantity, 100.0)
        self.assertEqual(line.rejected_quantity, 10.0)
        self.assertEqual(line.outstanding_confirmed_quantity, 0.0)
        self.assertEqual(line.remaining_unfulfilled_quantity, 0.0)

        # Dock line representation
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=110.0,
            accepted_quantity=100.0,
            rejected_quantity=10.0,
            excess_quantity=10.0,
            unit_cost=30.0,
            remaining_shelf_life_days=5,
            total_shelf_life_days=7,
            discrepancy_reasons=(DiscrepancyReason.EXCESS_DELIVERY,),
            batch_references=("BATCH-100",),
        )
        self.assertEqual(dock_line.delivered_quantity, 110.0)
        self.assertEqual(dock_line.accepted_quantity, 100.0)
        self.assertEqual(dock_line.excess_quantity, 10.0)

    # FIX A: Physical over-delivery PO 100 / ASN 80 / delivered 85
    def test_fix_a_physical_over_delivery_po100_asn80_del85(self):
        line = FulfilmentOrderLine(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            ordered_quantity=100.0,
            unit_cost=30.0,
            supplier_accepted_quantity=100.0,
            shipped_quantity=80.0,
            delivered_quantity=85.0,  # 85 physical delivered vs 80 ASN
            accepted_quantity=80.0,   # 80 accepted (respecting shipment declaration)
            rejected_quantity=5.0,    # 5 excess units rejected against shipment
        )
        self.assertEqual(line.delivered_quantity, 85.0)
        self.assertEqual(line.accepted_quantity, 80.0)
        self.assertEqual(line.rejected_quantity, 5.0)
        self.assertEqual(line.outstanding_confirmed_quantity, 15.0)  # 100 confirmed - 85 delivered = 15 remaining
        self.assertEqual(line.remaining_unfulfilled_quantity, 20.0)   # 100 ordered - 80 accepted = 20 remaining

    # FIX A: Physical delivery below ASN quantity
    def test_fix_a_delivery_below_asn(self):
        line = FulfilmentOrderLine(
            sku_id="SKU_MILK_1L",
            category="Dairy",
            ordered_quantity=100.0,
            unit_cost=30.0,
            supplier_accepted_quantity=100.0,
            shipped_quantity=100.0,
            delivered_quantity=90.0,  # 90 delivered vs 100 ASN
            accepted_quantity=90.0,
            rejected_quantity=0.0,
        )
        self.assertEqual(line.delivered_quantity, 90.0)
        self.assertEqual(line.accepted_quantity, 90.0)
        self.assertEqual(line.outstanding_confirmed_quantity, 10.0)
        self.assertEqual(line.remaining_unfulfilled_quantity, 10.0)

    # FIX E: Boundary test for exactly ₹5,000 threshold
    def test_fix_e_boundary_exactly_5000(self):
        # Exactly ₹5,000 damaged cost: 100 units at ₹50.0 = ₹5,000.00
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=100.0,
            accepted_quantity=0.0,
            rejected_quantity=100.0,
            damaged_quantity=100.0,
            unit_cost=50.0,  # 100 * 50 = ₹5,000.00
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        # Exactly ₹5,000 does NOT strictly exceed ₹5,000 -> sign-off is NOT required
        grn = GoodsReceiptNote(
            grn_id="GRN-EXACT-5000",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            shipment_id="ASN-001",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
            max_dock_discrepancy_cost_inr=5000.0,
        )
        self.assertIsNone(grn.supervisor_signoff_by)

    # FIX E: Boundary test for strictly greater than ₹5,000 (e.g. ₹5,000.01)
    def test_fix_e_boundary_strictly_greater_than_5000(self):
        # 100.0002 units at ₹50.0 = ₹5,000.01
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=100.001,
            accepted_quantity=0.0,
            rejected_quantity=100.001,
            damaged_quantity=100.001,
            unit_cost=50.0,  # 100.001 * 50 = ₹5,000.05 (> ₹5,000)
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        # Strictly greater than ₹5,000 -> requires supervisor signoff
        with self.assertRaises(ValidationError) as ctx:
            GoodsReceiptNote(
                grn_id="GRN-OVER-5000",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                shipment_id="ASN-001",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                receipt_at="2024-11-16T08:30:00Z",
                receiver_id="RECEIVER_ARUN",
                inspection_lines=(dock_line,),
                max_dock_discrepancy_cost_inr=5000.0,
            )
        self.assertIn("Mandatory supervisor sign-off", str(ctx.exception))

    # FIX E: Configurable independent threshold
    def test_fix_e_configurable_dock_threshold_independent(self):
        from src.agents.common.config import Lead2Config, FulfilmentThresholds, ProcurementApprovalThresholds
        cfg = Lead2Config(
            fulfilment_thresholds=FulfilmentThresholds(max_dock_discrepancy_cost_inr=3000.0),
            procurement_thresholds=ProcurementApprovalThresholds(max_line_item_cost_inr=5000.0),
        )
        # Procurement threshold remains 5000
        self.assertEqual(cfg.procurement_thresholds.max_line_item_cost_inr, 5000.0)
        # Dock discrepancy threshold is independently 3000
        self.assertEqual(cfg.fulfilment_thresholds.max_dock_discrepancy_cost_inr, 3000.0)

        # Damage of ₹3,500 with custom 3000 threshold requires signoff
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=70.0,
            accepted_quantity=0.0,
            rejected_quantity=70.0,
            damaged_quantity=70.0,
            unit_cost=50.0,  # 70 * 50 = ₹3,500 (> ₹3,000)
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        with self.assertRaises(ValidationError) as ctx:
            GoodsReceiptNote(
                grn_id="GRN-CUSTOM-3000",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                shipment_id="ASN-001",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                receipt_at="2024-11-16T08:30:00Z",
                receiver_id="RECEIVER_ARUN",
                inspection_lines=(dock_line,),
                max_dock_discrepancy_cost_inr=cfg.fulfilment_thresholds.max_dock_discrepancy_cost_inr,
            )
        self.assertIn("Mandatory supervisor sign-off", str(ctx.exception))

    # FulfilmentBatch test
    def test_fulfilment_batch_conversion(self):
        fb = FulfilmentBatch(
            batch_id="BATCH-001",
            sku_id="SKU_MILK_1L",
            units=24.0,
            remaining_shelf_life_days=5,
            receipt_date="2024-11-16",
            fo_id=self.fo_id,
            grn_id="GRN-001",
            supplier_lot_ref="LOT-456",
        )
        core_batch = fb.to_inventory_batch()
        self.assertEqual(core_batch.batch_id, "BATCH-001")
        self.assertEqual(core_batch.units, 24.0)
        self.assertEqual(core_batch.remaining_shelf_life_days, 5)
        self.assertEqual(core_batch.creation_date, "2024-11-16")

    def test_grn_rejected_goods_disposition_validation(self):
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=0.0,
            rejected_quantity=24.0,
            damaged_quantity=24.0,
            unit_cost=30.0,
            remaining_shelf_life_days=0,
            total_shelf_life_days=5,
            discrepancy_reasons=(DiscrepancyReason.DAMAGED_TRANSIT,),
        )
        # Valid disposition string
        grn = GoodsReceiptNote(
            grn_id="GRN-DISP-001",
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            shipment_id="ASN-001",
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            receipt_at="2024-11-16T08:30:00Z",
            receiver_id="RECEIVER_ARUN",
            inspection_lines=(dock_line,),
            rejected_goods_disposition="RTO-DISP-TRUCK-88",
        )
        self.assertEqual(grn.rejected_goods_disposition, "RTO-DISP-TRUCK-88")

        # Whitespace disposition rejected
        with self.assertRaises(ValidationError) as ctx:
            GoodsReceiptNote(
                grn_id="GRN-DISP-002",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                shipment_id="ASN-001",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                receipt_at="2024-11-16T08:30:00Z",
                receiver_id="RECEIVER_ARUN",
                inspection_lines=(dock_line,),
                rejected_goods_disposition="   ",
            )
        self.assertIn("rejected_goods_disposition, if provided, cannot be empty", str(ctx.exception))

    def test_fulfilment_order_active_shipment_ids_scoping(self):
        base_fo = FulfilmentOrder(
            fo_id=self.fo_id,
            po_id=self.approved_po.po_id,
            po_snapshot=self.approved_po,
            store_id="DS_BLR_KORAMANGALA",
            supplier_id="SIM_SUPP_DAIRY_PRIMARY",
            supplier_type=SupplierType.PRIMARY,
            status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
            lines=(
                FulfilmentOrderLine(
                    sku_id="SKU_MILK_1L",
                    category="Dairy",
                    ordered_quantity=24.0,
                    unit_cost=30.0,
                    supplier_unconfirmed_quantity=24.0,
                ),
            ),
            created_at="2024-11-15T11:00:00Z",
            updated_at="2024-11-15T11:00:00Z",
        )

        # 1. Multiple shipments without resolution evidence -> all active
        fo_multi = base_fo.model_copy(update={
            "shipment_ids": ("ASN-01", "ASN-02"),
        })
        self.assertEqual(fo_multi.active_shipment_ids, ("ASN-01", "ASN-02"))

        # 2. Multiple shipments with shipment_cancellation_ref but order_wide_cancellation_scope=False -> NOT order-wide resolved
        fo_no_scope = fo_multi.model_copy(update={
            "shipment_cancellation_ref": "RTO-REF-01",
            "order_wide_cancellation_scope": False,
        })
        self.assertEqual(fo_no_scope.active_shipment_ids, ("ASN-01", "ASN-02"))

        # 3. Multiple shipments with order_wide_cancellation_scope=True -> all resolved
        fo_order_wide = fo_multi.model_copy(update={
            "shipment_cancellation_ref": "RTO-REF-01",
            "order_wide_cancellation_scope": True,
        })
        self.assertEqual(fo_order_wide.active_shipment_ids, ())

        # 4. Multiple shipments with shipment_resolution_evidence for one shipment only
        fo_partial = fo_multi.model_copy(update={
            "shipment_resolution_evidence": {"ASN-01": "RTO-ASN-01"},
        })
        self.assertEqual(fo_partial.active_shipment_ids, ("ASN-02",))

        # 5. Single shipment with shipment_cancellation_ref -> resolved
        fo_single = base_fo.model_copy(update={
            "shipment_ids": ("ASN-01",),
            "shipment_cancellation_ref": "RTO-SINGLE",
        })
        self.assertEqual(fo_single.active_shipment_ids, ())

    def test_batch_receipt_date_and_lot_ref_validation(self):
        # Valid FulfilmentBatch
        batch = FulfilmentBatch(
            batch_id="BATCH-001",
            sku_id="SKU_MILK_1L",
            units=24.0,
            remaining_shelf_life_days=4,
            receipt_date="2024-11-16",
            fo_id=self.fo_id,
            grn_id="GRN-001",
            supplier_lot_ref="LOT-456",
        )
        self.assertEqual(batch.receipt_date, "2024-11-16")
        self.assertEqual(batch.supplier_lot_ref, "LOT-456")

        # Invalid date formats
        for invalid_date in ["not-a-date", "2024/11/16", "16-11-2024", "2024-13-40", "   "]:
            with self.assertRaises(ValidationError):
                FulfilmentBatch(
                    batch_id="BATCH-001",
                    sku_id="SKU_MILK_1L",
                    units=24.0,
                    remaining_shelf_life_days=4,
                    receipt_date=invalid_date,
                    fo_id=self.fo_id,
                    grn_id="GRN-001",
                )

        # Whitespace lot ref
        with self.assertRaises(ValidationError) as ctx:
            FulfilmentBatch(
                batch_id="BATCH-001",
                sku_id="SKU_MILK_1L",
                units=24.0,
                remaining_shelf_life_days=4,
                receipt_date="2024-11-16",
                fo_id=self.fo_id,
                grn_id="GRN-001",
                supplier_lot_ref="   ",
            )
        self.assertIn("cannot be empty or whitespace", str(ctx.exception))

    def test_optional_string_whitespace_rejections_across_schemas(self):
        # 1. AdvanceShipmentNotice
        for field, val in [("carrier", "  "), ("tracking_reference", "  "), ("notes", "  ")]:
            kwargs = {
                "shipment_id": "ASN-001",
                "fo_id": self.fo_id,
                "po_id": self.approved_po.po_id,
                "supplier_id": "SIM_SUPP_DAIRY_PRIMARY",
                "dispatch_at": "2024-11-15T12:00:00Z",
                "lines": (ShipmentLineItem(sku_id="SKU_MILK_1L", shipped_quantity=24.0),),
                field: val,
            }
            with self.assertRaises(ValidationError):
                AdvanceShipmentNotice(**kwargs)

        # 2. SupplierAcknowledgmentLine notes
        with self.assertRaises(ValidationError):
            SupplierAcknowledgmentLine(
                sku_id="SKU_MILK_1L",
                accepted_quantity=24.0,
                declined_quantity=0.0,
                notes="   ",
            )

        # 3. DockInspectionLine notes & empty batch ref
        with self.assertRaises(ValidationError):
            DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=24.0,
                rejected_quantity=0.0,
                unit_cost=30.0,
                remaining_shelf_life_days=4,
                batch_references=("BATCH-01",),
                notes="   ",
            )
        with self.assertRaises(ValidationError):
            DockInspectionLine(
                sku_id="SKU_MILK_1L",
                delivered_quantity=24.0,
                accepted_quantity=24.0,
                rejected_quantity=0.0,
                unit_cost=30.0,
                remaining_shelf_life_days=4,
                batch_references=("BATCH-01", "   "),
            )

        # 4. GoodsReceiptNote voluntary signoff without timestamp
        dock_line = DockInspectionLine(
            sku_id="SKU_MILK_1L",
            delivered_quantity=24.0,
            accepted_quantity=24.0,
            rejected_quantity=0.0,
            unit_cost=30.0,
            remaining_shelf_life_days=4,
            batch_references=("BATCH-01",),
        )
        with self.assertRaises(ValidationError) as ctx:
            GoodsReceiptNote(
                grn_id="GRN-001",
                fo_id=self.fo_id,
                po_id=self.approved_po.po_id,
                shipment_id="ASN-001",
                store_id="DS_BLR_KORAMANGALA",
                supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                receipt_at="2024-11-15T15:00:00Z",
                receiver_id="RECEIVER_ARUN",
                inspection_lines=(dock_line,),
                supervisor_signoff_by="MGR_PRIYA",
                supervisor_signoff_at=None,
            )
        self.assertIn("requires 'supervisor_signoff_at' timestamp", str(ctx.exception))

        # 5. FulfilmentEvent whitespace optional fields
        with self.assertRaises(ValidationError):
            FulfilmentEvent(
                event_id="EVT-001",
                idempotency_key="KEY-001",
                fo_id=self.fo_id,
                event_type=FulfilmentEventType.CREATE_FULFILMENT_ORDER,
                actor_type=FulfilmentActorType.FULFILMENT_AGENT,
                actor_id="AGENT",
                timestamp="2024-11-15T10:00:00Z",
                previous_status=None,
                new_status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
                sequence_number=1,
                related_reference_id="   ",
            )
        with self.assertRaises(ValidationError):
            FulfilmentEvent(
                event_id="EVT-001",
                idempotency_key="KEY-001",
                fo_id=self.fo_id,
                event_type=FulfilmentEventType.CREATE_FULFILMENT_ORDER,
                actor_type=FulfilmentActorType.FULFILMENT_AGENT,
                actor_id="AGENT",
                timestamp="2024-11-15T10:00:00Z",
                previous_status=None,
                new_status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
                sequence_number=1,
                payload_fingerprint="   ",
            )

        # 6. FulfilmentOrder optional whitespace fields
        for field in [
            "cancellation_reason",
            "cancelled_by",
            "supplier_cancellation_ref",
            "shipment_cancellation_ref",
            "closed_by",
            "closure_reason",
            "open_exception_reason",
        ]:
            with self.assertRaises(ValidationError):
                FulfilmentOrder(
                    fo_id=self.fo_id,
                    po_id=self.approved_po.po_id,
                    po_snapshot=self.approved_po,
                    store_id="DS_BLR_KORAMANGALA",
                    supplier_id="SIM_SUPP_DAIRY_PRIMARY",
                    supplier_type=SupplierType.PRIMARY,
                    status=FulfilmentOrderStatus.READY_TO_TRANSMIT,
                    lines=(
                        FulfilmentOrderLine(
                            sku_id="SKU_MILK_1L",
                            category="Dairy",
                            ordered_quantity=24.0,
                            unit_cost=30.0,
                            supplier_unconfirmed_quantity=24.0,
                        ),
                    ),
                    created_at="2024-11-15T11:00:00Z",
                    updated_at="2024-11-15T11:00:00Z",
                    **{field: "   "},
                )


if __name__ == "__main__":
    unittest.main()
