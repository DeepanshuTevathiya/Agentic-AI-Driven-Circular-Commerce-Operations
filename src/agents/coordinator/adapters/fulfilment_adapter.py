"""
Lead 2 Coordinator Adapter: Fulfilment Service
=============================================
Wraps the existing FulfilmentService to provide standardized execution for:
- FULFILMENT_TRANSMISSION (gateway 2PC order transmission)
- DOCK_RECEIVING (authorized physical dock intake & inspection)
- INVENTORY_POSTING (durable posting of accepted batches to inventory ledger)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from ..schemas import CoordinatorStage, TaskContext
from ..registry import CoordinatorAgent, TaskDelegationError
from ...fulfilment.service import FulfilmentService
from ...fulfilment.schemas import GoodsReceiptNote
from ...procurement.schemas import PurchaseOrder


class FulfilmentServiceAdapter:
    """
    Adapter wrapping FulfilmentService for Coordinator stages:
    - FULFILMENT_TRANSMISSION
    - DOCK_RECEIVING
    - INVENTORY_POSTING

    CRITICAL GOVERNANCE INVARIANTS:
    1. Supplier transmission strictly mandates verified human approval on the PO.
    2. Physical receipt strictly enforces dock inspection evidence.
    3. Inventory posting strictly posts only accepted sellable stock; damaged or held stock is blocked.
    4. The coordinator never writes directly to database tables.
    """

    def __init__(self, service: Optional[FulfilmentService] = None, db_path: Optional[Union[str, Path]] = None) -> None:
        self.service = service or FulfilmentService(db_path=db_path)

    @property
    def agent_id(self) -> str:
        return "fulfilment_service_adapter"

    @property
    def agent_name(self) -> str:
        return "Fulfilment Service Adapter"

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        return (
            CoordinatorStage.FULFILMENT_TRANSMISSION,
            CoordinatorStage.DOCK_RECEIVING,
            CoordinatorStage.INVENTORY_POSTING,
        )

    def can_handle(self, task: TaskContext) -> bool:
        return task.stage in self.handled_stages

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates input payload requirements based on task stage."""
        payload = task.input_payload

        if task.stage == CoordinatorStage.FULFILMENT_TRANSMISSION:
            if "purchase_order" not in payload:
                raise ValueError(f"Task '{task.task_id}' missing 'purchase_order' in input payload.")

        elif task.stage == CoordinatorStage.DOCK_RECEIVING:
            if "fo_id" not in payload:
                raise ValueError(f"Task '{task.task_id}' missing 'fo_id' in input payload.")
            if "grn" not in payload:
                raise ValueError(f"Task '{task.task_id}' missing 'grn' in input payload.")

        elif task.stage == CoordinatorStage.INVENTORY_POSTING:
            if "fo_id" not in payload:
                raise ValueError(f"Task '{task.task_id}' missing 'fo_id' in input payload.")
            if "grn_id" not in payload:
                raise ValueError(f"Task '{task.task_id}' missing 'grn_id' in input payload.")

        return True

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """Dispatches task execution to appropriate FulfilmentService method based on stage."""
        payload = task.input_payload

        try:
            # 1. Supplier Transmission
            if task.stage == CoordinatorStage.FULFILMENT_TRANSMISSION:
                raw_po = payload["purchase_order"]
                po = PurchaseOrder.model_validate(raw_po) if isinstance(raw_po, dict) else raw_po

                fo_id = payload.get("fo_id") or f"FO-{po.po_id.replace('PO-', '')}"
                idem_key = payload.get("idempotency_key") or task.idempotency_key
                actor_id = payload.get("actor_id", "FULFILMENT_COORDINATOR")
                override_scenario = payload.get("override_scenario")

                # Step A: Create order if not existing
                existing_fo = self.service.repository.get(fo_id)
                if not existing_fo:
                    fo = self.service.create_order(
                        po=po,
                        actor_id=actor_id,
                        fo_id=fo_id,
                        idempotency_key=f"CREATE:{fo_id}",
                    )
                else:
                    fo = existing_fo

                # Boundary 4 domain pre-check: if order was already acknowledged at supplier or beyond
                if fo.status.value in ("ACKNOWLEDGED", "DISPATCHED", "IN_TRANSIT", "PARTIALLY_DELIVERED", "DELIVERED", "CLOSED"):
                    raw_asn = payload.get("asn") or payload.get("advance_shipment_notice")
                    updated_fo = fo
                    if raw_asn:
                        from ...fulfilment.schemas import AdvanceShipmentNotice
                        asn = AdvanceShipmentNotice.model_validate(raw_asn) if isinstance(raw_asn, dict) else raw_asn
                        if asn.shipment_id not in updated_fo.shipment_ids:
                            updated_fo = self.service.dispatch_shipment(
                                fo_id=updated_fo.fo_id,
                                asn=asn,
                                actor_id=actor_id,
                            )
                    supp_ref = None
                    for evt in updated_fo.event_history:
                        if hasattr(evt, "metadata") and isinstance(evt.metadata, dict):
                            supp_ref = evt.metadata.get("supplier_reference")
                            if supp_ref:
                                break
                    if not supp_ref:
                        supp_ref = f"SUP-REF-{updated_fo.fo_id}"

                    return {
                        "fo_id": updated_fo.fo_id,
                        "po_id": po.po_id,
                        "status": updated_fo.status.value,
                        "transmission_status": "CONFIRMED_ACCEPTANCE",
                        "transmission_id": f"TX-REUSED-{updated_fo.fo_id}",
                        "supplier_reference": supp_ref,
                        "shipment_ids": list(updated_fo.shipment_ids),
                    }

                # Step B: Transmit order via gateway
                updated_fo, tx_result = self.service.transmit_purchase_order(
                    fo_id=fo.fo_id,
                    po=po,
                    idempotency_key=idem_key,
                    actor_id=actor_id,
                    override_scenario=override_scenario,
                )

                # Step C: Dispatch shipment if advance shipment notice provided
                raw_asn = payload.get("asn") or payload.get("advance_shipment_notice")
                if raw_asn:
                    from ...fulfilment.schemas import AdvanceShipmentNotice
                    asn = AdvanceShipmentNotice.model_validate(raw_asn) if isinstance(raw_asn, dict) else raw_asn
                    updated_fo = self.service.dispatch_shipment(
                        fo_id=updated_fo.fo_id,
                        asn=asn,
                        actor_id=actor_id,
                    )

                return {
                    "fo_id": updated_fo.fo_id,
                    "po_id": po.po_id,
                    "status": updated_fo.status.value,
                    "transmission_status": tx_result.status.value,
                    "transmission_id": tx_result.transmission_id,
                    "supplier_reference": tx_result.supplier_reference,
                    "shipment_ids": list(updated_fo.shipment_ids),
                }

            # 2. Dock Receiving
            elif task.stage == CoordinatorStage.DOCK_RECEIVING:
                fo_id = payload["fo_id"]
                raw_grn = payload["grn"]
                grn = GoodsReceiptNote.model_validate(raw_grn) if isinstance(raw_grn, dict) else raw_grn
                actor_id = payload.get("actor_id", "STORE_RECEIVER")
                auto_post = payload.get("auto_post_inventory", False)

                # Boundary 4 domain pre-check: if GRN was already received and persisted
                existing_grn = None
                try:
                    for g in self.service.repository.get_grns_for_order(fo_id):
                        if g.grn_id == grn.grn_id:
                            existing_grn = g
                            break
                except Exception:
                    existing_grn = None

                if existing_grn is not None:
                    current_fo = self.service.repository.get(fo_id)
                    return {
                        "fo_id": current_fo.fo_id if current_fo else fo_id,
                        "grn_id": existing_grn.grn_id,
                        "shipment_id": existing_grn.shipment_id,
                        "order_status": current_fo.status.value if current_fo else "DELIVERED",
                        "total_delivered_quantity": existing_grn.total_delivered_quantity,
                        "total_accepted_quantity": existing_grn.total_accepted_quantity,
                        "total_rejected_quantity": existing_grn.total_rejected_quantity,
                        "has_discrepancy": (existing_grn.total_rejected_quantity > 0),
                    }

                updated_fo, persisted_grn = self.service.process_dock_intake(
                    fo_id=fo_id,
                    grn=grn,
                    actor_id=actor_id,
                    idempotency_key=task.idempotency_key,
                    auto_post_inventory=auto_post,
                )

                return {
                    "fo_id": updated_fo.fo_id,
                    "grn_id": persisted_grn.grn_id,
                    "shipment_id": persisted_grn.shipment_id,
                    "order_status": updated_fo.status.value,
                    "total_delivered_quantity": persisted_grn.total_delivered_quantity,
                    "total_accepted_quantity": persisted_grn.total_accepted_quantity,
                    "total_rejected_quantity": persisted_grn.total_rejected_quantity,
                    "has_discrepancy": (persisted_grn.total_rejected_quantity > 0),
                }

            # 3. Inventory Posting
            elif task.stage == CoordinatorStage.INVENTORY_POSTING:
                fo_id = payload["fo_id"]
                grn_id = payload["grn_id"]
                actor_id = payload.get("actor_id", "STORE_RECEIVER")
                idem_key = payload.get("idempotency_key") or task.idempotency_key

                # Boundary 4 domain pre-check: if GRN was already posted to inventory ledger
                try:
                    existing_entries = self.service.inventory_repository.get_ledger_history(grn_id=grn_id)
                    if existing_entries:
                        return {
                            "fo_id": fo_id,
                            "grn_id": grn_id,
                            "posted_entries_count": len(existing_entries),
                            "posting_ids": [e.posting_id for e in existing_entries],
                            "total_posted_quantity": sum(e.quantity for e in existing_entries),
                        }
                except Exception:
                    pass

                ledger_entries = self.service.post_grn_to_inventory(
                    fo_id=fo_id,
                    grn_id=grn_id,
                    actor_id=actor_id,
                    idempotency_key=idem_key,
                )

                return {
                    "fo_id": fo_id,
                    "grn_id": grn_id,
                    "posted_entries_count": len(ledger_entries),
                    "posting_ids": [e.posting_id for e in ledger_entries],
                    "total_posted_quantity": sum(e.quantity for e in ledger_entries),
                }

            else:
                raise ValueError(f"Unsupported stage '{task.stage.value}' for FulfilmentServiceAdapter.")

        except Exception as exc:
            raise TaskDelegationError(
                f"FulfilmentService execution failed for task '{task.task_id}': {exc}"
            ) from exc

