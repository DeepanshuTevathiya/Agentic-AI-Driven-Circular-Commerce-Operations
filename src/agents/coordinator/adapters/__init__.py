"""
Lead 2 Multi-Agent Coordinator Adapters
=======================================
Lightweight adapters wrapping existing Lead 1 and Lead 2 services:
- DataLoaderAdapter (INGESTION)
- InventoryEngineAdapter (INVENTORY_EVALUATION)
- ProcurementEngineAdapter (PROCUREMENT_PLANNING)
- ApprovalWorkflowAdapter (HUMAN_APPROVAL_GATE)
- FulfilmentServiceAdapter (FULFILMENT_TRANSMISSION, DOCK_RECEIVING, INVENTORY_POSTING)
- TrackingServiceAdapter (SHIPMENT_TRACKING)
- CircularityAdapter (CIRCULARITY_DISPOSITION — prototype wrapper)
"""

from .data_loader_adapter import DataLoaderAdapter
from .inventory_adapter import InventoryEngineAdapter
from .procurement_adapter import ProcurementEngineAdapter
from .approval_adapter import ApprovalWorkflowAdapter
from .fulfilment_adapter import FulfilmentServiceAdapter
from .tracking_adapter import TrackingServiceAdapter
from .circularity_adapter import CircularityAdapter

__all__ = [
    "DataLoaderAdapter",
    "InventoryEngineAdapter",
    "ProcurementEngineAdapter",
    "ApprovalWorkflowAdapter",
    "FulfilmentServiceAdapter",
    "TrackingServiceAdapter",
    "CircularityAdapter",
]

