"""
Application Lifecycle & Context Manager
=======================================
Provides a unified, robust application startup and shutdown lifecycle for both
the FastAPI REST server and the operational CLI.
Ensures single-instance dependency injection, SQLite connection safety,
WAL checkpointing, crash recovery invocation, and resource teardown.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from .config import RuntimeConfig, get_runtime_config
from ..agents.coordinator.repository import CoordinatorRepository
from ..agents.coordinator.registry import AgentRegistry
from ..agents.coordinator.service import MultiAgentCoordinator
from ..agents.coordinator.adapters.data_loader_adapter import DataLoaderAdapter
from ..agents.coordinator.adapters.inventory_adapter import InventoryEngineAdapter
from ..agents.coordinator.adapters.procurement_adapter import ProcurementEngineAdapter
from ..agents.coordinator.adapters.approval_adapter import ApprovalWorkflowAdapter
from ..agents.coordinator.adapters.fulfilment_adapter import FulfilmentServiceAdapter
from ..agents.coordinator.adapters.tracking_adapter import TrackingServiceAdapter
from ..agents.coordinator.adapters.circularity_adapter import CircularityAdapter
from ..agents.fulfilment.service import FulfilmentService, ShipmentTrackingService
from ..agents.inventory.repository import OperationalInventoryRepository

logger = logging.getLogger("pipeline.runtime")


def _checkpoint_db(db_path: Path) -> None:
    """Performs a WAL truncate checkpoint on a SQLite database file if it exists."""
    if db_path.is_file() and str(db_path) != ":memory:":
        try:
            conn = sqlite3.connect(str(db_path), timeout=10.0)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
        except Exception as e:
            logger.debug(f"WAL checkpoint skipped for {db_path}: {e}")


class ApplicationContext:
    """
    Central dependency injection and lifecycle manager.
    Encapsulates repository connections, domain services, coordinator adapters,
    and the MultiAgentCoordinator service.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or get_runtime_config()
        self._lock = threading.RLock()
        self._is_initialized = False

        # Core services and repositories (initialized in startup)
        self.coordinator_repository: Optional[CoordinatorRepository] = None
        self.fulfilment_service: Optional[FulfilmentService] = None
        self.tracking_service: Optional[ShipmentTrackingService] = None
        self.inventory_repository: Optional[OperationalInventoryRepository] = None
        self.registry: Optional[AgentRegistry] = None
        self.coordinator: Optional[MultiAgentCoordinator] = None

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def startup(self) -> "ApplicationContext":
        """
        Initializes repositories, services, adapters, and multi-agent coordinator.
        Executes startup crash recovery for interrupted executions.
        """
        with self._lock:
            if self._is_initialized:
                return self

            logger.info("Starting up Demand-Consumption Pipeline application context...")
            self.config.ensure_directories()

            # 1. Initialize Coordinator Persistence Repository
            self.coordinator_repository = CoordinatorRepository(
                db_path=self.config.coordinator_db_path,
            )

            # 2. Crash Recovery Boundary: recover any interrupted executions
            recovered_ids = self.coordinator_repository.recover_interrupted_executions()
            if recovered_ids:
                logger.warning(
                    f"Recovered {len(recovered_ids)} interrupted execution(s) from previous abnormal shutdown: {recovered_ids}"
                )

            # 3. Initialize Fulfilment Service and Domain Repositories
            self.fulfilment_service = FulfilmentService(
                db_path=self.config.fulfilment_db_path,
            )
            self.tracking_service = self.fulfilment_service.tracking_service
            self.inventory_repository = self.fulfilment_service.inventory_repository

            # 4. Build Agent Registry and register all 7 thin adapters
            self.registry = AgentRegistry()
            self.registry.register(DataLoaderAdapter(config=self.config))
            self.registry.register(InventoryEngineAdapter(config=self.config))
            self.registry.register(ProcurementEngineAdapter(config=self.config))
            self.registry.register(ApprovalWorkflowAdapter(config=self.config))
            self.registry.register(FulfilmentServiceAdapter(service=self.fulfilment_service))
            self.registry.register(TrackingServiceAdapter(tracking_service=self.tracking_service))
            self.registry.register(CircularityAdapter())

            # 5. Initialize Central Multi-Agent Coordinator
            self.coordinator = MultiAgentCoordinator(
                config=self.config,
                registry=self.registry,
                repository=self.coordinator_repository,
                max_workers=self.config.max_workers,
            )

            self._is_initialized = True
            logger.info(
                f"Application context initialized successfully (env={self.config.env}, workers={self.config.max_workers})."
            )
            return self

    def shutdown(self) -> None:
        """
        Gracefully tears down the coordinator, flushes database connections,
        checkpoints SQLite WAL logs, and terminates executors.
        """
        with self._lock:
            if not self._is_initialized:
                return

            logger.info("Shutting down application context and closing database connections...")

            # 1. Close Coordinator Service and workers
            if self.coordinator is not None:
                try:
                    self.coordinator.close()
                except Exception as e:
                    logger.error(f"Error closing coordinator: {e}")
                finally:
                    self.coordinator = None

            # 2. Close Fulfilment Service & Inventory Repository
            if self.fulfilment_service is not None:
                try:
                    self.fulfilment_service.close()
                except Exception as e:
                    logger.error(f"Error closing fulfilment service: {e}")
                finally:
                    self.fulfilment_service = None
                    self.tracking_service = None
                    self.inventory_repository = None

            # 3. Close Coordinator Repository
            if self.coordinator_repository is not None:
                try:
                    self.coordinator_repository.close()
                except Exception as e:
                    logger.error(f"Error closing coordinator repository: {e}")
                finally:
                    self.coordinator_repository = None

            # 4. Checkpoint SQLite WAL databases to disk
            _checkpoint_db(self.config.coordinator_db_path)
            _checkpoint_db(self.config.fulfilment_db_path)

            self._is_initialized = False
            logger.info("Application context shut down cleanly.")

    def __enter__(self) -> "ApplicationContext":
        return self.startup()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def get_coordinator(self) -> MultiAgentCoordinator:
        """Returns the active MultiAgentCoordinator or raises RuntimeError if uninitialized."""
        if not self._is_initialized or self.coordinator is None:
            raise RuntimeError("ApplicationContext is not initialized. Call startup() first.")
        return self.coordinator

    def get_inventory_repository(self) -> OperationalInventoryRepository:
        """Returns active OperationalInventoryRepository."""
        if not self._is_initialized or self.inventory_repository is None:
            raise RuntimeError("ApplicationContext is not initialized. Call startup() first.")
        return self.inventory_repository

    def get_fulfilment_service(self) -> FulfilmentService:
        """Returns active FulfilmentService."""
        if not self._is_initialized or self.fulfilment_service is None:
            raise RuntimeError("ApplicationContext is not initialized. Call startup() first.")
        return self.fulfilment_service

    def verify_readiness(self) -> Dict[str, Any]:
        """
        Executes comprehensive readiness checks for:
        - SQLite database paths and writability.
        - Processed forecast, master, and preorder parquet files.
        - Pre-trained LightGBM model artifact.
        """
        readiness: Dict[str, Any] = {
            "ready": True,
            "checks": {},
            "errors": [],
        }

        # Check Coordinator DB directory
        try:
            self.config.coordinator_db_path.parent.mkdir(parents=True, exist_ok=True)
            readiness["checks"]["coordinator_db"] = "OK"
        except Exception as e:
            readiness["ready"] = False
            readiness["checks"]["coordinator_db"] = f"FAILED: {e}"
            readiness["errors"].append(str(e))

        # Check Fulfilment DB directory
        try:
            self.config.fulfilment_db_path.parent.mkdir(parents=True, exist_ok=True)
            readiness["checks"]["fulfilment_db"] = "OK"
        except Exception as e:
            readiness["ready"] = False
            readiness["checks"]["fulfilment_db"] = f"FAILED: {e}"
            readiness["errors"].append(str(e))

        # Check Datasets
        for name, path in [
            ("forecast_file", self.config.forecast_file),
            ("master_file", self.config.master_file),
            ("preorders_file", self.config.preorders_file),
        ]:
            if path.is_file():
                readiness["checks"][name] = "EXISTS"
            else:
                readiness["ready"] = False
                readiness["checks"][name] = f"MISSING ({path})"
                readiness["errors"].append(f"Required dataset '{name}' not found at {path}")

        # Check Model Artifact
        if self.config.model_file.is_file():
            readiness["checks"]["model_file"] = "EXISTS"
        else:
            readiness["ready"] = False
            readiness["checks"]["model_file"] = f"MISSING ({self.config.model_file})"
            readiness["errors"].append(f"Required model artifact not found at {self.config.model_file}")

        return readiness


# Global application context instance
_GLOBAL_CONTEXT: Optional[ApplicationContext] = None
_GLOBAL_LOCK = threading.Lock()


def get_application_context(config: Optional[RuntimeConfig] = None) -> ApplicationContext:
    """Returns singleton ApplicationContext."""
    global _GLOBAL_CONTEXT
    with _GLOBAL_LOCK:
        if _GLOBAL_CONTEXT is None:
            _GLOBAL_CONTEXT = ApplicationContext(config=config)
        return _GLOBAL_CONTEXT

