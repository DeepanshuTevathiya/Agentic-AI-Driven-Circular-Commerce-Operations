"""
Runtime Configuration Module
============================
Extends Lead2Config with operational deployment parameters, environment variable
overrides, .env loading, and absolute path resolution.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..agents.common.config import (
    CATEGORY_PROCUREMENT_PARAMS,
    CATEGORY_SUPPLIER_PARAMS,
    DEFAULT_COORDINATOR_DB_PATH,
    DEFAULT_FULFILMENT_DB_PATH,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    CategoryProcurementParams,
    FulfilmentThresholds,
    Lead2Config,
    ProcurementApprovalThresholds,
    SupplierSimulationParams,
)


def _load_dotenv(env_path: Path) -> Dict[str, str]:
    """Simple, dependency-free .env file parser."""
    values: Dict[str, str] = {}
    if not env_path.is_file():
        return values

    try:
        with open(env_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped:
                    key, val = stripped.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    values[key] = val
    except Exception:
        pass
    return values


@dataclass
class RuntimeConfig(Lead2Config):
    """
    Master runtime configuration for Lead 2 API, CLI, and multi-agent coordinator daemon.
    Inherits all business engine parameters from Lead2Config and layers operational/deployment settings.
    """
    env: str = "production"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_workers: int = 4
    operator_api_key: Optional[str] = None
    authorized_operators: List[str] = field(
        default_factory=lambda: [
            "OPERATOR_DIR_01",
            "MGR_JOHN",
            "LEAD_OPERATOR",
            "STORE_MANAGER_BLR",
            "DIR_OPERATIONS",
        ]
    )
    model_file: Path = OUTPUTS_DIR / "lgb_p10_p50_p90_corrected.joblib"
    enable_simulation_gateways: bool = True

    def __post_init__(self) -> None:
        """Coerces string paths to Path objects if passed directly."""
        if isinstance(self.coordinator_db_path, str):
            self.coordinator_db_path = Path(self.coordinator_db_path)
        if isinstance(self.fulfilment_db_path, str):
            self.fulfilment_db_path = Path(self.fulfilment_db_path)
        if isinstance(self.forecast_file, str):
            self.forecast_file = Path(self.forecast_file)
        if isinstance(self.master_file, str):
            self.master_file = Path(self.master_file)
        if isinstance(self.preorders_file, str):
            self.preorders_file = Path(self.preorders_file)
        if isinstance(self.model_file, str):
            self.model_file = Path(self.model_file)

    def ensure_directories(self) -> None:
        """Ensures that all writable directories for databases and outputs exist."""
        self.coordinator_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fulfilment_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.forecast_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_from_env(
        cls,
        env_file: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> "RuntimeConfig":
        """
        Constructs a RuntimeConfig by layering:
        1. Defaults from Lead2Config
        2. .env file variables (if present)
        3. Operating system environment variable overrides
        """
        root = Path(base_dir).resolve() if base_dir else PROJECT_ROOT

        # 1. Parse optional .env
        dotenv_path = Path(env_file).resolve() if env_file else root / ".env"
        env_vars = _load_dotenv(dotenv_path)

        def get_val(key: str, default: Optional[str] = None) -> Optional[str]:
            return os.environ.get(key, env_vars.get(key, default))

        # 2. Extract values
        env_mode = get_val("PIPELINE_ENV", "production") or "production"
        log_lvl = get_val("LOG_LEVEL", "INFO") or "INFO"
        host = get_val("API_HOST", "0.0.0.0") or "0.0.0.0"
        port = int(get_val("API_PORT", "8000") or 8000)
        workers = int(get_val("MAX_WORKERS", "4") or 4)
        api_key = get_val("OPERATOR_API_KEY", None)

        raw_ops = get_val("AUTHORIZED_OPERATORS", "")
        if raw_ops and raw_ops.strip():
            auth_operators = [op.strip() for op in raw_ops.split(",") if op.strip()]
        else:
            auth_operators = [
                "OPERATOR_DIR_01",
                "MGR_JOHN",
                "LEAD_OPERATOR",
                "STORE_MANAGER_BLR",
                "DIR_OPERATIONS",
            ]

        # Path resolutions
        def resolve_p(raw: Optional[str], default_p: Path) -> Path:
            if not raw or not raw.strip():
                return default_p.resolve()
            p = Path(raw.strip())
            return p.resolve() if p.is_absolute() else (root / p).resolve()

        coord_db = resolve_p(get_val("COORDINATOR_DB_PATH"), DEFAULT_COORDINATOR_DB_PATH)
        fulfil_db = resolve_p(get_val("FULFILMENT_DB_PATH"), DEFAULT_FULFILMENT_DB_PATH)
        forecast_p = resolve_p(
            get_val("FORECAST_FILE_PATH"),
            PROCESSED_DATA_DIR / "forecast_payloads_step2_corrected.parquet",
        )
        master_p = resolve_p(
            get_val("MASTER_FILE_PATH"),
            PROCESSED_DATA_DIR / "df_master.parquet",
        )
        preorders_p = resolve_p(
            get_val("PREORDERS_FILE_PATH"),
            PROCESSED_DATA_DIR / "committed_preorders_step3.parquet",
        )
        model_p = resolve_p(
            get_val("MODEL_FILE_PATH"),
            OUTPUTS_DIR / "lgb_p10_p50_p90_corrected.joblib",
        )

        sim_gateways = (get_val("ENABLE_SIMULATION_GATEWAYS", "true") or "true").lower() in ("1", "true", "yes")

        cfg = cls(
            forecast_file=forecast_p,
            master_file=master_p,
            preorders_file=preorders_p,
            fulfilment_db_path=fulfil_db,
            coordinator_db_path=coord_db,
            env=env_mode,
            log_level=log_lvl,
            api_host=host,
            api_port=port,
            max_workers=max(1, workers),
            operator_api_key=api_key,
            authorized_operators=auth_operators,
            model_file=model_p,
            enable_simulation_gateways=sim_gateways,
        )
        cfg.ensure_directories()
        return cfg


def get_runtime_config() -> RuntimeConfig:
    """Returns singleton/standard RuntimeConfig initialized from environment."""
    return RuntimeConfig.load_from_env()

