from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"

METADATA_DIR = PROJECT_ROOT / "metadata"

RAW_METADATA_DIR = METADATA_DIR / "raw"

PROCESSED_METADATA_DIR = METADATA_DIR / "processed"

LOG_DIR = METADATA_DIR / "logs"

REPORTS_DIR = METADATA_DIR / "reports"

LOGS_DIR = METADATA_DIR / "logs"

MASTER_METADATA_DIR = METADATA_DIR / "master"

