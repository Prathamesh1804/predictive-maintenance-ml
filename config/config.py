"""
Central configuration for the Predictive Maintenance Intelligence System.
All paths are relative to the project root. No hard-coded machine paths.
"""
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Accepted raw CSV filenames (user may rename the file)
RAW_CSV_CANDIDATES = [
    "ai4i2020.csv",
    "predictive_maintenance.csv",
    "ai4i_2020_predictive_maintenance_dataset.csv",
]

# ── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"

# ── Reports ──────────────────────────────────────────────────────────────────
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_FILE = REPORTS_DIR / "metrics.json"
COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"

# ── Features ─────────────────────────────────────────────────────────────────
# Input features used for training (no leakage columns)
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_BASE_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
BASE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_BASE_FEATURES

# Engineered features (derived, no leakage)
ENGINEERED_FEATURES = [
    "Temperature difference [K]",
    "Mechanical power [W]",
    "Wear-torque interaction",
]

# All features after engineering (used for training)
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# Target column
TARGET_COLUMN = "Machine failure"

# Leakage columns — NEVER used as inputs
LEAKAGE_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

# Non-predictive identifiers to drop
ID_COLUMNS = ["UDI", "Product ID"]

# ── Model training ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Models to train
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Gradient Boosting",
]

# Default decision threshold for binary classification
DEFAULT_THRESHOLD = 0.5

# ── Type label encoder mapping ────────────────────────────────────────────────
PRODUCT_TYPE_OPTIONS = ["L", "M", "H"]

# ── Typical operating ranges (for UI validation) ─────────────────────────────
FEATURE_RANGES = {
    "Air temperature [K]":      (295.0, 305.0),
    "Process temperature [K]":  (305.0, 315.0),
    "Rotational speed [rpm]":   (1168, 2886),
    "Torque [Nm]":              (3.8, 76.6),
    "Tool wear [min]":          (0, 253),
}

# ── Flask API ─────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
