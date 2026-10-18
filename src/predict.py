"""
Shared prediction service.

Used by BOTH the Streamlit dashboard and the Flask REST API so that
inference always uses the same trained pipeline.

Supports:
  - Single-record prediction (dict → label + probability)
  - Batch prediction (DataFrame or list[dict] → augmented DataFrame)
  - Model switching at runtime
  - Graceful handling of missing model artifacts
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.config import (
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_THRESHOLD,
    MODELS_DIR,
    MODEL_REGISTRY_FILE,
    PRODUCT_TYPE_OPTIONS,
)
from src.utils import get_logger, load_json

logger = get_logger(__name__)

# ── Input field name normalisation map (API-friendly → DataFrame column names) ─
_FIELD_MAP = {
    # API camelCase / snake_case → exact dataset column names
    "type":                   "Type",
    "Type":                   "Type",
    "air_temperature_k":      "Air temperature [K]",
    "Air temperature [K]":    "Air temperature [K]",
    "process_temperature_k":  "Process temperature [K]",
    "Process temperature [K]": "Process temperature [K]",
    "rotational_speed_rpm":   "Rotational speed [rpm]",
    "Rotational speed [rpm]": "Rotational speed [rpm]",
    "torque_nm":              "Torque [Nm]",
    "Torque [Nm]":            "Torque [Nm]",
    "tool_wear_min":          "Tool wear [min]",
    "Tool wear [min]":        "Tool wear [min]",
}


def _normalise_input(raw: dict) -> dict:
    """Map API field names to DataFrame column names."""
    normalised = {}
    for k, v in raw.items():
        mapped_key = _FIELD_MAP.get(k, k)
        normalised[mapped_key] = v
    return normalised


def load_registry() -> Optional[dict]:
    """Load model registry JSON; return None if not found."""
    if MODEL_REGISTRY_FILE.exists():
        return load_json(MODEL_REGISTRY_FILE)
    return None


def list_available_models() -> list[str]:
    """Return list of model names that have saved artifacts."""
    registry = load_registry()
    if registry is None:
        return []
    return [
        name for name, info in registry["models"].items()
        if Path(info["path"]).exists()
    ]


def get_best_model_name() -> Optional[str]:
    """Return the name of the best model from the registry."""
    registry = load_registry()
    if registry is None:
        return None
    return registry.get("best_model")


class PredictionService:
    """
    Stateful prediction service that caches the loaded pipeline in memory.

    Usage
    -----
    service = PredictionService()
    result = service.predict_single({"Type": "M", "Air temperature [K]": 300.0, ...})
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name: Optional[str] = None
        self._pipeline = None
        self._threshold: float = DEFAULT_THRESHOLD
        self._load(model_name)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, model_name: Optional[str] = None):
        """Load a pipeline from disk. Falls back to best model if None."""
        registry = load_registry()
        if registry is None:
            logger.warning("No model registry found. Train models first.")
            return

        name = model_name or registry.get("best_model")
        if name is None:
            logger.warning("Registry exists but no best_model entry found.")
            return

        model_info = registry["models"].get(name)
        if model_info is None:
            logger.error("Model '%s' not in registry.", name)
            return

        path = Path(model_info["path"])
        if not path.exists():
            logger.error("Model artifact not found: %s", path)
            return

        try:
            self._pipeline = joblib.load(path)
            self._model_name = name
            logger.info("Loaded model: %s from %s", name, path)
        except Exception as exc:
            logger.error("Failed to load model artifact: %s", exc)

    def _check_ready(self):
        if self._pipeline is None:
            raise RuntimeError(
                "No trained model available. Run 'python scripts/run_pipeline.py' first."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 < value < 1:
            raise ValueError("Threshold must be in (0, 1).")
        self._threshold = value

    def switch_model(self, model_name: str):
        """Load a different model by name."""
        self._pipeline = None
        self._model_name = None
        self._load(model_name)

    def predict_single(self, raw_input: dict, threshold: Optional[float] = None) -> dict:
        """
        Predict for one machine observation.

        Parameters
        ----------
        raw_input : dict
            Keys may use API snake_case or exact DataFrame column names.
        threshold : float, optional
            Decision threshold. Defaults to self.threshold.

        Returns
        -------
        dict with keys: predicted_label, failure_probability, model_name, threshold
        """
        self._check_ready()
        th = threshold if threshold is not None else self._threshold

        normalised = _normalise_input(raw_input)
        missing = [f for f in BASE_FEATURES if f not in normalised]
        if missing:
            raise ValueError(f"Missing required input features: {missing}")

        row = pd.DataFrame([normalised])[BASE_FEATURES]

        prob = float(self._pipeline.predict_proba(row)[0, 1])
        label = int(prob >= th)

        return {
            "predicted_label":    label,
            "predicted_class":    "Failure" if label == 1 else "No Failure",
            "failure_probability": round(prob, 6),
            "failure_probability_pct": round(prob * 100, 2),
            "model_name":         self._model_name,
            "threshold":          th,
        }

    def predict_batch(
        self,
        data: pd.DataFrame | list[dict],
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Batch prediction.

        Parameters
        ----------
        data : pd.DataFrame or list[dict]
            Must contain the six BASE_FEATURES columns.
        threshold : float, optional

        Returns
        -------
        pd.DataFrame — original columns plus 'predicted_label',
                        'predicted_class', 'failure_probability',
                        'failure_probability_pct', 'model_name'.
        """
        self._check_ready()
        th = threshold if threshold is not None else self._threshold

        if isinstance(data, list):
            data = pd.DataFrame([_normalise_input(r) for r in data])

        missing = [f for f in BASE_FEATURES if f not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns in batch input: {missing}")

        X = data[BASE_FEATURES].copy()
        probs = self._pipeline.predict_proba(X)[:, 1]
        labels = (probs >= th).astype(int)

        result = data.copy()
        result["failure_probability"] = probs.round(6)
        result["failure_probability_pct"] = (probs * 100).round(2)
        result["predicted_label"] = labels
        result["predicted_class"] = np.where(labels == 1, "Failure", "No Failure")
        result["model_name"] = self._model_name
        result["threshold"] = th
        return result

    def validate_batch_columns(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Check that all required columns are present. Return (ok, missing_list)."""
        missing = [f for f in BASE_FEATURES if f not in df.columns]
        return len(missing) == 0, missing


# ── Module-level singleton (lazy) ─────────────────────────────────────────────
_default_service: Optional[PredictionService] = None


def get_prediction_service(model_name: Optional[str] = None) -> PredictionService:
    """Return (or create) the module-level singleton PredictionService."""
    global _default_service
    if _default_service is None or (model_name and model_name != _default_service.model_name):
        _default_service = PredictionService(model_name)
    return _default_service
