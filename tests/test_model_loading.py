"""
Tests for cross-platform model path resolution, model registry integrity,
and artifact loading.
"""
import json
from pathlib import Path
import pytest
import pandas as pd

from config.config import MODELS_DIR, MODEL_REGISTRY_FILE, RAW_DIR, BASE_FEATURES
from src.predict import (
    resolve_model_path,
    list_available_models,
    get_best_model_name,
    PredictionService,
)
from src.data_loader import load_raw_data, validate_dataset


@pytest.fixture
def sample_input():
    return {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }


def test_model_registry_no_absolute_paths():
    """Verify model registry contains only portable relative paths, not machine-specific ones."""
    assert MODEL_REGISTRY_FILE.exists(), "model_registry.json must exist in models/"
    with open(MODEL_REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)

    for name, info in registry["models"].items():
        path_str = info["path"]
        assert not path_str.startswith("C:"), f"Model '{name}' has hardcoded Windows drive path: {path_str}"
        assert "\\" not in path_str, f"Model '{name}' contains Windows backslash: {path_str}"
        resolved = resolve_model_path(path_str)
        assert resolved.exists(), f"Model artifact for '{name}' does not exist at resolved path: {resolved}"


def test_resolve_model_path_relative():
    """Test resolution of relative POSIX path."""
    resolved = resolve_model_path("models/gradient_boosting.joblib")
    assert resolved.exists()
    assert resolved.name == "gradient_boosting.joblib"


def test_resolve_model_path_legacy_windows_path():
    """Test resolution of legacy absolute Windows path extracts filename correctly."""
    legacy_path = r"C:\Users\hp\Desktop\ML-mini-project\models\gradient_boosting.joblib"
    resolved = resolve_model_path(legacy_path)
    assert resolved.exists()
    assert resolved.name == "gradient_boosting.joblib"


def test_resolve_model_path_by_model_name():
    """Test resolution directly by model display name."""
    resolved = resolve_model_path("Random Forest")
    assert resolved.exists()
    assert resolved.name == "random_forest.joblib"


def test_list_available_models():
    """All 4 expected models should be discovered."""
    available = list_available_models()
    expected = ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"]
    for model in expected:
        assert model in available, f"Expected model '{model}' not found in available models: {available}"


def test_best_model_name():
    """Best model name should be one of the available models."""
    best = get_best_model_name()
    assert best is not None
    assert best in list_available_models()


@pytest.mark.parametrize("model_name", [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Gradient Boosting",
])
def test_all_models_load_and_predict(model_name, sample_input):
    """Test that every trained model artifact can be loaded and predict successfully."""
    service = PredictionService(model_name=model_name)
    assert service.is_ready, f"Failed to load model '{model_name}': {service.load_error}"
    assert service.model_name == model_name

    result = service.predict_single(sample_input)
    assert "predicted_label" in result
    assert result["predicted_label"] in (0, 1)
    assert "failure_probability" in result
    assert 0.0 <= result["failure_probability"] <= 1.0


def test_missing_model_reports_clean_error():
    """Loading a non-existent model should set a clear load_error without unhandled crash."""
    service = PredictionService(model_name="NonExistentModel")
    assert not service.is_ready
    assert service.load_error is not None
    with pytest.raises(RuntimeError, match="No trained model|was not found"):
        service.predict_single({"Type": "M"})


def test_dataset_loading_and_validation():
    """Verify that the official dataset loads and passes validation."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000, f"Expected 10,000 rows, got {len(df)}"
    report = validate_dataset(df)
    assert report["valid"] is True, f"Validation failed: {report['errors']}"
