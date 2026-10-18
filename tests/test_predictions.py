"""
Tests for single and batch prediction, model loading, and invalid inputs.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config.config import BASE_FEATURES, TARGET_COLUMN
from src.predict import PredictionService
from src.preprocessing import build_full_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build and save a minimal trained pipeline for tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def temp_pipeline(tmp_path_factory):
    """Build a tiny fitted pipeline and return it (not saved to disk)."""
    tmp = tmp_path_factory.mktemp("models")
    np.random.seed(0)
    n = 300
    X = pd.DataFrame({
        "Type":                     np.random.choice(["L", "M", "H"], n),
        "Air temperature [K]":      np.random.uniform(295, 305, n),
        "Process temperature [K]":  np.random.uniform(305, 315, n),
        "Rotational speed [rpm]":   np.random.randint(1168, 2886, n),
        "Torque [Nm]":              np.random.uniform(3.8, 76.6, n),
        "Tool wear [min]":          np.random.randint(0, 253, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.966, 0.034]))
    # Ensure at least one positive class
    y.iloc[0] = 1

    clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    pipeline = build_full_pipeline(clf)
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def valid_input():
    return {
        "Type":                     "M",
        "Air temperature [K]":      300.0,
        "Process temperature [K]":  310.0,
        "Rotational speed [rpm]":   1500,
        "Torque [Nm]":              40.0,
        "Tool wear [min]":          100,
    }


@pytest.fixture
def api_style_input():
    """Input using API snake_case field names."""
    return {
        "type":                   "M",
        "air_temperature_k":      300.0,
        "process_temperature_k":  310.0,
        "rotational_speed_rpm":   1500,
        "torque_nm":              40.0,
        "tool_wear_min":          100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PredictionService tests using monkey-patched pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionService:

    def _make_service(self, pipeline):
        """Create a PredictionService with a pre-loaded pipeline (bypass registry)."""
        svc = PredictionService.__new__(PredictionService)
        svc._model_name = "Test Model"
        svc._pipeline = pipeline
        svc._threshold = 0.5
        return svc

    def test_predict_single_returns_dict(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        result = svc.predict_single(valid_input)
        assert isinstance(result, dict)

    def test_predict_single_keys(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        result = svc.predict_single(valid_input)
        assert "predicted_label" in result
        assert "predicted_class" in result
        assert "failure_probability" in result
        assert "failure_probability_pct" in result
        assert "model_name" in result
        assert "threshold" in result

    def test_predict_single_label_is_binary(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        result = svc.predict_single(valid_input)
        assert result["predicted_label"] in (0, 1)

    def test_probability_in_range(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        result = svc.predict_single(valid_input)
        assert 0.0 <= result["failure_probability"] <= 1.0
        assert 0.0 <= result["failure_probability_pct"] <= 100.0

    def test_predicted_class_consistent(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        result = svc.predict_single(valid_input)
        if result["predicted_label"] == 1:
            assert result["predicted_class"] == "Failure"
        else:
            assert result["predicted_class"] == "No Failure"

    def test_api_style_input_works(self, temp_pipeline, api_style_input):
        """API snake_case keys should be normalised correctly."""
        svc = self._make_service(temp_pipeline)
        result = svc.predict_single(api_style_input)
        assert "predicted_label" in result

    def test_missing_feature_raises(self, temp_pipeline):
        svc = self._make_service(temp_pipeline)
        incomplete = {
            "Type": "M",
            "Air temperature [K]": 300.0,
            # Missing other features
        }
        with pytest.raises(ValueError, match="Missing required input features"):
            svc.predict_single(incomplete)

    def test_threshold_respected(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        result_low = svc.predict_single(valid_input, threshold=0.01)
        result_high = svc.predict_single(valid_input, threshold=0.99)
        assert result_low["predicted_label"] == 1   # Very low threshold → always failure
        assert result_high["predicted_label"] == 0  # Very high threshold → always no failure

    def test_batch_prediction_returns_dataframe(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        batch = [valid_input.copy() for _ in range(5)]
        result = svc.predict_batch(batch)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_batch_has_required_output_columns(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        batch = pd.DataFrame([valid_input.copy() for _ in range(3)])
        result = svc.predict_batch(batch)
        for col in ["predicted_label", "predicted_class", "failure_probability", "model_name"]:
            assert col in result.columns

    def test_batch_invalid_missing_column(self, temp_pipeline):
        svc = self._make_service(temp_pipeline)
        bad_df = pd.DataFrame({
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            # Missing all other columns
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            svc.predict_batch(bad_df)

    def test_service_not_ready_raises(self):
        svc = PredictionService.__new__(PredictionService)
        svc._model_name = None
        svc._pipeline = None
        svc._threshold = 0.5
        with pytest.raises(RuntimeError, match="No trained model"):
            svc.predict_single({"Type": "M"})

    def test_validate_batch_columns_ok(self, temp_pipeline, valid_input):
        svc = self._make_service(temp_pipeline)
        df = pd.DataFrame([valid_input])
        ok, missing = svc.validate_batch_columns(df)
        assert ok is True
        assert missing == []

    def test_validate_batch_columns_missing(self, temp_pipeline):
        svc = self._make_service(temp_pipeline)
        df = pd.DataFrame({"Type": ["M"], "Air temperature [K]": [300.0]})
        ok, missing = svc.validate_batch_columns(df)
        assert ok is False
        assert len(missing) > 0

    def test_feature_consistency(self, temp_pipeline, valid_input):
        """Training and inference must use the same feature columns."""
        svc = self._make_service(temp_pipeline)
        # Both should use BASE_FEATURES (no leakage)
        r1 = svc.predict_single(valid_input)
        row = pd.DataFrame([valid_input])[BASE_FEATURES]
        prob_direct = temp_pipeline.predict_proba(row)[0, 1]
        assert abs(r1["failure_probability"] - prob_direct) < 1e-6
