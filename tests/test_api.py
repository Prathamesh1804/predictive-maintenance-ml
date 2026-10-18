"""
Tests for the Flask REST API.
"""
import sys
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import app
from src.predict import PredictionService
from src.preprocessing import build_full_pipeline
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_pipeline():
    """Build and fit a minimal LR pipeline for API testing."""
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
    y.iloc[0] = 1  # ensure positive class exists
    clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    pipeline = build_full_pipeline(clf)
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="module")
def client(trained_pipeline):
    """Return a Flask test client with the model monkey-patched into the service."""
    import src.predict as predict_module

    # Inject a ready service without needing disk artifacts
    svc = PredictionService.__new__(PredictionService)
    svc._model_name = "Logistic Regression"
    svc._pipeline = trained_pipeline
    svc._threshold = 0.5
    predict_module._default_service = svc

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def valid_payload():
    return {
        "type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# /health tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["model_loaded"] is True


# ─────────────────────────────────────────────────────────────────────────────
# /model-info tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_model_info_has_input_features(self, client):
        resp = client.get("/model-info")
        data = resp.get_json()
        assert "input_features" in data
        assert "active_model" in data


# ─────────────────────────────────────────────────────────────────────────────
# /predict tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:

    def test_predict_returns_200(self, client, valid_payload):
        resp = client.post(
            "/predict",
            data=json.dumps(valid_payload),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_predict_response_structure(self, client, valid_payload):
        resp = client.post(
            "/predict",
            data=json.dumps(valid_payload),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "predicted_label" in data
        assert "predicted_class" in data
        assert "failure_probability" in data
        assert "model_name" in data

    def test_predict_label_binary(self, client, valid_payload):
        resp = client.post(
            "/predict",
            data=json.dumps(valid_payload),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["predicted_label"] in (0, 1)

    def test_predict_probability_range(self, client, valid_payload):
        resp = client.post(
            "/predict",
            data=json.dumps(valid_payload),
            content_type="application/json",
        )
        data = resp.get_json()
        assert 0.0 <= data["failure_probability"] <= 1.0

    def test_predict_missing_field_returns_400(self, client):
        payload = {
            "type": "M",
            "air_temperature_k": 300.0,
            # Missing other fields
        }
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_predict_invalid_type_returns_400(self, client, valid_payload):
        payload = {**valid_payload, "type": "Z"}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_predict_non_numeric_temperature_returns_400(self, client, valid_payload):
        payload = {**valid_payload, "air_temperature_k": "hot"}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_predict_wrong_content_type_returns_400(self, client, valid_payload):
        resp = client.post(
            "/predict",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_predict_out_of_range_value_returns_400(self, client, valid_payload):
        payload = {**valid_payload, "air_temperature_k": 9999}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_predict_custom_threshold(self, client, valid_payload):
        payload = {**valid_payload, "threshold": 0.01}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["predicted_label"] == 1  # Very low threshold → failure

    def test_predict_invalid_threshold_returns_400(self, client, valid_payload):
        payload = {**valid_payload, "threshold": 1.5}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# /predict-batch tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:

    def test_batch_predict_returns_200(self, client, valid_payload):
        batch = [valid_payload.copy(), valid_payload.copy()]
        resp = client.post(
            "/predict-batch",
            data=json.dumps(batch),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_batch_predict_response_structure(self, client, valid_payload):
        batch = [valid_payload.copy(), valid_payload.copy()]
        resp = client.post(
            "/predict-batch",
            data=json.dumps(batch),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "count" in data
        assert "predictions" in data
        assert data["count"] == 2

    def test_batch_predict_empty_array_returns_400(self, client):
        resp = client.post(
            "/predict-batch",
            data=json.dumps([]),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_predict_non_array_returns_400(self, client, valid_payload):
        resp = client.post(
            "/predict-batch",
            data=json.dumps(valid_payload),  # dict not list
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_each_record_has_prediction(self, client, valid_payload):
        batch = [valid_payload.copy() for _ in range(3)]
        resp = client.post(
            "/predict-batch",
            data=json.dumps(batch),
            content_type="application/json",
        )
        data = resp.get_json()
        for pred in data["predictions"]:
            assert "predicted_label" in pred
            assert "failure_probability" in pred


# ─────────────────────────────────────────────────────────────────────────────
# 404 / 405 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandlers:

    def test_404_returns_json(self, client):
        resp = client.get("/nonexistent-endpoint")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data

    def test_health_post_returns_405(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405
