"""
Flask REST API for Predictive Maintenance Intelligence System.

Endpoints
---------
GET  /health          — service status and model availability
GET  /model-info      — active model name and input schema
POST /predict         — single-record failure prediction
POST /predict-batch   — batch prediction from JSON array

Run
---
    python api.py

Test with PowerShell
--------------------
    # Health check
    Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET

    # Single prediction
    $body = @{
        type = "M"
        air_temperature_k = 300.1
        process_temperature_k = 310.5
        rotational_speed_rpm = 1500
        torque_nm = 40.0
        tool_wear_min = 100
    } | ConvertTo-Json
    Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST `
        -ContentType "application/json" -Body $body
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Project root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request
from config.config import API_HOST, API_PORT, API_DEBUG, BASE_FEATURES, DEFAULT_THRESHOLD
from src.predict import get_prediction_service, list_available_models, get_best_model_name
from src.utils import get_logger

logger = get_logger("api")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def _error(message: str, status: int = 400) -> tuple:
    """Return a standardised JSON error response."""
    return jsonify({"error": message, "status": status}), status


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    service = get_prediction_service()
    return jsonify({
        "status": "ok",
        "service": "Predictive Maintenance Intelligence System",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": service.is_ready,
        "active_model": service.model_name,
        "available_models": list_available_models(),
    })


# ── Model info ────────────────────────────────────────────────────────────────

@app.route("/model-info", methods=["GET"])
def model_info():
    service = get_prediction_service()
    if not service.is_ready:
        return _error(
            "No trained model available. Run 'python scripts/run_pipeline.py' first.", 503
        )
    return jsonify({
        "active_model": service.model_name,
        "available_models": list_available_models(),
        "input_features": {
            "type": "string — one of L, M, H",
            "air_temperature_k": "float — Air temperature in Kelvin (e.g. 300.0)",
            "process_temperature_k": "float — Process temperature in Kelvin (e.g. 310.0)",
            "rotational_speed_rpm": "int — Rotational speed in rpm (e.g. 1500)",
            "torque_nm": "float — Torque in Nm (e.g. 40.0)",
            "tool_wear_min": "int — Tool wear in minutes (e.g. 100)",
        },
        "output_fields": [
            "predicted_label", "predicted_class",
            "failure_probability", "failure_probability_pct",
            "model_name", "threshold",
        ],
    })


# ── Single prediction ─────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return _error("Request Content-Type must be application/json.")

    data = request.get_json(silent=True)
    if data is None:
        return _error("Could not parse JSON body.")

    # Validate required fields
    required = [
        "type", "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return _error(f"Missing required fields: {missing}")

    # Type validation
    if data.get("type") not in ("L", "M", "H"):
        return _error("Field 'type' must be one of: L, M, H")

    numeric_fields = {
        "air_temperature_k": (270, 340),
        "process_temperature_k": (270, 350),
        "rotational_speed_rpm": (0, 5000),
        "torque_nm": (0, 200),
        "tool_wear_min": (0, 500),
    }
    for field, (lo, hi) in numeric_fields.items():
        try:
            val = float(data[field])
        except (TypeError, ValueError):
            return _error(f"Field '{field}' must be numeric.")
        if not (lo <= val <= hi):
            return _error(
                f"Field '{field}' value {val} is outside plausible range [{lo}, {hi}]."
            )

    # Threshold (optional override)
    threshold = DEFAULT_THRESHOLD
    if "threshold" in data:
        try:
            threshold = float(data["threshold"])
            if not 0 < threshold < 1:
                return _error("Field 'threshold' must be between 0 and 1 (exclusive).")
        except (TypeError, ValueError):
            return _error("Field 'threshold' must be numeric.")

    service = get_prediction_service()
    if not service.is_ready:
        return _error(
            "No trained model available. Run 'python scripts/run_pipeline.py' first.", 503
        )

    try:
        result = service.predict_single(data, threshold=threshold)
        return jsonify(result)
    except ValueError as exc:
        return _error(str(exc))
    except Exception:
        logger.exception("Unhandled error in /predict")
        return _error("Internal prediction error. Check server logs.", 500)


# ── Batch prediction ──────────────────────────────────────────────────────────

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    if not request.is_json:
        return _error("Request Content-Type must be application/json.")

    data = request.get_json(silent=True)
    if data is None:
        return _error("Could not parse JSON body.")

    if not isinstance(data, list):
        return _error("Request body must be a JSON array of observation objects.")

    if len(data) == 0:
        return _error("Input array is empty.")

    if len(data) > 10_000:
        return _error("Batch size exceeds maximum (10,000 records).")

    service = get_prediction_service()
    if not service.is_ready:
        return _error(
            "No trained model available. Run 'python scripts/run_pipeline.py' first.", 503
        )

    try:
        import pandas as pd
        result_df = service.predict_batch(data)
        records = result_df.to_dict(orient="records")
        return jsonify({
            "count": len(records),
            "model_name": service.model_name,
            "predictions": records,
        })
    except ValueError as exc:
        return _error(str(exc))
    except Exception:
        logger.exception("Unhandled error in /predict-batch")
        return _error("Internal batch prediction error. Check server logs.", 500)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return _error(f"Endpoint not found. Available: /health /model-info /predict /predict-batch", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return _error(f"Method not allowed.", 405)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Flask API on http://%s:%d", API_HOST, API_PORT)
    service = get_prediction_service()
    if not service.is_ready:
        logger.warning(
            "No trained model found. API will start but /predict will return 503. "
            "Run: python scripts/run_pipeline.py"
        )
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
