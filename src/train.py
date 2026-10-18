"""
Model training module.

Workflow
--------
1. Load and validate the raw dataset.
2. Drop ID and leakage columns; select input features + target.
3. Stratified 80/20 train/test split (seed=42).
4. For each model: build full pipeline → fit on training data → save artifact.
5. Evaluate every model on the held-out test set.
6. Select the best model by PR-AUC (failure-detection priority).
7. Save evaluation metrics and a model registry.

Leakage prevention
------------------
- TWF, HDF, PWF, OSF, RNF, Machine failure are NEVER in the feature matrix.
- The ColumnTransformer and scalers are fit ONLY on X_train.
- SMOTE (if enabled) is applied ONLY inside the training fold.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC

from config.config import (
    ALL_FEATURES,
    BASE_FEATURES,
    LEAKAGE_COLUMNS,
    ID_COLUMNS,
    MODELS_DIR,
    MODEL_REGISTRY_FILE,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_and_validate
from src.evaluate import evaluate_model, build_comparison_table
from src.preprocessing import build_full_pipeline
from src.utils import ensure_dirs, get_logger, save_json

logger = get_logger(__name__)


# ── Model definitions ─────────────────────────────────────────────────────────

def _get_classifiers() -> dict:
    """Return a dict of {name: unfitted_classifier}."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "SVM": CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                probability=False,
            ),
            cv=3,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_SEED,
            subsample=0.8,
        ),
    }


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Drop leakage and ID columns, return (X, y).

    Parameters
    ----------
    df : pd.DataFrame  Raw dataset including all columns.

    Returns
    -------
    X : pd.DataFrame  Feature matrix (BASE_FEATURES only — leakage-free).
    y : pd.Series     Binary target (Machine failure).
    """
    drop_cols = [c for c in LEAKAGE_COLUMNS + ID_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN].astype(int)

    # Keep only expected base features (guard against unexpected columns)
    available = [f for f in BASE_FEATURES if f in X.columns]
    missing = [f for f in BASE_FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Required feature columns missing from dataset: {missing}")

    X = X[available]
    logger.info("Feature matrix: %d rows × %d cols | Target positives: %d (%.2f%%)",
                len(X), len(X.columns),
                int(y.sum()), float(y.mean() * 100))
    return X, y


# ── Main training function ────────────────────────────────────────────────────

def train_all_models(raw_dir: Path | None = None) -> dict:
    """
    Full training run. Returns a registry dict with paths to saved models
    and their evaluation metrics.
    """
    ensure_dirs(MODELS_DIR)

    # 1. Load data
    df, validation_report = load_and_validate(raw_dir) if raw_dir else load_and_validate()

    # 2. Prepare features / target
    X, y = prepare_data(df)

    # 3. Stratified split — fit only on train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    logger.info("Train positives: %d | Test positives: %d",
                int(y_train.sum()), int(y_test.sum()))

    classifiers = _get_classifiers()
    registry: dict = {
        "trained_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "models": {},
        "best_model": None,
    }

    all_metrics: list[dict] = []

    # 4. Train each model
    for name, clf in classifiers.items():
        logger.info("─── Training: %s ───", name)
        t0 = time.time()

        pipeline = build_full_pipeline(clf)
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info("  Fit time: %.1f s", elapsed)

        # 5. Evaluate on held-out test set
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        metrics["fit_time_s"] = round(elapsed, 2)
        all_metrics.append(metrics)

        logger.info(
            "  ROC-AUC=%.4f | PR-AUC=%.4f | Recall=%.4f | F1=%.4f",
            metrics["roc_auc"], metrics["pr_auc"],
            metrics["recall"], metrics["f1"],
        )

        # 6. Save model artifact
        safe_name = name.replace(" ", "_").lower()
        model_path = MODELS_DIR / f"{safe_name}.joblib"
        joblib.dump(pipeline, model_path)
        logger.info("  Saved: %s", model_path)

        registry["models"][name] = {
            "path": str(model_path),
            "metrics": metrics,
        }

    # 7. Select best model by PR-AUC
    best_name = max(
        registry["models"],
        key=lambda n: registry["models"][n]["metrics"]["pr_auc"],
    )
    registry["best_model"] = best_name
    logger.info("━━━ Best model: %s (PR-AUC=%.4f) ━━━",
                best_name,
                registry["models"][best_name]["metrics"]["pr_auc"])

    # 8. Save registry
    save_json(registry, MODEL_REGISTRY_FILE)
    logger.info("Registry saved: %s", MODEL_REGISTRY_FILE)

    # 9. Save comparison table
    from config.config import COMPARISON_FILE, REPORTS_DIR
    ensure_dirs(REPORTS_DIR)
    comparison_df = build_comparison_table(all_metrics)
    comparison_df.to_csv(COMPARISON_FILE, index=False)
    logger.info("Comparison table saved: %s", COMPARISON_FILE)

    return registry


if __name__ == "__main__":
    train_all_models()
