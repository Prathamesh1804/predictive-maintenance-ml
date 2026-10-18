"""
Model explainability module.

Provides:
  - Feature importance for tree-based models (Random Forest, Gradient Boosting)
  - Permutation importance on held-out data (model-agnostic)
  - SHAP explanations (global + local) — gracefully disabled if unavailable
  - Logistic Regression coefficient magnitudes

Never fabricates importance values. If a method is unavailable, returns
a descriptive dict with an 'error' key so the UI can display a message.
"""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config.config import FIGURES_DIR
from src.preprocessing import get_feature_names
from src.utils import ensure_dirs, get_logger

logger = get_logger(__name__)


def _get_feature_names_safe(pipeline) -> list[str]:
    """Try to extract feature names from the fitted pipeline."""
    names = get_feature_names(pipeline)
    if not names:
        names = [f"feature_{i}" for i in range(100)]  # fallback
    return names


# ── Tree-based feature importance ─────────────────────────────────────────────

def get_tree_feature_importance(pipeline) -> dict:
    """
    Extract feature importance from a fitted tree-based model inside a pipeline.

    Returns
    -------
    dict with keys: 'importances' (list), 'feature_names' (list), 'method', or 'error'
    """
    try:
        clf = pipeline.named_steps["classifier"]
        # Handle CalibratedClassifierCV (e.g. SVM)
        if hasattr(clf, "estimator"):
            inner = clf.estimator
        elif hasattr(clf, "base_estimator"):
            inner = clf.base_estimator
        else:
            inner = clf

        if not hasattr(inner, "feature_importances_"):
            return {"error": f"'{type(inner).__name__}' does not provide feature_importances_."}

        importances = inner.feature_importances_
        feature_names = _get_feature_names_safe(pipeline)
        feature_names = feature_names[:len(importances)]

        idx = np.argsort(importances)[::-1]
        return {
            "method": "Tree Feature Importance (MDI)",
            "feature_names": [feature_names[i] for i in idx],
            "importances":   [float(importances[i]) for i in idx],
        }
    except Exception as exc:
        logger.warning("Tree importance failed: %s", exc)
        return {"error": str(exc)}


# ── Logistic Regression coefficients ─────────────────────────────────────────

def get_lr_coefficients(pipeline) -> dict:
    """
    Extract absolute coefficient magnitudes from a LogisticRegression model.

    Returns
    -------
    dict with keys: 'importances', 'feature_names', 'method', or 'error'
    """
    try:
        clf = pipeline.named_steps["classifier"]
        if not hasattr(clf, "coef_"):
            return {"error": "Not a LogisticRegression or model has no coef_."}

        coefs = np.abs(clf.coef_[0])
        feature_names = _get_feature_names_safe(pipeline)
        feature_names = feature_names[:len(coefs)]

        idx = np.argsort(coefs)[::-1]
        return {
            "method": "Logistic Regression |Coefficients|",
            "feature_names": [feature_names[i] for i in idx],
            "importances":   [float(coefs[i]) for i in idx],
        }
    except Exception as exc:
        logger.warning("LR coefficients failed: %s", exc)
        return {"error": str(exc)}


# ── Permutation importance (model-agnostic) ───────────────────────────────────

def get_permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Compute permutation importance on the held-out test set.

    Returns
    -------
    dict with keys: 'importances_mean', 'importances_std', 'feature_names', 'method', or 'error'
    """
    try:
        result = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="average_precision",
            n_jobs=-1,
        )
        feature_names = list(X_test.columns)
        idx = np.argsort(result.importances_mean)[::-1]
        return {
            "method":           "Permutation Importance (PR-AUC scoring)",
            "feature_names":    [feature_names[i] for i in idx],
            "importances_mean": [float(result.importances_mean[i]) for i in idx],
            "importances_std":  [float(result.importances_std[i]) for i in idx],
        }
    except Exception as exc:
        logger.warning("Permutation importance failed: %s", exc)
        return {"error": str(exc)}


# ── SHAP explanations ─────────────────────────────────────────────────────────

def get_shap_values(
    pipeline,
    X_sample: pd.DataFrame,
    max_samples: int = 200,
) -> dict:
    """
    Compute SHAP values using an appropriate explainer for the model type.

    Returns
    -------
    dict with keys: 'shap_values', 'feature_names', 'base_value', 'method', or 'error'
    """
    try:
        import shap
    except ImportError:
        return {"error": "SHAP is not installed. Run: pip install shap"}

    try:
        from src.preprocessing import build_preprocessor
        from src.feature_engineering import FeatureEngineer

        # Transform X through the pipeline steps up to (but not including) the classifier
        fe = pipeline.named_steps["feature_engineer"]
        prep = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]

        X_fe = fe.transform(X_sample.copy())
        X_transformed = prep.transform(X_fe)
        feature_names = _get_feature_names_safe(pipeline)

        sample = X_transformed[:min(max_samples, len(X_transformed))]

        # Choose explainer based on model type
        inner_clf = clf
        if hasattr(clf, "estimator"):
            inner_clf = clf.estimator

        model_type = type(inner_clf).__name__

        if model_type in ("RandomForestClassifier", "GradientBoostingClassifier",
                          "ExtraTreesClassifier"):
            explainer = shap.TreeExplainer(inner_clf)
            shap_vals = explainer.shap_values(sample)
            # For binary classification, shap_values is a list [class0, class1]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            base_value = float(explainer.expected_value[1]) if isinstance(
                explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
        else:
            # Kernel SHAP — slower but works for any model
            bg = shap.kmeans(sample, min(50, sample.shape[0]))
            explainer = shap.KernelExplainer(
                lambda x: clf.predict_proba(x)[:, 1], bg
            )
            shap_vals = explainer.shap_values(sample[:50])
            base_value = float(explainer.expected_value)

        return {
            "method":        f"SHAP ({type(explainer).__name__})",
            "shap_values":   shap_vals.tolist() if hasattr(shap_vals, "tolist") else shap_vals,
            "feature_names": feature_names[:shap_vals.shape[1]],
            "base_value":    base_value,
            "n_samples":     sample.shape[0],
        }
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return {"error": f"SHAP computation failed: {exc}"}


# ── Auto-select importance method ─────────────────────────────────────────────

def get_best_importance(pipeline, X_test: pd.DataFrame = None, y_test=None) -> dict:
    """
    Return the most informative importance method available for the given pipeline.
    Priority: tree importance → LR coefficients → permutation importance.
    """
    clf = pipeline.named_steps.get("classifier")
    inner = getattr(clf, "estimator", clf)

    if hasattr(inner, "feature_importances_"):
        result = get_tree_feature_importance(pipeline)
        if "error" not in result:
            return result

    if hasattr(inner, "coef_"):
        result = get_lr_coefficients(pipeline)
        if "error" not in result:
            return result

    if X_test is not None and y_test is not None:
        return get_permutation_importance(pipeline, X_test, y_test)

    return {"error": "No importance method available for this model type."}


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_feature_importance(importance_dict: dict, title: str = "Feature Importance",
                             top_n: int = 15) -> Optional[Path]:
    """Plot a horizontal bar chart for importance. Return save path or None."""
    if "error" in importance_dict:
        return None
    ensure_dirs(FIGURES_DIR)

    names = importance_dict.get("feature_names", [])[:top_n]
    values = importance_dict.get("importances", importance_dict.get("importances_mean", []))[:top_n]

    names = names[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5)))
    bars = ax.barh(names, values, color="#4E79A7", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
