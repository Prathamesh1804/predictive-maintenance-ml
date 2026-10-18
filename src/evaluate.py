"""
Model evaluation utilities.

All metrics are computed on the held-out test set ONLY.
Never reports training-set metrics as test metrics.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.config import FIGURES_DIR, DEFAULT_THRESHOLD
from src.utils import ensure_dirs, get_logger

logger = get_logger(__name__)

PALETTE = {
    "Logistic Regression": "#4E79A7",
    "Random Forest":       "#F28E2B",
    "SVM":                 "#E15759",
    "Gradient Boosting":   "#76B7B2",
}


# ── Core metric computation ───────────────────────────────────────────────────

def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Compute all evaluation metrics for a fitted pipeline.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    X_test     : feature DataFrame (raw, before any preprocessing)
    y_test     : true binary labels
    model_name : display name for the model
    threshold  : decision threshold for binary classification

    Returns
    -------
    dict with all key metrics.
    """
    # Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    # Threshold-based predictions
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "model_name":    model_name,
        "threshold":     threshold,
        "accuracy":      float(report["accuracy"]),
        "precision":     float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":        float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":            float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":       float(roc_auc_score(y_test, y_prob)),
        "pr_auc":        float(average_precision_score(y_test, y_prob)),
        "true_positives":  int(tp),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
        "y_test": y_test.tolist(),
    }
    return metrics


def build_comparison_table(all_metrics: list[dict]) -> pd.DataFrame:
    """Convert list of metric dicts into a tidy comparison DataFrame."""
    rows = []
    for m in all_metrics:
        rows.append({
            "Model":     m["model_name"],
            "Accuracy":  round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"], 4),
            "F1-Score":  round(m["f1"], 4),
            "ROC-AUC":   round(m["roc_auc"], 4),
            "PR-AUC":    round(m["pr_auc"], 4),
            "TP":        m["true_positives"],
            "TN":        m["true_negatives"],
            "FP":        m["false_positives"],
            "FN":        m["false_negatives"],
        })
    return pd.DataFrame(rows)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _fig_path(filename: str) -> Path:
    ensure_dirs(FIGURES_DIR)
    return FIGURES_DIR / filename


def plot_roc_curves(all_metrics: list[dict]) -> Path:
    """Plot and save ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in all_metrics:
        name = m["model_name"]
        fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
        color = PALETTE.get(name, None)
        ax.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})",
                color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_path("roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr_curves(all_metrics: list[dict]) -> Path:
    """Plot and save Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in all_metrics:
        name = m["model_name"]
        prec, rec, _ = precision_recall_curve(m["y_test"], m["y_prob"])
        color = PALETTE.get(name, None)
        ax.plot(rec, prec, label=f"{name} (PR-AUC={m['pr_auc']:.3f})",
                color=color, linewidth=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_path("pr_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrices(all_metrics: list[dict]) -> Path:
    """Plot and save confusion matrices for all models in a grid."""
    n = len(all_metrics)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes_flat = axes.flatten() if n > 1 else [axes]
    for i, m in enumerate(all_metrics):
        cm = np.array(m["confusion_matrix"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Failure", "Failure"])
        disp.plot(ax=axes_flat[i], colorbar=False, cmap="Blues")
        axes_flat[i].set_title(m["model_name"], fontsize=12, fontweight="bold")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = _fig_path("confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(comparison_df: pd.DataFrame) -> Path:
    """Bar chart comparing key metrics across models."""
    metrics_to_plot = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    melted = comparison_df.melt(
        id_vars="Model", value_vars=metrics_to_plot,
        var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = list(PALETTE.values())
    models = comparison_df["Model"].tolist()
    palette = {m: PALETTE.get(m, colors[i % len(colors)])
               for i, m in enumerate(models)}
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model",
                palette=palette, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Metric Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _fig_path("metric_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(all_metrics: list[dict], comparison_df: pd.DataFrame) -> dict:
    """Generate and save all evaluation plots. Return dict of plot paths."""
    paths = {}
    paths["roc"] = str(plot_roc_curves(all_metrics))
    paths["pr"] = str(plot_pr_curves(all_metrics))
    paths["cm"] = str(plot_confusion_matrices(all_metrics))
    paths["comparison"] = str(plot_metric_comparison(comparison_df))
    logger.info("Evaluation plots saved to: %s", FIGURES_DIR)
    return paths
