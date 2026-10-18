"""
End-to-end pipeline runner.

Steps:
  1. Validate environment
  2. Download dataset (or print instructions)
  3. Load and validate dataset
  4. Train all models
  5. Generate evaluation plots
  6. Print summary report

Run from the project root:
    python scripts/run_pipeline.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from config.config import (
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    MODEL_REGISTRY_FILE, COMPARISON_FILE, METRICS_FILE
)
from src.utils import get_logger, ensure_dirs, save_json
from src.data_loader import find_raw_csv

logger = get_logger("run_pipeline")

SEP = "=" * 60
SEP2 = "-" * 60


def check_environment():
    """Verify required packages are importable."""
    required = [
        "pandas", "numpy", "sklearn", "matplotlib",
        "seaborn", "plotly", "joblib",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing packages: %s", missing)
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)
    logger.info("Environment check passed.")


def ensure_dataset():
    """Check for dataset; try auto-download if missing."""
    csv_path = find_raw_csv()
    if csv_path is not None:
        logger.info("Dataset found: %s", csv_path)
        return True

    logger.warning("Dataset not found. Attempting download...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "download_data.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        return False

    csv_path = find_raw_csv()
    return csv_path is not None


def main():
    print("\n" + SEP)
    print("  Predictive Maintenance Intelligence System")
    print("  Training Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP + "\n")

    # 1. Environment check
    logger.info("Step 1/5: Checking environment...")
    check_environment()

    # 2. Dataset check
    logger.info("Step 2/5: Checking dataset...")
    if not ensure_dataset():
        logger.error("Dataset unavailable. See instructions above.")
        sys.exit(1)

    # 3. Train models
    logger.info("Step 3/5: Training models...")
    ensure_dirs(MODELS_DIR, REPORTS_DIR, FIGURES_DIR)

    from src.train import train_all_models
    registry = train_all_models()

    # 4. Generate plots
    logger.info("Step 4/5: Generating evaluation plots...")
    from src.data_loader import load_and_validate
    from src.train import prepare_data
    from sklearn.model_selection import train_test_split
    from config.config import RANDOM_SEED, TEST_SIZE
    from src.evaluate import (
        evaluate_model, build_comparison_table, generate_all_plots
    )
    import joblib

    df, _ = load_and_validate()
    X, y = prepare_data(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    all_metrics = []
    for name, info in registry["models"].items():
        pipeline = joblib.load(info["path"])
        m = evaluate_model(pipeline, X_test, y_test, name)
        all_metrics.append(m)

    comparison_df = build_comparison_table(all_metrics)
    plot_paths = generate_all_plots(all_metrics, comparison_df)

    # Save full metrics JSON
    metrics_json = {
        "generated_at": datetime.now().isoformat(),
        "best_model": registry["best_model"],
        "models": {
            m["model_name"]: {k: v for k, v in m.items()
                              if k not in ("y_prob", "y_pred", "y_test",
                                           "confusion_matrix", "classification_report")}
            for m in all_metrics
        }
    }
    save_json(metrics_json, METRICS_FILE)

    # 5. Print summary
    logger.info("Step 5/5: Generating summary report...")
    print("\n" + SEP)
    print("  MODEL COMPARISON RESULTS")
    print(SEP)
    print(comparison_df.to_string(index=False))
    print("\n" + SEP2)
    print(f"  Best model (by PR-AUC): {registry['best_model']}")
    print(SEP2)
    print("\n  Output files:")
    print(f"  Models:     {MODELS_DIR.resolve()}")
    print(f"  Metrics:    {METRICS_FILE.resolve()}")
    print(f"  Comparison: {COMPARISON_FILE.resolve()}")
    print(f"  Figures:    {FIGURES_DIR.resolve()}")
    print("\n  Next steps:")
    print("  * Start dashboard: streamlit run app.py")
    print("  * Start API:       python api.py")
    print("  * Run tests:       pytest tests/ -v")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
