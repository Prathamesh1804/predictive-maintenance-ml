"""
Tests for data loading and preprocessing pipeline.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import (
    BASE_FEATURES, CATEGORICAL_FEATURES, NUMERIC_BASE_FEATURES,
    TARGET_COLUMN, LEAKAGE_COLUMNS, ID_COLUMNS,
)
from src.data_loader import validate_dataset
from src.feature_engineering import FeatureEngineer
from src.preprocessing import build_preprocessor, build_full_pipeline
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    """Create a minimal synthetic DataFrame that mimics the UCI dataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "UDI":                       range(1, n + 1),
        "Product ID":                [f"M{i:05d}" for i in range(n)],
        "Type":                      np.random.choice(["L", "M", "H"], n),
        "Air temperature [K]":       np.random.uniform(295, 305, n),
        "Process temperature [K]":   np.random.uniform(305, 315, n),
        "Rotational speed [rpm]":    np.random.randint(1168, 2886, n),
        "Torque [Nm]":               np.random.uniform(3.8, 76.6, n),
        "Tool wear [min]":           np.random.randint(0, 253, n),
        "Machine failure":           np.random.choice([0, 1], n, p=[0.966, 0.034]),
        "TWF":                       np.zeros(n, dtype=int),
        "HDF":                       np.zeros(n, dtype=int),
        "PWF":                       np.zeros(n, dtype=int),
        "OSF":                       np.zeros(n, dtype=int),
        "RNF":                       np.zeros(n, dtype=int),
    })
    return df


@pytest.fixture
def feature_df(synthetic_df):
    """Return just the base feature columns (no leakage, no IDs)."""
    return synthetic_df[BASE_FEATURES].copy()


@pytest.fixture
def labels(synthetic_df):
    return synthetic_df[TARGET_COLUMN].astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetValidation:

    def test_valid_dataset_passes(self, synthetic_df):
        report = validate_dataset(synthetic_df)
        assert report["valid"] is True

    def test_missing_column_fails(self, synthetic_df):
        broken = synthetic_df.drop(columns=["Machine failure"])
        report = validate_dataset(broken)
        assert report["valid"] is False
        assert "Machine failure" in str(report["missing_columns"])

    def test_row_count_reported(self, synthetic_df):
        report = validate_dataset(synthetic_df)
        assert report["row_count"] == len(synthetic_df)

    def test_target_distribution_reported(self, synthetic_df):
        report = validate_dataset(synthetic_df)
        assert 0 in report["target_distribution"]
        assert 1 in report["target_distribution"]

    def test_missing_values_detected(self, synthetic_df):
        df_mv = synthetic_df.copy()
        df_mv.loc[0, "Air temperature [K]"] = float("nan")
        report = validate_dataset(df_mv)
        assert "Air temperature [K]" in report["missing_values"]

    def test_too_few_rows_fails(self):
        tiny_df = pd.DataFrame({
            "UDI": [1], "Product ID": ["X"],
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [100],
            "Machine failure": [0],
            "TWF": [0], "HDF": [0], "PWF": [0], "OSF": [0], "RNF": [0],
        })
        report = validate_dataset(tiny_df)
        assert report["valid"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_columns_added(self, feature_df):
        fe = FeatureEngineer()
        out = fe.transform(feature_df)
        assert "Temperature difference [K]" in out.columns
        assert "Mechanical power [W]" in out.columns
        assert "Wear-torque interaction" in out.columns

    def test_temp_diff_correct(self, feature_df):
        fe = FeatureEngineer(add_temp_diff=True, add_power=False, add_wear_torque=False)
        out = fe.transform(feature_df)
        expected = feature_df["Process temperature [K]"] - feature_df["Air temperature [K]"]
        pd.testing.assert_series_equal(
            out["Temperature difference [K]"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_power_calculation(self, feature_df):
        fe = FeatureEngineer(add_temp_diff=False, add_power=True, add_wear_torque=False)
        out = fe.transform(feature_df)
        import math
        expected = feature_df["Torque [Nm]"] * feature_df["Rotational speed [rpm]"] * (2 * math.pi / 60)
        pd.testing.assert_series_equal(
            out["Mechanical power [W]"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_mutation_of_input(self, feature_df):
        original_cols = list(feature_df.columns)
        fe = FeatureEngineer()
        fe.transform(feature_df)
        assert list(feature_df.columns) == original_cols

    def test_fit_returns_self(self, feature_df):
        fe = FeatureEngineer()
        result = fe.fit(feature_df)
        assert result is fe

    def test_all_features_off(self, feature_df):
        fe = FeatureEngineer(add_temp_diff=False, add_power=False, add_wear_torque=False)
        out = fe.transform(feature_df)
        assert "Temperature difference [K]" not in out.columns
        assert "Mechanical power [W]" not in out.columns
        assert "Wear-torque interaction" not in out.columns


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingPipeline:

    def test_pipeline_builds(self):
        clf = LogisticRegression(max_iter=10)
        pipeline = build_full_pipeline(clf)
        assert pipeline is not None

    def test_pipeline_fits(self, feature_df, labels):
        clf = LogisticRegression(max_iter=200, random_state=42)
        pipeline = build_full_pipeline(clf)
        pipeline.fit(feature_df, labels)
        # Check pipeline has the expected steps
        assert "feature_engineer" in pipeline.named_steps
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_no_leakage_columns_used(self, synthetic_df):
        """Assert that leakage columns are NOT in the feature matrix passed to the pipeline."""
        leakage = set(LEAKAGE_COLUMNS)
        input_cols = set(BASE_FEATURES)
        assert leakage.isdisjoint(input_cols), (
            f"Leakage columns found in features: {leakage & input_cols}"
        )

    def test_output_shape(self, feature_df, labels):
        clf = LogisticRegression(max_iter=200, random_state=42)
        pipeline = build_full_pipeline(clf)
        pipeline.fit(feature_df, labels)
        preds = pipeline.predict(feature_df)
        assert len(preds) == len(feature_df)

    def test_predict_proba_available(self, feature_df, labels):
        clf = LogisticRegression(max_iter=200, random_state=42)
        pipeline = build_full_pipeline(clf)
        pipeline.fit(feature_df, labels)
        proba = pipeline.predict_proba(feature_df)
        assert proba.shape == (len(feature_df), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_unknown_type_handled(self, feature_df, labels):
        """OHE with handle_unknown='ignore' should not crash on unseen categories."""
        clf = LogisticRegression(max_iter=200, random_state=42)
        pipeline = build_full_pipeline(clf)
        pipeline.fit(feature_df, labels)
        # Create a row with unseen Type = "Z"
        row = feature_df.iloc[[0]].copy()
        row["Type"] = "Z"
        proba = pipeline.predict_proba(row)
        assert proba.shape == (1, 2)
