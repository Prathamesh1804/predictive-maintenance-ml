"""
Preprocessing pipeline builder.

Builds a scikit-learn ColumnTransformer that:
  1. One-hot encodes the 'Type' categorical column (handle_unknown='ignore').
  2. Applies StandardScaler to all numeric base + engineered features.

The pipeline is fitted ONLY on training data and saved alongside the model.
"""
import sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_BASE_FEATURES,
    ENGINEERED_FEATURES,
)
from src.feature_engineering import FeatureEngineer
from src.utils import get_logger

logger = get_logger(__name__)

# All numeric columns after feature engineering
ALL_NUMERIC = NUMERIC_BASE_FEATURES + ENGINEERED_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that handles the full feature set.

    Steps:
        - categorical: OHE 'Type' → [L, M, H] one-hot columns
        - numeric:     StandardScaler on numeric base + engineered features
    """
    # sparse_output=False was added in sklearn 1.2; use the appropriate parameter
    import sklearn
    from sklearn import __version__ as _sklearn_version
    _ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        # sklearn >= 1.2
        OneHotEncoder(sparse_output=False)
        _ohe_kwargs["sparse_output"] = False
    except TypeError:
        _ohe_kwargs["sparse"] = False

    categorical_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(**_ohe_kwargs)),
    ])

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, ALL_NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_pipeline(classifier) -> Pipeline:
    """
    Return an end-to-end Pipeline:
        FeatureEngineer → ColumnTransformer → Classifier

    Parameters
    ----------
    classifier : scikit-learn estimator
        A fitted or unfitted scikit-learn classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline(steps=[
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
    return pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract feature names after the preprocessor step of a fitted pipeline."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())
    except Exception as exc:
        logger.warning("Could not extract feature names: %s", exc)
        return []
