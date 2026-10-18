"""
Feature engineering module.

Derives physics-informed features from base input columns.
ALL transformations are implemented as a scikit-learn compatible
FunctionTransformer / custom transformer so they can be slotted into a Pipeline.

Derived features (no target leakage):
    - Temperature difference [K]  = Process temp − Air temp
    - Mechanical power [W]        = Torque × RPM × 2π / 60
    - Wear-torque interaction      = Tool wear × Torque

These features are computed AFTER the train/test split during training
and applied identically at inference time via the saved pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import NUMERIC_BASE_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that derives new columns from the
    base numeric features.

    Expected input: a pandas DataFrame (or array-like) whose column order
    matches NUMERIC_BASE_FEATURES:
        [Air temperature [K], Process temperature [K],
         Rotational speed [rpm], Torque [Nm], Tool wear [min]]

    After encoding the categorical 'Type' column is handled separately;
    this transformer operates on the numeric block only.
    """

    def __init__(self, add_temp_diff: bool = True,
                 add_power: bool = True,
                 add_wear_torque: bool = True):
        self.add_temp_diff = add_temp_diff
        self.add_power = add_power
        self.add_wear_torque = add_wear_torque

    def fit(self, X, y=None):  # noqa: N803
        return self  # stateless — nothing to fit

    def transform(self, X, y=None):  # noqa: N803
        """Append derived columns and return the enriched DataFrame."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=NUMERIC_BASE_FEATURES)
        else:
            X = X.copy()

        air_temp = X["Air temperature [K]"]
        proc_temp = X["Process temperature [K]"]
        rpm = X["Rotational speed [rpm]"]
        torque = X["Torque [Nm]"]
        wear = X["Tool wear [min]"]

        if self.add_temp_diff:
            X["Temperature difference [K]"] = proc_temp - air_temp

        if self.add_power:
            # Mechanical power P = T × ω = T × (2π × n / 60)
            X["Mechanical power [W]"] = torque * rpm * (2 * np.pi / 60)

        if self.add_wear_torque:
            X["Wear-torque interaction"] = wear * torque

        return X

    def get_feature_names_out(self, input_features=None):
        names = list(NUMERIC_BASE_FEATURES)
        if self.add_temp_diff:
            names.append("Temperature difference [K]")
        if self.add_power:
            names.append("Mechanical power [W]")
        if self.add_wear_torque:
            names.append("Wear-torque interaction")
        return np.array(names)


def engineer_features(df: pd.DataFrame,
                      add_temp_diff: bool = True,
                      add_power: bool = True,
                      add_wear_torque: bool = True) -> pd.DataFrame:
    """
    Convenience function: apply feature engineering to a raw DataFrame.

    Operates on a copy; does NOT modify the original.
    Should be called ONLY on the training set (or applied identically
    via the saved pipeline at inference time).
    """
    fe = FeatureEngineer(
        add_temp_diff=add_temp_diff,
        add_power=add_power,
        add_wear_torque=add_wear_torque,
    )
    return fe.transform(df)
