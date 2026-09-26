"""Feature engineering, packaged as sklearn transformers so it ships inside the model artifact."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from churn.schema import CATEGORICAL_COLUMNS, CATEGORY_VALUES, NUMERIC_COLUMNS

ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ENGINEERED_COLUMNS = ["avg_monthly_spend", "num_addons", "is_new_customer"]


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Derives business features from the raw customer record (stateless)."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["avg_monthly_spend"] = out["TotalCharges"] / np.maximum(out["tenure"], 1)
        out["num_addons"] = (out[ADDON_COLUMNS] == "Yes").sum(axis=1)
        out["is_new_customer"] = (out["tenure"] <= 6).astype(int)
        return out


def build_preprocessor() -> ColumnTransformer:
    encoder = OneHotEncoder(
        categories=[list(CATEGORY_VALUES[c]) for c in CATEGORICAL_COLUMNS],
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        [
            ("categorical", encoder, CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS + ENGINEERED_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )
