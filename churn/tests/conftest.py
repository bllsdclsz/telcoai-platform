from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from churn.config import Settings
from churn.schema import CATEGORY_VALUES


def make_customers(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic customers in cleaned form, with churn driven by contract type and tenure."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(values, n) for col, values in CATEGORY_VALUES.items()})
    df.insert(0, "customerID", [f"C{i:05d}" for i in range(n)])
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
    logit = 1.5 * (df["Contract"] == "Month-to-month") - 0.05 * df["tenure"] + 0.5
    df["Churn"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse :func:`churn.data.clean` so tests exercise the real raw format."""
    raw = df.copy()
    raw["Churn"] = raw["Churn"].map({1: "Yes", 0: "No"})
    raw["TotalCharges"] = raw["TotalCharges"].astype(str)
    raw.loc[raw["tenure"] == 0, "TotalCharges"] = " "
    return raw


@pytest.fixture
def customers() -> pd.DataFrame:
    return make_customers()


@pytest.fixture
def raw_csv(tmp_path: Path, customers: pd.DataFrame) -> Path:
    path = tmp_path / "raw" / "telco_churn.csv"
    path.parent.mkdir(parents=True)
    to_raw(customers).to_csv(path, index=False)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated MLflow store (sqlite + local artifacts) inside the test's tmp dir."""
    s = Settings(
        data_dir=tmp_path,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        min_roc_auc=0.0,
    )
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    MlflowClient().create_experiment(
        s.experiment_name, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return s
