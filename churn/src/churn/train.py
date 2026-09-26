"""Train a churn model, track it in MLflow and register it if it passes the quality gate."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.config import Settings
from churn.data import load
from churn.features import FeatureEngineer, build_preprocessor
from churn.schema import FEATURE_COLUMNS, TARGET

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

# Non-builtin classes skops must be allowed to deserialize from the model artifact.
SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "churn.features.FeatureEngineer",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]


class QualityGateError(RuntimeError):
    """Raised when a trained model is not good enough to be registered."""


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_version: str
    metrics: dict[str, float]


def build_pipeline(params: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("classifier", LGBMClassifier(**params, random_state=seed, verbose=-1)),
        ]
    )


def evaluate(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
    }


def file_md5(path: Path) -> str:
    """Same hash DVC records in ``<file>.dvc``, linking each run to an exact data version."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(
    settings: Settings,
    params: dict[str, Any] | None = None,
    data_path: Path | None = None,
) -> TrainResult:
    params = {**DEFAULT_PARAMS, **(params or {})}
    data_path = data_path or settings.raw_data_path
    df = load(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET],
        test_size=settings.test_size,
        stratify=df[TARGET],
        random_state=settings.random_seed,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params({**params, "seed": settings.random_seed, "test_size": settings.test_size})
        mlflow.set_tag("data_md5", file_md5(data_path))
        mlflow.log_input(
            from_pandas(df, source=str(data_path), name="telco_churn", targets=TARGET),
            context="training",
        )

        model = build_pipeline(params, settings.random_seed).fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        if metrics["roc_auc"] < settings.min_roc_auc:
            mlflow.set_tag("quality_gate", "failed")
            raise QualityGateError(
                f"roc_auc {metrics['roc_auc']:.3f} < required {settings.min_roc_auc:.3f}"
            )
        mlflow.set_tag("quality_gate", "passed")

        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_test.head(5),
            registered_model_name=settings.registered_model_name,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        version = str(info.registered_model_version)
        MlflowClient().set_registered_model_alias(
            settings.registered_model_name, settings.register_alias, version
        )

    return TrainResult(run_id=run.info.run_id, model_version=version, metrics=metrics)
