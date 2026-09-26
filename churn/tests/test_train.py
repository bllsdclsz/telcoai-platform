from pathlib import Path

import mlflow.sklearn
import pytest
from mlflow import MlflowClient

from churn.config import Settings
from churn.schema import FEATURE_COLUMNS
from churn.train import QualityGateError, train

FAST = {"n_estimators": 20}


def test_train_registers_model_under_dev_alias(settings: Settings, raw_csv: Path) -> None:
    result = train(settings, params=FAST, data_path=raw_csv)

    assert result.metrics["roc_auc"] > 0.6  # synthetic data has a real signal
    mv = MlflowClient().get_model_version_by_alias(settings.registered_model_name, "dev")
    assert str(mv.version) == result.model_version
    assert mv.run_id == result.run_id


def test_registered_model_round_trips(settings: Settings, raw_csv: Path, customers) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    model = mlflow.sklearn.load_model(f"models:/{settings.registered_model_name}@dev")
    proba = model.predict_proba(customers[FEATURE_COLUMNS].head(3))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_quality_gate_blocks_registration(settings: Settings, raw_csv: Path) -> None:
    strict = settings.model_copy(update={"min_roc_auc": 1.01})
    with pytest.raises(QualityGateError):
        train(strict, params=FAST, data_path=raw_csv)

    runs = mlflow.search_runs(experiment_names=[settings.experiment_name])
    assert runs["tags.quality_gate"].tolist() == ["failed"]
    assert MlflowClient().search_registered_models() == []
