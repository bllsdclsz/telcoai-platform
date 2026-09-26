from pathlib import Path

import pytest
from mlflow import MlflowClient

from churn.config import Settings
from churn.registry import promote
from churn.train import train

FAST = {"n_estimators": 20}


def aliases(settings: Settings) -> dict[str, str]:
    model = MlflowClient().get_registered_model(settings.registered_model_name)
    return {k: str(v) for k, v in model.aliases.items()}


def test_promote_and_rollback(settings: Settings, raw_csv: Path) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    assert promote(settings, "dev", "prod") == "1"

    train(settings, params=FAST, data_path=raw_csv)
    assert promote(settings, "dev", "prod") == "2"
    assert aliases(settings) == {"dev": "2", "prod": "2", "prod-previous": "1"}

    assert promote(settings, "prod-previous", "prod") == "1"
    assert aliases(settings)["prod"] == "1"


def test_promote_rejects_unknown_target(settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown target"):
        promote(settings, "dev", "production")
