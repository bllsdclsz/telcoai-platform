import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient
from prefect.testing.utilities import prefect_test_harness

from churn.config import Settings
from churn.flows import drift_monitoring_flow, training_flow
from churn.schema import FEATURE_COLUMNS
from churn.simulate import shift
from churn.train import QualityGateError, file_md5

from .conftest import make_customers

FAST = {"n_estimators": 20}


@pytest.fixture(scope="module", autouse=True)
def prefect_backend() -> Iterator[None]:
    with prefect_test_harness():
        yield


@pytest.fixture
def flow_settings(settings: Settings, raw_csv: Path) -> Settings:
    # fetch_data looks for the dataset at settings.raw_data_path.
    assert raw_csv == settings.raw_data_path
    return settings


def test_flow_trains_and_promotes_to_staging(flow_settings: Settings) -> None:
    result = training_flow(promote_to="staging", params=FAST, settings=flow_settings)

    model = MlflowClient().get_registered_model(flow_settings.registered_model_name)
    assert {k: str(v) for k, v in model.aliases.items()} == {
        "dev": result.model_version,
        "staging": result.model_version,
    }
    run = MlflowClient().get_run(result.run_id)
    assert run.data.tags["data_md5"] == file_md5(flow_settings.raw_data_path)


def test_flow_stops_at_quality_gate(flow_settings: Settings) -> None:
    strict = flow_settings.model_copy(update={"min_roc_auc": 1.01})
    with pytest.raises(QualityGateError):
        training_flow(promote_to="staging", params=FAST, settings=strict)
    assert MlflowClient().search_registered_models() == []


def test_flow_refuses_automatic_prod_promotion(flow_settings: Settings) -> None:
    with pytest.raises(ValueError, match="manual approval"):
        training_flow(promote_to="prod", params=FAST, settings=flow_settings)


def test_file_md5_matches_dvc(tmp_path: Path) -> None:
    dvc_file = Path(__file__).parents[2] / "data" / "raw" / "telco_churn.csv.dvc"
    data = dvc_file.with_suffix("")
    if not data.exists():
        pytest.skip("dataset not pulled (dvc pull)")
    copy = shutil.copy(data, tmp_path / "copy.csv")
    assert f"md5: {file_md5(Path(copy))}" in dvc_file.read_text()


def log_predictions(settings: Settings, customers) -> None:
    assert settings.prediction_log_dir is not None
    settings.prediction_log_dir.mkdir(parents=True, exist_ok=True)
    rows = customers[FEATURE_COLUMNS].to_dict(orient="records")
    now = datetime.now(UTC).isoformat()
    lines = [json.dumps({**r, "scored_at": now}, default=int) for r in rows]
    (settings.prediction_log_dir / "predictions-today.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture
def monitor_settings(flow_settings: Settings, tmp_path: Path) -> Settings:
    return flow_settings.model_copy(update={"prediction_log_dir": tmp_path / "predictions"})


def test_monitoring_gives_no_verdict_on_too_little_traffic(monitor_settings: Settings) -> None:
    log_predictions(monitor_settings, make_customers(n=50, seed=1))
    assert drift_monitoring_flow(settings=monitor_settings) is None


def test_monitoring_without_drift_does_not_retrain(monitor_settings: Settings) -> None:
    log_predictions(monitor_settings, make_customers(n=300, seed=1))

    result = drift_monitoring_flow(settings=monitor_settings)

    assert result is not None and not result.dataset_drift
    assert MlflowClient().search_registered_models() == []


def test_monitoring_on_drift_logs_report_and_retrains(monitor_settings: Settings) -> None:
    log_predictions(monitor_settings, shift(make_customers(n=300, seed=1)))

    result = drift_monitoring_flow(settings=monitor_settings)

    assert result is not None and result.dataset_drift
    (run,) = mlflow.search_runs(
        experiment_names=[monitor_settings.monitoring_experiment_name], output_format="list"
    )
    assert run.data.tags["dataset_drift"] == "true"
    assert run.data.metrics["drift_share"] == result.drift_share
    artifacts = [a.path for a in MlflowClient().list_artifacts(run.info.run_id)]
    assert any(a.endswith(".html") for a in artifacts)
    # Retraining ran and the candidate landed in staging, never prod.
    aliases = MlflowClient().get_registered_model(monitor_settings.registered_model_name).aliases
    assert set(aliases) == {"dev", "staging"}


def test_flow_tunes_before_training(flow_settings: Settings) -> None:
    result = training_flow(promote_to=None, tune_trials=2, settings=flow_settings)

    runs = mlflow.search_runs(experiment_names=[flow_settings.experiment_name])
    assert (runs["tags.run_type"] == "tuning").sum() == 1
    assert result.model_version == "1"
