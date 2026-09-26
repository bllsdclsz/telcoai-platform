import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from mlflow import MlflowClient
from prefect.testing.utilities import prefect_test_harness

from churn.config import Settings
from churn.flows import training_flow
from churn.train import QualityGateError, file_md5

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
