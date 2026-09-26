import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from churn.config import Settings
from churn.schema import FEATURE_COLUMNS
from churn.serve.app import CustomerFeatures, LoadedModel, create_app
from churn.train import build_pipeline

from .conftest import make_customers


@pytest.fixture(scope="module")
def log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("predictions")


@pytest.fixture(scope="module")
def client(log_dir: Path) -> Iterator[TestClient]:
    df = make_customers()
    model = build_pipeline({"n_estimators": 20}, seed=0).fit(df[FEATURE_COLUMNS], df["Churn"])
    app = create_app(
        loader=lambda: LoadedModel(model=model, version="7"),
        settings=Settings(prediction_log_dir=log_dir),
    )
    with TestClient(app) as c:
        yield c


def customer_payload(**overrides: Any) -> dict[str, Any]:
    row = make_customers(n=1)[FEATURE_COLUMNS].iloc[0]
    payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    return {**payload, **overrides}


def test_health_reports_model_version(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_version": "7"}


def test_predict_returns_probability_per_customer(client: TestClient) -> None:
    body = {"customers": [customer_payload(), customer_payload(Contract="Two year")]}
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "7"
    assert len(data["predictions"]) == 2
    for p in data["predictions"]:
        assert 0 <= p["churn_probability"] <= 1
        assert p["will_churn"] == (p["churn_probability"] >= 0.5)


@pytest.mark.parametrize(
    "overrides",
    [{"Contract": "Lifetime"}, {"tenure": -1}, {"MonthlyCharges": -3.0}, {"SeniorCitizen": 2}],
)
def test_predict_rejects_invalid_input(client: TestClient, overrides: dict[str, Any]) -> None:
    resp = client.post("/predict", json={"customers": [customer_payload(**overrides)]})
    assert resp.status_code == 422


def test_predict_rejects_empty_batch(client: TestClient) -> None:
    assert client.post("/predict", json={"customers": []}).status_code == 422


def test_request_schema_matches_training_features() -> None:
    assert set(CustomerFeatures.model_fields) == set(FEATURE_COLUMNS)


def metric_value(text: str, name: str, **labels: str) -> float:
    """Value of one sample in Prometheus text exposition format."""
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    prefix = f"{name}{{{label_str}}} " if labels else f"{name} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.split()[-1])
    raise AssertionError(f"{prefix!r} not found")


def test_metrics_count_requests_and_predictions(client: TestClient) -> None:
    before = client.get("/metrics").text
    scored = metric_value(before, "churn_prediction_probability_count")
    ok = metric_value(
        before, "churn_http_requests_total", method="POST", route="/predict", status="200"
    )

    client.post("/predict", json={"customers": [customer_payload(), customer_payload()]})
    client.post("/predict", json={"customers": [customer_payload(tenure=-1)]})

    after = client.get("/metrics").text
    assert metric_value(after, "churn_prediction_probability_count") == scored + 2
    assert (
        metric_value(
            after, "churn_http_requests_total", method="POST", route="/predict", status="200"
        )
        == ok + 1
    )
    assert (
        metric_value(
            after, "churn_http_requests_total", method="POST", route="/predict", status="422"
        )
        >= 1
    )
    assert metric_value(after, "churn_model_info", alias="prod", model_version="7") == 1


def test_predictions_are_logged_for_drift_monitoring(client: TestClient, log_dir: Path) -> None:
    client.post("/predict", json={"customers": [customer_payload(Contract="One year")]})

    (log_file,) = log_dir.glob("predictions-*.jsonl")
    last = json.loads(log_file.read_text().splitlines()[-1])
    assert last["Contract"] == "One year"
    assert last["model_version"] == "7"
    assert 0 <= last["churn_probability"] <= 1
    assert set(FEATURE_COLUMNS) <= set(last)
