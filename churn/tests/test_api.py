from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from churn.schema import FEATURE_COLUMNS
from churn.serve.app import CustomerFeatures, LoadedModel, create_app
from churn.train import build_pipeline

from .conftest import make_customers


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    df = make_customers()
    model = build_pipeline({"n_estimators": 20}, seed=0).fit(df[FEATURE_COLUMNS], df["Churn"])
    app = create_app(loader=lambda: LoadedModel(model=model, version="7"))
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
