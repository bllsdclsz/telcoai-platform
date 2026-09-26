"""Real-time churn scoring API. Serves the registered model version behind an alias."""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request, Response
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from churn.config import Settings
from churn.schema import (
    FEATURE_COLUMNS,
    Contract,
    Gender,
    InternetAddon,
    InternetService,
    MultipleLines,
    PaymentMethod,
    YesNo,
)
from churn.serve.monitoring import Metrics, PredictionLog

DECISION_THRESHOLD = 0.5


class CustomerFeatures(BaseModel):
    gender: Gender
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


@dataclass(frozen=True)
class LoadedModel:
    model: Any  # anything with sklearn's predict_proba
    version: str


def load_registered_model(settings: Settings | None = None) -> LoadedModel:
    settings = settings or Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name, alias = settings.registered_model_name, settings.serving_alias
    version = MlflowClient().get_model_version_by_alias(name, alias).version
    return LoadedModel(mlflow.sklearn.load_model(f"models:/{name}@{alias}"), str(version))


def create_app(
    loader: Callable[[], LoadedModel] = load_registered_model,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or Settings()
    metrics = Metrics()
    prediction_log = (
        PredictionLog(settings.prediction_log_dir) if settings.prediction_log_dir else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.loaded = loader()
        metrics.model_info.labels(app.state.loaded.version, settings.serving_alias).set(1)
        yield

    app = FastAPI(title="Telco Churn API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def record_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # Route template, not the raw path, keeps label cardinality bounded.
            route = getattr(request.scope.get("route"), "path", "unmatched")
            metrics.latency.labels(route, request.method).observe(time.perf_counter() - start)
            metrics.requests.labels(route, request.method, str(status)).inc()

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "model_version": request.app.state.loaded.version}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        loaded: LoadedModel = request.app.state.loaded
        customers = [c.model_dump() for c in body.customers]
        X = pd.DataFrame(customers)[FEATURE_COLUMNS]
        proba = loaded.model.predict_proba(X)[:, 1]
        metrics.observe_predictions(proba, DECISION_THRESHOLD)
        if prediction_log:
            prediction_log.write(customers, proba, loaded.version)
        return PredictResponse(
            model_version=loaded.version,
            predictions=[
                Prediction(churn_probability=float(p), will_churn=bool(p >= DECISION_THRESHOLD))
                for p in proba
            ],
        )

    return app


app = create_app()
