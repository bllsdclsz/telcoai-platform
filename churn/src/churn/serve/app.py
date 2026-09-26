"""Real-time churn scoring API. Serves the registered model version behind an alias."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from mlflow import MlflowClient
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


def create_app(loader: Callable[[], LoadedModel] = load_registered_model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.loaded = loader()
        yield

    app = FastAPI(title="Telco Churn API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "model_version": request.app.state.loaded.version}

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        loaded: LoadedModel = request.app.state.loaded
        X = pd.DataFrame([c.model_dump() for c in body.customers])[FEATURE_COLUMNS]
        proba = loaded.model.predict_proba(X)[:, 1]
        return PredictResponse(
            model_version=loaded.version,
            predictions=[
                Prediction(churn_probability=float(p), will_churn=bool(p >= DECISION_THRESHOLD))
                for p in proba
            ],
        )

    return app


app = create_app()
