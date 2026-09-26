"""Prefect orchestration of the training pipeline: fetch -> train (+ quality gate) -> promote."""

from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from churn.config import Settings
from churn.data import download
from churn.registry import promote
from churn.train import TrainResult, train


@task(retries=3, retry_delay_seconds=10, cache_policy=NO_CACHE)
def fetch_data(settings: Settings) -> Path:
    path = settings.raw_data_path
    if path.exists():
        get_run_logger().info("Using existing dataset %s (versioned with DVC)", path)
        return path
    return download(settings.raw_data_url, path)


@task(cache_policy=NO_CACHE)
def train_and_register(
    settings: Settings, params: dict[str, Any] | None, data: Path
) -> TrainResult:
    result = train(settings, params=params, data_path=data)
    get_run_logger().info("Registered v%s with metrics %s", result.model_version, result.metrics)
    return result


@task(cache_policy=NO_CACHE)
def promote_model(settings: Settings, source: str, target: str) -> str:
    version = promote(settings, source, target)
    get_run_logger().info("Promoted v%s: %s -> %s", version, source, target)
    return version


@flow(name="churn-training")
def training_flow(
    promote_to: str | None = "staging",
    params: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> TrainResult:
    """Train a candidate and, if it passes the quality gate, promote it to ``promote_to``.

    Promotion to ``prod`` is intentionally not automated: it needs a human approval.
    """
    if promote_to == "prod":
        raise ValueError("promotion to prod requires manual approval: `churn promote`")
    settings = settings or Settings()
    data = fetch_data(settings)
    result = train_and_register(settings, params, data)
    if promote_to:
        promote_model(settings, settings.register_alias, promote_to)
    return result
