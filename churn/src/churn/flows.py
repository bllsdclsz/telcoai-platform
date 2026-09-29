"""Prefect flows: training (fetch -> [tune] -> train + gate -> promote) and drift monitoring."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from churn.config import Settings
from churn.data import download, load
from churn.drift import DriftResult, detect_drift, load_predictions
from churn.registry import promote
from churn.train import TrainResult, train
from churn.tune import tune


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
def tune_params(settings: Settings, n_trials: int, data: Path) -> dict[str, Any]:
    result = tune(settings, n_trials=n_trials, data_path=data)
    get_run_logger().info(
        "Tuned %d trials: CV ROC AUC %.4f (baseline %.4f)",
        result.n_trials,
        result.best_cv_roc_auc,
        result.baseline_cv_roc_auc,
    )
    return result.best_params


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
    tune_trials: int = 0,
) -> TrainResult:
    """Train a candidate and, if it passes the quality gate, promote it to ``promote_to``.

    With ``tune_trials`` > 0, hyperparameters are searched first (explicit ``params`` still win).
    Promotion to ``prod`` is intentionally not automated: it needs a human approval.
    """
    if promote_to == "prod":
        raise ValueError("promotion to prod requires manual approval: `churn promote`")
    settings = settings or Settings()
    data = fetch_data(settings)
    if tune_trials > 0:
        params = {**tune_params(settings, tune_trials, data), **(params or {})}
    result = train_and_register(settings, params, data)
    if promote_to:
        promote_model(settings, settings.register_alias, promote_to)
    return result


@task(cache_policy=NO_CACHE)
def check_drift(settings: Settings, report: Path) -> DriftResult | None:
    logger = get_run_logger()
    if settings.prediction_log_dir is None:
        raise ValueError("drift monitoring needs CHURN_PREDICTION_LOG_DIR")
    current = load_predictions(settings.prediction_log_dir, settings.drift_window_days)
    if len(current) < settings.drift_min_rows:
        logger.info("Only %d predictions (< %d): no verdict", len(current), settings.drift_min_rows)
        return None
    reference = load(fetch_data(settings))
    result = detect_drift(
        reference, current, settings.drift_threshold, settings.drift_share, html_report=report
    )
    logger.info(
        "Drift share %.2f (%d/%d columns: %s), dataset drift: %s",
        result.drift_share,
        len(result.drifted_columns),
        len(result.column_scores),
        ", ".join(result.drifted_columns) or "none",
        result.dataset_drift,
    )
    return result


@task(cache_policy=NO_CACHE)
def log_drift(settings: Settings, result: DriftResult, report: Path) -> str:
    """Keep every drift check in MLflow so the history is auditable next to the models."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.monitoring_experiment_name)
    with mlflow.start_run(run_name=f"drift-{datetime.now(UTC):%Y%m%dT%H%M}") as run:
        mlflow.log_params(
            {"threshold": result.threshold, "drift_share_threshold": result.drift_share_threshold}
        )
        mlflow.log_metrics(
            {
                "drift_share": result.drift_share,
                "n_current": result.n_current,
                **{f"drift_{col}": score for col, score in result.column_scores.items()},
            }
        )
        mlflow.set_tags(
            {
                "dataset_drift": str(result.dataset_drift).lower(),
                "drifted_columns": ",".join(result.drifted_columns),
            }
        )
        mlflow.log_artifact(str(report))
    return run.info.run_id


@flow(name="churn-drift-monitoring")
def drift_monitoring_flow(
    retrain_on_drift: bool = True, settings: Settings | None = None
) -> DriftResult | None:
    """Compare recent traffic with the training data; on drift, retrain and promote to staging.

    In production the retraining input would be the newest labeled window, not the original
    dataset; the trigger wiring (and the human gate before prod) is the same.
    """
    settings = settings or Settings()
    report = settings.data_dir / "reports" / f"drift-{datetime.now(UTC):%Y%m%dT%H%M%S}.html"
    result = check_drift(settings, report)
    if result is None:
        return None
    log_drift(settings, result, report)
    if result.dataset_drift and retrain_on_drift:
        get_run_logger().warning("Dataset drift detected: triggering retraining")
        training_flow(promote_to="staging", settings=settings)
    return result
