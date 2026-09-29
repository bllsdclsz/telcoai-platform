"""Hyperparameter search with Optuna, scored by cross-validation on the training split only.

The test split is never seen during tuning, so the test metrics reported by ``train`` (and the
quality gate) stay an unbiased estimate. Each trial is a nested MLflow run under one parent.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn.config import Settings
from churn.data import load
from churn.train import DEFAULT_PARAMS, build_pipeline, split


@dataclass(frozen=True)
class TuneResult:
    best_params: dict[str, Any]
    best_cv_roc_auc: float
    baseline_cv_roc_auc: float
    n_trials: int
    run_id: str


def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def tune(
    settings: Settings,
    n_trials: int = 30,
    data_path: Path | None = None,
    cv_folds: int = 5,
) -> TuneResult:
    X_train, _, y_train, _ = split(load(data_path or settings.raw_data_path), settings)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=settings.random_seed)

    def cv_roc_auc(params: dict[str, Any]) -> float:
        model = build_pipeline(params, settings.random_seed)
        return float(cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean())

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_name = f"tune-{datetime.now(UTC):%Y%m%dT%H%M%S}"

    with mlflow.start_run(run_name=run_name, tags={"run_type": "tuning"}) as parent:
        baseline = cv_roc_auc(DEFAULT_PARAMS)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                score = cv_roc_auc(params)
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", score)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
            study_name=run_name,
        )
        study.optimize(objective, n_trials=n_trials)

        best = suggest_params(optuna.trial.FixedTrial(study.best_params))
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metrics(
            {
                "best_cv_roc_auc": study.best_value,
                "baseline_cv_roc_auc": baseline,
                "n_trials": len(study.trials),
            }
        )

    return TuneResult(
        best_params=best,
        best_cv_roc_auc=float(study.best_value),
        baseline_cv_roc_auc=baseline,
        n_trials=len(study.trials),
        run_id=parent.info.run_id,
    )
