import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from churn.config import Settings
from churn.model_card import (
    DataInfo,
    build_model_card,
    fairness_gaps,
    render_markdown,
    slice_metrics,
)
from churn.train import train

from .conftest import make_customers


def frame(gender: list[str], senior: list[int]) -> pd.DataFrame:
    df = make_customers(n=len(gender), seed=0)
    df["gender"], df["SeniorCitizen"] = gender, senior
    return df


def test_slice_metrics_by_hand() -> None:
    X = frame(["Male", "Male", "Female", "Female"], [0, 0, 0, 0])
    y = pd.Series([1, 0, 1, 0])
    proba = np.array([0.9, 0.2, 0.3, 0.6])  # male: TP, TN; female: FN, FP

    by_gender = {s.group: s for s in slice_metrics(X, y, proba) if s.attribute == "gender"}

    assert by_gender["Male"].recall == 1.0 and by_gender["Male"].false_positive_rate == 0.0
    assert by_gender["Female"].recall == 0.0 and by_gender["Female"].false_positive_rate == 1.0
    assert by_gender["Male"].predicted_churn_rate == by_gender["Female"].predicted_churn_rate == 0.5


def test_fairness_gaps_measure_between_group_differences() -> None:
    X = frame(["Male", "Male", "Female", "Female"], [1, 1, 0, 0])
    y = pd.Series([1, 1, 1, 0])
    proba = np.array([0.9, 0.8, 0.1, 0.1])  # everyone in group 1 flagged, nobody in group 0

    gaps = fairness_gaps(slice_metrics(X, y, proba))

    assert gaps["gender"] == {"recall_gap": 1.0, "predicted_rate_gap": 1.0}
    assert gaps["SeniorCitizen"] == {"recall_gap": 1.0, "predicted_rate_gap": 1.0}


def card_for(proba: np.ndarray, X: pd.DataFrame, y: pd.Series) -> dict:
    return build_model_card(
        model_name="telco-churn",
        version="3",
        run_id="abc",
        params={"n_estimators": 10},
        metrics={"roc_auc": 0.84},
        data=DataInfo("data.csv", "md5", 100, 0.26, 80, 20),
        X_test=X,
        y_test=y,
        proba=proba,
    )


def test_review_required_only_when_a_gap_exceeds_tolerance() -> None:
    X = frame(["Male", "Female"] * 50, [0, 1] * 50)
    y = pd.Series([1, 0] * 50)

    fair = card_for(np.full(100, 0.4), X, y)  # identical scores for everyone
    assert fair["fairness"]["review_required"] is False

    skewed = card_for(np.where(X["gender"] == "Male", 0.9, 0.1), X, y)
    assert skewed["fairness"]["review_required"] is True
    assert "gender.predicted_rate_gap" in skewed["fairness"]["flagged"]


def test_card_is_json_serialisable_and_renders_all_sections() -> None:
    X = make_customers(n=200, seed=1)
    card = card_for(np.random.default_rng(0).random(200), X, X["Churn"])

    json.dumps(card)
    md = render_markdown(card)
    for heading in (
        "# Model card: telco-churn v3",
        "## Intended use",
        "## Training data",
        "## Performance by group",
        "## Fairness",
        "## Limitations",
        "## Privacy",
    ):
        assert heading in md


@pytest.mark.parametrize("fmt", ["md", "json"])
def test_training_stores_model_card_with_version(
    settings: Settings, raw_csv: Path, fmt: str
) -> None:
    result = train(settings, params={"n_estimators": 20}, data_path=raw_csv)

    local = mlflow.artifacts.download_artifacts(
        run_id=result.run_id, artifact_path=f"model_card/model_card.{fmt}"
    )
    text = Path(local).read_text(encoding="utf-8")
    assert f"v{result.model_version}" in text or f'"version": "{result.model_version}"' in text

    mv = MlflowClient().get_model_version(settings.registered_model_name, result.model_version)
    assert mv.tags["model_card"] == "model_card/model_card.md"
    assert mv.tags["fairness_review"] in {"required", "not_required"}
    assert "ROC AUC" in mv.description


def test_blind_spots_list_groups_whose_churners_are_never_flagged() -> None:
    X = frame(["Male", "Female"] * 50, [0, 1] * 50)
    y = pd.Series([1, 1] * 50)
    proba = np.where(X["gender"] == "Male", 0.9, 0.1)  # female churners never flagged

    card = card_for(proba, X, y)

    spots = [line for line in card["limitations"] if line.startswith("Rarely flags")]
    assert any("gender=Female" in line for line in spots)
    assert not any("gender=Male" in line for line in spots)
