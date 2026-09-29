from pathlib import Path

import pytest
from mlflow import MlflowClient

from churn.config import Settings
from churn.registry import PromotionRefusedError, promote
from churn.train import train

FAST = {"n_estimators": 20}
APPROVAL = {"approved_by": "Jane Reviewer", "fairness_reviewed": True}


def aliases(settings: Settings) -> dict[str, str]:
    model = MlflowClient().get_registered_model(settings.registered_model_name)
    return {k: str(v) for k, v in model.aliases.items()}


def version_tags(settings: Settings, version: str) -> dict[str, str]:
    return MlflowClient().get_model_version(settings.registered_model_name, version).tags


def test_promote_and_rollback(settings: Settings, raw_csv: Path) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    assert promote(settings, "dev", "prod", **APPROVAL) == "1"

    train(settings, params=FAST, data_path=raw_csv)
    assert promote(settings, "dev", "prod", **APPROVAL) == "2"
    assert aliases(settings) == {"dev": "2", "prod": "2", "prod-previous": "1"}

    assert promote(settings, "prod-previous", "prod", **APPROVAL) == "1"
    assert aliases(settings)["prod"] == "1"


def test_staging_needs_no_approval(settings: Settings, raw_csv: Path) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    assert promote(settings, "dev", "staging") == "1"


def test_prod_requires_named_approver(settings: Settings, raw_csv: Path) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    for approver in (None, "  "):
        with pytest.raises(PromotionRefusedError, match="approver"):
            promote(settings, "dev", "prod", approved_by=approver, fairness_reviewed=True)
    assert "prod" not in aliases(settings)


def test_prod_approval_is_recorded_on_the_version(settings: Settings, raw_csv: Path) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    promote(settings, "dev", "prod", **APPROVAL)

    tags = version_tags(settings, "1")
    assert tags["prod_approved_by"] == "Jane Reviewer"
    assert tags["prod_approved_at"]


def test_flagged_fairness_review_must_be_acknowledged(settings: Settings, raw_csv: Path) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    client = MlflowClient()
    client.set_model_version_tag(settings.registered_model_name, "1", "fairness_review", "required")

    with pytest.raises(PromotionRefusedError, match="fairness review"):
        promote(settings, "dev", "prod", approved_by="Jane Reviewer")

    promote(settings, "dev", "prod", approved_by="Jane Reviewer", fairness_reviewed=True)
    assert version_tags(settings, "1")["fairness_review"] == "acknowledged"


def test_prod_refuses_version_without_model_card(settings: Settings, raw_csv: Path) -> None:
    train(settings, params=FAST, data_path=raw_csv)
    MlflowClient().delete_model_version_tag(settings.registered_model_name, "1", "model_card")

    with pytest.raises(PromotionRefusedError, match="model card"):
        promote(settings, "dev", "prod", **APPROVAL)


def test_promote_rejects_unknown_target(settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown target"):
        promote(settings, "dev", "production")
