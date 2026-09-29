"""Model promotion between environments via MLflow registry aliases (dev -> staging -> prod).

Promotion to prod is governed: the version must have a model card, a named person must approve
it, and a version flagged for fairness review needs that review acknowledged explicitly. Every
prod approval is written onto the model version as an audit trail.
"""

from datetime import UTC, datetime

import mlflow
from mlflow import MlflowClient

from churn.config import Settings

ENVIRONMENTS = ("dev", "staging", "prod")


class PromotionRefusedError(RuntimeError):
    """Raised when a governance check blocks a promotion."""


def promote(
    settings: Settings,
    source: str,
    target: str,
    approved_by: str | None = None,
    fairness_reviewed: bool = False,
) -> str:
    """Point ``target`` at the version currently behind ``source``; returns that version.

    The previous target version is recorded as ``<target>-previous`` so a rollback is one call:
    ``promote(settings, "prod-previous", "prod", approved_by=...)``.
    """
    if target not in ENVIRONMENTS:
        raise ValueError(f"unknown target alias {target!r}, expected one of {ENVIRONMENTS}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = settings.registered_model_name
    mv = client.get_model_version_by_alias(name, source)
    version = str(mv.version)

    if target == "prod":
        _check_prod_governance(mv.tags, version, approved_by, fairness_reviewed)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        client.set_model_version_tag(name, version, "prod_approved_by", approved_by)
        client.set_model_version_tag(name, version, "prod_approved_at", now)
        if fairness_reviewed:
            client.set_model_version_tag(name, version, "fairness_review", "acknowledged")

    aliases = client.get_registered_model(name).aliases
    current = aliases.get(target)
    if current is not None and str(current) != version:
        client.set_registered_model_alias(name, f"{target}-previous", str(current))
    client.set_registered_model_alias(name, target, version)
    return version


def _check_prod_governance(
    tags: dict[str, str], version: str, approved_by: str | None, fairness_reviewed: bool
) -> None:
    if not approved_by or not approved_by.strip():
        raise PromotionRefusedError(f"v{version} -> prod needs a named approver (--approved-by)")
    if "model_card" not in tags:
        raise PromotionRefusedError(f"v{version} has no model card; retrain to generate one")
    if tags.get("fairness_review") == "required" and not fairness_reviewed:
        raise PromotionRefusedError(
            f"v{version} is flagged for fairness review (see its model card); "
            "confirm the review with --fairness-reviewed"
        )
