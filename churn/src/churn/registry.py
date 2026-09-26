"""Model promotion between environments via MLflow registry aliases (dev -> staging -> prod)."""

import mlflow
from mlflow import MlflowClient

from churn.config import Settings

ENVIRONMENTS = ("dev", "staging", "prod")


def promote(settings: Settings, source: str, target: str) -> str:
    """Point ``target`` at the version currently behind ``source``; returns that version.

    The previous target version is recorded as ``<target>-previous`` so a rollback is one call:
    ``promote(settings, "prod-previous", "prod")``.
    """
    if target not in ENVIRONMENTS:
        raise ValueError(f"unknown target alias {target!r}, expected one of {ENVIRONMENTS}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = settings.registered_model_name
    version = str(client.get_model_version_by_alias(name, source).version)

    aliases = client.get_registered_model(name).aliases
    current = aliases.get(target)
    if current is not None and str(current) != version:
        client.set_registered_model_alias(name, f"{target}-previous", str(current))
    client.set_registered_model_alias(name, target, version)
    return version
