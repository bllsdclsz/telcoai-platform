"""Replay customers against the API, optionally with a realistic distribution shift (demo)."""

import json
import urllib.request

import numpy as np
import pandas as pd

from churn.features import ADDON_COLUMNS
from churn.schema import FEATURE_COLUMNS


def shift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Scenario: a price increase during an aggressive fiber acquisition campaign.

    More new, month-to-month fiber customers paying by electronic check, all paying more.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    new = rng.random(n) < 0.6
    out.loc[new, "tenure"] = rng.integers(0, 13, new.sum())
    out.loc[rng.random(n) < 0.7, "Contract"] = "Month-to-month"
    out.loc[rng.random(n) < 0.5, "PaymentMethod"] = "Electronic check"

    fiber = rng.random(n) < 0.6
    out.loc[fiber, "InternetService"] = "Fiber optic"
    for col in ADDON_COLUMNS:  # keep records consistent: fiber customers have internet add-ons
        out.loc[fiber & (out[col] == "No internet service"), col] = "No"

    out["MonthlyCharges"] = (out["MonthlyCharges"] * 1.25).round(2)
    out["TotalCharges"] = (out["MonthlyCharges"] * out["tenure"]).round(2)
    return out


def replay(
    customers: pd.DataFrame, api_url: str, batch_size: int = 50, timeout: float = 10.0
) -> int:
    """POST customers to ``/predict`` in batches; returns how many were scored."""
    records = customers[FEATURE_COLUMNS].to_dict(orient="records")
    scored = 0
    for start in range(0, len(records), batch_size):
        body = json.dumps({"customers": records[start : start + batch_size]}).encode()
        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/predict",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            scored += len(json.load(resp)["predictions"])
    return scored
