"""Download, clean and validate the raw Telco churn dataset."""

import urllib.request
from pathlib import Path

import pandas as pd

from churn.schema import TARGET, CustomerSchema


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix known issues in the raw export.

    ``TotalCharges`` is blank for brand-new customers (tenure 0); they have not been billed yet,
    so 0 is the correct value rather than a missing one.
    """
    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load(path: Path) -> pd.DataFrame:
    """Read, clean and validate; raises ``pandera.errors.SchemaErrors`` on bad data."""
    return CustomerSchema.validate(clean(pd.read_csv(path)), lazy=True)
