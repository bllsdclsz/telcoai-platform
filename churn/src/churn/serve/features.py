"""HTTP client for the Feast feature server (keeps the Feast SDK out of the serving image)."""

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from churn.schema import FEATURE_COLUMNS, FEATURE_SERVICE, ID_COLUMN


@dataclass(frozen=True)
class OnlineFeatures:
    found: dict[str, dict[str, Any]]  # customer id -> feature values
    missing: list[str]


class FeatureServerClient:
    def __init__(self, url: str, timeout: float = 2.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def get(self, customer_ids: list[str]) -> OnlineFeatures:
        body = {"feature_service": FEATURE_SERVICE, "entities": {ID_COLUMN: customer_ids}}
        req = urllib.request.Request(
            f"{self.url}/get-online-features",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return parse_response(json.load(resp), customer_ids)


def parse_response(payload: dict[str, Any], customer_ids: list[str]) -> OnlineFeatures:
    """Turn Feast's column-oriented response into one record per customer.

    Feast reports unknown entities as PRESENT with null values, so a customer counts as
    missing when any model feature is null.
    """
    columns = {
        name: result["values"]
        for name, result in zip(
            payload["metadata"]["feature_names"], payload["results"], strict=True
        )
    }
    found: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for i, cid in enumerate(customer_ids):
        record = {col: columns[col][i] for col in FEATURE_COLUMNS}
        if any(v is None for v in record.values()):
            missing.append(cid)
        else:
            found[cid] = record
    return OnlineFeatures(found=found, missing=missing)
