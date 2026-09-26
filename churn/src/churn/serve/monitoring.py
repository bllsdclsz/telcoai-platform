"""Operational telemetry for the scoring API: Prometheus metrics and the prediction log."""

import json
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5)
PROBABILITY_BUCKETS = tuple(round(0.1 * i, 1) for i in range(1, 11))


class Metrics:
    """All API metrics, on a private registry so tests and multiple apps don't collide."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "churn_http_requests",
            "HTTP requests by route and status code",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "churn_http_request_duration_seconds",
            "HTTP request latency",
            ["route", "method"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "churn_predictions",
            "Scored customers by predicted outcome",
            ["outcome"],
            registry=self.registry,
        )
        self.probability = Histogram(
            "churn_prediction_probability",
            "Distribution of predicted churn probabilities",
            buckets=PROBABILITY_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "churn_model_info",
            "Model version currently served (value is always 1)",
            ["model_version", "alias"],
            registry=self.registry,
        )
        # Export known series at 0 from startup, so rate()/increase() also see the first burst.
        for outcome in ("churn", "stay"):
            self.predictions.labels(outcome=outcome)
        for status in ("200", "422", "500"):
            self.requests.labels("/predict", "POST", status)

    def observe_predictions(self, probabilities: Sequence[float], threshold: float) -> None:
        for p in probabilities:
            self.probability.observe(p)
            self.predictions.labels(outcome="churn" if p >= threshold else "stay").inc()


class PredictionLog:
    """Appends scored requests as JSON lines, one file per UTC day (input to drift checks).

    A local file keeps the demo self-contained; in production this would be a topic or table.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()

    def write(
        self, customers: list[dict[str, Any]], probabilities: Sequence[float], model_version: str
    ) -> None:
        now = datetime.now(UTC)
        lines = [
            json.dumps(
                {
                    **customer,
                    "churn_probability": float(p),
                    "model_version": model_version,
                    "scored_at": now.isoformat(),
                }
            )
            for customer, p in zip(customers, probabilities, strict=True)
        ]
        path = self.directory / f"predictions-{now:%Y-%m-%d}.jsonl"
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
