"""Model card: what the model is for, how it was built and evaluated, and where it falls short.

Generated for every registered version and stored next to it in MLflow (Markdown for people,
JSON for tooling), so an auditor can trace any prod model to its data, metrics and risks.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Groups every slice table reports on.
SLICE_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "InternetService"]
# Sensitive attributes: large gaps between their groups require a human risk review before prod.
SENSITIVE_ATTRIBUTES = ["gender", "SeniorCitizen"]
MAX_GAP = 0.10
# Blind spot: a group with real churners that the model (almost) never flags.
BLIND_SPOT_MIN_N = 30
BLIND_SPOT_MAX_RECALL = 0.10


@dataclass(frozen=True)
class DataInfo:
    source: str
    md5: str
    n_rows: int
    churn_rate: float
    n_train: int
    n_test: int


@dataclass(frozen=True)
class SliceMetrics:
    attribute: str
    group: str
    n: int
    churn_rate: float
    predicted_churn_rate: float
    recall: float | None  # share of real churners caught (None: no churners in the slice)
    false_positive_rate: float | None
    roc_auc: float | None  # None when the slice has a single class


def _rate(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if mask.size else None


def slice_metrics(
    X: pd.DataFrame, y: pd.Series, proba: np.ndarray, threshold: float = 0.5
) -> list[SliceMetrics]:
    y_arr, pred = y.to_numpy(), proba >= threshold
    out = []
    for attribute in SLICE_COLUMNS:
        values = X[attribute].to_numpy()
        for group in sorted(pd.unique(values), key=str):
            m = values == group
            ys, ps, pr = y_arr[m], pred[m], proba[m]
            out.append(
                SliceMetrics(
                    attribute=attribute,
                    group=str(group),
                    n=int(m.sum()),
                    churn_rate=float(ys.mean()),
                    predicted_churn_rate=float(ps.mean()),
                    recall=_rate(ps[ys == 1]),
                    false_positive_rate=_rate(ps[ys == 0]),
                    roc_auc=float(roc_auc_score(ys, pr)) if len(set(ys)) == 2 else None,
                )
            )
    return out


def fairness_gaps(slices: list[SliceMetrics]) -> dict[str, dict[str, float]]:
    """Largest between-group difference per sensitive attribute.

    ``recall_gap`` is equal opportunity (are real churners caught equally often?) and
    ``predicted_rate_gap`` is demographic parity (are groups flagged equally often?).
    """
    gaps = {}
    for attribute in SENSITIVE_ATTRIBUTES:
        rows = [s for s in slices if s.attribute == attribute]
        recalls = [s.recall for s in rows if s.recall is not None]
        rates = [s.predicted_churn_rate for s in rows]
        gaps[attribute] = {
            "recall_gap": max(recalls) - min(recalls) if recalls else 0.0,
            "predicted_rate_gap": max(rates) - min(rates),
        }
    return gaps


def blind_spots(slices: list[SliceMetrics]) -> list[str]:
    return [
        f"Rarely flags {s.attribute}={s.group}: recall {s.recall:.0%} although "
        f"{s.churn_rate:.0%} of these {s.n} test customers churned."
        for s in slices
        if s.n >= BLIND_SPOT_MIN_N and s.recall is not None and s.recall < BLIND_SPOT_MAX_RECALL
    ]


def build_model_card(
    *,
    model_name: str,
    version: str,
    run_id: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    data: DataInfo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    slices = slice_metrics(X_test, y_test, proba, threshold)
    gaps = fairness_gaps(slices)
    flagged = sorted(
        f"{attr}.{name}" for attr, g in gaps.items() for name, v in g.items() if v > MAX_GAP
    )
    return {
        "model": {
            "name": model_name,
            "version": version,
            "mlflow_run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "algorithm": "LightGBM gradient-boosted trees in an sklearn pipeline",
            "decision_threshold": threshold,
            "params": params,
        },
        "intended_use": {
            "primary": "Rank existing residential customers by churn risk so retention teams can "
            "prioritise outreach (call center lookup and nightly campaign lists).",
            "users": "Retention and call center teams; a person always decides the action.",
            "out_of_scope": [
                "Automated decisions with legal or similar effect (pricing, credit, contract "
                "termination) without human review",
                "Business customers or markets outside the training population",
                "Individual-level explanations presented to customers as facts",
            ],
        },
        "data": asdict(data),
        "metrics": {"test": metrics},
        "slices": [asdict(s) for s in slices],
        "fairness": {
            "sensitive_attributes": SENSITIVE_ATTRIBUTES,
            "max_allowed_gap": MAX_GAP,
            "gaps": gaps,
            "flagged": flagged,
            "review_required": bool(flagged),
        },
        "limitations": [
            *blind_spots(slices),
            "Trained on a single public snapshot (IBM Telco sample); no time dimension, so "
            "seasonality and recent market changes are not represented.",
            "Uses gender and senior-citizen status as inputs; kept for transparency of this demo "
            "and monitored above. A production model should justify or drop them.",
            "Probabilities are not calibrated; use them for ranking, not as exact likelihoods.",
            "Performance degrades under distribution shift. Drift is monitored (Evidently); "
            "retraining is automatic, but promotion to prod needs human approval.",
        ],
        "privacy": "No direct identifiers are used as features (customerID is only a join key). "
        "Prediction logs contain customer attributes and must follow the data retention policy.",
    }


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def render_markdown(card: dict[str, Any]) -> str:
    m, d, f = card["model"], card["data"], card["fairness"]
    lines = [
        f"# Model card: {m['name']} v{m['version']}",
        "",
        f"MLflow run `{m['mlflow_run_id']}` · created {m['created_at']} · {m['algorithm']} · "
        f"decision threshold {m['decision_threshold']}",
        "",
        "## Intended use",
        "",
        card["intended_use"]["primary"],
        "",
        f"**Users:** {card['intended_use']['users']}",
        "",
        "**Out of scope:**",
        *(f"- {item}" for item in card["intended_use"]["out_of_scope"]),
        "",
        "## Training data",
        "",
        f"- Source: {d['source']} (md5 `{d['md5']}`, versioned with DVC)",
        f"- {d['n_rows']:,} customers, churn rate {d['churn_rate']:.1%}",
        f"- Stratified split: {d['n_train']:,} train / {d['n_test']:,} test",
        "",
        "## Performance (test set)",
        "",
        "| Metric | Value |",
        "|---|---|",
        *(f"| {k} | {v:.3f} |" for k, v in card["metrics"]["test"].items()),
        "",
        "## Performance by group",
        "",
        "| Attribute | Group | n | Churn rate | Predicted churn | Recall | FPR | ROC AUC |",
        "|---|---|---|---|---|---|---|---|",
        *(
            f"| {s['attribute']} | {s['group']} | {s['n']} | {_pct(s['churn_rate'])} | "
            f"{_pct(s['predicted_churn_rate'])} | {_pct(s['recall'])} | "
            f"{_pct(s['false_positive_rate'])} | {_num(s['roc_auc'])} |"
            for s in card["slices"]
        ),
        "",
        "## Fairness",
        "",
        "Largest gap between groups of each sensitive attribute "
        f"(tolerance {f['max_allowed_gap']:.0%}):",
        "",
        "| Attribute | Recall gap (equal opportunity) | Predicted-rate gap (demographic parity) |",
        "|---|---|---|",
        *(
            f"| {attr} | {_pct(g['recall_gap'])} | {_pct(g['predicted_rate_gap'])} |"
            for attr, g in f["gaps"].items()
        ),
        "",
        (
            f"**Review required** before prod: {', '.join(f['flagged'])} above tolerance."
            if f["review_required"]
            else "All gaps within tolerance."
        ),
        "",
        "## Limitations",
        "",
        *(f"- {item}" for item in card["limitations"]),
        "",
        "## Privacy",
        "",
        card["privacy"],
        "",
    ]
    return "\n".join(lines)
