"""Safety evaluation of the input guardrails: `rag eval-safety`.

Injection detection is scored for the rules alone and for rules + learned classifier:
- tuning recall: attacks in eval/safety.yaml ``injection`` (rules were tuned on these,
  the classifier trained on them): an upper bound, not a generalization estimate;
- held-out recall: ``injection_holdout``, never used for tuning or training: the honest estimate;
- held-out false-positive rate: ``benign`` + the off-topic set, never used for training;
- golden false-positive rate: the retrieval golden questions (in-domain training data for the
  classifier), reported to catch regressions on real customer questions.
PII redaction is scored by recall and precision over labeled items.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yaml

from support_rag.config import Settings
from support_rag.embeddings import DenseEncoder
from support_rag.evaluate import load_golden
from support_rag.guardrails import detect_injection, redact_pii
from support_rag.injection_model import InjectionClassifier


@dataclass
class DetectorScores:
    tuning_recall: float
    holdout_recall: float
    holdout_fpr: float
    golden_fpr: float
    missed_holdout: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)


@dataclass
class SafetyReport:
    rules: DetectorScores
    combined: DetectorScores | None  # rules OR classifier; None when no classifier
    n_tuning: int
    n_holdout: int
    n_benign: int
    n_golden: int
    pii_recall: float
    pii_precision: float
    pii_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def shipped(self) -> DetectorScores:
        return self.combined or self.rules


def _score(detect: Callable[[str], bool], tuning, holdout, benign, golden) -> DetectorScores:
    rate = lambda xs: sum(map(detect, xs)) / len(xs)  # noqa: E731
    return DetectorScores(
        tuning_recall=rate(tuning),
        holdout_recall=rate(holdout),
        holdout_fpr=rate(benign),
        golden_fpr=rate(golden),
        missed_holdout=[h for h in holdout if not detect(h)],
        false_positives=[b for b in benign + golden if detect(b)],
    )


def evaluate_safety(
    settings: Settings,
    classifier: InjectionClassifier | None = None,
    encoder: DenseEncoder | None = None,
) -> SafetyReport:
    spec = yaml.safe_load((settings.eval_dir / "safety.yaml").read_text(encoding="utf-8"))
    flat = lambda d: [x for v in d.values() for x in v]  # noqa: E731
    tuning, holdout = flat(spec["injection"]), flat(spec["injection_holdout"])
    off_topic = yaml.safe_load(
        (settings.eval_dir / "out_of_scope.yaml").read_text(encoding="utf-8")
    )
    benign = list(spec["benign"]) + [q["q"] for q in off_topic]
    golden = [q.question for q in load_golden(settings.eval_dir / "retrieval_golden.yaml")]

    rules = _score(lambda t: bool(detect_injection(t)), tuning, holdout, benign, golden)
    combined = None
    if classifier is not None and encoder is not None:
        cache: dict[str, bool] = {}

        def detect(text: str) -> bool:
            if text not in cache:
                cache[text] = bool(detect_injection(text)) or classifier.is_injection(
                    encoder.embed_query(text)
                )
            return cache[text]

        combined = _score(detect, tuning, holdout, benign, golden)

    tp = fp = fn = 0
    pii_errors = []
    for item in spec["pii"]:
        expected, found = Counter(item["types"]), Counter(redact_pii(item["text"]).found)
        tp += sum((expected & found).values())
        fp += sum((found - expected).values())
        fn += sum((expected - found).values())
        if expected != found:
            pii_errors.append(
                {"text": item["text"], "expected": item["types"], "found": sorted(found.elements())}
            )

    return SafetyReport(
        rules=rules,
        combined=combined,
        n_tuning=len(tuning),
        n_holdout=len(holdout),
        n_benign=len(benign),
        n_golden=len(golden),
        pii_recall=tp / (tp + fn) if tp + fn else 1.0,
        pii_precision=tp / (tp + fp) if tp + fp else 1.0,
        pii_errors=pii_errors,
    )


def check_safety_thresholds(report: SafetyReport, thresholds: dict[str, float]) -> list[str]:
    """Gate on the shipped detector (rules + classifier when present) and PII redaction."""
    s = report.shipped
    values = {
        "tuning_recall": (s.tuning_recall, ">="),
        "holdout_recall": (s.holdout_recall, ">="),
        "holdout_fpr": (s.holdout_fpr, "<="),
        "golden_fpr": (s.golden_fpr, "<="),
        "pii_recall": (report.pii_recall, ">="),
        "pii_precision": (report.pii_precision, ">="),
    }
    violations = []
    for name, limit in thresholds.items():
        value, op = values[name]
        if (op == ">=" and value < limit) or (op == "<=" and value > limit):
            violations.append(f"{name} {value:.3f} (required {op} {limit})")
    return violations
