"""Learned prompt-injection classifier: logistic regression on the multilingual query embedding.

Why: hand-written patterns caught 100% of the attacks they were tuned on but 0 of 16 new
phrasings. A classifier on the e5 embedding (already computed for retrieval) generalizes across
wording and even across languages it has no training data for (FR/IT).

Training data (``rag train-injection``):
- deepset/prompt-injections train split (Apache-2.0; EN/DE, injections + generic benign),
- the attack set in eval/safety.yaml (``injection``),
- the retrieval golden questions as in-domain benign examples, so real support questions
  are not blocked. The benign/off-topic lists in eval/safety.yaml and the held-out attacks are
  NOT used for training: they measure false positives and recall on unseen text.

The shipped artifact is a small JSON (weights + metadata + metrics), versioned in git like a
prompt; inference is a dot product, no scikit-learn needed at serving time.
"""

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from support_rag.config import Settings
from support_rag.embeddings import DenseEncoder
from support_rag.evaluate import load_golden

DATASET = "deepset/prompt-injections"
DATASET_FILES = {
    "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test": "data/test-00000-of-00001-701d16158af87368.parquet",
}


@dataclass(frozen=True)
class InjectionClassifier:
    weights: list[float]
    bias: float
    threshold: float
    embedding_model: str
    metadata: dict[str, Any]

    def probability(self, vector: list[float]) -> float:
        z = self.bias + sum(w * x for w, x in zip(self.weights, vector, strict=True))
        return 1 / (1 + math.exp(-z))

    def is_injection(self, vector: list[float]) -> bool:
        return self.probability(vector) > self.threshold

    @classmethod
    def load(cls, path: Path) -> InjectionClassifier:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            doc["weights"], doc["bias"], doc["threshold"], doc["embedding_model"], doc["metadata"]
        )


def _load_deepset(split: str) -> tuple[list[str], list[int]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATASET_FILES[split], repo_type="dataset")
    df = pd.read_parquet(path)
    return df["text"].tolist(), df["label"].astype(int).tolist()


def train_injection_classifier(
    settings: Settings, encoder: DenseEncoder, threshold: float = 0.7
) -> InjectionClassifier:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    spec = yaml.safe_load((settings.eval_dir / "safety.yaml").read_text(encoding="utf-8"))
    attacks = [a for v in spec["injection"].values() for a in v]
    in_domain = [q.question for q in load_golden(settings.eval_dir / "retrieval_golden.yaml")]
    ds_text, ds_label = _load_deepset("train")

    texts = ds_text + attacks + in_domain
    labels = ds_label + [1] * len(attacks) + [0] * len(in_domain)
    X = np.array([encoder.embed_query(t) for t in texts])
    model = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0).fit(X, labels)

    test_text, test_label = _load_deepset("test")
    pred = model.predict_proba(np.array([encoder.embed_query(t) for t in test_text]))[:, 1]
    flagged = pred > threshold
    positives = sum(test_label)
    true_pos = sum(f and y for f, y in zip(flagged, test_label, strict=True))

    return InjectionClassifier(
        weights=[round(float(w), 6) for w in model.coef_[0]],
        bias=round(float(model.intercept_[0]), 6),
        threshold=threshold,
        embedding_model=getattr(encoder, "model_name", "unknown"),
        metadata={
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "training_data": {
                f"{DATASET} (train)": len(ds_text),
                "eval/safety.yaml injection": len(attacks),
                "retrieval golden questions (benign)": len(in_domain),
            },
            "deepset_test": {
                "n": len(test_text),
                "recall": round(float(true_pos / positives), 3),
                "precision": round(float(true_pos / max(int(flagged.sum()), 1)), 3),
            },
        },
    )


def save(classifier: InjectionClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "embedding_model": classifier.embedding_model,
        "threshold": classifier.threshold,
        "metadata": classifier.metadata,
        "bias": classifier.bias,
        "weights": classifier.weights,
    }
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
