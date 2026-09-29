"""Retrieval evaluation: golden questions per language -> hit@k, MRR, language match, gate.

A query counts as answered at rank r when the r-th retrieved chunk belongs to the expected help
topic. Metrics are reported overall and per language, with the UI language known (filtered
search, the production path) and unknown (unfiltered, cross-lingual robustness).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from qdrant_client import QdrantClient

from support_rag.chunking import chunk_article
from support_rag.config import LANGUAGES, Settings
from support_rag.corpus import load_corpus
from support_rag.embeddings import FastEmbedDense, FastEmbedSparse
from support_rag.index import build_index
from support_rag.retrieve import Retriever


@dataclass(frozen=True)
class GoldenQuery:
    question: str
    lang: str
    topic: str


@dataclass
class RetrievalReport:
    config: dict[str, Any]
    n_queries: int
    overall: dict[str, float]
    by_lang: dict[str, dict[str, float]]
    unfiltered: dict[str, float]  # same queries without the language filter
    failures: list[dict[str, Any]] = field(default_factory=list)
    ms_per_query: float = 0.0


def load_golden(path: Path) -> list[GoldenQuery]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenQuery(question=r["q"], lang=r["lang"], topic=r["topic"]) for r in rows]


def _rank(retriever: Retriever, q: GoldenQuery, k: int, lang: str | None) -> tuple[int | None, str]:
    hits = retriever.search(q.question, k=k, lang=lang)
    rank = next((i + 1 for i, h in enumerate(hits) if h.topic == q.topic), None)
    return rank, hits[0].lang if hits else ""


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    return {
        "hit@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr@10": sum(1 / r for r in ranks if r is not None) / n,
    }


def evaluate_retrieval(
    retriever: Retriever, queries: list[GoldenQuery], config: dict[str, Any], k: int = 10
) -> RetrievalReport:
    filtered: list[int | None] = []
    unfiltered: list[int | None] = []
    same_lang_top1 = 0
    failures = []
    start = time.perf_counter()
    for q in queries:
        rank, _ = _rank(retriever, q, k, q.lang)
        filtered.append(rank)
        if rank is None or rank > 3:
            top = retriever.search(q.question, k=1, lang=q.lang)
            failures.append(
                {
                    "q": q.question,
                    "lang": q.lang,
                    "expected": q.topic,
                    "rank": rank,
                    "top1": top[0].topic if top else None,
                }
            )
        rank_u, top_lang = _rank(retriever, q, k, None)
        unfiltered.append(rank_u)
        same_lang_top1 += top_lang == q.lang
    elapsed = time.perf_counter() - start

    by_lang = {
        lang: _metrics([r for r, q in zip(filtered, queries, strict=True) if q.lang == lang])
        for lang in LANGUAGES
    }
    return RetrievalReport(
        config=config,
        n_queries=len(queries),
        overall=_metrics(filtered),
        by_lang=by_lang,
        unfiltered={**_metrics(unfiltered), "same_lang@1": same_lang_top1 / (len(queries) or 1)},
        failures=failures,
        ms_per_query=1000 * elapsed / (2 * len(queries) or 1),
    )


def run_retrieval_eval(
    settings: Settings, dense_model: str, sparse_model: str | None
) -> RetrievalReport:
    """Index the corpus in memory with the given models and score the golden set."""
    chunks = [
        c
        for a in load_corpus(settings.corpus_dir)
        for c in chunk_article(a, settings.chunk_max_words)
    ]
    dense = FastEmbedDense(dense_model)
    sparse = FastEmbedSparse(sparse_model) if sparse_model else None
    client = QdrantClient(":memory:")
    build_index(client, "eval", chunks, dense, sparse)
    retriever = Retriever(client, "eval", dense, sparse)
    queries = load_golden(settings.eval_dir / "retrieval_golden.yaml")
    config = {"dense_model": dense_model, "sparse_model": sparse_model, "chunks": len(chunks)}
    return evaluate_retrieval(retriever, queries, config)


def check_thresholds(report: RetrievalReport, thresholds: dict[str, Any]) -> list[str]:
    """Violations of the acceptance thresholds (empty list = gate passes)."""
    violations = [
        f"overall {metric} {report.overall[metric]:.3f} < {minimum}"
        for metric, minimum in thresholds.get("overall", {}).items()
        if report.overall[metric] < minimum
    ]
    for metric, minimum in thresholds.get("per_language", {}).items():
        violations += [
            f"{lang} {metric} {m[metric]:.3f} < {minimum}"
            for lang, m in report.by_lang.items()
            if m[metric] < minimum
        ]
    return violations
