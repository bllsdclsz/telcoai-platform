import pytest
from qdrant_client import QdrantClient
from rag_fakes import HashingDense, HashingSparse

from support_rag.chunking import Chunk
from support_rag.evaluate import (
    GoldenQuery,
    RetrievalReport,
    check_thresholds,
    evaluate_retrieval,
)
from support_rag.index import build_index
from support_rag.retrieve import Retriever


@pytest.fixture(params=["dense", "hybrid"])
def retriever(request: pytest.FixtureRequest, client: QdrantClient) -> Retriever:
    return Retriever(
        client, "test", HashingDense(), HashingSparse() if request.param == "hybrid" else None
    )


def test_finds_article_by_its_own_wording(retriever: Retriever) -> None:
    hits = retriever.search("Replay 250 channels recordings 500 hours", k=3, lang="en")
    assert hits[0].topic == "tv-replay"
    assert hits[0].url == "https://help.nordalp.example/en/tv-replay"


@pytest.mark.parametrize("lang", ["de", "fr", "it", "en"])
def test_language_filter_returns_only_that_language(retriever: Retriever, lang: str) -> None:
    hits = retriever.search("CHF 40 SIM", k=10, lang=lang)
    assert hits and {h.lang for h in hits} == {lang}


def test_reingest_overwrites_instead_of_duplicating(chunks: list[Chunk]) -> None:
    qc = QdrantClient(":memory:")
    build_index(qc, "c", chunks, HashingDense())
    build_index(qc, "c", chunks, HashingDense())
    assert qc.count("c").count == len(chunks)


class StubRetriever:
    """Returns a fixed ranking of topics, to test the metric arithmetic."""

    def __init__(self, ranking: dict[str, list[str]]) -> None:
        self.ranking = ranking

    def search(self, query: str, k: int = 5, lang: str | None = None) -> list:
        from support_rag.retrieve import Hit

        return [Hit(f"{t}#0", t, t, lang or "en", t, "", "", 1.0) for t in self.ranking[query][:k]]


def test_metrics_arithmetic() -> None:
    queries = [
        GoldenQuery("q1", "en", "a"),  # rank 1
        GoldenQuery("q2", "en", "a"),  # rank 2
        GoldenQuery("q3", "de", "a"),  # rank 4
        GoldenQuery("q4", "de", "a"),  # not found
    ]
    stub = StubRetriever(
        {"q1": ["a", "b"], "q2": ["b", "a"], "q3": ["b", "c", "d", "a"], "q4": ["b", "c"]}
    )
    report = evaluate_retrieval(stub, queries, config={})  # type: ignore[arg-type]

    assert report.overall == {
        "hit@1": 0.25,
        "hit@3": 0.5,
        "hit@5": 0.75,
        "mrr@10": (1 + 0.5 + 0.25) / 4,
    }
    assert report.by_lang["en"]["hit@3"] == 1.0
    assert report.by_lang["de"]["hit@3"] == 0.0
    assert {f["q"] for f in report.failures} == {"q3", "q4"}


def test_thresholds_report_every_violation() -> None:
    report = RetrievalReport(
        config={},
        n_queries=4,
        overall={"hit@1": 0.7, "hit@3": 0.96, "hit@5": 1.0, "mrr@10": 0.9},
        by_lang={"de": {"hit@3": 0.8}, "en": {"hit@3": 1.0}},
        unfiltered={},
    )
    thresholds = {"overall": {"hit@1": 0.8, "hit@3": 0.95}, "per_language": {"hit@3": 0.9}}
    assert check_thresholds(report, thresholds) == [
        "overall hit@1 0.700 < 0.8",
        "de hit@3 0.800 < 0.9",
    ]
