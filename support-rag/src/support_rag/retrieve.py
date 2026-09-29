"""Search the help center: dense or hybrid (dense + BM25, fused with reciprocal rank fusion)."""

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from support_rag.embeddings import DenseEncoder, SparseEncoder
from support_rag.index import DENSE, SPARSE


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    article_id: str
    topic: str
    lang: str
    title: str
    url: str
    text: str
    score: float

    @classmethod
    def from_point(cls, point: models.ScoredPoint) -> Hit:
        p: dict[str, Any] = point.payload or {}
        return cls(
            chunk_id=p["chunk_id"],
            article_id=p["article_id"],
            topic=p["topic"],
            lang=p["lang"],
            title=p["title"],
            url=p["url"],
            text=p["text"],
            score=point.score,
        )


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense: DenseEncoder,
        sparse: SparseEncoder | None = None,
        candidates: int = 30,
    ) -> None:
        self.client, self.collection = client, collection
        self.dense, self.sparse = dense, sparse
        self.candidates = candidates  # per retriever, before fusion

    def search(self, query: str, k: int = 5, lang: str | None = None) -> list[Hit]:
        """Top ``k`` chunks; ``lang`` restricts results to one language (e.g. the UI locale)."""
        where = (
            models.Filter(
                must=[models.FieldCondition(key="lang", match=models.MatchValue(value=lang))]
            )
            if lang
            else None
        )
        dense_query = self.dense.embed_query(query)
        if self.sparse is None:
            points = self.client.query_points(
                self.collection, query=dense_query, using=DENSE, query_filter=where, limit=k
            ).points
        else:
            points = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_query, using=DENSE, filter=where, limit=self.candidates
                    ),
                    models.Prefetch(
                        query=self.sparse.embed_query(query),
                        using=SPARSE,
                        filter=where,
                        limit=self.candidates,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
            ).points
        return [Hit.from_point(p) for p in points]
