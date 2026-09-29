"""Build the Qdrant collection: one point per chunk, dense (+ optional sparse) vectors, payload."""

import uuid
import warnings
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from support_rag.chunking import Chunk
from support_rag.config import Settings
from support_rag.embeddings import DenseEncoder, SparseEncoder

DENSE, SPARSE = "dense", "sparse"
_ID_NAMESPACE = uuid.UUID("5b1d2f0e-6a4c-4e8f-9b1a-2c3d4e5f6a7b")


def connect(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def point_id(chunk: Chunk) -> str:
    """Stable ID, so re-ingesting the same chunk overwrites instead of duplicating."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk.id))


def build_index(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Chunk],
    dense: DenseEncoder,
    sparse: SparseEncoder | None = None,
    batch_size: int = 64,
) -> int:
    """(Re)create ``collection`` from ``chunks``; returns the number of points written."""
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={
            DENSE: models.VectorParams(size=dense.dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config=(
            {SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)} if sparse else None
        ),
    )
    with warnings.catch_warnings():  # embedded Qdrant ignores payload indexes; the server uses them
        warnings.simplefilter("ignore", UserWarning)
        for field in ("lang", "topic"):
            client.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        dense_vecs = dense.embed_passages(texts)
        sparse_vecs = sparse.embed_passages(texts) if sparse else [None] * len(batch)
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=point_id(c),
                    vector={DENSE: d, **({SPARSE: s} if s is not None else {})},
                    payload={
                        "chunk_id": c.id,
                        "article_id": c.article.id,
                        "topic": c.article.topic,
                        "lang": c.article.lang,
                        "title": c.article.title,
                        "category": c.article.category,
                        "url": c.article.url,
                        "updated": c.article.updated,
                        "text": c.text,
                    },
                )
                for c, d, s in zip(batch, dense_vecs, sparse_vecs, strict=True)
            ],
        )
    return len(chunks)
