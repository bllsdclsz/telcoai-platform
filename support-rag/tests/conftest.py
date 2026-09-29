import pytest
from qdrant_client import QdrantClient
from rag_fakes import HashingDense, HashingSparse

from support_rag.chunking import Chunk, chunk_article
from support_rag.config import Settings
from support_rag.corpus import Article, load_corpus
from support_rag.index import build_index


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def articles(settings: Settings) -> list[Article]:
    return load_corpus(settings.corpus_dir)


@pytest.fixture(scope="session")
def chunks(articles: list[Article], settings: Settings) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, settings.chunk_max_words)]


@pytest.fixture
def client(chunks: list[Chunk]) -> QdrantClient:
    qc = QdrantClient(":memory:")
    build_index(qc, "test", chunks, HashingDense(), HashingSparse())
    return qc
