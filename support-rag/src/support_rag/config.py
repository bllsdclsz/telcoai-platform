"""Runtime configuration, overridable through ``RAG_*`` environment variables."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LANGUAGES = ("de", "fr", "it", "en")
Language = Literal["de", "fr", "it", "en"]
PACKAGE_DIR = Path(__file__).resolve().parents[2]  # support-rag/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    corpus_dir: Path = PACKAGE_DIR / "corpus"
    eval_dir: Path = PACKAGE_DIR / "eval"

    # Qdrant: a server URL (Docker), otherwise an embedded on-disk store at qdrant_path.
    qdrant_url: str | None = None
    qdrant_path: Path = Path("data/qdrant")
    collection: str = "support_articles"

    # Chosen by benchmark with the retrieval evaluation (see support-rag/README.md).
    dense_model: str = "intfloat/multilingual-e5-large"
    # Hybrid BM25 lowered hit@1 for e5-large in the benchmark (English stemming on DE/FR/IT),
    # so the default is dense-only; set RAG_SPARSE_MODEL=Qdrant/bm25 to enable hybrid search.
    sparse_model: str | None = None

    chunk_max_words: int = 120
    top_k: int = 5
