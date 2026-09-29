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

    # Generation. Any LiteLLM model string works, e.g. "azure/<deployment>" with its API key.
    prompts_dir: Path = PACKAGE_DIR / "prompts"
    prompt_version: int | None = None  # None -> latest version
    # Chosen by comparison (support-rag/README.md): newest model with clean, correct answers.
    llm_model: str = "ollama_chat/granite4.2:8b"
    llm_api_base: str | None = "http://127.0.0.1:11434"
    temperature: float = 0.0
    reasoning_effort: str | None = "none"  # thinking off; see llm.LiteLLMChat
    max_tokens: int = 400
    answer_top_k: int = 4
    # First, cheap scope filter: below this best-match score the question is answered with the
    # fallback, without an LLM call. Set just under the lowest in-scope score (0.788 on the golden
    # set), so no real question is refused; it stops 12/24 off-topic questions. The rest are left
    # to the model's NO_ANSWER instruction (second layer). See support-rag/README.md.
    min_retrieval_score: float = 0.78

    # Learned prompt-injection classifier (rag train-injection); None disables it (rules only).
    injection_classifier: Path | None = PACKAGE_DIR / "models" / "injection_classifier.json"
