from collections import Counter
from pathlib import Path

import pytest

from support_rag.chunking import chunk_article
from support_rag.config import LANGUAGES, Settings
from support_rag.corpus import Article, CorpusError, load_corpus
from support_rag.evaluate import load_golden


def test_every_topic_exists_in_every_language(articles: list[Article]) -> None:
    per_topic = Counter(a.topic for a in articles)
    assert len(per_topic) >= 20
    assert set(per_topic.values()) == {len(LANGUAGES)}


def test_article_ids_and_urls_are_unique(articles: list[Article]) -> None:
    assert len({a.id for a in articles}) == len(articles)
    assert len({a.url for a in articles}) == len(articles)


def test_missing_translation_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "x.yaml").write_text(
        "category: c\nupdated: 2026-01-01\ntranslations:\n  en: {title: T, body: B}\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="missing languages"):
        load_corpus(tmp_path)


def test_golden_set_covers_every_topic_in_every_language(
    settings: Settings, articles: list[Article]
) -> None:
    golden = load_golden(settings.eval_dir / "retrieval_golden.yaml")
    topics = {a.topic for a in articles}
    assert {q.topic for q in golden} <= topics
    covered = {(q.topic, q.lang) for q in golden}
    assert {(t, lang) for t in topics for lang in LANGUAGES} <= covered


def _article(body: str) -> Article:
    return Article("t", "en", "Title", body, "c", "2026-01-01")


def test_chunks_pack_whole_paragraphs_and_keep_the_title() -> None:
    body = "\n\n".join(" ".join(["word"] * 40) for _ in range(5))  # 5 x 40 words
    chunks = chunk_article(_article(body), max_words=100)
    assert [c.text.count("word") for c in chunks] == [80, 80, 40]
    assert all(c.text.startswith("Title\n\n") for c in chunks)
    assert [c.id for c in chunks] == ["t.en#0", "t.en#1", "t.en#2"]


def test_oversized_paragraph_becomes_its_own_chunk() -> None:
    body = "short one\n\n" + " ".join(["long"] * 300) + "\n\nshort two"
    chunks = chunk_article(_article(body), max_words=100)
    assert len(chunks) == 3
    assert chunks[1].text.count("long") == 300
