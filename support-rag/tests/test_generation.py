from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from rag_fakes import HashingDense

from support_rag.api import create_app
from support_rag.assistant import FALLBACK, NO_ANSWER, Assistant, cited_numbers, number_sources
from support_rag.config import LANGUAGES, Settings
from support_rag.llm import Completion, clean_output
from support_rag.prompts import load_prompt
from support_rag.retrieve import Hit, Retriever


class ScriptedChat:
    """Returns a fixed reply and records what it was asked."""

    model = "fake/scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> Completion:
        self.calls.append(messages)
        return Completion(clean_output(self.reply), self.model, 100, 20, 5.0)


def assistant(client: QdrantClient, settings: Settings, reply: str, min_score: float = 0.0):
    chat = ScriptedChat(reply)
    prompt = load_prompt(settings.prompts_dir, "answer")
    retriever = Retriever(client, "test", HashingDense())
    return Assistant(retriever, chat, prompt, top_k=4, min_score=min_score), chat


QUESTION = "What does a replacement SIM cost after losing my phone?"


def test_prompt_contains_numbered_sources_language_and_question(
    client: QdrantClient, settings: Settings
) -> None:
    bot, chat = assistant(client, settings, "A replacement SIM costs CHF 40 [1].")
    bot.ask(QUESTION, "de")

    system, user = chat.calls[0]
    assert "German" in system["content"] and NO_ANSWER in system["content"]
    assert user["content"].count("\n[") >= 1 and "[1]" in user["content"]
    assert QUESTION in user["content"]


def test_answer_returns_only_the_sources_it_cites(client: QdrantClient, settings: Settings) -> None:
    bot, _ = assistant(client, settings, "It costs CHF 40 [2]. Block it first [2].")
    answer = bot.ask(QUESTION, "en")

    assert answer.answered and answer.reason == "answered"
    assert [s.n for s in answer.cited] == [2]
    assert len(answer.retrieved) >= 2
    assert answer.prompt == "answer@v1" and len(answer.prompt_sha256) == 64
    assert (answer.model, answer.input_tokens, answer.output_tokens) == ("fake/scripted", 100, 20)


def test_citations_to_unknown_sources_are_dropped(client: QdrantClient, settings: Settings) -> None:
    bot, _ = assistant(client, settings, "It costs CHF 40 [1] [99].")
    assert [s.n for s in bot.ask(QUESTION, "en").cited] == [1]


@pytest.mark.parametrize("lang", LANGUAGES)
def test_model_no_answer_becomes_localized_fallback(
    client: QdrantClient, settings: Settings, lang: str
) -> None:
    bot, _ = assistant(client, settings, NO_ANSWER)
    answer = bot.ask(QUESTION, lang)
    assert answer.reason == "model_no_answer" and answer.text == FALLBACK[lang]
    assert not answer.cited


def test_empty_reply_is_a_generation_failure_not_a_decline(
    client: QdrantClient, settings: Settings
) -> None:
    bot, _ = assistant(client, settings, "<think>long reasoning, then cut off</think>")
    answer = bot.ask(QUESTION, "fr")
    assert answer.reason == "generation_failed" and answer.text == FALLBACK["fr"]
    assert not answer.answered


def test_irrelevant_question_is_not_sent_to_the_model(
    client: QdrantClient, settings: Settings
) -> None:
    bot, chat = assistant(client, settings, "should not be used", min_score=0.99)
    answer = bot.ask("Qual è la ricetta del tiramisù?", "it")
    assert answer.reason == "no_relevant_sources" and answer.text == FALLBACK["it"]
    assert chat.calls == []


def test_reasoning_blocks_are_stripped() -> None:
    assert clean_output("<think>\nplan...\n</think>\nCHF 40 [1].") == "CHF 40 [1]."
    assert clean_output("<think></think>Ciao") == "Ciao"
    # Dangling closing tag: the opening one was part of the chat template.
    dangling = "We need one sentence. [1]\nOk.\n</think>\nCHF 30 [1]."
    assert clean_output(dangling) == "CHF 30 [1]."


def test_number_sources_keeps_best_chunk_per_article() -> None:
    def hit(article: str, score: float) -> Hit:
        return Hit(f"{article}#0", article, "t", "en", article, "u", "text", score)

    sources = number_sources([hit("a.en", 0.9), hit("a.en", 0.8), hit("b.en", 0.7)])
    assert [(s.n, s.article_id, s.score) for s in sources] == [(1, "a.en", 0.9), (2, "b.en", 0.7)]
    assert cited_numbers("x [2] y [1] z [2] [10]") == [1, 2, 10]


def test_prompt_versions_load_by_number_and_latest(tmp_path: Path, settings: Settings) -> None:
    latest = load_prompt(settings.prompts_dir, "answer")
    assert load_prompt(settings.prompts_dir, "answer", latest.version) == latest

    folder = tmp_path / "answer"
    folder.mkdir()
    (folder / "v1.yaml").write_text(
        "id: answer\nversion: 2\nsystem: s\nuser: u\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="do not match"):
        load_prompt(tmp_path, "answer")


def test_ask_endpoint(client: QdrantClient, settings: Settings) -> None:
    bot, _ = assistant(client, settings, "Ersatz-SIM: CHF 40 [1].")
    with TestClient(create_app(lambda: bot)) as http:
        assert http.get("/health").json() == {"status": "ok", "prompt": "answer@v1"}
        body = http.post("/ask", json={"question": QUESTION, "lang": "de"}).json()

    assert body["answered"] is True and body["answer"] == "Ersatz-SIM: CHF 40 [1]."
    assert body["sources"][0]["n"] == 1 and body["sources"][0]["url"].startswith("https://")
    assert body["prompt"] == "answer@v1" and len(body["request_id"]) == 36


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "lang": "de"},
        {"question": "x" * 1001, "lang": "de"},
        {"question": "hi", "lang": "es"},
    ],
)
def test_ask_rejects_invalid_requests(
    client: QdrantClient, settings: Settings, payload: dict
) -> None:
    bot, _ = assistant(client, settings, "unused")
    with TestClient(create_app(lambda: bot)) as http:
        assert http.post("/ask", json=payload).status_code == 422
