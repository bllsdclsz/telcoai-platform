"""Answer customer questions: retrieve articles, generate a cited answer, or fall back safely."""

import re
from dataclasses import dataclass, field, replace
from typing import Literal

from support_rag.llm import ChatModel
from support_rag.prompts import PromptTemplate
from support_rag.retrieve import Hit, Retriever

NO_ANSWER = "NO_ANSWER"
LANGUAGE_NAMES = {
    "de": "German (Swiss standard German: write 'ss' instead of 'ß')",
    "fr": "French",
    "it": "Italian",
    "en": "English",
}
# Shown instead of a generated answer when the help center does not cover the question.
FALLBACK = {
    "de": "Dazu habe ich in unserem Hilfe-Center leider nichts gefunden. Bitte kontaktieren Sie "
    "uns im Chat der Nordalp-App oder unter 0800 700 700.",
    "fr": "Je n'ai malheureusement rien trouvé à ce sujet dans notre centre d'aide. Contactez-"
    "nous dans le chat de l'app Nordalp ou au 0800 700 700.",
    "it": "Purtroppo non ho trovato nulla su questo argomento nel nostro centro assistenza. "
    "Contattaci nella chat dell'app Nordalp o al numero 0800 700 700.",
    "en": "I couldn't find this in our help center. Please contact us in the Nordalp app chat "
    "or at 0800 700 700.",
}
_CITATION = re.compile(r"\[(\d+)\]")

Reason = Literal["answered", "no_relevant_sources", "model_no_answer"]


@dataclass(frozen=True)
class Source:
    n: int
    article_id: str
    title: str
    url: str
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    lang: str
    reason: Reason
    cited: list[Source] = field(default_factory=list)
    retrieved: list[Source] = field(default_factory=list)
    prompt: str = ""
    prompt_sha256: str = ""
    model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def answered(self) -> bool:
        return self.reason == "answered"


def number_sources(hits: list[Hit]) -> list[Source]:
    """One numbered source per article (best chunk first), as the model will cite them."""
    seen: set[str] = set()
    sources: list[Source] = []
    for h in hits:
        if h.article_id in seen:
            continue
        seen.add(h.article_id)
        sources.append(Source(len(sources) + 1, h.article_id, h.title, h.url, h.text, h.score))
    return sources


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.n}] {s.text}" for s in sources)


def cited_numbers(text: str) -> list[int]:
    return sorted({int(n) for n in _CITATION.findall(text)})


class Assistant:
    def __init__(
        self,
        retriever: Retriever,
        llm: ChatModel,
        prompt: PromptTemplate,
        *,
        top_k: int = 4,
        min_score: float = 0.0,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> None:
        self.retriever, self.llm, self.prompt = retriever, llm, prompt
        self.top_k, self.min_score = top_k, min_score
        self.temperature, self.max_tokens = temperature, max_tokens

    def ask(self, question: str, lang: str) -> Answer:
        hits = self.retriever.search(question, k=self.top_k, lang=lang)
        sources = number_sources(hits)
        fallback = Answer(
            FALLBACK[lang],
            lang,
            "no_relevant_sources",
            retrieved=sources,
            prompt=self.prompt.ref,
            prompt_sha256=self.prompt.sha256,
        )
        # Nothing relevant in the help center: answer with the fallback, don't call the model.
        if not sources or sources[0].score < self.min_score:
            return fallback

        completion = self.llm.complete(
            self.prompt.render(
                language_name=LANGUAGE_NAMES[lang],
                no_answer=NO_ANSWER,
                sources=format_sources(sources),
                question=question.strip(),
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        generated = replace(
            fallback,
            model=completion.model,
            latency_ms=completion.latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        if NO_ANSWER in completion.text or not completion.text:
            return replace(generated, reason="model_no_answer")

        by_number = {s.n: s for s in sources}
        cited = [by_number[n] for n in cited_numbers(completion.text) if n in by_number]
        return replace(generated, text=completion.text, reason="answered", cited=cited)
