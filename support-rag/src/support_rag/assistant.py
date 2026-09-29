"""Answer customer questions: retrieve articles, generate a cited answer, or fall back safely."""

import re
from dataclasses import dataclass, field, replace
from typing import Literal

from support_rag.guardrails import CANARY, detect_injection, redact_pii, unsupported_numbers
from support_rag.injection_model import InjectionClassifier
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
# Shown when a question tries to manipulate the assistant (prompt injection).
BLOCKED = {
    "de": "Ich kann nur Fragen zu den Produkten und Diensten von Nordalp beantworten.",
    "fr": "Je peux uniquement répondre aux questions sur les produits et services Nordalp.",
    "it": "Posso rispondere solo a domande sui prodotti e servizi di Nordalp.",
    "en": "I can only help with questions about Nordalp products and services.",
}
_CITATION = re.compile(r"\[(\d+)\]")

Reason = Literal[
    "answered",
    "no_relevant_sources",  # retrieval score below the scope threshold
    "model_no_answer",  # the model said the sources don't cover the question
    "generation_failed",  # empty model output
    "blocked_input",  # prompt-injection attempt, never sent to the model
    "blocked_output",  # the answer leaked the system prompt
    "ungrounded",  # no valid citation, or numbers not found in the cited sources
]


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
    pii_redacted: list[str] = field(default_factory=list)
    guard_detail: str = ""

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
        injection_classifier: InjectionClassifier | None = None,
    ) -> None:
        self.retriever, self.llm, self.prompt = retriever, llm, prompt
        self.top_k, self.min_score = top_k, min_score
        self.temperature, self.max_tokens = temperature, max_tokens
        self.injection_classifier = injection_classifier

    def ask(self, question: str, lang: str) -> Answer:
        # Input guards: redact personal data first, so it reaches neither the model nor logs.
        redaction = redact_pii(question.strip())
        base = Answer(
            FALLBACK[lang],
            lang,
            "no_relevant_sources",
            prompt=self.prompt.ref,
            prompt_sha256=self.prompt.sha256,
            pii_redacted=redaction.found,
        )
        if pattern := detect_injection(question):
            return replace(base, text=BLOCKED[lang], reason="blocked_input", guard_detail=pattern)
        question = redaction.text

        # One embedding serves both the learned injection check and retrieval.
        vector = self.retriever.dense.embed_query(question)
        if self.injection_classifier is not None:
            p = self.injection_classifier.probability(vector)
            if p > self.injection_classifier.threshold:
                detail = f"injection classifier p={p:.2f}"
                return replace(
                    base, text=BLOCKED[lang], reason="blocked_input", guard_detail=detail
                )

        hits = self.retriever.search(question, k=self.top_k, lang=lang, vector=vector)
        sources = number_sources(hits)
        fallback = replace(base, retrieved=sources)
        # Nothing relevant in the help center: answer with the fallback, don't call the model.
        if not sources or sources[0].score < self.min_score:
            return fallback

        completion = self.llm.complete(
            self.prompt.render(
                language_name=LANGUAGE_NAMES[lang],
                no_answer=NO_ANSWER,
                canary=CANARY,
                sources=format_sources(sources),
                question=question,
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
        # An empty reply is a failure (e.g. cut off by max_tokens), not the model declining:
        # the customer still gets the fallback, but evals and logs must be able to tell them apart.
        if not completion.text:
            return replace(generated, reason="generation_failed")
        if NO_ANSWER in completion.text:
            return replace(generated, reason="model_no_answer")

        # Output guards: never show a leaked system prompt or an answer its sources don't support.
        if CANARY in completion.text:
            return replace(generated, reason="blocked_output", guard_detail="canary leaked")
        by_number = {s.n: s for s in sources}
        cited = [by_number[n] for n in cited_numbers(completion.text) if n in by_number]
        if not cited:
            return replace(generated, reason="ungrounded", guard_detail="no valid citation")
        unsupported = unsupported_numbers(completion.text, [c.text for c in cited], question)
        if unsupported:
            detail = f"numbers not in cited sources: {', '.join(unsupported)}"
            return replace(generated, reason="ungrounded", cited=cited, guard_detail=detail)
        return replace(generated, text=completion.text, reason="answered", cited=cited)
