"""Answer customer questions: retrieve articles, generate a cited answer, or fall back safely."""

import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from support_rag.actions import (
    CREATED,
    DUPLICATE,
    GOODWILL_TOOL,
    TOOL_INSTRUCTIONS,
    ActionStore,
    propose_goodwill_credit,
)
from support_rag.guardrails import CANARY, detect_injection, redact_pii, unsupported_numbers
from support_rag.injection_model import InjectionClassifier
from support_rag.llm import ChatModel, Completion
from support_rag.prompts import PromptTemplate
from support_rag.retrieve import Hit, Retriever
from support_rag.tracing import NOOP_TRACER, Tracer

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
    "action_proposed",  # a pending action was filed for a human agent
    "action_duplicate",  # the customer already has an open request
    "action_invalid",  # the model's tool call failed validation
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
    question: str = ""  # redacted: safe to log and trace
    request_id: str = ""
    total_ms: float = 0.0
    action_id: str = ""

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
        tracer: Tracer = NOOP_TRACER,
        actions: ActionStore | None = None,
    ) -> None:
        self.retriever, self.llm, self.prompt = retriever, llm, prompt
        self.top_k, self.min_score = top_k, min_score
        self.temperature, self.max_tokens = temperature, max_tokens
        self.injection_classifier = injection_classifier
        self.tracer = tracer
        self.actions = actions

    def ask(
        self,
        question: str,
        lang: str,
        request_id: str | None = None,
        customer_id: str | None = None,
    ) -> Answer:
        """Answer one question; with tracing on, this is one MLflow trace (the audit record)."""
        request_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()
        with self.tracer.span("support_answer", "CHAIN") as root:
            answer = self._answer(question, lang, request_id, customer_id)
            answer = replace(
                answer,
                request_id=request_id,
                total_ms=round(1000 * (time.perf_counter() - start), 1),
            )
            # Only the redacted question is recorded: personal data never reaches the traces.
            root.set_inputs({"question": answer.question, "lang": lang})
            root.set_outputs({"answer": answer.text, "reason": answer.reason})
            self.tracer.tag_trace(request_id, audit_tags(answer), answer.question, answer.text)
        return answer

    def _answer(self, question: str, lang: str, request_id: str, customer_id: str | None) -> Answer:
        tracer = self.tracer
        # Actions are offered only to an identified customer and when an action store exists.
        tools_on = bool(customer_id) and self.actions is not None
        # Input guards: redact personal data first, so it reaches neither the model nor logs.
        with tracer.span("input_guards", "GUARDRAIL") as span:
            redaction = redact_pii(question.strip())
            pattern = detect_injection(question)
            span.set_outputs({"pii_types": redaction.found, "injection_rule": pattern})
        base = Answer(
            FALLBACK[lang],
            lang,
            "no_relevant_sources",
            prompt=self.prompt.ref,
            prompt_sha256=self.prompt.sha256,
            pii_redacted=redaction.found,
            question=redaction.text,
        )
        if pattern:
            return replace(base, text=BLOCKED[lang], reason="blocked_input", guard_detail=pattern)
        question = redaction.text

        # One embedding serves both the learned injection check and retrieval.
        with tracer.span("embed_query", "EMBEDDING", {"text": question}):
            vector = self.retriever.dense.embed_query(question)
        if self.injection_classifier is not None:
            with tracer.span("injection_classifier", "GUARDRAIL") as span:
                p = self.injection_classifier.probability(vector)
                blocked = p > self.injection_classifier.threshold
                span.set_outputs({"probability": round(p, 4), "blocked": blocked})
            if blocked:
                detail = f"injection classifier p={p:.2f}"
                return replace(
                    base, text=BLOCKED[lang], reason="blocked_input", guard_detail=detail
                )

        with tracer.span("retrieve", "RETRIEVER", {"query": question, "lang": lang}) as span:
            hits = self.retriever.search(question, k=self.top_k, lang=lang, vector=vector)
            sources = number_sources(hits)
            span.set_outputs(
                [
                    {"n": s.n, "article_id": s.article_id, "score": round(s.score, 4)}
                    for s in sources
                ]
            )
        fallback = replace(base, retrieved=sources)
        # Nothing relevant in the help center: answer with the fallback, don't call the model.
        if not sources or sources[0].score < self.min_score:
            return fallback

        with tracer.span(
            "generate", "CHAT_MODEL", {"model": self.llm.model, "prompt": self.prompt.ref}
        ) as span:
            # A provider error (timeout, 5xx, malformed tool call) must never crash a customer
            # conversation: it becomes the fallback answer, recorded as generation_failed.
            error = ""
            try:
                completion = self.llm.complete(
                    self.prompt.render(
                        language_name=LANGUAGE_NAMES[lang],
                        no_answer=NO_ANSWER,
                        canary=CANARY,
                        tool_instructions=TOOL_INSTRUCTIONS if tools_on else "",
                        sources=format_sources(sources),
                        question=question,
                    ),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **({"tools": [GOODWILL_TOOL]} if tools_on else {}),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                completion = Completion("", self.llm.model, 0, 0, 0.0, finish_reason="error")
            span.set_outputs(
                {
                    "text": completion.text,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                    "finish_reason": completion.finish_reason,
                    "tool_calls": [c.name for c in completion.tool_calls],
                }
            )
        generated = replace(
            fallback,
            guard_detail=error,
            model=completion.model,
            latency_ms=completion.latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        tool_call = next(
            (c for c in completion.tool_calls if c.name == GOODWILL_TOOL["function"]["name"]), None
        )
        if tools_on and tool_call is not None and customer_id and self.actions is not None:
            return self._propose(generated, tool_call.arguments, question, request_id, customer_id)
        with tracer.span("output_guards", "GUARDRAIL") as span:
            result = self._check_output(generated, completion.text, sources, question)
            span.set_outputs({"reason": result.reason, "detail": result.guard_detail})
        return result

    def _propose(
        self,
        generated: Answer,
        arguments: dict[str, Any],
        question: str,
        request_id: str,
        customer_id: str,
    ) -> Answer:
        """The model asked for an action: file it for a human, never execute it."""
        assert self.actions is not None
        with self.tracer.span("propose_action", "TOOL", {"arguments": arguments}) as span:
            proposal = propose_goodwill_credit(
                self.actions,
                customer_id=customer_id,
                arguments=arguments,
                question=question,
                request_id=request_id,
            )
            span.set_outputs(
                {
                    "outcome": proposal.outcome,
                    "action_id": proposal.action.id if proposal.action else None,
                    "checks": proposal.action.checks if proposal.action else None,
                    "problems": proposal.problems,
                }
            )
        lang = generated.lang
        if proposal.outcome == "invalid" or proposal.action is None:
            detail = "; ".join(proposal.problems)
            return replace(generated, reason="action_invalid", guard_detail=detail)
        template = CREATED if proposal.outcome == "created" else DUPLICATE
        reason: Reason = "action_proposed" if proposal.outcome == "created" else "action_duplicate"
        return replace(
            generated,
            text=template[lang].format(ref=proposal.action.id),
            reason=reason,
            action_id=proposal.action.id,
        )

    def _check_output(
        self, generated: Answer, text: str, sources: list[Source], question: str
    ) -> Answer:
        # An empty reply is a failure (e.g. cut off by max_tokens), not the model declining:
        # the customer still gets the fallback, but evals and logs must be able to tell them apart.
        if not text:
            return replace(generated, reason="generation_failed")
        if NO_ANSWER in text:
            return replace(generated, reason="model_no_answer")
        # Never show a leaked system prompt or an answer its sources don't support.
        if CANARY in text:
            return replace(generated, reason="blocked_output", guard_detail="canary leaked")
        by_number = {s.n: s for s in sources}
        cited = [by_number[n] for n in cited_numbers(text) if n in by_number]
        if not cited:
            return replace(generated, reason="ungrounded", guard_detail="no valid citation")
        unsupported = unsupported_numbers(text, [c.text for c in cited], question)
        if unsupported:
            detail = f"numbers not in cited sources: {', '.join(unsupported)}"
            return replace(generated, reason="ungrounded", cited=cited, guard_detail=detail)
        return replace(generated, text=text, reason="answered", cited=cited)


def audit_tags(answer: Answer) -> dict[str, str]:
    """Trace tags for audit queries (e.g. all blocked requests); PII only by type, never value."""
    return {
        "lang": answer.lang,
        "reason": answer.reason,
        "guard": answer.guard_detail[:250],
        "pii_types": ",".join(answer.pii_redacted),
        "prompt": answer.prompt,
        "prompt_sha256": answer.prompt_sha256[:12],
        "model": answer.model,
        "cited": ",".join(s.article_id for s in answer.cited),
        "total_ms": str(answer.total_ms),
        "action_id": answer.action_id,
    }
