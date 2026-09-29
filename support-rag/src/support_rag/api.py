"""HTTP API for the support assistant."""

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from support_rag.assistant import Answer, Assistant
from support_rag.config import Language, Settings
from support_rag.injection_model import InjectionClassifier


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    lang: Language


class SourceOut(BaseModel):
    n: int
    title: str
    url: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    answered: bool
    reason: str
    sources: list[SourceOut]
    prompt: str
    model: str
    latency_ms: float


def build_assistant(settings: Settings | None = None) -> Assistant:
    from support_rag.embeddings import FastEmbedDense, FastEmbedSparse
    from support_rag.index import connect
    from support_rag.llm import LiteLLMChat
    from support_rag.prompts import load_prompt
    from support_rag.retrieve import Retriever

    s = settings or Settings()
    sparse = FastEmbedSparse(s.sparse_model) if s.sparse_model else None
    classifier = None
    if s.injection_classifier and s.injection_classifier.exists():
        classifier = InjectionClassifier.load(s.injection_classifier)
        if classifier.embedding_model != s.dense_model:
            raise ValueError(
                f"injection classifier was trained on {classifier.embedding_model}, "
                f"but retrieval uses {s.dense_model}: retrain with `rag train-injection`"
            )
    return Assistant(
        Retriever(connect(s), s.collection, FastEmbedDense(s.dense_model), sparse),
        LiteLLMChat(s.llm_model, api_base=s.llm_api_base, reasoning_effort=s.reasoning_effort),
        load_prompt(s.prompts_dir, "answer", s.prompt_version),
        top_k=s.answer_top_k,
        min_score=s.min_retrieval_score,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        injection_classifier=classifier,
    )


def to_response(answer: Answer) -> AskResponse:
    return AskResponse(
        request_id=str(uuid.uuid4()),
        answer=answer.text,
        answered=answer.answered,
        reason=answer.reason,
        sources=[SourceOut(n=s.n, title=s.title, url=s.url) for s in answer.cited],
        prompt=answer.prompt,
        model=answer.model,
        latency_ms=round(answer.latency_ms, 1),
    )


def create_app(factory: Callable[[], Assistant] = build_assistant) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.assistant = factory()
        yield

    app = FastAPI(title="Nordalp Support Assistant", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "prompt": request.app.state.assistant.prompt.ref}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        return to_response(request.app.state.assistant.ask(body.question, body.lang))

    return app
