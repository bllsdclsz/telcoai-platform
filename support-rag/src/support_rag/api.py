"""HTTP API for the support assistant, plus the agent API for approving proposed actions."""

import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from support_rag.actions import ActionRequest, ActionStore, InvalidTransitionError
from support_rag.assistant import Answer, Assistant
from support_rag.config import Language, Settings
from support_rag.injection_model import InjectionClassifier
from support_rag.tracing import Tracer


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    lang: Language
    # Set by the authenticated channel (app, web) for logged-in customers; enables actions.
    customer_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{3,40}$")


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
    action_id: str | None = None


class Decision(BaseModel):
    agent: str = Field(min_length=2, max_length=80)
    note: str = Field(default="", max_length=500)


class ActionOut(BaseModel):
    id: str
    type: str
    status: str
    customer_id: str
    params: dict[str, Any]
    checks: dict[str, Any]
    question: str
    request_id: str
    created_at: str
    decided_by: str | None
    decided_at: str | None
    note: str | None
    events: list[dict[str, Any]] = []


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
        tracer=Tracer(s.trace_mlflow_uri, s.trace_experiment),
        actions=ActionStore(s.actions_db) if s.actions_db else None,
    )


def to_response(answer: Answer) -> AskResponse:
    return AskResponse(
        request_id=answer.request_id,
        answer=answer.text,
        answered=answer.answered,
        reason=answer.reason,
        sources=[SourceOut(n=s.n, title=s.title, url=s.url) for s in answer.cited],
        prompt=answer.prompt,
        model=answer.model,
        latency_ms=round(answer.latency_ms, 1),
        action_id=answer.action_id or None,
    )


def to_action_out(action: ActionRequest, store: ActionStore) -> ActionOut:
    return ActionOut(
        id=action.id,
        type=action.type,
        status=action.status,
        customer_id=action.customer_id,
        params=action.params,
        checks=action.checks,
        question=action.question,
        request_id=action.request_id,
        created_at=action.created_at,
        decided_by=action.decided_by,
        decided_at=action.decided_at,
        note=action.note,
        events=store.events(action.id),
    )


def create_app(
    factory: Callable[[], Assistant] = build_assistant, agent_token: str | None = None
) -> FastAPI:
    token = agent_token if agent_token is not None else Settings().agent_token

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.assistant = factory()
        yield

    app = FastAPI(title="Nordalp Support Assistant", version="0.1.0", lifespan=lifespan)

    def store_of(request: Request) -> ActionStore:
        store: ActionStore | None = request.app.state.assistant.actions
        if store is None:
            raise HTTPException(503, "actions are disabled (RAG_ACTIONS_DB)")
        return store

    def require_agent(x_agent_token: Annotated[str | None, Header()] = None) -> None:
        if not token:
            raise HTTPException(503, "agent API disabled: set RAG_AGENT_TOKEN")
        if not x_agent_token or not secrets.compare_digest(x_agent_token, token):
            raise HTTPException(401, "invalid agent token")

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "prompt": request.app.state.assistant.prompt.ref}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        answer = request.app.state.assistant.ask(
            body.question, body.lang, customer_id=body.customer_id
        )
        return to_response(answer)

    @app.get("/actions", dependencies=[Depends(require_agent)])
    def list_actions(
        request: Request, status: Literal["pending", "approved", "rejected"] | None = None
    ) -> list[ActionOut]:
        store = store_of(request)
        return [to_action_out(a, store) for a in store.find(status=status)]

    def decide(action_id: str, decision: str, body: Decision, request: Request) -> ActionOut:
        store = store_of(request)
        try:
            action = store.decide(action_id, decision, body.agent, body.note)
        except KeyError:
            raise HTTPException(404, f"no action {action_id}") from None
        except InvalidTransitionError as exc:
            raise HTTPException(409, str(exc)) from None
        return to_action_out(action, store)

    @app.post("/actions/{action_id}/approve", dependencies=[Depends(require_agent)])
    def approve(action_id: str, body: Decision, request: Request) -> ActionOut:
        return decide(action_id, "approved", body, request)

    @app.post("/actions/{action_id}/reject", dependencies=[Depends(require_agent)])
    def reject(action_id: str, body: Decision, request: Request) -> ActionOut:
        return decide(action_id, "rejected", body, request)

    return app
