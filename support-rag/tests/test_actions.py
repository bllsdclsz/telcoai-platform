from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from rag_fakes import HashingDense

from support_rag.actions import (
    CREATED,
    ActionStore,
    InvalidTransitionError,
    parse_duration_hours,
    propose_goodwill_credit,
)
from support_rag.api import create_app
from support_rag.assistant import Assistant
from support_rag.config import Settings
from support_rag.llm import Completion, ToolCall
from support_rag.prompts import load_prompt
from support_rag.retrieve import Retriever

GOOD_ARGS = {"service": "internet", "outage_hours": 72, "summary": "3-day outage, asks credit"}


@pytest.fixture
def store() -> ActionStore:
    return ActionStore(":memory:")


def propose(
    store: ActionStore,
    args: dict[str, Any],
    question: str = "Internet 3 Tage weg",
    customer: str = "C-1001",
):
    return propose_goodwill_credit(
        store, customer_id=customer, arguments=args, question=question, request_id="req-1"
    )


@pytest.mark.parametrize(
    ("text", "hours"),
    [
        ("Mein Internet war 3 Tage lang weg", 72),
        ("down for about 30 hours", 30),
        ("depuis 2 jours", 48),
        ("da 36 ore", 36),
        ("für 1,5 Tage", 36),
        ("seit gestern", None),
    ],
)
def test_duration_is_parsed_from_the_customer_text(text: str, hours: float | None) -> None:
    assert parse_duration_hours(text) == hours


def test_valid_proposal_is_filed_as_pending(store: ActionStore) -> None:
    p = propose(store, GOOD_ARGS)
    assert p.outcome == "created" and p.action is not None
    assert p.action.status == "pending" and p.action.id.startswith("GC-")
    assert p.action.checks["eligible_by_claim"] is True
    assert p.action.checks["extraction_mismatch"] is False
    assert [e["event"] for e in store.events(p.action.id)] == ["proposed"]


def test_model_extraction_that_contradicts_the_customer_is_flagged(store: ActionStore) -> None:
    # Seen with qwen2.5: "3 Tage" became outage_hours=3.
    p = propose(store, {**GOOD_ARGS, "outage_hours": 3}, question="Mein Internet war 3 Tage weg")
    assert p.outcome == "created" and p.action is not None  # not auto-rejected ...
    assert p.action.checks["extraction_mismatch"] is True  # ... but flagged for the agent
    assert p.action.checks["eligible_by_claim"] is False


@pytest.mark.parametrize(
    "args",
    [
        {**GOOD_ARGS, "service": "fridge"},
        {**GOOD_ARGS, "outage_hours": -5},
        {**GOOD_ARGS, "outage_hours": 99999},
        {**GOOD_ARGS, "outage_hours": "a lot"},
        {"_invalid": "{broken"},
    ],
)
def test_invalid_tool_arguments_are_refused(store: ActionStore, args: dict[str, Any]) -> None:
    p = propose(store, args)
    assert p.outcome == "invalid" and p.problems and store.find() == []


def test_one_open_request_per_customer(store: ActionStore) -> None:
    first = propose(store, GOOD_ARGS)
    second = propose(store, GOOD_ARGS)
    assert second.outcome == "duplicate" and second.action is not None
    assert first.action is not None and second.action.id == first.action.id
    assert propose(store, GOOD_ARGS, customer="C-2002").outcome == "created"


def test_state_machine_and_event_log(store: ActionStore) -> None:
    p = propose(store, GOOD_ARGS)
    assert p.action is not None
    approved = store.decide(p.action.id, "approved", "Anna Agent", "confirmed in network logs")
    assert (approved.status, approved.decided_by) == ("approved", "Anna Agent")
    with pytest.raises(InvalidTransitionError):
        store.decide(p.action.id, "rejected", "Ben Agent")  # already decided
    events = store.events(p.action.id)
    assert [(e["event"], e["actor"]) for e in events] == [
        ("proposed", "assistant"),
        ("approved", "Anna Agent"),
    ]


def test_decision_needs_a_named_agent(store: ActionStore) -> None:
    p = propose(store, GOOD_ARGS)
    assert p.action is not None
    with pytest.raises(InvalidTransitionError, match="named agent"):
        store.decide(p.action.id, "approved", "  ")


# --- through the assistant and the API


class ToolChat:
    """Calls the tool when it is offered; otherwise answers normally."""

    model = "fake/tools"

    def __init__(self, arguments: dict[str, Any] | None = None) -> None:
        self.arguments, self.offered = arguments or GOOD_ARGS, []

    def complete(self, messages, *, temperature, max_tokens, tools=None) -> Completion:  # type: ignore[no-untyped-def]
        self.offered.append(bool(tools))
        if tools:
            return Completion(
                "",
                self.model,
                10,
                5,
                1.0,
                tool_calls=[ToolCall("request_goodwill_credit", self.arguments)],
            )
        return Completion("Restart the box first [1].", self.model, 10, 5, 1.0)


QUESTION = "My internet was down for 3 days, I want a credit for the outage"


def assistant(
    client: QdrantClient, settings: Settings, store: ActionStore, chat: ToolChat
) -> Assistant:
    return Assistant(
        Retriever(client, "test", HashingDense()),
        chat,
        load_prompt(settings.prompts_dir, "answer"),
        actions=store,
    )


def test_logged_in_customer_gets_a_reference(
    client: QdrantClient, settings: Settings, store: ActionStore
) -> None:
    chat = ToolChat()
    answer = assistant(client, settings, store, chat).ask(QUESTION, "de", customer_id="C-1001")
    assert answer.reason == "action_proposed" and chat.offered == [True]
    assert answer.text == CREATED["de"].format(ref=answer.action_id)
    (action,) = store.find(status="pending")
    assert action.id == answer.action_id and action.request_id == answer.request_id


def test_anonymous_customer_is_never_offered_the_tool(
    client: QdrantClient, settings: Settings, store: ActionStore
) -> None:
    chat = ToolChat()
    answer = assistant(client, settings, store, chat).ask(QUESTION, "en")
    assert chat.offered == [False] and answer.reason == "answered" and store.find() == []


def test_invalid_tool_call_files_nothing(
    client: QdrantClient, settings: Settings, store: ActionStore
) -> None:
    chat = ToolChat({"service": "fridge", "outage_hours": 72, "summary": "x"})
    answer = assistant(client, settings, store, chat).ask(QUESTION, "fr", customer_id="C-1001")
    assert answer.reason == "action_invalid" and store.find() == []


def test_agent_api(client: QdrantClient, settings: Settings, store: ActionStore) -> None:
    bot = assistant(client, settings, store, ToolChat())
    with TestClient(create_app(lambda: bot, agent_token="s3cret")) as http:
        body = http.post(
            "/ask", json={"question": QUESTION, "lang": "en", "customer_id": "C-1001"}
        ).json()
        action_id = body["action_id"]
        assert body["reason"] == "action_proposed" and action_id

        assert http.get("/actions").status_code == 401
        assert http.get("/actions", headers={"X-Agent-Token": "wrong"}).status_code == 401
        auth = {"X-Agent-Token": "s3cret"}
        (pending,) = http.get("/actions?status=pending", headers=auth).json()
        assert pending["id"] == action_id and pending["checks"]["extraction_mismatch"] is False

        decision = {"agent": "Anna Agent", "note": "outage confirmed"}
        ok = http.post(f"/actions/{action_id}/approve", json=decision, headers=auth)
        assert ok.status_code == 200 and ok.json()["status"] == "approved"
        assert [e["event"] for e in ok.json()["events"]] == ["proposed", "approved"]
        assert (
            http.post(f"/actions/{action_id}/reject", json=decision, headers=auth).status_code
            == 409
        )
        assert http.post("/actions/GC-NOPE/approve", json=decision, headers=auth).status_code == 404


def test_agent_api_disabled_without_token(
    client: QdrantClient, settings: Settings, store: ActionStore
) -> None:
    bot = assistant(client, settings, store, ToolChat())
    with TestClient(create_app(lambda: bot, agent_token="")) as http:
        assert http.get("/actions", headers={"X-Agent-Token": "x"}).status_code == 503


def test_customer_id_format_is_validated(
    client: QdrantClient, settings: Settings, store: ActionStore
) -> None:
    bot = assistant(client, settings, store, ToolChat())
    with TestClient(create_app(lambda: bot, agent_token="t")) as http:
        bad = {"question": QUESTION, "lang": "en", "customer_id": "x; DROP TABLE"}
        assert http.post("/ask", json=bad).status_code == 422
