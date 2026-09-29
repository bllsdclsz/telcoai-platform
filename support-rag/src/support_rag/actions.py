"""Human-in-the-loop actions: the assistant may *propose* a goodwill credit, only a human approves.

Flow: the model calls the ``request_goodwill_credit`` tool -> the arguments are validated ->
deterministic checks run -> a *pending* request is stored -> an agent approves or rejects it
through the agent API. Every state change is appended to an event log (who, when, why).

Deliberate choices:
- Nothing is executed or auto-approved by the model. Code enforces the state machine.
- Policy checks are advice for the agent, not automatic rejections: the model extracts the
  outage duration, and extractions can be wrong (qwen2.5 turned "3 Tage" into 3 hours). The
  duration is therefore cross-checked against the customer's own words, and disagreements are
  flagged for the agent instead of silently deciding.
- Only schema violations (unknown service, implausible duration) are refused outright.
"""

import json
import re
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

GOODWILL_CREDIT = "goodwill_credit"
SERVICES = ("internet", "mobile", "tv")
POLICY_MIN_OUTAGE_HOURS = 24  # help center: credit on request if an outage lasts > 24 hours
MAX_OUTAGE_HOURS = 24 * 60

GOODWILL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_goodwill_credit",
        "description": (
            "Forward the customer's request for a goodwill credit after an outage to a human "
            "agent for review. Call it only when the customer explicitly asks for a credit, "
            "refund or compensation because a Nordalp service was down."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "enum": list(SERVICES)},
                "outage_hours": {
                    "type": "number",
                    "description": "How long the outage lasted, in HOURS (e.g. 3 days = 72)",
                },
                "summary": {"type": "string", "description": "One sentence, in English"},
            },
            "required": ["service", "outage_hours", "summary"],
        },
    },
}

TOOL_INSTRUCTIONS = (
    "You have one tool, request_goodwill_credit. If the customer asks for a credit, refund, "
    "compensation or goodwill gesture because a Nordalp service (internet, mobile or TV) was "
    "down, you MUST call request_goodwill_credit (outage_hours as a plain number of hours, "
    "e.g. 3 days = 72) instead of answering or replying NO_ANSWER. A human agent reviews the "
    "request. For every other question, do not call the tool: answer from the sources."
)

CREATED = {
    "de": "Ich habe Ihre Anfrage für eine Gutschrift an unser Team weitergeleitet (Referenz "
    "{ref}). Eine Mitarbeiterin oder ein Mitarbeiter prüft sie und meldet sich bei Ihnen.",
    "fr": "J'ai transmis votre demande de crédit à notre équipe (référence {ref}). Un "
    "collaborateur ou une collaboratrice l'examinera et reviendra vers vous.",
    "it": "Ho inoltrato la tua richiesta di accredito al nostro team (riferimento {ref}). Un "
    "collaboratore o una collaboratrice la esaminerà e ti ricontatterà.",
    "en": "I've forwarded your request for a credit to our team (reference {ref}). An agent "
    "will review it and get back to you.",
}
DUPLICATE = {
    "de": "Für Sie ist bereits eine Anfrage offen (Referenz {ref}). Sie wird von unserem Team "
    "geprüft.",
    "fr": "Une demande est déjà ouverte à votre nom (référence {ref}). Notre équipe l'examine.",
    "it": "C'è già una richiesta aperta a tuo nome (riferimento {ref}). Il nostro team la sta "
    "esaminando.",
    "en": "You already have an open request (reference {ref}). Our team is reviewing it.",
}

Status = Literal["pending", "approved", "rejected"]
_TRANSITIONS: dict[str, set[str]] = {"pending": {"approved", "rejected"}}

_HOURS = r"h|hr|hrs|hours?|std|stunden?|heures?|ore|ora"
_DAYS = r"d|days?|tage?n?|jours?|giorn[oi]"
_DURATION = re.compile(rf"(\d+(?:[.,]\d+)?)\s*({_HOURS}|{_DAYS})\b", re.IGNORECASE)


def parse_duration_hours(text: str) -> float | None:
    """Longest duration the customer wrote ("3 Tage", "30 hours", "2 jours"), in hours."""
    found = []
    for number, unit in _DURATION.findall(text):
        value = float(number.replace(",", "."))
        found.append(value * 24 if re.fullmatch(_DAYS, unit, re.IGNORECASE) else value)
    return max(found) if found else None


@dataclass
class ActionRequest:
    id: str
    type: str
    status: Status
    customer_id: str
    params: dict[str, Any]
    checks: dict[str, Any]
    request_id: str
    question: str  # redacted
    created_at: str
    decided_by: str | None = None
    decided_at: str | None = None
    note: str | None = None


@dataclass
class Proposal:
    outcome: Literal["created", "duplicate", "invalid"]
    action: ActionRequest | None = None
    problems: list[str] = field(default_factory=list)


class InvalidTransitionError(ValueError):
    """Raised when a decision does not follow pending -> approved | rejected."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ActionStore:
    """SQLite store: current state per action plus an append-only event log."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY, type TEXT, status TEXT, customer_id TEXT, params TEXT,
                checks TEXT, request_id TEXT, question TEXT, created_at TEXT,
                decided_by TEXT, decided_at TEXT, note TEXT);
            CREATE TABLE IF NOT EXISTS action_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT, at TEXT, actor TEXT,
                event TEXT, detail TEXT);
            """
        )

    def _row(self, row: sqlite3.Row) -> ActionRequest:
        d = dict(row)
        d["params"], d["checks"] = json.loads(d["params"]), json.loads(d["checks"])
        return ActionRequest(**d)

    def _event(self, action_id: str, actor: str, event: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO action_events (action_id, at, actor, event, detail) VALUES (?,?,?,?,?)",
            (action_id, _now(), actor, event, detail),
        )

    def create(self, action: ActionRequest) -> ActionRequest:
        with self.db:
            self.db.execute(
                "INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    action.id,
                    action.type,
                    action.status,
                    action.customer_id,
                    json.dumps(action.params),
                    json.dumps(action.checks),
                    action.request_id,
                    action.question,
                    action.created_at,
                    None,
                    None,
                    None,
                ),
            )
            self._event(action.id, "assistant", "proposed", json.dumps(action.params))
        return action

    def get(self, action_id: str) -> ActionRequest | None:
        row = self.db.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
        return self._row(row) if row else None

    def find(
        self, status: str | None = None, customer_id: str | None = None
    ) -> list[ActionRequest]:
        query, args = "SELECT * FROM actions WHERE 1=1", list[str]()
        if status:
            query, args = query + " AND status = ?", [*args, status]
        if customer_id:
            query, args = query + " AND customer_id = ?", [*args, customer_id]
        rows = self.db.execute(query + " ORDER BY created_at DESC", args).fetchall()
        return [self._row(r) for r in rows]

    def decide(self, action_id: str, decision: str, agent: str, note: str = "") -> ActionRequest:
        if not agent.strip():
            raise InvalidTransitionError("a decision needs a named agent")
        with self.db:
            action = self.get(action_id)
            if action is None:
                raise KeyError(action_id)
            if decision not in _TRANSITIONS.get(action.status, set()):
                raise InvalidTransitionError(
                    f"{action_id} is {action.status}, cannot become {decision}"
                )
            self.db.execute(
                "UPDATE actions SET status=?, decided_by=?, decided_at=?, note=? WHERE id=?",
                (decision, agent.strip(), _now(), note, action_id),
            )
            self._event(action_id, agent.strip(), decision, note)
        updated = self.get(action_id)
        assert updated is not None
        return updated

    def events(self, action_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT at, actor, event, detail FROM action_events WHERE action_id = ? ORDER BY seq",
            (action_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def _as_hours(value: Any) -> Any:
    """Normalize how models send numbers: 30, "30", or a schema echo {"value": 30}."""
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, str):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return value
    return value


def propose_goodwill_credit(
    store: ActionStore,
    *,
    customer_id: str,
    arguments: dict[str, Any],
    question: str,
    request_id: str,
) -> Proposal:
    """Validate the model's tool call and file a pending request for a human agent."""
    problems = []
    service = arguments.get("service")
    hours = _as_hours(arguments.get("outage_hours"))
    if "_invalid" in arguments:
        problems.append("tool arguments are not valid JSON")
    if service not in SERVICES:
        problems.append(f"unknown service {service!r}")
    if not isinstance(hours, int | float) or not 0 < hours <= MAX_OUTAGE_HOURS:
        problems.append(f"implausible outage_hours {hours!r}")
    if problems or not isinstance(hours, int | float):
        return Proposal("invalid", problems=problems)
    hours = float(hours)

    open_requests = store.find(status="pending", customer_id=customer_id)
    if open_requests:
        return Proposal("duplicate", action=open_requests[0])

    in_text = parse_duration_hours(question)
    checks = {
        "policy_min_hours": POLICY_MIN_OUTAGE_HOURS,
        "claimed_hours": hours,
        "eligible_by_claim": hours > POLICY_MIN_OUTAGE_HOURS,
        "hours_in_customer_text": in_text,
        # The model's number disagrees with what the customer wrote: the agent must look.
        "extraction_mismatch": in_text is not None
        and abs(in_text - hours) > max(1.0, 0.1 * in_text),
    }
    action = ActionRequest(
        id=f"GC-{secrets.token_hex(3).upper()}",
        type=GOODWILL_CREDIT,
        status="pending",
        customer_id=customer_id,
        params={
            "service": service,
            "outage_hours": hours,
            "summary": str(arguments.get("summary", ""))[:300],
        },
        checks=checks,
        request_id=request_id,
        question=question,
        created_at=_now(),
    )
    return Proposal("created", action=store.create(action))


def evaluate_actions(assistant: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Proposal recall/precision and duration extraction on eval/actions.yaml.

    ``assistant`` must have an (ideally in-memory) action store; each case uses its own
    customer id so the one-open-request rule does not interfere.
    """
    rows = []
    for i, case in enumerate(cases):
        answer = assistant.ask(case["q"], case["lang"], customer_id=f"EVAL-{i:03d}")
        action = assistant.actions.get(answer.action_id) if answer.action_id else None
        hours = action.params["outage_hours"] if action else None
        rows.append(
            {
                "q": case["q"],
                "lang": case["lang"],
                "expected": case["propose"],
                "proposed": answer.reason in ("action_proposed", "action_duplicate"),
                "reason": answer.reason,
                "expected_hours": case.get("hours"),
                "extracted_hours": hours,
                "hours_ok": None
                if not (action and case.get("hours"))
                else abs(hours - case["hours"]) <= max(1.0, 0.1 * case["hours"]),
                "flagged_mismatch": action.checks["extraction_mismatch"] if action else None,
            }
        )
    positives = [r for r in rows if r["expected"]]
    proposed = [r for r in rows if r["proposed"]]
    extracted = [r for r in rows if r["hours_ok"] is not None]
    wrong = [r for r in extracted if not r["hours_ok"]]
    return {
        "recall": sum(r["proposed"] for r in positives) / len(positives),
        "precision": sum(r["expected"] for r in proposed) / len(proposed) if proposed else 1.0,
        "hours_accuracy": sum(r["hours_ok"] for r in extracted) / len(extracted)
        if extracted
        else None,
        "wrong_hours_flagged": sum(bool(r["flagged_mismatch"]) for r in wrong) / len(wrong)
        if wrong
        else None,
        "rows": rows,
    }
