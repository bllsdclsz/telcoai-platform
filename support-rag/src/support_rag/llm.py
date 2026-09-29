"""Provider-agnostic chat models: LiteLLM routes one interface to local and hosted providers.

Switching provider is configuration, e.g. ``ollama_chat/granite4.2:8b`` (local, default) or
``azure/<deployment>`` with that provider's API key set.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]  # parsed JSON; {"_invalid": raw} if the model sent broken JSON


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str = "stop"  # "length" = cut off by max_tokens
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion: ...


def clean_output(text: str) -> str:
    """Drop reasoning some models emit before the answer.

    Handles complete ``<think>...</think>`` blocks and a dangling ``</think>`` whose opening tag
    was consumed by the chat template (seen with Granite 4.2): everything before it is reasoning.
    """
    text = _THINK.sub("", text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


class LiteLLMChat:
    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        timeout: float = 120.0,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model, self.api_base, self.timeout = model, api_base, timeout
        # "none" turns off thinking on reasoning models (LiteLLM maps it per provider). Grounded
        # answers from given sources don't need it, and a thinking model can spend the whole
        # token budget reasoning and return no answer at all.
        self.reasoning_effort = reasoning_effort

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        import litellm

        start = time.perf_counter()
        response = litellm.completion(
            model=self.model,
            api_base=self.api_base,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
            **({"tools": tools} if tools else {}),
        )
        usage = getattr(response, "usage", None)
        message = response.choices[0].message
        return Completion(
            text=clean_output(message.content or ""),
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=1000 * (time.perf_counter() - start),
            finish_reason=response.choices[0].finish_reason or "stop",
            tool_calls=[_parse_tool_call(c) for c in (getattr(message, "tool_calls", None) or [])],
        )


def _parse_tool_call(call: Any) -> ToolCall:
    raw = call.function.arguments or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError:
        args = {"_invalid": raw}
    return ToolCall(call.function.name, args if isinstance(args, dict) else {"_invalid": raw})
