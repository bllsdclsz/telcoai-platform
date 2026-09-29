"""Provider-agnostic chat models: LiteLLM routes one interface to Ollama, Anthropic, OpenAI, Azure.

Switching provider is configuration, e.g. ``ollama_chat/granite4.2:3b`` (local, default) or
``anthropic/<model>`` with ``ANTHROPIC_API_KEY`` set.
"""

import re
import time
from dataclasses import dataclass
from typing import Protocol

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ChatModel(Protocol):
    model: str

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> Completion: ...


def clean_output(text: str) -> str:
    """Drop reasoning blocks some models emit before the answer."""
    return _THINK.sub("", text).strip()


class LiteLLMChat:
    def __init__(self, model: str, api_base: str | None = None, timeout: float = 120.0) -> None:
        self.model, self.api_base, self.timeout = model, api_base, timeout

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
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
        )
        usage = getattr(response, "usage", None)
        return Completion(
            text=clean_output(response.choices[0].message.content or ""),
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=1000 * (time.perf_counter() - start),
        )
