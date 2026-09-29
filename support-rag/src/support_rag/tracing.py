"""Request tracing with MLflow: one trace per question, one span per pipeline step.

Traces double as the audit log: each carries tags for the request id, language, outcome,
guardrail decision, PII *types* found, prompt version and hash, model and cited articles.
Only the redacted question is ever recorded, never the raw text, so personal data stays out of
the trace store.

Tracing is optional: without ``RAG_TRACE_MLFLOW_URI`` every call is a no-op, so tests and
deployments without MLflow are unaffected.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol


class Span(Protocol):
    def set_inputs(self, inputs: Any) -> None: ...
    def set_outputs(self, outputs: Any) -> None: ...
    def set_attributes(self, attributes: dict[str, Any]) -> None: ...


class _NoopSpan:
    def set_inputs(self, inputs: Any) -> None:
        pass

    def set_outputs(self, outputs: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass


class Tracer:
    """Thin wrapper over MLflow tracing that does nothing when tracing is not configured."""

    def __init__(self, tracking_uri: str | None = None, experiment: str | None = None) -> None:
        self.enabled = bool(tracking_uri)
        if tracking_uri:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment or "support-assistant-traces")

    @contextmanager
    def span(
        self, name: str, span_type: str, inputs: dict[str, Any] | None = None
    ) -> Iterator[Span]:
        if not self.enabled:
            yield _NoopSpan()
            return
        import mlflow

        with mlflow.start_span(name=name, span_type=span_type) as span:
            if inputs is not None:
                span.set_inputs(inputs)
            yield span

    def tag_trace(self, request_id: str, tags: dict[str, str], request: str, response: str) -> None:
        """Audit tags on the current trace; call inside the root span."""
        if not self.enabled:
            return
        import mlflow

        mlflow.update_current_trace(
            client_request_id=request_id,
            tags={k: v for k, v in tags.items() if v},
            request_preview=request[:300],
            response_preview=response[:300],
        )


NOOP_TRACER = Tracer()


def audit_log(
    tracking_uri: str,
    experiment: str,
    *,
    reason: str | None = None,
    lang: str | None = None,
    request_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Recent traces as audit records, newest first, filtered by outcome, language or request."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return []
    filters = [
        f"tags.{key} = '{value}'" for key, value in (("reason", reason), ("lang", lang)) if value
    ]
    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id],
        filter_string=" AND ".join(filters) or None,
        max_results=limit if request_id is None else 1000,
        order_by=["timestamp_ms DESC"],
        return_type="list",
        flush=True,  # traces are exported asynchronously (no latency on the request path)
    )
    records = []
    for trace in traces:
        info = trace.info
        if request_id and info.client_request_id != request_id:
            continue
        tags = info.tags
        records.append(
            {
                "time_ms": info.request_time,
                "request_id": info.client_request_id,
                "trace_id": info.trace_id,
                "duration_ms": info.execution_duration,
                **{
                    k: tags.get(k, "")
                    for k in ("lang", "reason", "guard", "pii_types", "prompt", "model", "cited")
                },
                "question": info.request_preview,
            }
        )
    return records[:limit]
