"""Aggregates overall compression stats across all prior guardrails."""

from __future__ import annotations

from typing import Any

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except Exception:  # pragma: no cover
    class CustomGuardrail:  # type: ignore[no-redef]
        pass


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "tool_result":
                    parts.append(_content_to_text(item.get("content", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        if isinstance(m, dict):
            total += len(_content_to_text(m.get("content"))) // 4 + 1
    return total


class CompressionSummaryGuardrail(CustomGuardrail):
    def __init__(self, guardrail_name: str | None = None, **kwargs: Any) -> None:
        self.guardrail_name = guardrail_name or "compression-summary"
        try:
            super().__init__(guardrail_name=self.guardrail_name, **kwargs)  # type: ignore[misc]
        except Exception:
            pass

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        if input_type != "request":
            return inputs
        if not isinstance(inputs, dict):
            return inputs
        structured = inputs.get("structured_messages")
        if not isinstance(structured, list):
            return inputs
        tokens_after = _estimate_tokens(structured)
        original = None
        if isinstance(request_data, dict):
            bucket = request_data.get("litellm_metadata")
            if isinstance(bucket, dict):
                original = bucket.get("_compression_original_tokens")
            if original is None:
                # Backwards compatibility with an older DCP release that wrote
                # the marker at the top level.
                original = request_data.get("_compression_original_tokens")
        if not isinstance(original, int) or original <= 0:
            original = tokens_after
        tokens_saved = original - tokens_after
        ratio = round(1 - (tokens_after / original), 4) if original > 0 else 0.0
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={
                    "tokens_before": original,
                    "tokens_after": tokens_after,
                    "tokens_saved": tokens_saved,
                    "compression_ratio": ratio,
                },
                request_data=request_data,
                guardrail_status="success",
            )
        except Exception:
            pass
        return inputs


compression_summary_guardrail = CompressionSummaryGuardrail()
guardrail = compression_summary_guardrail