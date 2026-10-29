"""LLM-driven summarization guardrail.

Only runs when the estimated prompt size exceeds the configured threshold.
Below the threshold it does nothing and never calls an LLM.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from fastapi import HTTPException
except Exception:  # pragma: no cover
    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int = 400, detail: Any = None) -> None:
            self.status_code = status_code
            self.detail = detail

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except Exception:  # pragma: no cover
    class CustomGuardrail:  # type: ignore[no-redef]
        pass

from litellm_plugin_collection._model_context import detect_context_limit


def _session_id(request_data: dict) -> str:
    if not isinstance(request_data, dict):
        return "default"
    for key in ("session_id", "x-opencode-session", "x-opencode-session-id"):
        val = request_data.get(key)
        if val:
            return str(val)
    for bucket_name in ("metadata", "litellm_metadata"):
        metadata = request_data.get(bucket_name)
        if isinstance(metadata, dict):
            for key in ("session_id", "user_api_key_session_id"):
                val = metadata.get(key)
                if val:
                    return str(val)
    return "default"


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


def _estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            total += len(_content_to_text(msg.get("content"))) // 4 + 1
    return total


def _split_index(messages: list[Any], preserve_recent_turns: int) -> int | None:
    if preserve_recent_turns <= 0:
        preserve_recent_turns = 1
    target = len(messages) - preserve_recent_turns
    if target <= 0:
        return None
    for i in range(min(target, len(messages) - 1), 0, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            return i
    return None


def _resolve_model_params(
    model_name: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str | None, str | None]:
    if api_base or api_key:
        return model_name, api_base, api_key
    try:
        import litellm

        for entry in getattr(litellm, "model_list", []) or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("model_name") != model_name and entry.get("model") != model_name:
                continue
            params = entry.get("litellm_params", {})
            if isinstance(params, dict):
                return (
                    params.get("model", model_name),
                    params.get("api_base"),
                    params.get("api_key"),
                )
    except Exception:
        pass
    return model_name, None, None


class LLMSummarizerGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str | None = None,
        summary_model: str = "qwen3.8-flash-next",
        summary_api_base: str | None = None,
        summary_api_key: str | None = None,
        summary_threshold_tokens: int = 0,
        summary_target_tokens: int = 4096,
        preserve_recent_turns: int = 4,
        max_summary_calls: int = 3,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.guardrail_name = guardrail_name or "llm-summarizer"
        self.summary_model = summary_model
        self.summary_api_base = summary_api_base
        self.summary_api_key = summary_api_key
        self.summary_threshold_tokens = max(0, int(summary_threshold_tokens))
        self.summary_target_tokens = max(64, int(summary_target_tokens))
        self.preserve_recent_turns = max(1, int(preserve_recent_turns))
        self.max_summary_calls = max(1, int(max_summary_calls))
        self.temperature = float(temperature)
        self._calls: dict[str, int] = {}
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
        if not isinstance(structured, list) or not structured:
            return inputs

        request_data = request_data or {}
        if request_data.get("skip_llm_summarizer") is True:
            return inputs
        metadata = request_data.get("metadata")
        if isinstance(metadata, dict) and metadata.get("skip_llm_summarizer") is True:
            return inputs
        litellm_metadata = request_data.get("litellm_metadata")
        if isinstance(litellm_metadata, dict) and litellm_metadata.get("skip_llm_summarizer") is True:
            return inputs

        total_tokens = _estimate_tokens(structured)
        threshold = self.summary_threshold_tokens
        if threshold <= 0:
            threshold = detect_context_limit(request_data, inputs)
        if threshold <= 0:
            threshold = 128000
        if total_tokens <= threshold:
            return inputs

        sid = _session_id(request_data)
        calls = self._calls.get(sid, 0)
        if calls >= self.max_summary_calls:
            return inputs

        split = _split_index(structured, self.preserve_recent_turns)
        if split is None:
            return inputs
        older = structured[:split]
        recent = structured[split:]
        if not older:
            return inputs

        older_text = "\n\n".join(
            f"{msg.get('role', 'user')}: {_content_to_text(msg.get('content'))}"
            for msg in older
            if isinstance(msg, dict)
        )
        if not older_text.strip():
            return inputs

        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the earlier conversation concisely. Preserve key "
                    "facts, decisions, constraints, errors, and tool results. "
                    "Output only the summary."
                ),
            },
            {"role": "user", "content": older_text},
        ]

        try:
            import litellm

            model, api_base, api_key = _resolve_model_params(
                self.summary_model,
                self.summary_api_base,
                self.summary_api_key,
            )
            if api_base and "/" not in model:
                model = f"openai/{model}"
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": prompt,
                "max_tokens": self.summary_target_tokens,
                "temperature": self.temperature,
                "reasoning_effort": "low",
            }
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key
            response = await litellm.acompletion(**kwargs)
            summary_text = ""
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    if not isinstance(content, str):
                        content = msg.get("reasoning_content")
                    if isinstance(content, str):
                        summary_text = content
            else:
                choices = getattr(response, "choices", None) or []
                if choices:
                    msg = getattr(choices[0], "message", None)
                    content = getattr(msg, "content", None) if msg is not None else None
                    if not isinstance(content, str):
                        content = getattr(msg, "reasoning_content", None) if msg is not None else None
                    if isinstance(content, str):
                        summary_text = content
        except Exception:
            return inputs

        if not summary_text or not summary_text.strip():
            return inputs

        summary_msg = {
            "role": "system",
            "content": f"[LLM Summary of earlier conversation]\n{summary_text.strip()}",
        }
        new_structured = [summary_msg] + [dict(m) for m in recent]
        self._calls[sid] = calls + 1
        return {**inputs, "structured_messages": new_structured}


llm_summarizer = LLMSummarizerGuardrail()
guardrail = llm_summarizer