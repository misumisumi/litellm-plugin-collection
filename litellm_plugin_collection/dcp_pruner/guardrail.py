"""DCP-like pruner: compresses tool / tool_result content only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except Exception:  # pragma: no cover - allow import in non-litellm envs
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


def _text_to_content(original: Any, text: str) -> Any:
    if isinstance(original, str):
        return text
    if isinstance(original, list):
        # replace text parts, keep structure
        out: list[Any] = []
        for item in original:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append({**item, "text": text})
            else:
                out.append(item)
        return out
    return text


def _prune_tool_text(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        # collapse whitespace only
        return "\n".join(line.rstrip() for line in lines).strip()
    head = lines[:20]
    tail = lines[-5:]
    elided = len(lines) - 25
    return "\n".join(head + [f"... [{elided} lines elided by DCP] ..."] + tail)


def _is_tool_message(msg: dict) -> bool:
    role = msg.get("role")
    if role in ("tool", "function"):
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(p, dict) and p.get("type") == "tool_result" for p in content)
    return False


class DCPPrunder(CustomGuardrail):
    def __init__(self, guardrail_name: str | None = None, targets: list[str] | None = None, **kwargs: Any) -> None:
        self.guardrail_name = guardrail_name or "dcp-pruner"
        self.targets = targets or ["tool", "tool_result"]
        try:
            super().__init__(guardrail_name=self.guardrail_name, **kwargs)  # type: ignore[misc]
        except Exception:
            pass

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        if input_type != "request":
            return inputs
        structured = inputs.get("structured_messages") if isinstance(inputs, dict) else None
        if not isinstance(structured, list):
            return inputs
        # snapshot original token count for overall stats (first compression guardrail)
        if isinstance(request_data, dict) and "_compression_original_tokens" not in request_data:
            request_data["_compression_original_tokens"] = sum(
                len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1 for m in structured
            )
        # dedup tracking
        seen: set[str] = set()
        out: list[Any] = []
        tokens_before = sum(len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1 for m in structured)
        for msg in structured:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            if not _is_tool_message(msg):
                out.append(msg)
                continue
            text = _content_to_text(msg.get("content"))
            key = text[:200]
            if key in seen:
                new_msg = {**msg, "content": "[DCP: duplicate tool result removed]"}
                out.append(new_msg)
                continue
            seen.add(key)
            pruned = _prune_tool_text(text)
            new_msg = {**msg, "content": _text_to_content(msg.get("content"), pruned)}
            out.append(new_msg)
        tokens_after = sum(len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1 for m in out)
        ratio = round(1 - (tokens_after / tokens_before), 4) if tokens_before > 0 else 0.0
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "tokens_saved": tokens_before - tokens_after,
                    "compression_ratio": ratio,
                },
                request_data=request_data,
                guardrail_status="success",
            )
        except Exception:
            pass
        if isinstance(inputs, dict):
            return {**inputs, "structured_messages": out}
        return inputs


dcp_guardrail = DCPPrunder()
"""Module-level instance for litellm config dotted path."""