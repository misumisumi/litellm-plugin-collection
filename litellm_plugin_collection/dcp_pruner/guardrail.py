"""DCP-like pruner: compresses tool / tool_result content only."""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Lines worth keeping when a long tool output is elided. Error/stack-trace
# lines, file paths and URLs are usually the reason the model asked for the
# output in the first place.
_IMPORTANT_LINE_RE = re.compile(
    r"(?:Traceback|(?:Error|Exception):|File \".{1,300}\", line \d+|"
    r"^\s*at\s+\S|\bhttps?://|(?<!\S)(?:\.{0,2}/|[A-Za-z]:\\[A-Za-z0-9_.\\/-]+)|"
    r"^#{1,6}\s+)",
    re.IGNORECASE,
)

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
        # Only rewrite a single text/tool_result slot. Splitting one combined
        # string across several parts would duplicate or reorder content.
        slots = [
            i
            for i, item in enumerate(original)
            if isinstance(item, dict) and item.get("type") in ("text", "tool_result")
        ]
        if len(slots) != 1:
            return original
        index = slots[0]
        item = original[index]
        out = list(original)
        if item.get("type") == "tool_result":
            out[index] = {**item, "content": text}
        else:
            out[index] = {**item, "text": text}
        return out
    return text


def _mask_duplicate(content: Any) -> Any:
    """Replace only the tool-result portion, preserving any user text parts."""
    placeholder = "[DCP: duplicate tool result removed]"
    if isinstance(content, str):
        return placeholder
    if isinstance(content, list):
        out: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                out.append({**item, "content": placeholder})
            else:
                out.append(item)
        return out
    return placeholder


def _prune_tool_text(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        # collapse whitespace only
        return "\n".join(line.rstrip() for line in lines).strip()
    head = lines[:20]
    tail = lines[-5:]
    middle = lines[20:-5]
    important: list[str] = []
    for line in middle:
        if _IMPORTANT_LINE_RE.search(line):
            important.append(line)
            if len(important) >= 10:
                break
    elided = len(middle) - len(important)
    if important:
        marker = (
            f"... [{elided} lines elided by DCP; "
            f"{len(important)} important lines kept] ..."
        )
        return "\n".join(head + [marker] + important + tail)
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
        tokens_before = sum(
            len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1
            for m in structured
        )
        # Snapshot the original token count for the later summary guardrail.
        # Keep it in LiteLLM's internal metadata bucket: a top-level custom key
        # would be part of the request that is sent to the upstream provider.
        if isinstance(request_data, dict):
            bucket = request_data.get("litellm_metadata")
            if not isinstance(bucket, dict):
                bucket = {}
                request_data["litellm_metadata"] = bucket
            bucket.setdefault("_compression_original_tokens", tokens_before)
        # dedup tracking
        seen: set[str] = set()
        out: list[Any] = []
        for msg in structured:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            if not _is_tool_message(msg):
                out.append(msg)
                continue
            text = _content_to_text(msg.get("content"))
            key = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            if key in seen:
                out.append({**msg, "content": _mask_duplicate(msg.get("content"))})
                continue
            seen.add(key)
            pruned = _prune_tool_text(text)
            if pruned == text:
                out.append(msg)
                continue
            new_content = _text_to_content(msg.get("content"), pruned)
            out.append(msg if new_content is msg.get("content") else {**msg, "content": new_content})
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