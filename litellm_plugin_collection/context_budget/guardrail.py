"""Budget-aware context compression (Quinella 6.2 style)."""

from __future__ import annotations

from typing import Any

from litellm_plugin_collection._preservation import truncate_preserved

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
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _text_to_content(original: Any, text: str) -> Any:
    if isinstance(original, str):
        return text
    if isinstance(original, list):
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


def _estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += len(_content_to_text(m.get("content"))) // 4 + 1
    return total


class ContextBudgetGuardrail(CustomGuardrail):
    def __init__(self, guardrail_name: str | None = None, safety_ratio: float = 0.90,
                 reserve_output_tokens: int = 8192, context_limit: int = 128000, **kwargs: Any) -> None:
        self.guardrail_name = guardrail_name or "context-budget"
        self.safety_ratio = safety_ratio
        self.reserve_output_tokens = reserve_output_tokens
        self.context_limit = context_limit
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
        budget = int(self.context_limit * self.safety_ratio) - self.reserve_output_tokens
        if budget <= 0:
            return inputs
        if _estimate_tokens(structured) <= budget:
            return inputs
        # Keep every system message and the whole latest user turn (including
        # the assistant/tool messages that follow it). Truncating or dropping
        # those would break tool_call -> tool_result pairing.
        last_user = -1
        for i in range(len(structured) - 1, -1, -1):
            if isinstance(structured[i], dict) and structured[i].get("role") == "user":
                last_user = i
                break
        protected_from = last_user if last_user >= 0 else len(structured)
        out: list[Any] = []
        for i, msg in enumerate(structured):
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            if msg.get("role") == "system" or i >= protected_from:
                out.append(msg)
                continue
            # truncate oversized older messages
            text = _content_to_text(msg.get("content"))
            if len(text) > 500:
                truncated = truncate_preserved(text, 500)
                new_content = _text_to_content(msg.get("content"), truncated)
                out.append(msg if new_content is msg.get("content") else {**msg, "content": new_content})
            else:
                out.append(msg)

        def _is_protected(index: int, messages: list[Any], start: int) -> bool:
            if index >= start:
                return True
            return isinstance(messages[index], dict) and messages[index].get("role") == "system"

        # If still over budget, drop the oldest complete turn instead of
        # individual messages. Dropping a tool result while keeping its
        # assistant tool_call (or vice versa) makes the request invalid.
        while _estimate_tokens(out) > budget:
            start = next(
                (
                    i
                    for i, msg in enumerate(out)
                    if i < protected_from
                    and not _is_protected(i, out, protected_from)
                    and isinstance(msg, dict)
                    and msg.get("role") == "user"
                ),
                None,
            )
            if start is None:
                break
            end = protected_from
            for j in range(start + 1, protected_from):
                if (
                    isinstance(out[j], dict)
                    and out[j].get("role") == "user"
                    and not _is_protected(j, out, protected_from)
                ):
                    end = j
                    break
            removed = end - start
            del out[start:end]
            protected_from -= removed
            if not out:
                break
        tokens_before = sum(len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1 for m in structured)
        tokens_after = sum(len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1 for m in out)
        if isinstance(request_data, dict):
            bucket = request_data.get("litellm_metadata")
            if not isinstance(bucket, dict):
                bucket = {}
                request_data["litellm_metadata"] = bucket
            bucket.setdefault("_compression_original_tokens", tokens_before)
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
        return {**inputs, "structured_messages": out}


context_budget_guardrail = ContextBudgetGuardrail()
guardrail = context_budget_guardrail