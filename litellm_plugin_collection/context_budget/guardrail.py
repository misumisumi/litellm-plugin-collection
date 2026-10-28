"""Budget-aware context compression (Quinella 6.2 style)."""

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
        out: list[Any] = []
        for item in original:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append({**item, "text": text})
            else:
                out.append(item)
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
        # keep system + last user turn; truncate middle
        # identify last user index
        last_user = -1
        for i in range(len(structured) - 1, -1, -1):
            if isinstance(structured[i], dict) and structured[i].get("role") == "user":
                last_user = i
                break
        out: list[Any] = []
        for i, msg in enumerate(structured):
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            if i == 0 and msg.get("role") == "system":
                out.append(msg)
                continue
            if i == last_user:
                out.append(msg)
                continue
            # truncate
            text = _content_to_text(msg.get("content"))
            if len(text) > 500:
                truncated = text[:500] + "\n[truncated]"
                new_msg = {**msg, "content": _text_to_content(msg.get("content"), truncated)}
                out.append(new_msg)
            else:
                out.append(msg)
        # if still over budget, drop oldest non-protected
        while _estimate_tokens(out) > budget and len(out) > 3:
            # drop the first non-system message
            for i in range(1, len(out)):
                if not (isinstance(out[i], dict) and out[i].get("role") == "system"):
                    del out[i]
                    break
            else:
                break
        tokens_before = sum(len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1 for m in structured)
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
        return {**inputs, "structured_messages": out}


context_budget_guardrail = ContextBudgetGuardrail()
guardrail = context_budget_guardrail