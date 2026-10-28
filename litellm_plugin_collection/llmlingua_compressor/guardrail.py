"""llmlingua-2 based compressor for text (system/user/assistant) only."""

from __future__ import annotations

from typing import Any

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except Exception:  # pragma: no cover
    class CustomGuardrail:  # type: ignore[no-redef]
        pass

# Optional llmlingua (PromptCompressor with use_llmlingua2=True)
try:
    from llmlingua import PromptCompressor as _PromptCompressor  # type: ignore
    _HAS_LLMLINGUA = True
except Exception:
    _PromptCompressor = None  # type: ignore
    _HAS_LLMLINGUA = False


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


_FILLER_WORDS = {
    "please", "basically", "actually", "just", "really", "very", "quite",
    "simply", "essentially", "literally", "of course", "certainly",
}


def _lite_compress(text: str) -> str:
    """Fallback lite compression when llmlingua-2 is unavailable."""
    words = text.split()
    out: list[str] = []
    for w in words:
        if w.lower().strip(".,!?") in _FILLER_WORDS:
            continue
        out.append(w)
    result = " ".join(out)
    # collapse whitespace
    lines = [ln.rstrip() for ln in result.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()).strip()


class LLMLingua2Compressor(CustomGuardrail):
    def __init__(self, guardrail_name: str | None = None, cache_aware: bool = True, **kwargs: Any) -> None:
        self.guardrail_name = guardrail_name or "llmlingua-2"
        self.cache_aware = cache_aware
        self._compressor = None
        if _HAS_LLMLINGUA:
            try:
                self._compressor = _PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    use_llmlingua2=True,
                )  # type: ignore
            except Exception:
                self._compressor = None
        try:
            super().__init__(guardrail_name=self.guardrail_name, **kwargs)  # type: ignore[misc]
        except Exception:
            pass

    def _is_cached_request(self, request_data: dict) -> bool:
        if not self.cache_aware:
            return False
        if request_data.get("cache_control") is not None:
            return True
        metadata = request_data.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("prompt_caching"):
            return True
        return False

    def _compress_text(self, text: str) -> str:
        if not text.strip():
            return text
        if self._compressor is not None:
            try:
                out = self._compressor.compress_prompt(text, ratio=0.75)  # type: ignore[attr-defined]
                if isinstance(out, dict):
                    return out.get("compressed_prompt", text)
                return str(out)
            except Exception:
                pass
        return _lite_compress(text)

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        if input_type != "request":
            return inputs
        if not isinstance(inputs, dict):
            return inputs
        structured = inputs.get("structured_messages")
        if not isinstance(structured, list):
            return inputs
        cached = self._is_cached_request(request_data or {})
        tokens_before = sum(len(_content_to_text(m.get("content") if isinstance(m, dict) else "")) // 4 + 1 for m in structured)
        out: list[Any] = []
        for msg in structured:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            role = msg.get("role")
            if role not in ("system", "user", "assistant"):
                out.append(msg)
                continue
            # cache-aware: skip system prompt compression for cached requests
            if cached and role == "system":
                out.append(msg)
                continue
            text = _content_to_text(msg.get("content"))
            if not text.strip():
                out.append(msg)
                continue
            compressed = self._compress_text(text)
            new_msg = {**msg, "content": _text_to_content(msg.get("content"), compressed)}
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
        return {**inputs, "structured_messages": out}


llmlingua_guardrail = LLMLingua2Compressor()
guardrail = llmlingua_guardrail