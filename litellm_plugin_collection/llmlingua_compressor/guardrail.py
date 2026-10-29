"""llmlingua-2 based compressor for user/assistant prose only.

System prompts are preserved by default, and code, instructions, identifiers,
paths, URLs, errors, secrets and negations are shielded from the compressor.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Any

try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass

from litellm_plugin_collection._preservation import split_prose_and_preserved

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
        slots = [
            i
            for i, item in enumerate(original)
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if len(slots) != 1:
            return original
        index = slots[0]
        out = list(original)
        out[index] = {**original[index], "text": text}
        return out
    return text


_FILLER_WORDS = {
    "please", "basically", "actually", "just", "really", "very", "quite",
    "simply", "essentially", "literally", "of course", "certainly",
}


def _lite_compress(text: str) -> str:
    """Fallback lite compression when llmlingua-2 is unavailable.

    Operates line by line so markdown structure survives. Preserved spans
    (code, paths, negations, ...) have already been split out by the caller.
    """
    if not text.strip():
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):] if text.rstrip() else ""
    end = len(text) - len(trailing) if trailing else len(text)
    core = text[len(leading):end]
    lines: list[str] = []
    for line in core.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        words = [
            word
            for word in stripped.split()
            if word.lower().strip(".,!?") not in _FILLER_WORDS
        ]
        lines.append((indent + " ".join(words)).rstrip())
    result = "\n".join(lines)
    return leading + result + trailing if result.strip() else text


class LLMLingua2Compressor(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str | None = None,
        cache_aware: bool = True,
        compress_system: bool = False,
        compress_assistant: bool = False,
        min_prompt_chars: int = 2000,
        compression_rate: float = 0.50,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        cache_size: int = 256,
        **kwargs: Any,
    ) -> None:
        self.guardrail_name = guardrail_name or "llmlingua-2"
        self.cache_aware = cache_aware
        self.compress_system = compress_system
        self.compress_assistant = compress_assistant
        self.min_prompt_chars = max(0, int(min_prompt_chars))
        self.compression_rate = min(0.99, max(0.05, float(compression_rate)))
        self.model_name = model_name
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._compressor = None
        if _HAS_LLMLINGUA:
            try:
                try:
                    import torch

                    device_map = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device_map = "cpu"
                self._compressor = _PromptCompressor(
                    model_name=self.model_name,
                    use_llmlingua2=True,
                    device_map=device_map,
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
        # Anthropic/OpenAI-style prompt caching is usually requested on a
        # message content part, not at the top level of the request.
        messages = request_data.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("cache_control") is not None:
                    return True
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("cache_control") is not None:
                            return True
        return False

    def _compress_prose(self, text: str) -> str:
        if not text.strip():
            return text
        compressed: str | None = None
        if self._compressor is not None:
            try:
                out = self._compressor.compress_prompt(text, rate=self.compression_rate)  # type: ignore[attr-defined]
                if isinstance(out, dict):
                    value = out.get("compressed_prompt", text)
                else:
                    value = str(out)
                if isinstance(value, str) and value.strip():
                    compressed = value
            except Exception:
                compressed = None
        if compressed is None:
            compressed = _lite_compress(text)
        if not compressed or not compressed.strip():
            return text
        # Never expand a segment; a no-op is safer than losing the original.
        return compressed if len(compressed) < len(text) else text

    def _compress_text(self, text: str) -> str:
        if not text.strip():
            return text
        key = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        segments = split_prose_and_preserved(text)
        if len(segments) == 1 and segments[0][0] == "prose":
            result = self._compress_prose(text)
        else:
            parts = [
                value if kind == "preserved" else self._compress_prose(value)
                for kind, value in segments
            ]
            result = "".join(parts)
            if not (len(result) < len(text)):
                result = text
        if result and result.strip() and len(result) < len(text):
            self._cache[key] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result if result and result.strip() else text

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
            # System and assistant prompts carry instructions and reasoning;
            # leave them alone unless explicitly opted in.
            if role == "system" and (not self.compress_system or cached):
                out.append(msg)
                continue
            if role == "assistant" and not self.compress_assistant:
                out.append(msg)
                continue
            text = _content_to_text(msg.get("content"))
            if not text.strip():
                out.append(msg)
                continue
            if self.min_prompt_chars > 0 and len(text) < self.min_prompt_chars:
                out.append(msg)
                continue
            compressed = await asyncio.to_thread(self._compress_text, text)
            if compressed == text:
                out.append(msg)
                continue
            new_content = _text_to_content(msg.get("content"), compressed)
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
        return {**inputs, "structured_messages": out}


llmlingua_guardrail = LLMLingua2Compressor()
guardrail = llmlingua_guardrail