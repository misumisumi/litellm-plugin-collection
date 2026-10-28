"""LoopGuard: basic session-aware loop detection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
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


def _session_id(request_data: dict) -> str:
    for key in ("session_id", "x-opencode-session", "x-opencode-session-id"):
        val = request_data.get(key)
        if val:
            return str(val)
    metadata = request_data.get("metadata")
    if isinstance(metadata, dict):
        for key in ("session_id", "user_api_key_session_id"):
            val = metadata.get(key)
            if val:
                return str(val)
    return "default"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


class LoopGuard(CustomGuardrail):
    def __init__(self, guardrail_name: str | None = None, max_repeats: int = 3, **kwargs: Any) -> None:
        self.guardrail_name = guardrail_name or "loop-guard"
        self.max_repeats = max_repeats
        self._user_hashes: dict[str, list[str]] = defaultdict(list)
        self._assistant_hashes: dict[str, list[str]] = defaultdict(list)
        try:
            super().__init__(guardrail_name=self.guardrail_name, **kwargs)  # type: ignore[misc]
        except Exception:
            pass

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        if input_type != "request":
            return inputs
        sid = _session_id(request_data or {})
        # extract last user text
        text = ""
        structured = inputs.get("structured_messages") if isinstance(inputs, dict) else None
        if isinstance(structured, list):
            for msg in reversed(structured):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    c = msg.get("content")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = " ".join(
                            p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                        )
                    break
        h = _hash_text(text)
        self._user_hashes[sid].append(h)
        # detect repeated identical user asks
        recent = self._user_hashes[sid][-self.max_repeats:]
        if len(recent) >= self.max_repeats and len(set(recent)) == 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "loop_guard_stopped",
                        "message": "[Quinella LoopGuard] repeated identical user request detected; stopping.",
                        "code": "loop_guard_stopped",
                    }
                },
            )
        if isinstance(inputs, dict):
            return inputs
        return inputs

    async def async_post_call_success_hook(self, user_api_key_dict, data, response):
        sid = _session_id(data or {})
        # record assistant text hash if available
        try:
            text = ""
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    if isinstance(content, str):
                        text = content
            if text:
                self._assistant_hashes[sid].append(_hash_text(text))
        except Exception:
            pass
        return response


loop_guard = LoopGuard()
guardrail = loop_guard