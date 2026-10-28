"""LoopGuard: session-aware detection of repeated user turns.

LiteLLM sends the whole conversation on every turn, so the "last user message"
stays identical while an agent iterates through tool calls or a model is still
thinking. Counting requests would therefore flag normal agentic work as a loop.
This guard records only *new* user turns and ignores tool-result-only messages.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict, deque
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


def _header(request_data: dict, name: str) -> str | None:
    """Read a header from the shapes LiteLLM exposes to guardrails."""
    for container in (
        request_data,
        request_data.get("headers"),
        (request_data.get("proxy_server_request") or {}).get("headers")
        if isinstance(request_data.get("proxy_server_request"), dict)
        else None,
    ):
        if not isinstance(container, dict):
            continue
        value = container.get(name)
        if not value:
            # HTTP headers are case-insensitive; LiteLLM may preserve the case.
            lower = name.lower()
            for key, candidate in container.items():
                if isinstance(key, str) and key.lower() == lower and candidate:
                    value = candidate
                    break
        if value:
            return str(value)
    return None


def _session_id(request_data: dict) -> str | None:
    if not isinstance(request_data, dict):
        return None
    for key in ("session_id", "x-opencode-session", "x-opencode-session-id"):
        val = request_data.get(key)
        if val:
            return str(val)
        header = _header(request_data, key)
        if header:
            return header
    for bucket_name in ("metadata", "litellm_metadata"):
        metadata = request_data.get(bucket_name)
        if isinstance(metadata, dict):
            for key in ("session_id", "user_api_key_session_id"):
                val = metadata.get(key)
                if val:
                    return str(val)
    return None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _user_text(content: Any) -> str:
    """Return human-authored text from a message, ignoring tool results.

    Tool results are frequently delivered as ``role: user`` messages with only
    ``tool_result`` content parts. Those are continuation of the current turn,
    not a new user request.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # A message carrying a tool result is part of the current turn, even if
        # the client also attached some text to it.
        if any(
            isinstance(part, dict) and part.get("type") == "tool_result"
            for part in content
        ):
            return ""
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text"):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _user_turn_hashes(messages: Any) -> tuple[str, ...]:
    if not isinstance(messages, list):
        return ()
    hashes: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _user_text(message.get("content"))
        if text.strip():
            hashes.append(_hash_text(text))
    return tuple(hashes)


def _appended_turns(previous: tuple[str, ...], current: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return user turns added since ``previous``.

    ``None`` means the history was rewritten (compaction, truncation, a new
    conversation reusing the session id). Only a strict extension of the
    previously seen sequence counts as forward progress; on a rewrite the
    caller resets its baseline and fails open.
    """
    if current == previous:
        return ()
    if current[: len(previous)] == previous:
        return current[len(previous):]
    return None


class _SessionState:
    __slots__ = ("signature", "hashes", "last_seen", "seeded")

    def __init__(self) -> None:
        self.signature: tuple[str, ...] = ()
        self.hashes: deque[str] = deque(maxlen=32)
        self.last_seen: float = time.monotonic()
        self.seeded: bool = False


class LoopGuard(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str | None = None,
        max_repeats: int = 3,
        max_sessions: int = 1024,
        session_ttl_seconds: int = 3600,
        **kwargs: Any,
    ) -> None:
        self.guardrail_name = guardrail_name or "loop-guard"
        # Fewer than two repetitions can never describe a loop.
        self.max_repeats = max(2, int(max_repeats))
        self.max_sessions = max(1, int(max_sessions))
        self.session_ttl_seconds = max(0, int(session_ttl_seconds))
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        try:
            super().__init__(guardrail_name=self.guardrail_name, **kwargs)  # type: ignore[misc]
        except Exception:
            pass

    def _state(self, sid: str) -> _SessionState:
        now = time.monotonic()
        state = self._sessions.get(sid)
        if state is not None:
            if self.session_ttl_seconds and now - state.last_seen > self.session_ttl_seconds:
                state = None
        if state is None:
            state = _SessionState()
            self._sessions[sid] = state
        state.last_seen = now
        self._sessions.move_to_end(sid)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return state

    def _check(self, sid: str, turns: tuple[str, ...]) -> None:
        if not turns:
            return
        state = self._state(sid)
        appended = _appended_turns(state.signature, turns)
        state.signature = turns

        # The first request only establishes a baseline. Counting the history
        # it arrived with would flag a pre-existing (and possibly legitimate)
        # repetition that this guard did not observe as it happened.
        if not state.seeded:
            state.seeded = True
            state.hashes.extend(turns[-self.max_repeats:])
            return
        if appended is None:
            # History was rewritten; start a fresh baseline instead of treating
            # the rewritten turns as newly asked questions.
            state.hashes.clear()
            state.hashes.extend(turns[-self.max_repeats:])
            return
        if not appended:
            # Continuation of the same user turn (tool call, reasoning retry):
            # nothing new was asked, so nothing can be a repeated request.
            return

        state.hashes.extend(appended)
        recent = list(state.hashes)[-self.max_repeats:]
        if len(recent) >= self.max_repeats and len(set(recent)) == 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "loop_guard_stopped",
                        "message": (
                            "[Quinella LoopGuard] repeated identical user request "
                            "detected; stopping."
                        ),
                        "code": "loop_guard_stopped",
                    }
                },
            )

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        if input_type != "request":
            return inputs
        sid = _session_id(request_data or {})
        if sid is None:
            # No session identity means no safe way to correlate requests.
            # Failing open is better than mixing unrelated clients together.
            return inputs
        structured = inputs.get("structured_messages") if isinstance(inputs, dict) else None
        self._check(sid, _user_turn_hashes(structured))
        return inputs


loop_guard = LoopGuard()
guardrail = loop_guard
