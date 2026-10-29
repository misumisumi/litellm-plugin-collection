"""LoopGuard: session-aware loop detection and staged recovery.

The guard detects several loop signals, combines them with noisy-OR, and injects
a recovery prompt on the next request. It never rewrites the user's request
silently beyond that recovery injection and duplicate tool-output masking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
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


def _normalize_text(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = text.split()
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content):
            return ""
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
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
    if current == previous:
        return ()
    if current[: len(previous)] == previous:
        return current[len(previous) :]
    return None


def _response_text(response: Any) -> str:
    try:
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        else:
            choices = getattr(response, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", None) if msg is not None else None
                if isinstance(content, str):
                    return content
    except Exception:
        return ""
    return ""


def _response_tool_calls(response: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                msg = choices[0].get("message", {}) or {}
                tool_calls = msg.get("tool_calls") or []
            else:
                tool_calls = []
        else:
            choices = getattr(response, "choices", None) or []
            msg = getattr(choices[0], "message", None) if choices else None
            tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None
        for tc in tool_calls or []:
            if isinstance(tc, dict):
                fn = tc.get("function", {}) or {}
                result.append(
                    {
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    }
                )
            else:
                fn = getattr(tc, "function", None)
                result.append(
                    {
                        "id": getattr(tc, "id", None),
                        "name": getattr(fn, "name", None) if fn is not None else None,
                        "arguments": getattr(fn, "arguments", None) if fn is not None else None,
                    }
                )
    except Exception:
        return result
    return result


def _tool_call_hash(tc: dict[str, Any]) -> str:
    name = tc.get("name")
    args = tc.get("arguments")
    try:
        if isinstance(args, str):
            parsed = json.loads(args)
        else:
            parsed = args
        normalized = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    except Exception:
        normalized = str(args)
    return _hash_text(f"{name}:{normalized}")


def _normalize_error(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", text)
    text = re.sub(r":\d{1,9}\b", ":<line>", text)
    text = re.sub(r"/tmp/[^\s]+", "<path>", text)
    text = re.sub(r"/var/[^\s]+", "<path>", text)
    return text


def _is_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("error", "exception", "traceback", "failed", "failure")
    )


def _tool_messages(data: dict) -> list[dict[str, Any]]:
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return []
    result: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in ("tool", "function"):
            result.append(msg)
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "tool_result" for part in content
        ):
            result.append(msg)
    return result


def _tool_names(request_tools: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(request_tools, list):
        return names
    for tool in request_tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
        elif isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def _tool_schema(request_tools: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(request_tools, list):
        return None
    for tool in request_tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(fn, dict) or fn.get("name") != name:
            continue
        params = fn.get("parameters")
        return params if isinstance(params, dict) else None
    return None


def _validate_tool_calls(request_tools: Any, tool_calls: list[dict[str, Any]], tool_choice: Any, parallel: Any) -> bool:
    if not tool_calls:
        return True
    if tool_choice == "none":
        return False
    if parallel is False and len(tool_calls) > 1:
        return False
    names = _tool_names(request_tools)
    for tc in tool_calls:
        name = tc.get("name")
        if not isinstance(name, str) or (names and name not in names):
            return False
        args = tc.get("arguments")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except Exception:
            return False
        if not isinstance(parsed, dict):
            return False
        schema = _tool_schema(request_tools, name)
        if isinstance(schema, dict):
            required = schema.get("required")
            if isinstance(required, list):
                for key in required:
                    if key not in parsed:
                        return False
    return True


@dataclass
class _SessionState:
    seeded: bool = False
    signature: tuple[str, ...] = ()
    user_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    last_assistant_text: str = ""
    assistant_texts: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    tool_call_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=32))
    tool_error_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=32))
    seen_tool_signatures: set[str] = field(default_factory=set)
    seen_content_hashes: set[str] = field(default_factory=set)
    seen_test_output: set[str] = field(default_factory=set)
    progress_streak: int = 0
    no_progress_streak: int = 0
    protocol_failures: int = 0
    intervention_count: int = 0
    pending_intervention: bool = False
    loop_detected: bool = False
    jev_result: dict[str, Any] | None = None
    jev_inflight: bool = False
    last_seen: float = field(default_factory=time.monotonic)


class LoopGuard(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str | None = None,
        max_repeats: int = 3,
        max_sessions: int = 1024,
        session_ttl_seconds: int = 3600,
        repeat_assistant_similarity: float = 0.92,
        similarity_shingle_size: int = 5,
        repeat_tool_call_window: int = 8,
        repeat_tool_call_count: int = 3,
        repeat_tool_error_count: int = 3,
        no_progress_turns: int = 3,
        max_interventions: int = 3,
        intervention_reset_progress_turns: int = 2,
        jev_trigger: float = 0.4,
        jev_url: str | None = None,
        jev_api_key: str | None = None,
        jev_timeout: float = 3.0,
        weights: dict[str, float] | None = None,
        threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        self.guardrail_name = guardrail_name or "loop-guard"
        self.max_repeats = max(2, int(max_repeats))
        self.max_sessions = max(1, int(max_sessions))
        self.session_ttl_seconds = max(0, int(session_ttl_seconds))
        self.repeat_assistant_similarity = float(repeat_assistant_similarity)
        self.similarity_shingle_size = max(1, int(similarity_shingle_size))
        self.repeat_tool_call_window = max(1, int(repeat_tool_call_window))
        self.repeat_tool_call_count = max(2, int(repeat_tool_call_count))
        self.repeat_tool_error_count = max(2, int(repeat_tool_error_count))
        self.no_progress_turns = max(1, int(no_progress_turns))
        self.max_interventions = max(1, int(max_interventions))
        self.intervention_reset_progress_turns = max(1, int(intervention_reset_progress_turns))
        self.jev_trigger = float(jev_trigger)
        self.jev_url = jev_url
        self.jev_api_key = jev_api_key
        self.jev_timeout = max(0.1, float(jev_timeout))
        self.threshold = float(threshold)
        self.weights = weights or {
            "assistant_similarity": 0.6,
            "repeated_tool_call": 0.7,
            "repeated_tool_error": 0.7,
            "no_progress": 0.5,
            "tool_protocol": 0.8,
            "jev_loop": 0.6,
            "repeated_user_turn": 0.8,
        }
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        try:
            super().__init__(guardrail_name=self.guardrail_name, **kwargs)  # type: ignore[misc]
        except Exception:
            pass

    def _state(self, sid: str) -> _SessionState:
        now = time.monotonic()
        state = self._sessions.get(sid)
        if state is not None and self.session_ttl_seconds and now - state.last_seen > self.session_ttl_seconds:
            state = None
        if state is None:
            state = _SessionState()
            self._sessions[sid] = state
        state.last_seen = now
        self._sessions.move_to_end(sid)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return state

    def _assistant_similarity(self, state: _SessionState, current: str) -> float:
        if not current.strip() or not state.last_assistant_text.strip():
            return 0.0
        a = _shingles(_normalize_text(state.last_assistant_text), self.similarity_shingle_size)
        b = _shingles(_normalize_text(current), self.similarity_shingle_size)
        return _jaccard(a, b)

    def _noisy_or(self, signals: dict[str, float]) -> float:
        score = 1.0
        for value in signals.values():
            score *= 1.0 - value
        return 1.0 - score

    async def _call_jev(self, sid: str, state: _SessionState, payload: dict[str, Any]) -> None:
        if not self.jev_url:
            return
        state.jev_inflight = True
        try:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.jev_api_key:
                headers["Authorization"] = f"Bearer {self.jev_api_key}"
            async with httpx.AsyncClient(timeout=self.jev_timeout) as client:
                response = await client.post(self.jev_url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    state.jev_result = data
        except Exception:
            state.jev_result = None
        finally:
            state.jev_inflight = False

    def _mask_duplicate_tool_outputs(self, messages: list[Any]) -> list[Any]:
        seen: set[str] = set()
        out: list[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            role = msg.get("role")
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
            if role in ("tool", "function") and text.strip():
                key = _hash_text(text)
                if key in seen:
                    out.append({**msg, "content": "[LoopGuard: duplicate tool output removed]"})
                    continue
                seen.add(key)
            out.append(msg)
        return out

    def _inject_recovery(self, messages: list[Any], level: int) -> list[Any]:
        instruction = (
            "[Quinella LoopGuard] Recovery: stop repeating the same action. "
            "Reassess the goal, change strategy, and make concrete progress. "
            f"Intervention level {level}/{self.max_interventions}."
        )
        if level >= 2:
            instruction += " Escalate reasoning depth and verify each result before continuing."
        # Append to an existing system message instead of inserting a new one.
        system_index = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "system":
                system_index = i
                break
        out = [dict(msg) for msg in messages]
        if system_index is not None:
            msg = out[system_index]
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content + "\n\n" + instruction
            elif isinstance(content, list):
                msg["content"] = content + [{"type": "text", "text": instruction}]
            else:
                msg["content"] = instruction
        else:
            out.insert(0, {"role": "system", "content": instruction})
        return out

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        if input_type != "request":
            return inputs
        sid = _session_id(request_data or {})
        if sid is None:
            return inputs
        state = self._state(sid)
        structured = inputs.get("structured_messages") if isinstance(inputs, dict) else None
        if not isinstance(structured, list):
            structured = []
        turns = _user_turn_hashes(structured)
        appended = _appended_turns(state.signature, turns)
        human_turn = appended not in (None, ())

        # Reset the intervention episode on a genuinely new human turn or enough progress.
        if human_turn:
            state.intervention_count = 0
            state.pending_intervention = False
            state.loop_detected = False
            state.protocol_failures = 0

        # Apply a pending recovery from the previous response.
        if state.pending_intervention:
            level = min(state.intervention_count or 1, self.max_interventions)
            if level >= self.max_interventions:
                state.pending_intervention = False
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "loop_guard_stopped",
                            "message": "[Quinella LoopGuard] loop detected; stopping.",
                            "code": "loop_guard_stopped",
                        }
                    },
                )
            structured = self._inject_recovery(structured, level)
            structured = self._mask_duplicate_tool_outputs(structured)
            if isinstance(inputs, dict):
                inputs = {**inputs, "structured_messages": structured}
            state.pending_intervention = False
            if isinstance(request_data, dict):
                request_data["loop_guard_intervention"] = f"recover-{level}"

        # Preserve the existing simple repeated-user-turn stop.
        if turns and state.seeded:
            if appended is None:
                state.user_hashes.clear()
                state.user_hashes.extend(turns[-self.max_repeats :])
            elif appended:
                state.user_hashes.extend(appended)
                recent = list(state.user_hashes)[-self.max_repeats :]
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
        if not state.seeded:
            state.seeded = True
            state.user_hashes.extend(turns[-self.max_repeats :])

        state.signature = turns
        return inputs

    async def async_post_call_success_hook(self, user_api_key_dict, data, response):
        sid = _session_id(data or {})
        if sid is None:
            return response
        state = self._state(sid)
        structured = data.get("structured_messages") if isinstance(data, dict) else None
        if not isinstance(structured, list):
            structured = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(structured, list):
            structured = []

        turns = _user_turn_hashes(structured)
        appended = _appended_turns(state.signature, turns)
        human_turn = appended not in (None, ())

        assistant_text = _response_text(response)
        tool_calls = _response_tool_calls(response)
        request_tools = data.get("tools") if isinstance(data, dict) else None
        tool_choice = data.get("tool_choice") if isinstance(data, dict) else None
        parallel = data.get("parallel_tool_calls") if isinstance(data, dict) else None

        signals: dict[str, float] = {}
        det_score = 0.0

        # Assistant similarity.
        assistant_similarity = 0.0
        if assistant_text.strip():
            assistant_similarity = self._assistant_similarity(state, assistant_text)
            if assistant_similarity >= self.repeat_assistant_similarity:
                signals["assistant_similarity"] = self.weights.get("assistant_similarity", 0.6)
        if assistant_text.strip():
            state.assistant_texts.append(assistant_text)
            state.last_assistant_text = assistant_text

        # Repeated tool calls.
        repeated_tool = 0.0
        current_hashes = [_tool_call_hash(tc) for tc in tool_calls]
        state.tool_call_hashes.extend(current_hashes)
        counts = Counter(state.tool_call_hashes)
        if current_hashes and any(counts[h] >= self.repeat_tool_call_count for h in current_hashes):
            repeated_tool = 1.0
            signals["repeated_tool_call"] = self.weights.get("repeated_tool_call", 0.7)

        # Repeated tool errors.
        repeated_error = 0.0
        error_hashes: list[str] = []
        for msg in _tool_messages(data or {}):
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
            if _is_error_text(text):
                error_hashes.append(_hash_text(_normalize_error(text)))
        state.tool_error_hashes.extend(error_hashes)
        error_counts = Counter(state.tool_error_hashes)
        if error_hashes and any(error_counts[h] >= self.repeat_tool_error_count for h in error_hashes):
            repeated_error = 1.0
            signals["repeated_tool_error"] = self.weights.get("repeated_tool_error", 0.7)

        # Tool protocol guard.
        protocol_failed = False
        if tool_calls:
            valid = _validate_tool_calls(request_tools, tool_calls, tool_choice, parallel)
            if not valid:
                state.protocol_failures += 1
                protocol_failed = True
            else:
                state.protocol_failures = 0
        else:
            state.protocol_failures = 0
        protocol_signal = 0.0
        if state.protocol_failures >= self.repeat_tool_call_count:
            protocol_signal = 1.0
            signals["tool_protocol"] = self.weights.get("tool_protocol", 0.8)

        # Progress.
        progress = False
        if human_turn:
            progress = True
        new_tool_signatures: set[str] = set()
        for tc in tool_calls:
            sig = _tool_call_hash(tc)
            if sig not in state.seen_tool_signatures:
                new_tool_signatures.add(sig)
                progress = True
        state.seen_tool_signatures.update(new_tool_signatures)

        new_content_hashes: set[str] = set()
        for msg in _tool_messages(data or {}):
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
            h = _hash_text(_normalize_text(text))
            if h not in state.seen_content_hashes:
                new_content_hashes.add(h)
                progress = True
        state.seen_content_hashes.update(new_content_hashes)

        for msg in _tool_messages(data or {}):
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
            lowered = text.lower()
            if any(marker in lowered for marker in ("test", "build", "pytest", "cargo", "npm", "gradle", "make")):
                h = _hash_text(_normalize_text(text))
                if h not in state.seen_test_output:
                    state.seen_test_output.add(h)
                    progress = True

        # Jev async supplement.
        if self.jev_url and not state.jev_inflight:
            det_score = max(signals.values()) if signals else 0.0
            if det_score >= self.jev_trigger:
                summary = {
                    "assistant_text": assistant_text[:1000],
                    "tool_calls": [{"name": tc.get("name"), "arguments": tc.get("arguments")} for tc in tool_calls],
                }
                asyncio.create_task(self._call_jev(sid, state, summary))

        if state.jev_result and isinstance(state.jev_result, dict):
            if state.jev_result.get("progress") is True:
                progress = True
            if state.jev_result.get("loop") is True:
                signals["jev_loop"] = self.weights.get("jev_loop", 0.6)

        # no-progress streak.
        if progress:
            state.progress_streak += 1
            state.no_progress_streak = 0
        else:
            state.no_progress_streak += 1
            state.progress_streak = 0
        no_progress_signal = 0.0
        if state.no_progress_streak >= self.no_progress_turns:
            no_progress_signal = 1.0
            signals["no_progress"] = self.weights.get("no_progress", 0.5)

        # Repeated identical user turn signal.
        if turns and state.seeded:
            appended_now = _appended_turns(state.signature, turns)
            if appended_now not in (None, ()):
                state.user_hashes.extend(appended_now)
                recent = list(state.user_hashes)[-self.max_repeats :]
                if len(recent) >= self.max_repeats and len(set(recent)) == 1:
                    signals["repeated_user_turn"] = self.weights.get("repeated_user_turn", 0.8)

        # Update state.
        if not state.seeded:
            state.seeded = True
            state.user_hashes.extend(turns[-self.max_repeats :])
        state.signature = turns

        if assistant_text.strip():
            state.last_assistant_text = assistant_text
        state.signature = turns

        # Episode reset.
        if human_turn or state.progress_streak >= self.intervention_reset_progress_turns:
            if state.progress_streak >= self.intervention_reset_progress_turns:
                state.intervention_count = 0
                state.pending_intervention = False
                state.loop_detected = False

        score = self._noisy_or(signals)
        if score >= self.threshold:
            state.loop_detected = True
            state.pending_intervention = True
            state.intervention_count += 1

        return response


loop_guard = LoopGuard()
guardrail = loop_guard
