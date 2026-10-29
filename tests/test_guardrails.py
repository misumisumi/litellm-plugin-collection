"""Behaviour tests for the guardrail plugins.

Run with ``python -m unittest discover -s tests -v``.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from litellm_plugin_collection.compression_summary.guardrail import CompressionSummaryGuardrail
from litellm_plugin_collection.context_budget.guardrail import ContextBudgetGuardrail
from litellm_plugin_collection.dcp_pruner.guardrail import DCPPrunder
from litellm_plugin_collection.llm_summarizer.guardrail import LLMSummarizerGuardrail
from litellm_plugin_collection.llmlingua_compressor.guardrail import LLMLingua2Compressor
from litellm_plugin_collection.loop_guard.guardrail import LoopGuard


def _request(messages: list[Any], session: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"messages": messages, "litellm_metadata": {}}
    if session is not None:
        data["x-opencode-session"] = session
    return data


def _apply(guardrail: Any, messages: list[Any], session: str | None = None):
    data = _request(messages, session)
    inputs = {"structured_messages": messages, "texts": []}
    result = asyncio.run(
        guardrail.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=None,
        )
    )
    return result, data


def _tool_result(text: str) -> list[dict[str, Any]]:
    return [{"type": "tool_result", "tool_use_id": "t1", "content": text}]


class LoopGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        # LoopGuard.__init__ passes kwargs to litellm's base class; those are benign.
        self.guard = LoopGuard(max_repeats=3)

    def _call(self, messages, session: str | None = "s1") -> None:
        _apply(self.guard, messages, session)

    def test_tool_result_only_continuations_do_not_trigger(self):
        # This is the production regression: role=user messages carrying only a
        # tool_result have no human text. They must not hash as empty requests.
        base = [{"role": "user", "content": "Fix the bug"}]
        for i in range(5):
            messages = [
                *base,
                {"role": "assistant", "content": f"step {i}"},
                {"role": "user", "content": _tool_result(f"tool output {i}")},
            ]
            self._call(messages)

    def _post(self, guard, messages, response, session="s1"):
        data = _request(messages, session)
        # Keep request tools empty by default; tests override when needed.
        asyncio.run(
            guard.async_post_call_success_hook(
                user_api_key_dict=None,
                data=data,
                response=response,
            )
        )
        return data

    def test_repeated_assistant_response_triggers_next_request(self):
        guard = LoopGuard(max_interventions=1, threshold=0.5, confirm_turns=1)
        response = {"choices": [{"message": {"role": "assistant", "content": "same answer"}}]}
        messages = [{"role": "user", "content": "hello"}]
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        with self.assertRaises(Exception) as ctx:
            _apply(guard, messages, "s1")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_repeated_tool_call_triggers_next_request(self):
        guard = LoopGuard(
            max_interventions=1,
            threshold=0.5,
            confirm_turns=1,
            repeat_tool_call_count=2,
        )
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }],
                }
            }]
        }
        messages = [{"role": "user", "content": "hello"}]
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        with self.assertRaises(Exception) as ctx:
            _apply(guard, messages, "s1")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_protocol_violation_triggers_next_request(self):
        guard = LoopGuard(
            max_interventions=1,
            threshold=0.5,
            confirm_turns=1,
            repeat_tool_call_count=2,
        )
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "1",
                        "type": "function",
                        "function": {"name": "missing_tool", "arguments": "not-json"},
                    }],
                }
            }]
        }
        messages = [{"role": "user", "content": "hello"}]
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        with self.assertRaises(Exception) as ctx:
            _apply(guard, messages, "s1")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_recovery_is_injected_before_stop(self):
        guard = LoopGuard(max_interventions=3, threshold=0.5)
        response = {"choices": [{"message": {"role": "assistant", "content": "same answer"}}]}
        messages = [{"role": "user", "content": "hello"}]
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        # Fourth apply should inject recovery instead of stopping.
        result, data = _apply(guard, messages, "s1")
        system = next(m for m in result["structured_messages"] if m.get("role") == "system")
        self.assertIn("Recovery", system["content"])
        self.assertIn("loop_guard_intervention", data)

    def test_confirm_turns_delays_noisy_or_intervention(self):
        guard = LoopGuard(
            max_interventions=3,
            threshold=0.5,
            confirm_turns=3,
            no_progress_turns=100,
        )
        response = {"choices": [{"message": {"role": "assistant", "content": "same answer"}}]}
        messages = [{"role": "user", "content": "hello"}]
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        self.assertFalse(guard._sessions["s1"].pending_intervention)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        self.assertFalse(guard._sessions["s1"].pending_intervention)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        self.assertFalse(guard._sessions["s1"].pending_intervention)
        _apply(guard, messages, "s1")
        self._post(guard, messages, response)
        self.assertTrue(guard._sessions["s1"].pending_intervention)

    def test_tool_result_with_extra_text_is_still_a_continuation(self):
        for i in range(5):
            messages = [
                {"role": "user", "content": "Use the tool"},
                {"role": "user", "content": [
                    {"type": "text", "text": "continue"},
                    {"type": "tool_result", "content": f"result {i}"},
                ]},
            ]
            self._call(messages)

    def test_same_turn_with_reasoning_and_tools_does_not_trigger(self):
        for i in range(6):
            messages = [
                {"role": "user", "content": "Investigate the failing test"},
                {"role": "assistant", "content": f"thinking {i}"},
            ]
            if i:
                messages.append({"role": "tool", "tool_call_id": "t", "content": f"result {i}"})
            self._call(messages)

    def test_three_distinct_identical_user_turns_trigger(self):
        turn = {"role": "user", "content": "same request"}
        self._call([turn])
        self._call([turn, {"role": "assistant", "content": "a"}, turn])
        with self.assertRaises(Exception) as ctx:
            self._call(
                [
                    turn,
                    {"role": "assistant", "content": "a"},
                    turn,
                    {"role": "assistant", "content": "b"},
                    turn,
                ]
            )
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_first_request_with_repeated_history_does_not_trigger(self):
        # A single request that happens to contain a repeated phrase in its
        # history must not be blocked before the guard has observed any loop.
        turn = {"role": "user", "content": "continue"}
        self._call([turn, {"role": "assistant", "content": "a"}, turn,
                    {"role": "assistant", "content": "b"}, turn])

    def test_rewritten_history_resets_baseline(self):
        a = {"role": "user", "content": "alpha"}
        b = {"role": "user", "content": "beta"}
        c = {"role": "user", "content": "gamma"}
        self._call([a, b])
        # Compaction removed the first turn. This must not be interpreted as a
        # new identical user request.
        self._call([c])
        self._call([c, {"role": "assistant", "content": "x"}, c])
        # Only the third genuinely appended identical turn trips the guard.
        with self.assertRaises(Exception):
            self._call([c, {"role": "assistant", "content": "x"}, c,
                        {"role": "assistant", "content": "y"}, c])

    def test_different_user_turns_do_not_trigger(self):
        for i in range(5):
            self._call([{"role": "user", "content": f"request {i}"}])

    def test_missing_session_fails_open(self):
        turn = {"role": "user", "content": "same request"}
        for _ in range(5):
            self._call([turn], session=None)

    def test_sessions_are_isolated(self):
        turn = {"role": "user", "content": "same request"}
        self._call([turn], session="a")
        self._call([turn], session="b")
        # Each session is on its first observed request; neither should trip.
        self._call([turn, {"role": "assistant", "content": "x"}, turn], session="a")
        self._call([turn, {"role": "assistant", "content": "x"}, turn], session="b")


class DCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = DCPPrunder()

    def test_duplicate_tool_outputs_are_replaced(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "1", "content": "same output"},
            {"role": "tool", "tool_call_id": "2", "content": "same output"},
        ]
        result, _ = _apply(self.guard, messages)
        out = result["structured_messages"]
        self.assertEqual(out[1]["content"], "same output")
        self.assertEqual(out[2]["content"], "[DCP: duplicate tool result removed]")

    def test_long_tool_output_is_truncated(self):
        text = "\n".join(f"line {i}" for i in range(60))
        result, _ = _apply(self.guard, [{"role": "tool", "content": text}])
        content = result["structured_messages"][0]["content"]
        self.assertIn("lines elided by DCP", content)
        self.assertIn("line 59", content)

    def test_nested_tool_result_is_pruned(self):
        text = "\n".join(f"line {i}" for i in range(60))
        messages = [{"role": "user", "content": [{"type": "tool_result", "content": text}]}]
        result, _ = _apply(self.guard, messages)
        part = result["structured_messages"][0]["content"][0]
        self.assertIn("lines elided by DCP", part["content"])

    def test_stats_do_not_leak_into_request_payload(self):
        _, data = _apply(self.guard, [{"role": "tool", "content": "x"}])
        self.assertNotIn("_compression_original_tokens", data)
        self.assertIn("_compression_original_tokens", data["litellm_metadata"])

    def test_non_tool_messages_are_untouched(self):
        messages = [{"role": "user", "content": "a" * 1000}]
        result, _ = _apply(self.guard, messages)
        self.assertEqual(result["structured_messages"][0]["content"], "a" * 1000)

    def test_important_error_line_survives_elision(self):
        lines = [f"line {i}" for i in range(60)]
        lines[30] = "TypeError: cannot read property 'x' of undefined"
        result, _ = _apply(self.guard, [{"role": "tool", "content": "\n".join(lines)}])
        content = result["structured_messages"][0]["content"]
        self.assertIn("TypeError: cannot read property", content)
        self.assertIn("important lines kept", content)
        self.assertIn("line 0", content)
        self.assertIn("line 59", content)

    def test_duplicate_masking_keeps_user_text_parts(self):
        content = [
            {"type": "text", "text": "keep me"},
            {"type": "tool_result", "content": "same output"},
        ]
        messages = [
            {"role": "user", "content": content},
            {"role": "user", "content": content},
        ]
        result, _ = _apply(self.guard, messages)
        out = result["structured_messages"][1]["content"]
        self.assertEqual(out[0]["text"], "keep me")
        self.assertEqual(out[1]["content"], "[DCP: duplicate tool result removed]")


class CompressionSummaryTests(unittest.TestCase):
    def test_reads_original_tokens_from_litellm_metadata(self):
        guard = CompressionSummaryGuardrail()
        captured: list[dict[str, Any]] = []
        guard.add_standard_logging_guardrail_information_to_request_data = (  # type: ignore[method-assign]
            lambda **kwargs: captured.append(kwargs)
        )
        messages = [{"role": "user", "content": "compressed"}]
        data = _request(messages)
        data["litellm_metadata"]["_compression_original_tokens"] = 1000
        result = asyncio.run(
            guard.apply_guardrail(
                inputs={"structured_messages": messages, "texts": []},
                request_data=data,
                input_type="request",
                logging_obj=None,
            )
        )
        self.assertEqual(result["structured_messages"], messages)
        stats = captured[0]["guardrail_json_response"]
        self.assertEqual(stats["tokens_before"], 1000)
        self.assertGreater(stats["tokens_saved"], 0)


class LLMLinguaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = LLMLingua2Compressor(cache_aware=True, min_prompt_tokens=0)
        self.guard._compressor = None  # exercise the deterministic fallback

    def test_filler_is_removed_without_losing_structure(self):
        text = "Please just fix this.\n- keep the list\n```python\nprint('hi')\n```"
        result, _ = _apply(self.guard, [{"role": "user", "content": text}])
        out = result["structured_messages"][0]["content"]
        self.assertNotIn("just", out)
        self.assertIn("\n- keep the list", out)
        self.assertIn("```python\nprint('hi')\n```", out)

    def test_tool_role_is_not_compressed(self):
        text = "please just " + "x" * 200
        result, _ = _apply(self.guard, [{"role": "tool", "content": text}])
        self.assertEqual(result["structured_messages"][0]["content"], text)

    def test_system_is_skipped_for_cached_request(self):
        messages = [
            {"role": "system", "content": "please just keep this system prompt"},
            {"role": "user", "content": "please just do it"},
        ]
        data = _request(messages)
        messages[0]["content"] = [
            {"type": "text", "text": "please just keep this system prompt", "cache_control": {"type": "ephemeral"}}
        ]
        result = asyncio.run(
            self.guard.apply_guardrail(
                inputs={"structured_messages": messages, "texts": []},
                request_data=data,
                input_type="request",
                logging_obj=None,
            )
        )
        self.assertEqual(
            result["structured_messages"][0]["content"][0]["text"],
            "please just keep this system prompt",
        )

    def test_real_compressor_is_called_with_rate(self):
        class FakeCompressor:
            def compress_prompt(self, text, rate=0.5):
                return {"compressed_prompt": "compressed output"}

        guard = LLMLingua2Compressor(min_prompt_tokens=0, compression_rate=0.33)
        guard._compressor = FakeCompressor()
        text = "Please just explain this with enough length to pass."
        result, _ = _apply(guard, [{"role": "user", "content": text}])
        self.assertEqual(
            result["structured_messages"][0]["content"], "compressed output"
        )

    def test_empty_or_short_text_is_not_blanked(self):
        result, _ = _apply(self.guard, [{"role": "user", "content": "OK"}])
        self.assertEqual(result["structured_messages"][0]["content"], "OK")

    def test_short_prompt_is_skipped_with_threshold(self):
        guard = LLMLingua2Compressor(min_prompt_tokens=2000)
        guard._compressor = None
        text = "Please just do this."
        result, _ = _apply(guard, [{"role": "user", "content": text}])
        self.assertEqual(result["structured_messages"][0]["content"], text)

    def test_threshold_uses_total_non_system_tokens(self):
        # Each turn is below the floor, but the conversation as a whole is above
        # it. OmniRoute judges the concatenated non-system text, so both must
        # still be compressed.
        guard = LLMLingua2Compressor(min_prompt_tokens=50)
        guard._compressor = None
        text = "Please just do this. " + ("x " * 50)
        messages = [
            {"role": "user", "content": text},
            {"role": "user", "content": text},
        ]
        result, _ = _apply(guard, messages)
        for message in result["structured_messages"]:
            self.assertNotIn("just", message["content"])

    def test_legacy_min_prompt_chars_is_accepted(self):
        guard = LLMLingua2Compressor(min_prompt_chars=0)
        guard._compressor = None
        text = "Please just do this. " + ("x " * 100)
        result, _ = _apply(guard, [{"role": "user", "content": text}])
        self.assertNotIn("just", result["structured_messages"][0]["content"])

    def test_long_prompt_is_compressed_with_threshold(self):
        guard = LLMLingua2Compressor(min_prompt_tokens=10)
        guard._compressor = None
        text = "Please just do this. " + ("x " * 100)
        result, _ = _apply(guard, [{"role": "user", "content": text}])
        self.assertNotIn("just", result["structured_messages"][0]["content"])

    def test_system_prompt_is_preserved_by_default(self):
        text = "Please just keep this system prompt and never drop it."
        result, _ = _apply(self.guard, [{"role": "system", "content": text}])
        self.assertEqual(result["structured_messages"][0]["content"], text)

    def test_system_prompt_can_be_compressed_when_enabled(self):
        guard = LLMLingua2Compressor(compress_system=True, min_prompt_tokens=0)
        guard._compressor = None
        text = "Please just keep this system prompt and never drop it."
        result, _ = _apply(guard, [{"role": "system", "content": text}])
        out = result["structured_messages"][0]["content"]
        self.assertNotIn("just", out)
        self.assertIn("never", out)

    def test_preserves_instructions_negations_and_technical_tokens(self):
        text = (
            "Please just explain this. You must not remove it or ever forget.\n"
            "Visit https://example.com/docs and read `config.yaml`.\n"
            "Use ./src/main.py with process.env.HOME and version 1.2.3.\n"
            "API_KEY=abcdef123456"
        )
        result, _ = _apply(self.guard, [{"role": "user", "content": text}])
        out = result["structured_messages"][0]["content"]
        self.assertNotIn("just", out)
        self.assertIn("must not", out)
        self.assertIn("ever", out)
        self.assertIn("https://example.com/docs", out)
        self.assertIn("`config.yaml`", out)
        self.assertIn("./src/main.py", out)
        self.assertIn("process.env.HOME", out)
        self.assertIn("1.2.3", out)
        self.assertIn("API_KEY=abcdef123456", out)

    def test_system_reminder_envelope_is_not_rewritten(self):
        text = "Please just keep <system-reminder>please just keep me</system-reminder> intact."
        result, _ = _apply(self.guard, [{"role": "user", "content": text}])
        out = result["structured_messages"][0]["content"]
        self.assertIn(
            "<system-reminder>please just keep me</system-reminder>", out
        )


class ContextBudgetTests(unittest.TestCase):
    def test_system_and_latest_turn_are_preserved(self):
        guard = ContextBudgetGuardrail(
            context_limit=200, safety_ratio=0.9, reserve_output_tokens=0
        )
        system = {"role": "system", "content": "system prompt " * 10}
        old_turn = [
            {"role": "user", "content": "old request " * 20},
            {"role": "assistant", "content": "old answer " * 20},
        ]
        latest = [
            {"role": "user", "content": "latest request"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "tool result"},
        ]
        messages = [system, *old_turn, *latest]
        result, _ = _apply(guard, messages)
        out = result["structured_messages"]
        self.assertIs(out[0], system)
        # The whole latest turn survives, including the tool result.
        self.assertEqual(out[-3:], latest)

    def test_truncation_keeps_fenced_code_intact(self):
        guard = ContextBudgetGuardrail(
            context_limit=400, safety_ratio=0.9, reserve_output_tokens=0
        )
        long_old = "```python\nprint('keep me')\n```\n" + ("prose " * 300)
        messages = [
            {"role": "user", "content": long_old},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest"},
        ]
        result, _ = _apply(guard, messages)
        old_content = result["structured_messages"][0]["content"]
        self.assertIn("```python\nprint('keep me')\n```", old_content)
        self.assertIn("[truncated]", old_content)

    def test_under_budget_is_untouched(self):
        guard = ContextBudgetGuardrail(context_limit=10000, safety_ratio=0.9)
        messages = [{"role": "user", "content": "hi"}]
        result, _ = _apply(guard, messages)
        self.assertEqual(result["structured_messages"], messages)

    def test_auto_detects_context_limit_from_request_data(self):
        guard = ContextBudgetGuardrail(context_limit=0, safety_ratio=0.9, reserve_output_tokens=0)
        messages = [
            {"role": "user", "content": "x " * 1000},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest"},
        ]
        data = _request(messages)
        data["model_info"] = {"max_input_tokens": 300}
        result = asyncio.run(
            guard.apply_guardrail(
                inputs={"structured_messages": messages, "texts": []},
                request_data=data,
                input_type="request",
                logging_obj=None,
            )
        )
        # The old turn should be dropped or truncated under the small budget.
        self.assertNotEqual(result["structured_messages"], messages)


class LLMSummarizerTests(unittest.TestCase):
    def test_under_threshold_does_not_call_llm(self):
        guard = LLMSummarizerGuardrail(summary_threshold_tokens=100000)
        messages = [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "short answer"},
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            result, _ = _apply(guard, messages)
            self.assertFalse(mock.called)
            self.assertEqual(result["structured_messages"], messages)

    def test_over_threshold_calls_llm_and_replaces_older_messages(self):
        guard = LLMSummarizerGuardrail(
            summary_threshold_tokens=10,
            preserve_recent_turns=1,
            max_summary_calls=3,
        )
        messages = [
            {"role": "user", "content": "old request " * 50},
            {"role": "assistant", "content": "old answer " * 50},
            {"role": "user", "content": "latest question"},
        ]
        fake = {"choices": [{"message": {"role": "assistant", "content": "summary text"}}]}
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = fake
            result, _ = _apply(guard, messages)
            self.assertTrue(mock.called)
            self.assertEqual(len(result["structured_messages"]), 2)
            self.assertIn("[LLM Summary", result["structured_messages"][0]["content"])
            self.assertEqual(result["structured_messages"][1], messages[2])

    def test_fail_open_on_llm_error(self):
        guard = LLMSummarizerGuardrail(
            summary_threshold_tokens=10,
            preserve_recent_turns=1,
        )
        messages = [
            {"role": "user", "content": "old request " * 50},
            {"role": "assistant", "content": "old answer " * 50},
            {"role": "user", "content": "latest question"},
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("boom")
            result, _ = _apply(guard, messages)
            self.assertTrue(mock.called)
            self.assertEqual(result["structured_messages"], messages)

    def test_max_summary_calls_is_respected(self):
        guard = LLMSummarizerGuardrail(
            summary_threshold_tokens=1,
            preserve_recent_turns=1,
            max_summary_calls=1,
        )
        messages = [
            {"role": "user", "content": "old request " * 50},
            {"role": "assistant", "content": "old answer " * 50},
            {"role": "user", "content": "latest question"},
        ]
        fake = {"choices": [{"message": {"role": "assistant", "content": "summary text"}}]}
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = fake
            _apply(guard, messages)
            self.assertEqual(mock.call_count, 1)
            # second call should not invoke the LLM again
            _apply(guard, messages)
            self.assertEqual(mock.call_count, 1)

    def test_skip_marker_prevents_llm_call(self):
        guard = LLMSummarizerGuardrail(summary_threshold_tokens=10, preserve_recent_turns=1)
        messages = [
            {"role": "user", "content": "old request " * 50},
            {"role": "assistant", "content": "old answer " * 50},
            {"role": "user", "content": "latest question"},
        ]
        data = _request(messages)
        data["skip_llm_summarizer"] = True
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            asyncio.run(
                guard.apply_guardrail(
                    inputs={"structured_messages": messages, "texts": []},
                    request_data=data,
                    input_type="request",
                    logging_obj=None,
                )
            )
            self.assertFalse(mock.called)


if __name__ == "__main__":
    unittest.main()
