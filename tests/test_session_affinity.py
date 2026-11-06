"""Behaviour tests for the OpenCode session affinity guardrail.

Run with ``python -m unittest discover -s tests -v``.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from litellm_plugin_collection.session_affinity.guardrail import OpenCodeSessionAffinity


def _apply(
    guardrail: OpenCodeSessionAffinity,
    data: dict[str, Any],
    input_type: str = "request",
) -> dict[str, Any]:
    asyncio.run(
        guardrail.apply_guardrail(
            inputs={"structured_messages": data.get("messages", [])},
            request_data=data,
            input_type=input_type,
            logging_obj=None,
        )
    )
    return data


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


class SessionAffinityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = OpenCodeSessionAffinity()

    def test_derives_stable_id_from_first_user_message(self) -> None:
        first = _apply(self.guard, {"messages": [_user("Implement the parser")]})
        later = _apply(
            self.guard,
            {
                "messages": [
                    _user("Implement the parser"),
                    {"role": "assistant", "content": "On it."},
                    _user("Now add tests"),
                ]
            },
        )

        self.assertRegex(first["litellm_session_id"], r"^oc-[0-9a-f]{32}$")
        self.assertEqual(first["litellm_session_id"], later["litellm_session_id"])
        self.assertEqual(
            first["metadata"]["session_id"], first["litellm_session_id"]
        )
        self.assertEqual(
            later["litellm_metadata"]["session_id"], later["litellm_session_id"]
        )

    def test_distinct_conversations_get_distinct_ids(self) -> None:
        one = _apply(self.guard, {"messages": [_user("Fix the build")]})
        two = _apply(self.guard, {"messages": [_user("Write the docs")]})

        self.assertNotEqual(one["litellm_session_id"], two["litellm_session_id"])

    def test_keeps_caller_litellm_session_id(self) -> None:
        data = _apply(
            self.guard,
            {"litellm_session_id": "caller-session", "messages": [_user("hello")]},
        )

        self.assertEqual(data["litellm_session_id"], "caller-session")
        self.assertIsNone((data.get("metadata") or {}).get("session_id"))

    def test_keeps_header_session_id(self) -> None:
        for header in ("x-litellm-session-id", "x-session-id", "x-opencode-session"):
            with self.subTest(header=header):
                data = _apply(
                    self.guard,
                    {"headers": {header: "ses_abcdef123456"}, "messages": [_user("hello")]},
                )
                self.assertNotIn("litellm_session_id", data)

    def test_keeps_proxy_request_header_session_id(self) -> None:
        data = _apply(
            self.guard,
            {
                "proxy_server_request": {"headers": {"x-opencode-session-id": "ses_abcdef123456"}},
                "messages": [_user("hello")],
            },
        )

        self.assertNotIn("litellm_session_id", data)

    def test_keeps_body_metadata_session_id(self) -> None:
        data = _apply(
            self.guard,
            {"metadata": {"session_id": "caller-metadata"}, "messages": [_user("hello")]},
        )

        self.assertEqual(data["metadata"]["session_id"], "caller-metadata")
        self.assertNotIn("litellm_session_id", data)

    def test_replaces_litellm_generated_id(self) -> None:
        data = _apply(
            self.guard,
            {
                "litellm_session_id": "generated",
                "litellm_metadata": {"litellm_session_id_generated": True},
                "messages": [_user("hello")],
            },
        )

        self.assertRegex(data["litellm_session_id"], r"^oc-")
        self.assertNotIn("litellm_session_id_generated", data["litellm_metadata"])
        self.assertEqual(
            data["litellm_metadata"]["session_id"], data["litellm_session_id"]
        )

    def test_prefers_prompt_cache_key(self) -> None:
        data = _apply(
            self.guard,
            {"prompt_cache_key": "session-abcdef1234", "messages": [_user("hello")]},
        )

        self.assertEqual(data["litellm_session_id"], "session-abcdef1234")

    def test_hashes_unsafe_prompt_cache_key(self) -> None:
        data = _apply(
            self.guard,
            {"prompt_cache_key": "not a valid header value!", "messages": [_user("hello")]},
        )

        self.assertRegex(data["litellm_session_id"], r"^oc-[0-9a-f]{32}$")

    def test_reads_text_parts(self) -> None:
        data = _apply(
            self.guard,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "part one"},
                            {"type": "text", "text": "part two"},
                        ],
                    }
                ]
            },
        )

        self.assertRegex(data["litellm_session_id"], r"^oc-")

    def test_falls_back_to_system_prompt(self) -> None:
        data = _apply(
            self.guard,
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful agent."},
                    {"role": "assistant", "content": "hi"},
                ]
            },
        )

        self.assertRegex(data["litellm_session_id"], r"^oc-")

    def test_ignores_non_request_inputs(self) -> None:
        data = _apply(self.guard, {"messages": [_user("hello")]}, input_type="response")

        self.assertNotIn("litellm_session_id", data)

    def test_no_session_without_messages(self) -> None:
        data = _apply(self.guard, {})

        self.assertNotIn("litellm_session_id", data)


if __name__ == "__main__":
    unittest.main()
