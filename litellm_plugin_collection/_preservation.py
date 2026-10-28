"""Spans that compression must not rewrite.

Only prose is safe to compress. Code, instructions, identifiers, paths, URLs,
error text, secrets and negations all change meaning (or break) when a
compressor rewrites them. This follows the preservation model used by
OmniRoute's compression engines (MIT, Copyright (c) 2026 diegosouzapw):
mask preserved spans, compress the remaining prose, then restore them
verbatim.

The output of :func:`extract_preserved` contains NUL-delimited placeholders;
:func:`restore_preserved` fails open when a compressor drops one.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Any

SENTINEL_PREFIX = "\x00OMNI_PRESERVE"


@dataclass(frozen=True)
class PreservedBlock:
    placeholder: str
    content: str
    kind: str


# ── whole-region patterns ────────────────────────────────────────────────────
_REGION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"<system-reminder>[\s\S]*?</system-reminder>"), "system_instruction"),
    (re.compile(r"<instructions?>[\s\S]*?</instructions?>"), "system_instruction"),
    (
        re.compile(r"<project[- ]instructions?>[\s\S]*?</project[- ]instructions?>"),
        "system_instruction",
    ),
]

# ── risk-gate patterns (never compress credentials / migrations / traces) ────
_RISK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----[\s\S]{1,4096}?"
            r"-----END [A-Z0-9 ]{0,40}PRIVATE KEY-----"
        ),
        "private_key",
    ),
    (
        re.compile(
            r"\b(?:api[_-]?key|secret|token|password|passwd|bearer|authorization|"
            r"client[_-]?secret)\b[ \t]{0,20}[:=][ \t]{0,20}[\"']?"
            r"[A-Za-z0-9._\-+/]{8,200}",
            re.IGNORECASE,
        ),
        "secret_assignment",
    ),
    (
        re.compile(
            r"^\s{0,8}(?:at\s+\S.{0,300}|File \".{1,300}\", line \d{1,9}|"
            r"Traceback \(most recent call last\):|"
            r"[A-Za-z_.]{1,80}(?:Error|Exception):.{0,300})$",
            re.MULTILINE,
        ),
        "stack_trace",
    ),
    (
        re.compile(
            r"\b(?:CREATE|ALTER|DROP)\s+(?:TABLE|INDEX|SCHEMA|DATABASE|COLUMN)\b",
            re.IGNORECASE,
        ),
        "db_migration",
    ),
]

# ── inline constructs ────────────────────────────────────────────────────────
_BUILTIN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\$\$[\s\S]*?\$\$"), "math_block"),
    (re.compile(r"\\\[[\s\S]*?\\\]"), "math_block"),
    (
        re.compile(
            r"(?<!\$)\$(?![\s$\d])(?:\\.|[^$\n\\]){1,160}?(?<!\s)\$(?!\$)"
        ),
        "math_inline",
    ),
    (re.compile(r"\\begin\{[A-Za-z*]+\}[\s\S]*?\\end\{[A-Za-z*]+\}"), "latex_block"),
    (re.compile(r"^#{1,6}\s+.+$", re.MULTILINE), "markdown_heading"),
    (re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE), "markdown_table"),
    (
        re.compile(
            r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
            re.MULTILINE,
        ),
        "markdown_table",
    ),
    (
        re.compile(r"^\s*#(?:set|show|let|import|include)\b.+$", re.MULTILINE),
        "typst_directive",
    ),
    (re.compile(r"`[^`\n]+`"), "inline_code"),
    (re.compile(r"\[[^\]\n]+\]\([^) \n]+(?:\s+\"[^\"]*\")?\)"), "markdown_link"),
    (re.compile(r"\bhttps?://[^\s)\]\"'>]+", re.IGNORECASE), "url"),
    (re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"), "const_case"),
    (re.compile(r"\bprocess\.env\.[A-Za-z_][A-Za-z0-9_]*\b"), "env_var"),
    (re.compile(r"\$[A-Z_][A-Z0-9_]*\b"), "env_var"),
    (re.compile(r"\b\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9.-]+)?\b"), "version"),
    (
        re.compile(r"\b[a-zA-Z_$][\w$]*(?:\.[a-zA-Z_$][\w$]*)+\(\)?"),
        "dotted_identifier",
    ),
    (re.compile(r"\b[A-Za-z_$][\w$]*\s*\([^()\n]*\)"), "function_call"),
    (
        re.compile(
            r"(?<!\S)(?:\.{0,2}/[A-Za-z0-9_@./-]+|[A-Za-z]:\\[A-Za-z0-9_.\\/-]+)"
        ),
        "file_path",
    ),
    (
        re.compile(
            r"\b(?:TypeError|ReferenceError|SyntaxError|RangeError|URIError|"
            r"EvalError|Error|Exception):[^\n]+"
        ),
        "error_message",
    ),
]

# XML tags and negations/absolutes are split out of prose before compression:
# dropping a "not"/"never"/"must" inverts an instruction, and an uncased
# compressor can break tag punctuation.
_PROTECTED_SPAN_RE = re.compile(
    r"(<\/?[A-Za-z][A-Za-z0-9._-]*(?:\s[^<>]*?)?\/?>)"
    r"|(\b\w+(?:n't|n’t)\b)"
    r"|((?<![A-Za-z0-9_-])(?:not|no|never|none|nothing|nobody|nowhere|neither|nor|"
    r"cannot|always|must)(?![A-Za-z0-9_-]))",
    re.IGNORECASE,
)


def _add_block_factory() -> tuple[Any, list[PreservedBlock]]:
    blocks: list[PreservedBlock] = []
    seed = secrets.token_hex(4)
    counter = 0

    def add(content: str, kind: str) -> str:
        nonlocal counter
        placeholder = f"{SENTINEL_PREFIX}_{seed}_{counter}\x00"
        blocks.append(PreservedBlock(placeholder, content, kind))
        counter += 1
        return placeholder

    return add, blocks


def _replace_pattern(
    text: str, pattern: re.Pattern[str], kind: str, add: Any
) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        if not value or SENTINEL_PREFIX in value:
            return value
        return add(value, kind)

    return pattern.sub(repl, text)


def _extract_frontmatter(text: str, add: Any) -> str:
    if not text.startswith("---\n"):
        return text
    close = text.find("\n---", 4)
    if close == -1:
        return text
    close_end = text.find("\n", close + 4)
    end = len(text) if close_end == -1 else close_end + 1
    return add(text[:end], "frontmatter") + text[end:]


_FENCE_OPEN_RE = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})[^\n]*(?:\n|$)")
_FENCE_CLOSE_RE = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})\s*(?:\n|$)")


def _extract_fenced_code(text: str, add: Any) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    while i < len(lines):
        opening = _FENCE_OPEN_RE.match(lines[i])
        if not opening:
            out.append(lines[i])
            i += 1
            continue
        fence = opening.group(2)
        block = [lines[i]]
        j = i + 1
        closed = False
        while j < len(lines):
            block.append(lines[j])
            closing = _FENCE_CLOSE_RE.match(lines[j])
            if (
                closing
                and closing.group(2)[0] == fence[0]
                and len(closing.group(2)) >= len(fence)
            ):
                closed = True
                break
            j += 1
        if not closed:
            out.append(lines[i])
            i += 1
            continue
        out.append(add("".join(block), "fenced_code"))
        i = j + 1
    return "".join(out)


def extract_preserved(
    text: str, extra_patterns: list[str | re.Pattern[str]] | None = None
) -> tuple[str, list[PreservedBlock]]:
    """Replace preserved spans with placeholders and return both."""
    if not text:
        return text, []
    add, blocks = _add_block_factory()
    result = text
    result = _extract_frontmatter(result, add)

    for raw in extra_patterns or []:
        try:
            pattern = re.compile(raw) if isinstance(raw, str) else raw
        except re.error:
            continue
        result = _replace_pattern(result, pattern, "custom", add)

    for pattern, kind in _REGION_PATTERNS:
        result = _replace_pattern(result, pattern, kind, add)
    result = _extract_fenced_code(result, add)
    for pattern, kind in _RISK_PATTERNS:
        result = _replace_pattern(result, pattern, kind, add)
    for pattern, kind in _BUILTIN_PATTERNS:
        result = _replace_pattern(result, pattern, kind, add)

    result = _replace_pattern(result, _PROTECTED_SPAN_RE, "protected_token", add)
    return result, blocks


def restore_preserved(
    text: str, blocks: list[PreservedBlock], fallback: str | None = None
) -> str:
    """Restore placeholders; fail open if any placeholder disappeared."""
    if not blocks:
        return text
    if any(block.placeholder not in text for block in blocks):
        return fallback if fallback is not None else text
    for block in blocks:
        text = text.replace(block.placeholder, block.content)
    return text


def split_prose_and_preserved(
    text: str, extra_patterns: list[str | re.Pattern[str]] | None = None
) -> list[tuple[str, str]]:
    """Split into ``(kind, value)`` where kind is ``prose`` or ``preserved``.

    Preserved values are the original spans; they must never be sent to a
    compressor. Prose values may be compressed independently.
    """
    masked, blocks = extract_preserved(text, extra_patterns)
    if not blocks:
        return [("prose", text)]
    by_placeholder = {block.placeholder: block for block in blocks}
    pattern = re.compile(
        "(" + "|".join(re.escape(block.placeholder) for block in blocks) + ")"
    )
    segments: list[tuple[str, str]] = []
    for part in pattern.split(masked):
        if not part:
            continue
        block = by_placeholder.get(part)
        if block is not None:
            segments.append(("preserved", block.content))
        else:
            segments.append(("prose", part))
    return segments


def truncate_preserved(text: str, limit: int) -> str:
    """Truncate prose while keeping preserved spans near the retained head."""
    if limit <= 0 or len(text) <= limit:
        return text
    masked, blocks = extract_preserved(text)
    if not blocks:
        return text[:limit].rstrip() + "\n[truncated]"

    preserved_len = sum(len(block.content) for block in blocks)
    budget = max(limit - preserved_len, max(64, limit // 3))
    cut = masked[:budget]

    # Drop a partially included placeholder.
    start = cut.rfind(SENTINEL_PREFIX)
    if start != -1 and cut.find("\x00", start + len(SENTINEL_PREFIX)) == -1:
        cut = cut[:start]

    kept = [block for block in blocks if block.placeholder in cut]
    kept_set = set(kept)
    for block in blocks:
        if block not in kept_set:
            cut = cut.replace(block.placeholder, "")
    for block in kept:
        cut = cut.replace(block.placeholder, block.content)

    result = cut.rstrip() + "\n[truncated]"
    if len(result) >= len(text):
        return text[:limit].rstrip() + "\n[truncated]"
    return result
