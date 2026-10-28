# litellm-plugin-collection

A collection of litellm plugins/guardrails providing context compression,
loop detection, and budget-aware context management.

## Plugins

| Plugin | Target | Description |
|---|---|---|
| `dcp_pruner` | `tool` / `tool_result` only | Dynamic context pruning inspired by [OpenCode DCP](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning). Removes duplicate tool outputs and elides oversized ones. |
| `llmlingua_compressor` | `user` / `assistant` prose only | Prompt compression using [LLMLingua-2](https://github.com/microsoft/LLMLingua) (base model). System prompts and meaningful spans are preserved. Falls back to lite compression if unavailable. |
| `context_budget` | All messages | Budget-aware context truncation and pruning. |
| `compression_summary` | All messages (read-only) | Aggregates overall compression stats (tokens before/after, ratio). |
| `loop_guard` | Request-level | Session-aware detection of repeated identical user turns. Ignores tool results and continuations of the current turn. |

## DCP (Dynamic Context Pruning)

The DCP-like pruning in `dcp_pruner` is inspired by
[OpenCode Dynamic Context Pruning (DCP)](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning),
which reduces context size by replacing duplicate tool outputs with placeholders
and eliding very long ones. This plugin reimplements the pruning logic as a
litellm `pre_call` guardrail, targeting only `tool` and `tool_result` messages.

## Installation

```bash
# Editable install
cd litellm-plugin-collection
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/litellm-plugin-collection:$PYTHONPATH
```

### Optional: LLMLingua-2

```bash
pip install llmlingua
```

If not installed, `llmlingua_compressor` falls back to lite compression.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Configuration

See `examples/config.yaml` for a full litellm proxy config.

```yaml
guardrails:
  - guardrail_name: dcp-pruner
    litellm_params:
      guardrail: litellm_plugin_collection.dcp_pruner.guardrail
      mode: pre_call
  - guardrail_name: llmlingua-2
    litellm_params:
      guardrail: litellm_plugin_collection.llmlingua_compressor.guardrail
      mode: pre_call
  - guardrail_name: compression-summary
    litellm_params:
      guardrail: litellm_plugin_collection.compression_summary.guardrail
      mode: pre_call
  - guardrail_name: loop-guard
    litellm_params:
      guardrail: litellm_plugin_collection.loop_guard.guardrail
      mode: pre_call
```

## Nix / NixOS

A flake is provided. Build with:

```bash
nix build .#default   # litellm with plugin on PYTHONPATH
nix build .#plugin    # plugin package only
nix build .#llmlingua # llmlingua package
```

NixOS `services.litellm`:

```nix
services.litellm = {
  enable = true;
  package = inputs.litellm-plugin-collection.packages.x86_64-linux.default;
};
```

## Preserved content

Compression is only safe for prose. Following
[OmniRoute](https://github.com/diegosouzapw/OmniRoute)'s preservation model, the
compressors keep these spans verbatim:

- system prompts (opt in with `compress_system: true`)
- agent instruction envelopes (`<system-reminder>`, `<instructions>`, `<project-instructions>`)
- fenced code, inline code, frontmatter
- math / LaTeX, Markdown headings, tables and links
- URLs, file paths, env vars, versions, dotted identifiers and function calls
- error messages, stack traces, secrets / private keys and DDL
- XML tags, negations and absolutes (`not`, `never`, `must`, `don't`, ...)

System prompts are never compressed by default. Set `compress_system: true` to
opt in; cached requests always keep the system prompt unchanged.

## Loop guard

`loop_guard` only counts **new user turns**. A request that continues the same
turn (tool calls, tool results, reasoning retries) is not a loop, even though
LiteLLM resends the identical last user message every time. Tool-result-only
messages are ignored.

The guard needs a stable session identity. It reads, in order:

1. `session_id` / `x-opencode-session` / `x-opencode-session-id`
2. `metadata.session_id` / `litellm_metadata.session_id`
3. `metadata.user_api_key_session_id`

When none is present the guard fails open (does not block), because correlating
unrelated clients under a shared `default` bucket causes false positives.

Declare `loop-guard` **before** the compression guardrails so it sees the
original user text (see `examples/config.yaml`).

## Scopes

- **DCP**: `tool` / `tool_result` only
- **llmlingua-2**: `user` / `assistant` prose only (system prompt preserved by default)
- **Headroom**: upstream standard (tool + context)

## License

MIT
