# litellm-plugin-collection

A collection of litellm plugins/guardrails providing context compression,
loop detection, and budget-aware context management.

## Plugins

| Plugin | Target | Description |
|---|---|---|
| `dcp_pruner` | `tool` / `tool_result` only | Dynamic context pruning inspired by [OpenCode DCP](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning). Removes duplicate and stale tool outputs. |
| `llmlingua_compressor` | `system` / `user` / `assistant` text only | Prompt compression using [LLMLingua-2](https://github.com/microsoft/LLMLingua) (base model). Falls back to lite compression if unavailable. |
| `context_budget` | All messages | Budget-aware context truncation and pruning. |
| `compression_summary` | All messages (read-only) | Aggregates overall compression stats (tokens before/after, ratio). |
| `loop_guard` | Request-level | Session-aware loop detection and intervention. |

## DCP (Dynamic Context Pruning)

The DCP-like pruning in `dcp_pruner` is inspired by
[OpenCode Dynamic Context Pruning (DCP)](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning),
which reduces context size by replacing stale tool outputs with placeholders.
This plugin reimplements the pruning logic as a litellm `pre_call` guardrail,
targeting only `tool` and `tool_result` messages.

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

## Scopes

- **DCP**: `tool` / `tool_result` only
- **llmlingua-2**: `system` / `user` / `assistant` text only
- **Headroom**: upstream standard (tool + context)

## License

MIT
```