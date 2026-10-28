# litellm-plugin-collection

Quinella の機能を litellm の plugin/guardrail として移植したコレクション。

## プラグイン一覧

- `dcp_pruner`: tool/tool_result のみを対象とする DCP-like 動的 pruning
- `llmlingua_compressor`: system/user/assistant テキストのみを対象とする llmlingua-2 圧縮
- `context_budget`: 予算ベースのコンテキスト圧縮
- `loop_guard`: セッション状態に基づくループ検知

## インストール

```bash
pip install -e /path/to/litellm-plugin-collection
# llmlingua-2 を使う場合
pip install -e "/path/to/litellm-plugin-collection[llmlingua]"
```

## 設定

`examples/config.yaml` を参照。litellm proxy の config.yaml に guardrails と model_list を追加する。

## 対象スコープ

- DCP: tool / tool_result のみ
- llmlingua-2: system / user / assistant テキストのみ
- headroom: 上流標準（tool + context）