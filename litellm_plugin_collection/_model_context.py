"""Best-effort detection of a model's context window from LiteLLM request data."""

from __future__ import annotations

from typing import Any


def _model_name(request_data: Any, inputs: Any) -> str | None:
    if isinstance(inputs, dict):
        model = inputs.get("model")
        if isinstance(model, str) and model:
            return model
    if isinstance(request_data, dict):
        model = request_data.get("model")
        if isinstance(model, str) and model:
            return model
        metadata = request_data.get("metadata")
        if isinstance(metadata, dict):
            model = metadata.get("model")
            if isinstance(model, str) and model:
                return model
    return None


def _provider_for(model: str) -> str | None:
    if "/" in model:
        prefix = model.split("/", 1)[0]
        if prefix in {"opencode_go", "opencode", "commandcode"}:
            return prefix
    return None


def detect_context_limit(
    request_data: Any = None,
    inputs: Any = None,
    default: int = 128000,
) -> int:
    """Return the model's input context limit, or ``default`` if unknown."""
    model = _model_name(request_data, inputs)

    # Explicit model_info in the request/config always wins.
    for bucket in (
        request_data.get("model_info") if isinstance(request_data, dict) else None,
        request_data.get("litellm_params", {}).get("model_info")
        if isinstance(request_data, dict) and isinstance(request_data.get("litellm_params"), dict)
        else None,
    ):
        if isinstance(bucket, dict):
            for key in ("max_input_tokens", "context_window", "max_tokens"):
                value = bucket.get(key)
                if isinstance(value, int) and value > 0:
                    return value

    if model:
        # LiteLLM proxy keeps the configured model list in memory, including
        # model_info from config.yaml. This is the most reliable source for
        # custom/local models.
        try:
            import litellm

            model_list = getattr(litellm, "model_list", None)
            if isinstance(model_list, list):
                for entry in model_list:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("model_name") != model and entry.get("model") != model:
                        continue
                    info = entry.get("model_info")
                    if isinstance(info, dict):
                        for key in ("max_input_tokens", "max_tokens", "context_window"):
                            value = info.get(key)
                            if isinstance(value, int) and value > 0:
                                return value
                    params = entry.get("litellm_params", {})
                    if isinstance(params, dict):
                        info = params.get("model_info")
                        if isinstance(info, dict):
                            for key in ("max_input_tokens", "max_tokens", "context_window"):
                                value = info.get(key)
                                if isinstance(value, int) and value > 0:
                                    return value
        except Exception:
            pass

        try:
            import litellm

            provider = _provider_for(model)
            kwargs = {"custom_llm_provider": provider} if provider else {}
            info = litellm.get_model_info(model, **kwargs)
            for key in ("max_input_tokens", "max_tokens", "context_window"):
                value = info.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        except Exception:
            pass

        # LiteLLM's cost table is keyed by the bare model name for some providers.
        try:
            import litellm

            cost = getattr(litellm, "model_cost", {}) or {}
            for key in (model, model.split("/")[-1]):
                entry = cost.get(key)
                if isinstance(entry, dict):
                    for field in ("max_input_tokens", "max_tokens", "context_window"):
                        value = entry.get(field)
                        if isinstance(value, int) and value > 0:
                            return value
        except Exception:
            pass

    if isinstance(request_data, dict):
        metadata = request_data.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("max_input_tokens")
            if isinstance(value, int) and value > 0:
                return value

    return default