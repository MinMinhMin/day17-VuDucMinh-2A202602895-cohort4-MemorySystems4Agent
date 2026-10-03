from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for providers, datasets, state, and compaction.

    The model and judge can use any supported provider, or run offline.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _api_key_for(provider: str) -> str | None:
    if provider == "gemini":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    key_name = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider)
    return os.getenv(key_name) if key_name else None


def _model_config(prefix: str, provider: str) -> ProviderConfig:
    model_defaults = {
        "offline": "offline",
        "openai": "gpt-4o-mini",
        "custom": "custom-model",
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.2",
        "openrouter": "openai/gpt-4o-mini",
    }
    configured_model = os.getenv(f"{prefix}_MODEL")
    if prefix == "LLM":
        configured_model = configured_model or os.getenv("MODEL_NAME")
    configured_key = os.getenv(f"{prefix}_API_KEY") or _api_key_for(provider)
    configured_url = os.getenv(f"{prefix}_BASE_URL")
    if not configured_url:
        configured_url = {
            "custom": os.getenv("CUSTOM_BASE_URL"),
            "ollama": os.getenv("OLLAMA_BASE_URL"),
            "openrouter": os.getenv("OPENROUTER_BASE_URL"),
        }.get(provider)
    return ProviderConfig(
        provider=provider,
        model_name=configured_model or model_defaults[provider],
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=configured_key,
        base_url=configured_url,
    )


def _positive_int(value: str | None, default: int, setting_name: str) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError as exc:
        raise ValueError(f"{setting_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{setting_name} must be a positive integer.")
    return parsed


def load_config(
    base_dir: Path | None = None, state_dir: Path | None = None
) -> LabConfig:
    """Load lab settings, defaulting to deterministic offline operation."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    model_provider = normalize_provider(os.getenv("LLM_PROVIDER", "offline"))
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", "offline"))

    configured_state = state_dir or Path(os.getenv("STATE_DIR", root / "state"))
    if not configured_state.is_absolute():
        configured_state = root / configured_state
    configured_state = configured_state.resolve()
    configured_state.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=(root / "data").resolve(),
        state_dir=configured_state,
        compact_threshold_tokens=_positive_int(
            os.getenv("COMPACT_THRESHOLD_TOKENS"), 900, "COMPACT_THRESHOLD_TOKENS"
        ),
        compact_keep_messages=_positive_int(
            os.getenv("COMPACT_KEEP_MESSAGES"), 4, "COMPACT_KEEP_MESSAGES"
        ),
        model=_model_config("LLM", model_provider),
        judge_model=_model_config("JUDGE", judge_provider),
    )
