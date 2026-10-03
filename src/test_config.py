from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from config import load_config
from model_provider import ProviderConfig, build_chat_model, normalize_provider


def test_normalize_provider_aliases_and_reject_unknown_names() -> None:
    assert normalize_provider(" OPEN_AI ") == "openai"
    assert normalize_provider("google") == "gemini"
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("offline") == "offline"

    with pytest.raises(ValueError, match="Unsupported provider"):
        normalize_provider("made-up-provider")


def test_load_config_uses_root_relative_paths_and_offline_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path)

    assert config.base_dir == tmp_path.resolve()
    assert config.data_dir == tmp_path.resolve() / "data"
    assert config.state_dir == tmp_path.resolve() / "state"
    assert config.state_dir.is_dir()
    assert config.compact_threshold_tokens > 0
    assert config.compact_keep_messages > 0
    assert config.model.provider == "offline"
    assert config.judge_model.provider == "offline"


def test_load_config_reads_provider_and_compact_settings_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "custom")
    monkeypatch.setenv("LLM_MODEL", "local-chat-model")
    monkeypatch.setenv("CUSTOM_API_KEY", "local-secret")
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("JUDGE_PROVIDER", "openrouter")
    monkeypatch.setenv("JUDGE_MODEL", "vendor/judge")
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "321")
    monkeypatch.setenv("COMPACT_KEEP_MESSAGES", "6")
    external_state = tmp_path / "isolated-state"

    config = load_config(tmp_path, state_dir=external_state)

    assert config.model.provider == "custom"
    assert config.model.model_name == "local-chat-model"
    assert config.model.api_key == "local-secret"
    assert config.model.base_url == "http://localhost:8000/v1"
    assert config.judge_model.provider == "openrouter"
    assert config.judge_model.model_name == "vendor/judge"
    assert config.compact_threshold_tokens == 321
    assert config.compact_keep_messages == 6
    assert config.state_dir == external_state.resolve()
    assert config.state_dir.is_dir()


def test_gemini_provider_accepts_google_api_key_environment_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config = load_config(tmp_path)

    assert config.model.api_key == "google-key"


@pytest.mark.parametrize(
    ("provider", "module_name", "class_name", "expected_kwargs"),
    [
        ("openai", "langchain_openai", "ChatOpenAI", {"api_key": "test-key"}),
        (
            "custom",
            "langchain_openai",
            "ChatOpenAI",
            {"api_key": "test-key", "base_url": "http://localhost/v1"},
        ),
        (
            "gemini",
            "langchain_google_genai",
            "ChatGoogleGenerativeAI",
            {"api_key": "test-key"},
        ),
        (
            "anthropic",
            "langchain_anthropic",
            "ChatAnthropic",
            {"anthropic_api_key": "test-key"},
        ),
        (
            "ollama",
            "langchain_ollama",
            "ChatOllama",
            {"base_url": "http://localhost:11434"},
        ),
        (
            "openrouter",
            "langchain_openrouter",
            "ChatOpenRouter",
            {"api_key": "test-key", "base_url": "https://openrouter.example/v1"},
        ),
    ],
)
def test_build_chat_model_passes_provider_specific_arguments(
    provider: str,
    module_name: str,
    class_name: str,
    expected_kwargs: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType(module_name)

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    setattr(module, class_name, FakeChatModel)
    monkeypatch.setitem(sys.modules, module_name, module)
    config = ProviderConfig(
        provider=provider,
        model_name="model-under-test",
        temperature=0.2,
        api_key="test-key",
        base_url=(
            "https://openrouter.example/v1"
            if provider == "openrouter"
            else "http://localhost:11434"
            if provider == "ollama"
            else "http://localhost/v1"
        ),
    )

    model = build_chat_model(config)

    assert model.kwargs["model"] == "model-under-test"
    assert model.kwargs["temperature"] == 0.2
    for key, value in expected_kwargs.items():
        assert model.kwargs[key] == value
