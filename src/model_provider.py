from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider settings shared by the baseline and advanced agents.

    ``custom`` targets an OpenAI-compatible endpoint; ``offline`` avoids API calls.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


SUPPORTED_PROVIDERS = {
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
    "offline",
}

_PROVIDER_ALIASES = {
    "open_ai": "openai",
    "google": "gemini",
    "google_genai": "gemini",
    "google-gemini": "gemini",
    "anthorpic": "anthropic",
    "open_router": "openrouter",
    "local": "ollama",
    "none": "offline",
}


def normalize_provider(value: str) -> str:
    """Return the canonical name for one of the supported model providers."""

    name = value.strip().lower().replace(" ", "_")
    name = _PROVIDER_ALIASES.get(name, name)
    if name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {value!r}; choose one of "
            f"{', '.join(sorted(SUPPORTED_PROVIDERS - {'offline'}))} or 'offline'."
        )
    return name


def build_chat_model(config: ProviderConfig):
    """Build an optional LangChain chat model for the selected provider.

    Provider SDKs are imported only when a live model is requested, so the
    deterministic offline benchmark has no optional dependency requirements.
    """

    provider = normalize_provider(config.provider)
    if provider == "offline":
        raise ValueError("The offline provider does not construct a chat model.")

    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if provider == "custom" and config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatGoogleGenerativeAI(**kwargs)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["anthropic_api_key"] = config.api_key
            return ChatAnthropic(**kwargs)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            kwargs = {"model": config.model_name, "temperature": config.temperature}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOllama(**kwargs)

        if provider == "openrouter":
            from langchain_openrouter import ChatOpenRouter

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenRouter(**kwargs)
    except ImportError as exc:
        package_by_provider = {
            "openai": "langchain-openai",
            "custom": "langchain-openai",
            "gemini": "langchain-google-genai",
            "anthropic": "langchain-anthropic",
            "ollama": "langchain-ollama",
            "openrouter": "langchain-openrouter",
        }
        raise RuntimeError(
            f"Provider '{provider}' needs the optional package "
            f"{package_by_provider[provider]!r}."
        ) from exc

    raise ValueError(f"Unsupported provider {provider!r}.")
