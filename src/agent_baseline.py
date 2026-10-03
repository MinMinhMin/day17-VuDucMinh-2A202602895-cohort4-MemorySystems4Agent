from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_from_profile, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """An agent whose short-term memory lasts only for one thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        del user_id  # Baseline deliberately has no per-user persistent state.
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in session.messages
        )
        prompt_tokens = estimate_tokens(prompt_text)
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        answer = answer_from_profile(message, facts, "cuộc trò chuyện này")
        if not answer:
            if "?" in message or any(
                phrase in message.casefold()
                for phrase in ("mình tên gì", "nhắc lại", "ở đâu", "nghề gì")
            ):
                answer = "Mình chưa có thông tin đó trong cuộc trò chuyện hiện tại."
            else:
                answer = "Mình đã ghi nhận trong cuộc trò chuyện này."

        answer_tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += answer_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_messages = [
            {"role": "system", "content": "Answer concisely using only this thread's context."},
            *session.messages,
        ]
        prompt_tokens = estimate_tokens(
            "\n".join(f"{item['role']}: {item['content']}" for item in prompt_messages)
        )
        response = self.langchain_agent.invoke(prompt_messages)
        answer = _response_text(response)
        answer_tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += answer_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        if self.force_offline or self.config.model.provider == "offline":
            return None
        try:
            return build_chat_model(self.config.model)
        except RuntimeError:
            # Provider packages are optional; absence keeps the lab runnable offline.
            return None


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()
