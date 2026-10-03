from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_baseline import _response_text
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    answer_from_profile,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Short-term, persistent profile, and compact memory in one agent."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        answer_tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + answer_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.prompt_token_usage(thread_id) + prompt_tokens
        )
        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        prompt_parts = [
            "Use the persistent user facts and current thread context to answer.",
            self.profile_store.read_text(user_id),
            str(context["summary"]),
        ]
        prompt_parts.extend(
            f"{item['role']}: {item['content']}" for item in context["messages"]
        )
        return estimate_tokens("\n".join(part for part in prompt_parts if part))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        del thread_id  # Persistent recall is user-scoped; thread memory is measured separately.
        answer = answer_from_profile(
            message, self.profile_store.facts(user_id), "hồ sơ người dùng"
        )
        if answer:
            return answer
        if "?" in message or any(
            phrase in message.casefold()
            for phrase in ("mình tên gì", "nhắc lại", "ở đâu", "nghề gì")
        ):
            return "Mình chưa có thông tin chắc chắn về điều đó."
        if self.profile_store.file_size(user_id):
            return "Mình đã cập nhật hồ sơ phù hợp và sẽ dùng khi cần."
        return "Mình đã ghi nhận trong ngữ cảnh cuộc trò chuyện này."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        system_prompt = (
            "Answer the user's request. Treat the latest explicit profile fact as current. "
            "Profile facts:\n"
            f"{self.profile_store.read_text(user_id)}\n"
            "Older thread summary:\n"
            f"{context['summary']}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *context["messages"],
        ]
        response = self.langchain_agent.invoke(messages)
        answer = _response_text(response)
        answer_tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + answer_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.prompt_token_usage(thread_id) + prompt_tokens
        )
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
            # Keep tests and the benchmark usable when optional SDKs are absent.
            return None
