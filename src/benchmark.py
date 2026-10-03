from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load and validate the conversation and recall-question schema."""

    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain at least one conversation.")

    required_conversation_fields = {"id", "user_id", "turns", "recall_questions"}
    for index, conversation in enumerate(data):
        label = f"conversation at index {index}"
        if not isinstance(conversation, dict):
            raise ValueError(f"{label} must be a JSON object.")
        missing = required_conversation_fields - conversation.keys()
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}.")
        if not isinstance(conversation["id"], str) or not conversation["id"].strip():
            raise ValueError(f"{label} must have a non-empty string id.")
        if not isinstance(conversation["user_id"], str) or not conversation["user_id"].strip():
            raise ValueError(f"{label} must have a non-empty string user_id.")
        turns = conversation["turns"]
        if not isinstance(turns, list) or not turns or any(
            not isinstance(turn, str) or not turn.strip() for turn in turns
        ):
            raise ValueError(f"{label} must have a non-empty list of text turns.")
        questions = conversation["recall_questions"]
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"{label} must have at least one recall question.")
        for question_index, question in enumerate(questions):
            question_label = f"{label}, recall question {question_index}"
            if not isinstance(question, dict):
                raise ValueError(f"{question_label} must be a JSON object.")
            if not isinstance(question.get("question"), str) or not question["question"].strip():
                raise ValueError(f"{question_label} must have a non-empty question.")
            expected = question.get("expected_contains")
            if not isinstance(expected, list) or not expected or any(
                not isinstance(item, str) or not item.strip() for item in expected
            ):
                raise ValueError(
                    f"{question_label} must have a non-empty expected_contains list."
                )
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Score no facts, partial facts, or all expected facts as 0/0.5/1."""

    if not expected:
        return 0.0
    normalized_answer = answer.casefold()
    matched = sum(1 for fact in expected if fact.casefold() in normalized_answer)
    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Apply a lightweight, identical answer-coverage/conciseness score."""

    text = answer.strip()
    if not text:
        return 0.0
    coverage = recall_points(text, expected)
    concise = 1.0 if 20 <= len(text) <= 400 else 0.5
    return round(0.85 * coverage + 0.15 * concise, 3)


def _answer_text(result: Any) -> str:
    if isinstance(result, dict):
        value = result.get("answer", result.get("output", ""))
    else:
        value = result
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _counter(agent: Any, method_name: str, thread_id: str) -> int:
    method = getattr(agent, method_name, None)
    return int(method(thread_id)) if callable(method) else 0


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Run each conversation and each recall question against one agent."""

    del config  # State ownership belongs to the injected agent, not the evaluator.
    user_ids = {conversation["user_id"] for conversation in conversations}
    size_method = getattr(agent, "memory_file_size", None)
    initial_sizes = (
        {user_id: int(size_method(user_id)) for user_id in user_ids}
        if callable(size_method)
        else {}
    )
    agent_tokens = 0
    prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    thread_ids: set[str] = set()

    def run_reply(user_id: str, thread_id: str, message: str) -> str:
        nonlocal agent_tokens, prompt_tokens
        before_agent = _counter(agent, "token_usage", thread_id)
        before_prompt = _counter(agent, "prompt_token_usage", thread_id)
        result = agent.reply(user_id=user_id, thread_id=thread_id, message=message)
        after_agent = _counter(agent, "token_usage", thread_id)
        after_prompt = _counter(agent, "prompt_token_usage", thread_id)
        if isinstance(result, dict) and "agent_tokens" in result:
            agent_tokens += int(result["agent_tokens"])
        else:
            agent_tokens += max(0, after_agent - before_agent)
        if isinstance(result, dict) and "prompt_tokens" in result:
            prompt_tokens += int(result["prompt_tokens"])
        else:
            prompt_tokens += max(0, after_prompt - before_prompt)
        thread_ids.add(thread_id)
        return _answer_text(result)

    for conversation in conversations:
        user_id = conversation["user_id"]
        conversation_thread = conversation["id"]
        for turn in conversation["turns"]:
            run_reply(user_id, conversation_thread, turn)
        for question_index, recall_question in enumerate(conversation["recall_questions"], start=1):
            recall_thread = f"{conversation_thread}:recall:{question_index}"
            answer = run_reply(user_id, recall_thread, recall_question["question"])
            expected = recall_question["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_sizes = (
        {user_id: int(size_method(user_id)) for user_id in user_ids}
        if callable(size_method)
        else {}
    )
    memory_growth = (
        sum(
            max(0, final_sizes[user_id] - initial_sizes[user_id])
            for user_id in user_ids
        )
        if callable(size_method)
        else 0
    )
    compactions = sum(
        _counter(agent, "compaction_count", thread_id) for thread_id in thread_ids
    )
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / len(recall_scores), 3),
        response_quality=round(sum(quality_scores) / len(quality_scores), 3),
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark data using the six required Markdown columns."""

    headers = (
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.agent_name,
                    f"{row.agent_tokens_only:,}",
                    f"{row.prompt_tokens_processed:,}",
                    f"{row.recall_score:.3f}",
                    f"{row.response_quality:.3f}",
                    f"{row.memory_growth_bytes:,}",
                    str(row.compactions),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _run_suite(title: str, dataset_path: Path, root: Path) -> None:
    conversations = load_conversations(dataset_path)
    with tempfile.TemporaryDirectory(prefix="memory-lab-benchmark-") as temp_state:
        config = load_config(root, state_dir=Path(temp_state) / "state")
        baseline = BaselineAgent(config=config, force_offline=True)
        advanced = AdvancedAgent(config=config, force_offline=True)
        rows = [
            run_agent_benchmark("Baseline", baseline, conversations, config),
            run_agent_benchmark("Advanced", advanced, conversations, config),
        ]
    print(f"## {title}")
    print(format_rows(rows))
    print()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    _run_suite("Standard Benchmark", root / "data" / "conversations.json", root)
    _run_suite(
        "Long-Context Stress Benchmark",
        root / "data" / "advanced_long_context.json",
        root,
    )


if __name__ == "__main__":
    main()
