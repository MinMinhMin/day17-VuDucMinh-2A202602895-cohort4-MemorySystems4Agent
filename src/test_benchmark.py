from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import (
    BenchmarkRow,
    format_rows,
    heuristic_quality,
    load_conversations,
    recall_points,
    run_agent_benchmark,
)


class RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, object]:
        self.calls.append((user_id, thread_id, message))
        return {"answer": "Lan likes Python and Huế", "agent_tokens": 4, "prompt_tokens": 8}

    def token_usage(self, thread_id: str) -> int:
        return 0

    def prompt_token_usage(self, thread_id: str) -> int:
        return 0

    def memory_file_size(self, user_id: str) -> int:
        return 0

    def compaction_count(self, thread_id: str) -> int:
        return 0


class ThreadOnlyAgent:
    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, object]:
        return {"answer": "No persistent profile", "agent_tokens": 2, "prompt_tokens": 3}

    def compaction_count(self, thread_id: str) -> int:
        return 0


def _conversation() -> list[dict[str, object]]:
    return [
        {
            "id": "sample-1",
            "user_id": "sample-user",
            "turns": ["Mình tên là Lan."],
            "recall_questions": [
                {"question": "Tên mình là gì?", "expected_contains": ["Lan"]},
                {"question": "Mình thích gì?", "expected_contains": ["Python", "Huế"]},
            ],
        }
    ]


def test_loader_validates_conversations_and_rejects_empty_datasets(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(_conversation()), encoding="utf-8")
    assert load_conversations(valid_path)[0]["id"] == "sample-1"

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one conversation"):
        load_conversations(empty_path)


def test_loader_rejects_missing_recall_fields(tmp_path: Path) -> None:
    invalid = _conversation()
    invalid[0]["recall_questions"] = [{"question": "A question without expected facts"}]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="expected_contains"):
        load_conversations(path)


def test_loader_rejects_non_text_turns(tmp_path: Path) -> None:
    invalid = _conversation()
    invalid[0]["turns"] = ["A valid turn", 42]
    path = tmp_path / "invalid_turns.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="list of text turns"):
        load_conversations(path)


def test_recall_points_has_three_levels_and_quality_uses_same_formula() -> None:
    assert recall_points("Lan is from Huế", ["Lan", "Huế"]) == 1.0
    assert recall_points("Lan", ["Lan", "Huế"]) == 0.5
    assert recall_points("Unknown", ["Lan", "Huế"]) == 0.0
    assert heuristic_quality("Lan and Python", ["Lan", "Python"]) > heuristic_quality(
        "", ["Lan", "Python"]
    )
    assert heuristic_quality("Lan", ["Lan"]) == heuristic_quality("Lan", ["Lan"])


def test_format_rows_includes_all_six_metrics() -> None:
    row = BenchmarkRow("Advanced", 12, 34, 0.75, 0.8, 56, 2)

    rendered = format_rows([row])

    for column in (
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ):
        assert column in rendered
    assert "Advanced" in rendered


def test_benchmark_uses_fresh_matching_recall_threads_for_both_agents() -> None:
    baseline = RecordingAgent()
    advanced = RecordingAgent()
    conversations = _conversation()

    run_agent_benchmark("Baseline", baseline, conversations, config=None)
    run_agent_benchmark("Advanced", advanced, conversations, config=None)

    assert [call[1] for call in baseline.calls] == [call[1] for call in advanced.calls]
    assert [call[1] for call in baseline.calls] == [
        "sample-1",
        "sample-1:recall:1",
        "sample-1:recall:2",
    ]


def test_benchmark_reports_zero_memory_growth_for_agent_without_profile_store() -> None:
    row = run_agent_benchmark("Baseline", ThreadOnlyAgent(), _conversation(), config=None)

    assert row.memory_growth_bytes == 0
