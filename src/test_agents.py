from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    extract_profile_updates,
)
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    offline_model = ProviderConfig(
        provider="offline", model_name="offline", temperature=0.0
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=40,
        compact_keep_messages=2,
        model=offline_model,
        judge_model=offline_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    initial = "# User profile\n\n- name: Lan\n- location: Huế\n"
    store.write_text("user-1", initial)

    assert store.read_text("user-1") == initial
    assert store.edit_text("user-1", "location: Huế", "location: Đà Nẵng")
    assert "location: Đà Nẵng" in store.read_text("user-1")
    assert not store.edit_text("user-1", "missing", "replacement")


def test_user_profile_paths_stay_inside_profile_root(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    store = UserProfileStore(root)

    assert store.path_for("../../outside\\nested").is_relative_to(root.resolve())
    assert store.path_for("team-a/alice") != store.path_for("team-b/alice")


def test_distinct_user_ids_cannot_share_a_profile_path(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.path_for("alpha/beta") != store.path_for("alpha_beta")
    assert store.path_for(r"alpha\beta") != store.path_for("alpha/beta")


def test_extract_profile_updates_uses_explicit_current_facts() -> None:
    updates = extract_profile_updates(
        "Mình tên là Lan. Mình hiện đang ở Huế và đang làm backend engineer."
    )

    assert updates["name"] == "Lan"
    assert updates["location"] == "Huế"
    assert updates["profession"] == "backend engineer"


def test_extract_profile_updates_rejects_question_and_joke_claims() -> None:
    assert extract_profile_updates("Hay là mình chuyển sang product manager không?") == {}
    assert extract_profile_updates(
        "Mình đùa là sẽ chuyển sang product manager cho vui."
    ) == {}
    assert extract_profile_updates("Mình có thích Python và AI không?") == {}
    assert extract_profile_updates("Nhắc lại giúp mình tên và style trả lời mình thích.") == {}


def test_hypothetical_future_location_does_not_replace_current_profile(tmp_path: Path) -> None:
    advanced = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    advanced.reply("user-1", "thread-1", "Mình hiện đang ở Huế.")
    advanced.reply(
        "user-1",
        "thread-2",
        "Mình hiện đang ở Huế nhưng nếu sau này mình ở Hà Nội, mình sẽ cân nhắc chuyển nhà.",
    )

    answer = advanced.reply(
        "user-1", "thread-3", "Nơi ở hiện tại của mình ở đâu?"
    )["answer"]

    assert "Huế" in answer
    assert "Hà Nội" not in answer


def test_uncertain_location_is_not_saved_as_a_current_fact() -> None:
    assert extract_profile_updates("Có thể mình đang ở Hà Nội, nhưng chưa chắc.") == {}


def test_explicit_location_correction_survives_historical_clause(tmp_path: Path) -> None:
    advanced = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    advanced.reply("user-1", "thread-1", "Mình hiện đang ở Huế.")
    advanced.reply(
        "user-1",
        "thread-2",
        "Lúc đầu mình nói hiện ở Huế nhưng thực ra hiện mình đang ở Đà Nẵng.",
    )

    answer = advanced.reply(
        "user-1", "thread-3", "Nơi ở hiện tại của mình ở đâu?"
    )["answer"]

    assert "Đà Nẵng" in answer
    assert "Huế" not in answer


def test_profile_upsert_replaces_a_corrected_fact(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("user-1", "location", "Huế")
    store.upsert_fact("user-1", "location", "Đà Nẵng")

    profile = store.read_text("user-1")
    assert "location: Đà Nẵng" in profile
    assert "location: Huế" not in profile


def test_extractor_keeps_pet_name_when_follow_up_omits_the_word_name() -> None:
    assert extract_profile_updates("Con corgi Bơ dạo này hay phá lúc mình họp online.")["pet"] == "corgi tên Bơ"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages,
    )
    for index in range(8):
        manager.append("long-thread", "user", f"Message {index}: " + "context " * 20)

    context = manager.context("long-thread")
    assert manager.compaction_count("long-thread") > 0
    assert len(context["messages"]) <= config.compact_keep_messages
    assert context["summary"]
    assert manager.compaction_count("long-thread") > 1
    assert len(context["summary"]) <= 6 * 130


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    advanced.reply("user-1", "first-thread", "Mình tên là Lan.")
    baseline.reply("user-1", "first-thread", "Mình tên là Lan.")
    baseline_same_thread = baseline.reply(
        "user-1", "first-thread", "Mình tên gì?"
    )["answer"]
    advanced_answer = advanced.reply("user-1", "new-thread", "Mình tên gì?")["answer"]
    baseline_answer = baseline.reply("user-1", "new-thread", "Mình tên gì?")["answer"]

    assert "Lan" in baseline_same_thread
    assert "Lan" in advanced_answer
    assert "Lan" not in baseline_answer


def test_correction_is_recalled_in_a_later_thread(tmp_path: Path) -> None:
    advanced = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    advanced.reply("user-1", "thread-1", "Mình hiện đang ở Huế.")
    advanced.reply(
        "user-1", "thread-2", "Mình hiện đang ở Đà Nẵng, không còn ở Huế."
    )

    answer = advanced.reply("user-1", "thread-3", "Nơi ở hiện tại của mình ở đâu?")["answer"]

    assert "Đà Nẵng" in answer
    assert "Huế" not in answer


def test_recall_detects_location_asked_as_a_negative_confirmation(tmp_path: Path) -> None:
    advanced = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    advanced.reply("user-1", "thread-1", "Mình hiện đang ở Huế.")

    answer = advanced.reply(
        "user-1",
        "thread-2",
        "Hiện tại mình làm nghề gì và mình còn ở Huế không?",
    )["answer"]

    assert "Huế" in answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)
    long_turns = [f"Mình đang thảo luận chủ đề {index}: " + "chi tiết quan trọng " * 35 for index in range(10)]

    for index, turn in enumerate(long_turns):
        advanced.reply("user-1", "long-thread", turn)
        baseline.reply("user-1", "long-thread", turn)

    assert advanced.compaction_count("long-thread") > 0
    assert advanced.prompt_token_usage("long-thread") < baseline.prompt_token_usage("long-thread")
