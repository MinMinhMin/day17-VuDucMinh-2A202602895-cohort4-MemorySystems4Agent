from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate tokens deterministically from character count."""

    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


@dataclass
class UserProfileStore:
    """Read and update one Markdown profile per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        identity = str(user_id).strip()
        raw = re.sub(r"/+", "_", identity.replace("\\", "/"))
        slug = re.sub(r"[^\w.-]+", "_", raw, flags=re.UNICODE).strip("._")
        if slug in {"", ".", ".."}:
            slug = "user"
        identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        root = self.root_dir.resolve()
        path = (root / f"{slug}--{identity_hash}" / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("User id resolves outside the profile directory.")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        self.write_text(user_id, updated)
        return True

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^\s*-\s*([a-z0-9_]+):\s*(.*?)\s*$", line)
            if match:
                facts[match.group(1)] = match.group(2)
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        normalized_key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")
        normalized_value = " ".join(value.split())
        if not normalized_key or not normalized_value:
            return self.path_for(user_id)

        facts = self.facts(user_id)
        if normalized_key in {"interests", "response_style"} and normalized_key in facts:
            prior_items = [part.strip() for part in facts[normalized_key].split(";")]
            new_items = [part.strip() for part in normalized_value.split(";")]
            combined = prior_items[:]
            for item in new_items:
                if item and item.casefold() not in {old.casefold() for old in combined}:
                    combined.append(item)
            normalized_value = "; ".join(combined)
        facts[normalized_key] = normalized_value

        content = "# User profile\n\n" + "".join(
            f"- {fact_key}: {fact_value}\n" for fact_key, fact_value in facts.items()
        )
        return self.write_text(user_id, content)

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0


_CITY_PATTERN = r"(?:Đà Nẵng|Hà Nội|Hồ Chí Minh|Sài Gòn|Huế|Cần Thơ|Đà Lạt|Nha Trang)"
_ROLE_PATTERN = (
    r"(?:MLOps engineer|backend engineer|product manager|software engineer|"
    r"data engineer|data scientist|AI engineer|machine learning engineer|"
    r"developer|engineer|student|designer|teacher|researcher)"
)


def _claim_fragments(message: str) -> list[str]:
    clauses = re.split(r"(?<=[.!?;])\s+|,\s+|\n+", message)
    fragments: list[str] = []
    for clause in clauses:
        fragments.extend(re.split(r"\s+nhưng\s+", clause, flags=re.IGNORECASE))
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def _is_unreliable_fragment(fragment: str) -> bool:
    if fragment.rstrip().endswith("?"):
        return True
    text = fragment.casefold()
    if re.search(
        r"\b(?:nếu|giả sử|tưởng tượng|trong trường hợp|có thể|chưa chắc|"
        r"không chắc|biết đâu|maybe|perhaps|possibly)\b",
        text,
    ):
        return True
    return any(
        marker in text
        for marker in (
            "đùa",
            "giả sử",
            "hay là",
            "nếu mình",
            "nếu tôi",
            "ví dụ như",
            "lúc đầu",
            "trước đó",
            "trước đây",
        )
    )


def _append_unique(items: list[str], item: str) -> None:
    if item.casefold() not in {existing.casefold() for existing in items}:
        items.append(item)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract explicit, stable profile facts from a user message.

    Extraction is intentionally conservative: it ignores questions and
    hypothetical/joking fragments, and later explicit assertions win for
    single-value fields such as location and profession.
    """

    updates: dict[str, str] = {}
    fragments = _claim_fragments(message)
    reliable_fragments = [
        fragment for fragment in fragments if not _is_unreliable_fragment(fragment)
    ]
    assertive_message = " ".join(reliable_fragments)

    for fragment in reliable_fragments:
        name = re.search(
            r"\b(?:mình|tôi)\s+tên\s+là\s+([\wÀ-ỹ-]+(?:\s+[\wÀ-ỹ-]+)?)",
            fragment,
            flags=re.IGNORECASE,
        )
        if name:
            updates["name"] = name.group(1).strip(" .,:;")

        location = re.search(
            rf"\b(?:(?:mình|tôi)\s+(?:(?:hiện|giờ|bây giờ)\s+)?(?:đang\s+)?|"
            rf"(?:hiện|giờ|bây giờ)\s+)(?:sống\s+(?:ở|tại)|làm việc\s+ở|ở)\s+({_CITY_PATTERN})\b",
            fragment,
            flags=re.IGNORECASE,
        )
        if location:
            updates["location"] = location.group(1)

        profession_patterns = (
            rf"\b(?:nghề nghiệp(?:\s+hiện tại)?|nghề)\s+(?:hiện tại\s+)?(?:của mình\s+)?(?:là|vẫn là)\s+({_ROLE_PATTERN})",
            rf"\b(?:mình|tôi)\b[^.!?;]{{0,60}}?\b(?:đang\s+)?làm\s+({_ROLE_PATTERN})",
            rf"\b(?:đang|hiện đang)\s+làm\s+({_ROLE_PATTERN})",
            rf"\b(?:giờ|hiện tại|bây giờ)\s+(?:(?:mình|tôi)\s+)?(?:đã\s+)?(?:chuyển sang|làm)\s+({_ROLE_PATTERN})",
        )
        for pattern in profession_patterns:
            profession = re.search(pattern, fragment, flags=re.IGNORECASE)
            if profession:
                updates["profession"] = profession.group(1).strip(" .,:;")

    style_markers = (
        (r"ngắn gọn|trả lời gọn|câu trả lời gọn", "ngắn gọn"),
        (r"rõ ý|rõ ràng", "rõ ý"),
        (r"3\s+bullet|ba\s+bullet", "3 bullet"),
        (r"bullet ngắn|bullet ngắn gọn", "bullet ngắn"),
        (r"ví dụ thực tế|ví dụ thực chiến|ví dụ minh họa", "ví dụ thực tế"),
        (r"trade[- ]off", "trade-off"),
        (r"có cấu trúc|thành bullet", "có cấu trúc"),
    )
    if re.search(
        r"trả lời|giải thích|style|phong cách|ưu tiên",
        assertive_message,
        re.IGNORECASE,
    ):
        styles: list[str] = []
        for pattern, label in style_markers:
            if re.search(pattern, assertive_message, re.IGNORECASE):
                _append_unique(styles, label)
        if styles:
            updates["response_style"] = "; ".join(styles)

    if re.search(
        r"quan tâm|thích|đang học|đọc về|làm về|làm việc với",
        assertive_message,
        re.IGNORECASE,
    ):
        interests: list[str] = []
        for pattern, label in (
            (r"\bPython\b", "Python"),
            (r"\bAI(?:\s+ứng dụng|\s+agent)?\b", "AI"),
            (r"\bMLOps\b", "MLOps"),
            (r"\bRAG\b", "RAG"),
            (r"\bbenchmark(?:ing)?\b", "benchmark"),
            (r"\bevaluation\b", "evaluation"),
            (r"\bmemory architecture\b", "memory architecture"),
            (r"\bpipeline\b", "pipeline"),
        ):
            if re.search(pattern, assertive_message, re.IGNORECASE):
                _append_unique(interests, label)
        if interests:
            updates["interests"] = "; ".join(interests)

    drink_patterns = (
        r"đồ uống yêu thích(?: của mình)?\s*(?:là|:)?\s*(cà phê sữa đá|trà sữa|cà phê đen)",
        r"\bmình\s+(?:vẫn\s+)?uống\s+(cà phê sữa đá|trà sữa|cà phê đen)",
    )
    for pattern in drink_patterns:
        drink = re.search(pattern, assertive_message, flags=re.IGNORECASE)
        if drink:
            updates["favorite_drink"] = drink.group(1)

    food = re.search(
        r"(?:món ăn yêu thích(?: của mình)?|món ruột(?: của mình)?)\s*(?:là|:)?\s*(mì Quảng|phở|bún bò Huế)",
        assertive_message,
        flags=re.IGNORECASE,
    )
    if food:
        updates["favorite_food"] = food.group(1)
    elif re.search(r"món ruột|món ăn yêu thích", assertive_message, re.IGNORECASE):
        for known_food in ("mì Quảng", "phở", "bún bò Huế"):
            if re.search(re.escape(known_food), assertive_message, re.IGNORECASE):
                updates["favorite_food"] = known_food
                break

    pet = re.search(
        r"(?:nuôi|[Cc]on)\s+(?:một\s+)?(?:bé\s+)?(corgi|mèo|chó)"
        r"(?:\s+(?:tên\s+)?([A-ZÀ-Ỹ][\wÀ-ỹ-]*))?",
        assertive_message,
    )
    if pet:
        updates["pet"] = pet.group(1) + (f" tên {pet.group(2)}" if pet.group(2) else "")

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a small deterministic summary from older messages."""

    if max_items <= 0:
        return ""
    summary_lines: list[str] = []
    for message in messages[-max_items:]:
        role = message.get("role", "message").capitalize()
        content = " ".join(message.get("content", "").split())
        if not content:
            continue
        if len(content) > 120:
            content = content[:117].rstrip() + "..."
        summary_lines.append(f"{role}: {content}")
    return "\n".join(summary_lines)


def answer_from_profile(question: str, facts: dict[str, str], source: str) -> str:
    """Answer only the profile fields requested in a recall question."""

    query = question.casefold()
    requested: list[tuple[str, str]] = []
    if any(term in query for term in ("tên", "name", "là ai", "who am i")):
        requested.append(("name", "Tên"))
    if any(
        term in query
        for term in ("ở đâu", "nơi ở", "đang ở", "còn ở", "địa điểm", "location")
    ):
        requested.append(("location", "Nơi ở hiện tại"))
    if any(term in query for term in ("nghề", "profession", "đang làm", "công việc")):
        requested.append(("profession", "Nghề hiện tại"))
    if any(term in query for term in ("style", "phong cách", "kiểu trả lời", "bullet")):
        requested.append(("response_style", "Style trả lời"))
    if any(term in query for term in ("đồ uống", "uống", "drink")):
        requested.append(("favorite_drink", "Đồ uống yêu thích"))
    if any(term in query for term in ("món ăn", "món ruột", "favorite food")):
        requested.append(("favorite_food", "Món ăn yêu thích"))
    if any(term in query for term in ("nuôi", "con gì", "thú cưng", "corgi")):
        requested.append(("pet", "Thú cưng"))
    if any(
        term in query
        for term in ("mối quan tâm", "quan tâm", "sở thích", "python", "rag", "kỹ thuật")
    ) or "là ai" in query:
        requested.append(("interests", "Mối quan tâm kỹ thuật"))

    if not requested and any(term in query for term in ("tóm tắt", "mô tả mình", "describe me")):
        requested = [
            ("name", "Tên"),
            ("profession", "Nghề hiện tại"),
            ("location", "Nơi ở hiện tại"),
            ("interests", "Mối quan tâm kỹ thuật"),
            ("response_style", "Style trả lời"),
        ]

    answer_parts: list[str] = []
    seen_keys: set[str] = set()
    for key, label in requested:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        value = facts.get(key)
        if value:
            answer_parts.append(f"{label}: {value}")
    if not answer_parts:
        return ""
    return f"Theo {source}, " + "; ".join(answer_parts) + "."


@dataclass
class CompactMemoryManager:
    """Retain recent messages and a bounded summary of older context."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread.get("summary", ""))
        prompt_text = summary + "\n" + "\n".join(
            item["content"] for item in messages if isinstance(item, dict)
        )
        if (
            estimate_tokens(prompt_text) > self.threshold_tokens
            and len(messages) > self.keep_messages
        ):
            split_at = max(1, len(messages) - self.keep_messages)
            old_messages = messages[:split_at]
            recent_messages = messages[split_at:]
            summary_inputs = ([{"role": "summary", "content": summary}] if summary else [])
            summary_inputs.extend(old_messages)
            thread["summary"] = summarize_messages(summary_inputs, max_items=6)
            thread["messages"] = recent_messages
            thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        messages = thread.get("messages", [])
        return {
            "messages": [dict(item) for item in messages if isinstance(item, dict)],
            "summary": str(thread.get("summary", "")),
            "compactions": int(thread.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
