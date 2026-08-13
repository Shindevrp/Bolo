from __future__ import annotations

from dataclasses import dataclass, field
from modules.memory.facts import Fact


@dataclass
class UserProfile:
    """Cross-session user profile built from accumulated facts."""

    facts: dict[str, Fact] = field(default_factory=dict)
    session_count: int = 0
    last_session: str = ""
    preferred_topics: list[str] = field(default_factory=list)
    conversation_style: str = "balanced"

    def update_from_session(self, facts: list[Fact], topic: str = "") -> None:
        for fact in facts:
            existing = self.facts.get(fact.key)
            if existing is None or fact.source_turn >= existing.source_turn:
                self.facts[fact.key] = fact
        self.session_count += 1
        if topic and topic not in self.preferred_topics:
            self.preferred_topics.append(topic)
            if len(self.preferred_topics) > 10:
                self.preferred_topics = self.preferred_topics[-10:]

    def to_prompt_block(self) -> str:
        if not self.facts and self.session_count <= 1:
            return ""

        lines: list[str] = []

        if self.session_count > 1:
            lines.append(
                f"This is your {self.session_count}th conversation with this user."
            )

        if self.facts:
            fact_lines = [
                f"- {f.key}: {f.value}"
                for f in sorted(
                    self.facts.values(),
                    key=lambda f: (-f.confidence, -f.source_turn),
                )[:15]
            ]
            if fact_lines:
                lines.append("Things you know about the user:")
                lines.extend(fact_lines)

        if self.preferred_topics:
            lines.append(f"Topics they enjoy: {', '.join(self.preferred_topics[-5:])}")

        return "\n".join(lines)
