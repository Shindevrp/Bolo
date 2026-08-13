from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpeakerProfile:
    """Defines a single speaker in a multi-speaker conversation."""

    name: str
    voice_model_path: str
    voice_config_path: str | None = None
    personality: str = ""
    speaking_style: str = ""
    perspective: str = ""
    backchannel_words: list[str] = field(default_factory=lambda: ["uh-huh", "right", "exactly"])
    color: str = "#3B82F6"

    def system_prompt_block(self) -> str:
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Personality: {self.personality}.")
        if self.speaking_style:
            parts.append(f"Speaking style: {self.speaking_style}.")
        if self.perspective:
            parts.append(f"Perspective: {self.perspective}.")
        return " ".join(parts)
