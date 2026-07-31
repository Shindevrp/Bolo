from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBERED_ITEM = re.compile(r"^\d+[.)]\s")
_ELLIPSIS = re.compile(r"\.\.\.|…")


@dataclass(frozen=True)
class ProsodyProfile:
    """Per-utterance TTS parameters (Piper SynthesisConfig equivalents)."""

    length_scale: float = 1.0
    noise_scale: float = 0.4
    noise_w: float = 0.5
    sentence_silence: float = 0.02
    label: str = "neutral"


class ProsodySelector:
    """Maps conversation state + content into a ProsodyProfile.

    - length_scale < 1.0 = faster speech, > 1.0 = slower speech
    - noise_scale = expressiveness / pitch variation
    - sentence_silence = pause weight after sentences
    """

    def __init__(self) -> None:
        self._last_label: str = "neutral"

    @property
    def last_label(self) -> str:
        return self._last_label

    def select(
        self,
        text: str,
        *,
        trajectory: str = "neutral",
        engagement: float = 0.5,
        turn_count: int = 0,
        topic_shift: bool = False,
        first: bool = False,
        responding_to_question: bool = False,
    ) -> ProsodyProfile:
        trimmed = text.strip()
        profile = self._base_profile(trajectory, engagement)

        if trimmed.endswith("?"):
            profile = ProsodyProfile(
                length_scale=profile.length_scale * 1.12,
                noise_scale=profile.noise_scale,
                noise_w=profile.noise_w,
                sentence_silence=profile.sentence_silence + 0.05,
                label="question",
            )
        elif trimmed.endswith("!") or (
            len(trimmed) < 12 and trimmed.upper() == trimmed
        ):
            profile = ProsodyProfile(
                length_scale=profile.length_scale * 0.9,
                noise_scale=min(0.65, profile.noise_scale + 0.1),
                noise_w=profile.noise_w,
                sentence_silence=profile.sentence_silence,
                label="emphatic",
            )
        elif _ELLIPSIS.search(trimmed):
            profile = ProsodyProfile(
                length_scale=profile.length_scale * 1.15,
                noise_scale=profile.noise_scale * 0.9,
                noise_w=profile.noise_w,
                sentence_silence=profile.sentence_silence + 0.1,
                label="thoughtful",
            )
        elif _NUMBERED_ITEM.match(trimmed):
            profile = ProsodyProfile(
                length_scale=1.0,
                noise_scale=profile.noise_scale * 0.9,
                noise_w=profile.noise_w,
                sentence_silence=profile.sentence_silence,
                label="list",
            )
        elif responding_to_question:
            profile = ProsodyProfile(
                length_scale=profile.length_scale * 1.05,
                noise_scale=profile.noise_scale,
                noise_w=profile.noise_w,
                sentence_silence=profile.sentence_silence + 0.03,
                label="answer",
            )

        if topic_shift and first:
            profile = ProsodyProfile(
                length_scale=profile.length_scale * 1.15,
                noise_scale=profile.noise_scale,
                noise_w=profile.noise_w,
                sentence_silence=profile.sentence_silence + 0.03,
                label=f"{profile.label}-intro",
            )

        self._last_label = profile.label
        return profile

    def _base_profile(
        self, trajectory: str, engagement: float
    ) -> ProsodyProfile:
        if trajectory == "rising" and engagement >= 0.6:
            return ProsodyProfile(
                length_scale=0.8,
                noise_scale=0.5,
                noise_w=0.6,
                sentence_silence=0.02,
                label="eager",
            )
        if trajectory == "falling" or engagement <= 0.35:
            return ProsodyProfile(
                length_scale=1.12,
                noise_scale=0.32,
                noise_w=0.5,
                sentence_silence=0.06,
                label="calm",
            )
        return ProsodyProfile(
            length_scale=0.95,
            noise_scale=0.4,
            noise_w=0.5,
            sentence_silence=0.03,
            label="conversational",
        )
