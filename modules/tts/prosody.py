from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBERED_ITEM = re.compile(r"^\d+[.)]\s")
_EMPHASIS_TOKEN = re.compile(r"\b[A-Z]{2,}\b")
_ELLIPSIS = re.compile(r"\.\.\.|…")

_POSITIVE_WORDS = {
    "great", "awesome", "amazing", "good", "nice", "love", "happy", "cool",
    "wonderful", "fantastic", "excellent", "perfect", "thanks", "thank",
    "please", "yes", "yeah", "sure", "glad", "beautiful", "fun",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "sad", "angry", "frustrated",
    "frustrating", "annoyed", "confusing", "confused", "wrong", "broken",
    "stupid", "no", "nope", "stop", "disappointed", "annoying",
}


@dataclass(frozen=True)
class ProsodyProfile:
    """Per-utterance TTS parameters (Piper SynthesisConfig equivalents)."""

    length_scale: float = 1.0
    noise_scale: float = 0.4
    noise_w: float = 0.5
    sentence_silence: float = 0.02
    label: str = "neutral"


def classify_sentiment(text: str) -> str:
    """Very light lexicon sentiment: 'negative', 'positive', or 'neutral'."""
    words = [w for w in text.lower().split() if w.isalpha()]
    if not words:
        return "neutral"
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def pause_for(terminator: str, base: float) -> float:
    """Map boundary punctuation to a pause duration (seconds).

    '.' -> long pause, ',' -> short, '...' -> thinking, '?' -> rising pause.
    """
    if terminator in ("...", "…"):
        return max(0.25, base * 2.5)
    if terminator == "?":
        return base * 0.9 + 0.02
    if terminator == "!":
        return base * 0.8
    if terminator == ".":
        return base
    return base * 0.4


def split_emphasis(sentence: str) -> list[tuple[str, bool]]:
    """Split a sentence into (segment, emphasized) pairs based on ALL-CAPS words."""
    segments: list[tuple[str, bool]] = []
    last = 0
    for m in _EMPHASIS_TOKEN.finditer(sentence):
        if m.start() > last:
            segments.append((sentence[last:m.start()], False))
        segments.append((m.group(0), True))
        last = m.end()
    if last < len(sentence):
        segments.append((sentence[last:], False))
    return segments


class ProsodySelector:
    """Maps conversation state + content into a ProsodyProfile.

    - length_scale < 1.0 = faster speech, > 1.0 = slower speech
    - noise_scale = expressiveness / pitch variation
    - sentence_silence = baseline pause weight after sentences
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
        complexity: str = "standard",
        user_sentiment: str = "neutral",
        user_repeated: bool = False,
        first_response: bool = False,
    ) -> ProsodyProfile:
        trimmed = text.strip()
        base = self._base_profile(trajectory, engagement)
        length = base.length_scale
        noise = base.noise_scale
        noise_w = base.noise_w
        silence = base.sentence_silence
        label = base.label

        # B3: sentiment / emotion context.
        if user_sentiment == "negative":
            length *= 1.1
            noise *= 0.9
            label = f"{label}-supportive"
        elif user_sentiment == "positive":
            length *= 0.95
            noise = min(0.6, noise + 0.05)
            label = f"{label}-warm"

        # B4: user repeating themselves -> patient, measured.
        if user_repeated:
            length *= 1.08
            noise *= 0.95
            label = f"{label}-patient"

        # B4: conversation pacing.
        if first_response and first:
            length *= 1.12
            silence += 0.04
            label = f"{label}-opening"
        elif complexity == "complex":
            length *= 1.06
            silence += 0.03
            label = f"{label}-structured"
        elif turn_count > 4:
            length *= 0.94
            label = f"{label}-casual"

        # B2: utterance type / rhythm.
        if trimmed.endswith("?"):
            length *= 1.12
            silence += 0.05
            label = "question"
        elif trimmed.endswith("!") or (
            len(trimmed) < 12 and trimmed.upper() == trimmed
        ):
            length *= 0.9
            noise = min(0.65, noise + 0.1)
            label = "emphatic"
        elif _ELLIPSIS.search(trimmed):
            length *= 1.15
            noise *= 0.9
            silence += 0.15
            label = "thoughtful"
        elif _NUMBERED_ITEM.match(trimmed):
            length = 1.0
            noise *= 0.9
            label = "list"
        elif responding_to_question:
            length *= 1.05
            silence += 0.03
            label = "answer"

        if topic_shift and first:
            length *= 1.15
            silence += 0.03
            label = f"{label}-intro"

        profile = ProsodyProfile(
            length_scale=max(0.6, min(1.5, length)),
            noise_scale=max(0.2, min(0.7, noise)),
            noise_w=noise_w,
            sentence_silence=silence,
            label=label,
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
                sentence_silence=0.06,
                label="eager",
            )
        if trajectory == "falling" or engagement <= 0.35:
            return ProsodyProfile(
                length_scale=1.12,
                noise_scale=0.32,
                noise_w=0.5,
                sentence_silence=0.16,
                label="calm",
            )
        return ProsodyProfile(
            length_scale=0.95,
            noise_scale=0.4,
            noise_w=0.5,
            sentence_silence=0.1,
            label="conversational",
        )
