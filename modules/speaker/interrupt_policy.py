from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class InterruptStyle(Enum):
    NONE = auto()       # No interrupt
    SENTENCE = auto()   # Finish current sentence, then switch
    HARD = auto()       # Immediate crossfade


@dataclass
class InterruptDecision:
    style: InterruptStyle
    confidence: float  # 0.0-1.0
    reason: str


# Words/phrases that signal disagreement or strong opinion
_DISAGREEMENT = {
    "actually", "but", "however", "i disagree", "not sure about",
    "well,", "hmm,", "wait,", "hold on", "on the contrary",
    "i think differently", "that's not quite right", "let me push back",
    "i see your point but", "respectfully",
}

# Words that signal excitement or urgency
_EXCITEMENT = {
    "wow", "amazing", "incredible", "wait!", "oh!", "yes!",
    "exactly!", "that's it!", "brilliant", "genius", "perfect",
    "i just realized", "oh wait", "hang on",
}

# Words that signal a topic shift
_TOPIC_SHIFT = {
    "anyway", "by the way", "speaking of", "changing topics",
    "that reminds me", "unrelated", "different topic",
}


class InterruptPolicy:
    """Decides when Speaker B should interrupt Speaker A.

    Uses urgency scoring based on conversation dynamics:
    - Disagreement detected → moderate interrupt
    - Excitement detected → moderate interrupt
    - Topic shift → low interrupt
    - Turn balance (one speaker dominating) → low interrupt
    """

    def __init__(
        self,
        urgency_threshold: float = 0.7,
        max_consecutive_turns: int = 2,
        interrupt_mode: str = "sentence",
    ) -> None:
        self.urgency_threshold = urgency_threshold
        self.max_consecutive_turns = max_consecutive_turns
        self.interrupt_mode = InterruptStyle.HARD if interrupt_mode == "hard" else InterruptStyle.SENTENCE

    def evaluate(
        self,
        incoming_text: str,
        current_speaker: str,
        incoming_speaker: str,
        current_speaker_turns: int,
        current_text: str = "",
    ) -> InterruptDecision:
        """Evaluate whether the incoming speaker should interrupt."""
        if current_speaker == incoming_speaker:
            return InterruptDecision(style=InterruptStyle.NONE, confidence=0.0, reason="same_speaker")

        score = 0.0
        reasons: list[str] = []

        # Factor 1: Disagreement
        if self._has_disagreement(incoming_text):
            score += 0.4
            reasons.append("disagreement")

        # Factor 2: Excitement
        if self._has_excitement(incoming_text):
            score += 0.3
            reasons.append("excitement")

        # Factor 3: Topic shift
        if self._has_topic_shift(incoming_text):
            score += 0.2
            reasons.append("topic_shift")

        # Factor 4: Turn dominance
        if current_speaker_turns >= self.max_consecutive_turns:
            score += 0.3
            reasons.append("turn_dominance")

        # Factor 5: Current speaker is at a natural pause (sentence end)
        if current_text.rstrip().endswith((".", "!", "?")):
            score += 0.2
            reasons.append("natural_pause")

        # Decide
        if score >= self.urgency_threshold:
            return InterruptDecision(
                style=self.interrupt_mode,
                confidence=min(1.0, score),
                reason="+".join(reasons),
            )
        elif score >= self.urgency_threshold * 0.7:
            # Moderate urgency — wait for sentence end
            return InterruptDecision(
                style=InterruptStyle.SENTENCE,
                confidence=min(1.0, score),
                reason="+".join(reasons),
            )
        else:
            return InterruptDecision(
                style=InterruptStyle.NONE,
                confidence=score,
                reason="below_threshold",
            )

    def _has_disagreement(self, text: str) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in _DISAGREEMENT)

    def _has_excitement(self, text: str) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in _EXCITEMENT)

    def _has_topic_shift(self, text: str) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in _TOPIC_SHIFT)
