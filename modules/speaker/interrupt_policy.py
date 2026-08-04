from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from modules.speaker.stream_state import StreamState
from modules.speaker.metrics import SpeakerMetrics


class InterruptStyle(Enum):
    NONE = auto()       # No interrupt
    SENTENCE = auto()   # Finish current sentence, then switch
    HARD = auto()       # Immediate crossfade


class InterruptType(Enum):
    """Personality-driven interrupt types."""
    CORRECTION = "correction"       # "Wait, that's not right."
    ENHANCEMENT = "enhancement"     # "Or even better—"
    SKEPTICISM = "skepticism"       # "Are we sure about that?"
    COMPLETION = "completion"       # Finishes the other speaker's sentence
    EXCITEMENT = "excitement"       # "Oh! That's amazing!"
    TOPIC_SHIFT = "topic_shift"     # "By the way—"


@dataclass
class InterruptDecision:
    style: InterruptStyle
    interrupt_type: InterruptType
    confidence: float  # 0.0-1.0
    reason: str
    overlap_ms: int = 200  # Adaptive overlap duration


# Words/phrases mapped to interrupt types
_CORRECTION_MARKERS = {
    "actually", "not quite", "that's not right", "well,",
    "hmm,", "i disagree", "that's incorrect", "let me correct",
    "on the contrary", "not exactly", "that's backwards",
}

_ENHANCEMENT_MARKERS = {
    "or even", "better yet", "additionally", "moreover",
    "i'd add", "that reminds me", "and also", "going further",
}

_SKEPTICISM_MARKERS = {
    "are we sure", "is that true", "really?", "do you think",
    "not sure about", "that seems", "i'm not convinced",
    "hold on", "wait,", "but what about",
}

_EXCITEMENT_MARKERS = {
    "wow", "amazing", "incredible", "yes!", "exactly!",
    "that's it!", "brilliant", "genius", "perfect",
    "oh!", "wait!", "i just realized",
}

_TOPIC_SHIFT_MARKERS = {
    "anyway", "by the way", "speaking of", "unrelated",
    "different topic", "that reminds me", "switching gears",
}


class InterruptPolicy:
    """Multi-factor interrupt decision system.

    Scoring:
        interrupt_score = (
            urgency * 0.35 +
            novelty * 0.20 +
            dominance_pressure * 0.20 +
            timing_window * 0.25
        )

    Interrupt only if:
        interrupt_score > dynamic_threshold

    Dynamic threshold adapts based on conversation flow.
    """

    def __init__(
        self,
        urgency_threshold: float = 0.6,
        max_consecutive_turns: int = 2,
        interrupt_mode: str = "sentence",
    ) -> None:
        self.urgency_threshold = urgency_threshold
        self.max_consecutive_turns = max_consecutive_turns
        self.interrupt_mode = (
            InterruptStyle.HARD if interrupt_mode == "hard"
            else InterruptStyle.SENTENCE
        )
        self._metrics: SpeakerMetrics | None = None
        self._dynamic_threshold = urgency_threshold

    def set_metrics(self, metrics: SpeakerMetrics) -> None:
        self._metrics = metrics

    def evaluate(
        self,
        incoming_text: str,
        current_speaker: str,
        incoming_speaker: str,
        current_speaker_turns: int,
        current_text: str = "",
        stream_state: StreamState | None = None,
        incoming_stream_state: StreamState | None = None,
    ) -> InterruptDecision:
        """Multi-factor interrupt evaluation."""
        if current_speaker == incoming_speaker:
            return InterruptDecision(
                style=InterruptStyle.NONE,
                interrupt_type=InterruptType.CORRECTION,
                confidence=0.0,
                reason="same_speaker",
            )

        # Compute individual factors
        urgency = self._compute_urgency(incoming_text, stream_state)
        novelty = self._compute_novelty(incoming_text, incoming_stream_state, current_text)
        dominance = self._compute_dominance(current_speaker_turns)
        timing = self._compute_timing(stream_state, current_text)

        # Weighted score
        interrupt_score = (
            urgency * 0.35 +
            novelty * 0.20 +
            dominance * 0.20 +
            timing * 0.25
        )

        # Detect interrupt type
        interrupt_type = self._detect_type(incoming_text)

        # Adaptive threshold (lower when conversation is flowing well)
        threshold = self._dynamic_threshold

        # Decide
        if interrupt_score >= threshold:
            style = self.interrupt_mode
            # If at a clause boundary, prefer sentence mode for naturalness
            if timing > 0.8:
                style = InterruptStyle.SENTENCE

            return InterruptDecision(
                style=style,
                interrupt_type=interrupt_type,
                confidence=min(1.0, interrupt_score),
                reason=f"score={interrupt_score:.2f} urgency={urgency:.2f} "
                       f"novelty={novelty:.2f} dominance={dominance:.2f} "
                       f"timing={timing:.2f}",
            )
        elif interrupt_score >= threshold * 0.7:
            # Moderate: wait for sentence end
            return InterruptDecision(
                style=InterruptStyle.SENTENCE,
                interrupt_type=interrupt_type,
                confidence=min(1.0, interrupt_score),
                reason=f"moderate score={interrupt_score:.2f}",
            )
        else:
            return InterruptDecision(
                style=InterruptStyle.NONE,
                interrupt_type=interrupt_type,
                confidence=interrupt_score,
                reason=f"below threshold={threshold:.2f}",
            )

    def _compute_urgency(self, text: str, stream_state: StreamState | None) -> float:
        """How urgent is the incoming message?"""
        score = 0.0

        # Linguistic urgency
        lower = text.lower()
        for marker in _CORRECTION_MARKERS:
            if marker in lower:
                score += 0.5
                break
        for marker in _EXCITEMENT_MARKERS:
            if marker in lower:
                score += 0.4
                break

        # Stream state urgency (if available)
        if stream_state:
            if stream_state.uncertainty_score > 0.5:
                score += 0.3
            if stream_state.claim_strength > 0.6:
                score += 0.2

        # Shorter text = more likely urgent interjection
        word_count = len(text.split())
        if word_count <= 3:
            score += 0.2
        elif word_count > 15:
            score -= 0.1

        return min(1.0, max(0.0, score))

    def _compute_novelty(
        self,
        incoming_text: str,
        incoming_state: StreamState | None,
        current_text: str,
    ) -> float:
        """How novel/different is the incoming message from what's being said?"""
        if not current_text:
            return 0.5

        # Simple word overlap (inverse novelty)
        current_words = set(current_text.lower().split())
        incoming_words = set(incoming_text.lower().split())
        if not incoming_words:
            return 0.0

        overlap = len(current_words & incoming_words) / len(incoming_words)
        lexical_novelty = 1.0 - overlap

        # Semantic novelty via topic vectors
        semantic_novelty = 0.5
        if incoming_state and incoming_state.topic_vector is not None:
            # Compare with a simple "current topic" estimate
            # For now, just use lexical as proxy
            semantic_novelty = lexical_novelty

        return 0.6 * lexical_novelty + 0.4 * semantic_novelty

    def _compute_dominance(self, current_speaker_turns: int) -> float:
        """How much has the current speaker dominated?"""
        return min(1.0, current_speaker_turns / self.max_consecutive_turns)

    def _compute_timing(
        self,
        stream_state: StreamState | None,
        current_text: str,
    ) -> float:
        """How good is the current timing for an interrupt?"""
        if stream_state and stream_state.is_at_clause_boundary():
            return 1.0

        # Check if current text ends at a natural pause point
        if current_text.rstrip().endswith((",", ";", "—")):
            return 0.9
        if current_text.rstrip().endswith((".", "!", "?")):
            return 0.8

        # Check if stream state shows a long clause
        if stream_state and stream_state.tokens_since_boundary() > 8:
            return 0.6

        return 0.3

    def _detect_type(self, text: str) -> InterruptType:
        """Detect the personality type of the interrupt."""
        lower = text.lower()

        for marker in _CORRECTION_MARKERS:
            if marker in lower:
                return InterruptType.CORRECTION

        for marker in _EXCITEMENT_MARKERS:
            if marker in lower:
                return InterruptType.EXCITEMENT

        for marker in _SKEPTICISM_MARKERS:
            if marker in lower:
                return InterruptType.SKEPTICISM

        for marker in _ENHANCEMENT_MARKERS:
            if marker in lower:
                return InterruptType.ENHANCEMENT

        for marker in _TOPIC_SHIFT_MARKERS:
            if marker in lower:
                return InterruptType.TOPIC_SHIFT

        # Default: likely an enhancement or completion
        if text.rstrip().endswith(("—", "...")):
            return InterruptType.COMPLETION

        return InterruptType.ENHANCEMENT
