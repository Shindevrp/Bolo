from __future__ import annotations

import numpy as np
from modules.speaker.stream_state import StreamState


class SpeakerPredictor:
    """Predicts which speaker will talk next before the tag appears.

    Uses token patterns and context to make early predictions,
    reducing silence gaps between speakers.
    """

    def __init__(self, known_speakers: list[str]) -> None:
        self.known_speakers = known_speakers
        self._history: list[tuple[str, str]] = []  # (speaker, text)
        self._prediction_confidence: float = 0.0

    def predict(
        self,
        current_speaker: str,
        recent_tokens: list[str],
        stream_state: StreamState,
    ) -> tuple[str | None, float]:
        """Predict the next speaker.

        Returns (predicted_speaker, confidence).
        None means no prediction.
        """
        if not self.known_speaker_pairs:
            return None, 0.0

        # Signal 1: Alternation pattern
        alt = self._predict_by_alternation(current_speaker)
        if alt:
            return alt, 0.5

        # Signal 2: Turn-taking patterns in history
        hist = self._predict_by_history(current_speaker)
        if hist:
            return hist, 0.6

        # Signal 3: Semantic trajectory suggests new speaker
        if stream_state.uncertainty_score > 0.5 and stream_state.claim_strength < 0.3:
            other = self._other_than(current_speaker)
            return other, 0.4

        return None, 0.0

    def record_turn(self, speaker: str, text: str) -> None:
        self._history.append((speaker, text))
        if len(self._history) > 20:
            self._history = self._history[-20:]

    def _predict_by_alternation(self, current: str) -> str | None:
        """Default: alternate speakers."""
        return self._other_than(current)

    def _predict_by_history(self, current: str) -> str | None:
        """Look at historical patterns."""
        if len(self._history) < 2:
            return None

        # Who usually responds to this speaker?
        responders: dict[str, int] = {}
        for i in range(1, len(self._history)):
            if self._history[i-1][0] == current:
                next_speaker = self._history[i][0]
                responders[next_speaker] = responders.get(next_speaker, 0) + 1

        if responders:
            return max(responders, key=responders.get)
        return None

    def _other_than(self, name: str) -> str:
        for s in self.known_speakers:
            if s != name:
                return s
        return name

    @property
    def known_speaker_pairs(self) -> bool:
        return len(self.known_speakers) >= 2
