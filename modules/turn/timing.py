from __future__ import annotations


class TurnTiming:
    """Track response timing and pacing heuristics."""

    def compute_delay(self, pause_duration: float) -> float:
        return max(0.1, pause_duration * 0.8)
