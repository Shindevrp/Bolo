from __future__ import annotations

from collections import deque


class ProsodyAnalyzer:
    """Tracks pitch/energy trends over a sliding window of voiced frames.

    Only voiced frames (pitch > 0) feed the pitch trend, and a window that is
    not voiced enough reports a neutral trajectory instead of guessing. The
    trend is a least-squares slope across the buffered pitches — far more
    robust to outliers than a first-vs-last delta — smoothed with a small EMA
    so single-frame jitter cannot flip the trajectory.
    """

    def __init__(
        self,
        window_size: int = 30,
        smoothing: float = 0.3,
        min_voiced: int = 3,
    ) -> None:
        self._window_size = window_size
        self._smoothing = smoothing
        self._min_voiced = min_voiced
        self._pitch_buffer: deque[float] = deque(maxlen=window_size)
        self._energy_buffer: deque[float] = deque(maxlen=window_size)
        self._zcr_buffer: deque[float] = deque(maxlen=window_size)
        self._voiced = 0
        self._ema_pitch_trend = 0.0

    def update(self, features: dict[str, float]) -> dict[str, float | str]:
        self._energy_buffer.append(float(features.get("energy", 0.0)))
        self._zcr_buffer.append(
            float(features.get("zero_crossing_rate", 0.0))
        )
        pitch = float(features.get("pitch", 0.0))
        if pitch > 0.0:
            self._pitch_buffer.append(pitch)
            self._voiced += 1
        return self.analyze()

    def analyze(self) -> dict[str, float | str]:
        pitch_vals = list(self._pitch_buffer)
        if self._voiced < self._min_voiced or len(pitch_vals) < 3:
            self._ema_pitch_trend = 0.0
            return {"trajectory": "neutral", "pitch_trend": 0, "energy_trend": 0}

        n = len(pitch_vals)
        mean_x = (n - 1) / 2.0
        mean_y = sum(pitch_vals) / n
        denom = sum((i - mean_x) ** 2 for i in range(n))
        if denom == 0.0:
            slope = 0.0
        else:
            slope = sum(
                (i - mean_x) * (p - mean_y) for i, p in enumerate(pitch_vals)
            ) / denom
        raw_trend = slope * (n - 1)
        self._ema_pitch_trend = (
            self._smoothing * raw_trend
            + (1.0 - self._smoothing) * self._ema_pitch_trend
        )
        pitch_trend = self._ema_pitch_trend

        if pitch_trend > 20:
            trajectory = "rising"
        elif pitch_trend < -20:
            trajectory = "falling"
        else:
            trajectory = "flat"

        energy_vals = list(self._energy_buffer)
        energy_trend = (
            energy_vals[-1] - energy_vals[0] if len(energy_vals) >= 2 else 0.0
        )

        return {
            "trajectory": trajectory,
            "pitch_trend": round(pitch_trend, 2),
            "energy_trend": round(energy_trend, 2),
        }

    def reset(self) -> None:
        self._pitch_buffer.clear()
        self._energy_buffer.clear()
        self._zcr_buffer.clear()
        self._voiced = 0
        self._ema_pitch_trend = 0.0