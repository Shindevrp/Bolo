from __future__ import annotations

import numpy as np

# Below this normalized RMS (int16 scale) a frame is treated as silence and
# never yields a pitch estimate. Quiet/noisy frames otherwise inject spurious
# autocorrelation peaks that dominate the pitch trend.
ENERGY_FLOOR = 25.0

# Minimum normalized autocorrelation peak (0..1, normalized by lag-0 energy)
# to declare a frame voiced. Voiceless segments and clipped/unvoiced noise
# rarely exceed this; voiced speech periods do. Output pitch becomes
# amplitude-independent because the peak is normalized.
VOICING_PEAK_THRESHOLD = 0.30


class ProsodyExtractor:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30) -> None:
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * frame_ms // 1000

    def extract(self, audio_chunk: bytes) -> dict[str, float]:
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return {"pitch": 0.0, "energy": 0.0, "zero_crossing_rate": 0.0}

        energy = float(np.sqrt(np.mean(samples**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2)
        pitch = self._estimate_pitch(samples, energy)

        return {"pitch": pitch, "energy": energy, "zero_crossing_rate": zcr}

    def _estimate_pitch(self, samples: np.ndarray, energy: float) -> float:
        if energy < ENERGY_FLOOR:
            return 0.0

        centered = samples - float(np.mean(samples))
        n = len(centered)
        min_lag = max(1, int(self.sample_rate / 500))
        max_lag = int(self.sample_rate / 50)
        if max_lag >= n:
            max_lag = n - 1
        if min_lag >= n or max_lag <= min_lag:
            return 0.0

        r0 = float(np.dot(centered, centered))
        if r0 <= 0.0:
            return 0.0

        def _norm(lag: int) -> float:
            if lag < 1 or lag >= n:
                return 0.0
            return float(np.dot(centered[: n - lag], centered[lag:])) / r0

        best_lag = -1
        best_norm = 0.0
        for lag in range(min_lag, max_lag + 1):
            norm = _norm(lag)
            if norm > best_norm:
                best_norm = norm
                best_lag = lag

        if best_lag < 0 or best_norm < VOICING_PEAK_THRESHOLD:
            return 0.0

        # Parabolic interpolation over the lag-1 neighbors for sub-sample
        # period precision (keeps 100Hz vs 110Hz distinguishable).
        r_left = _norm(best_lag - 1)
        r_right = _norm(best_lag + 1)
        denom = r_left - 2.0 * best_norm + r_right
        if denom != 0.0:
            delta = 0.5 * (r_left - r_right) / denom
            if -1.0 < delta < 1.0:
                best_lag = best_lag + delta

        if best_lag <= 0:
            return 0.0
        return float(self.sample_rate / best_lag)