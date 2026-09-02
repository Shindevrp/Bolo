import math

import numpy as np

from modules.prosody.analyzer import ProsodyAnalyzer
from modules.prosody.extractor import ProsodyExtractor
from modules.tts.prosody import ProsodySelector


def _sine(freq: float, amplitude: int, n: int, sr: int = 16000) -> bytes:
    t = np.arange(n) / sr
    samples = amplitude * np.sin(2.0 * np.pi * freq * t)
    return samples.astype(np.int16).tobytes()


class TestProsodyExtractor:
    def test_tone_estimates_pitch_at_target(self) -> None:
        sr = 16000
        ext = ProsodyExtractor(sample_rate=sr)
        n = sr * 30 // 1000  # 480 samples
        for freq in (100, 120, 200, 300):
            pitch = ext.extract(_sine(freq, 3000, n, sr))["pitch"]
            assert math.isclose(pitch, freq, abs_tol=5.0)

    def test_pitch_is_amplitude_invariant(self) -> None:
        sr = 16000
        ext = ProsodyExtractor(sample_rate=sr)
        n = sr * 30 // 1000
        soft = ext.extract(_sine(200, 400, n, sr))["pitch"]
        loud = ext.extract(_sine(200, 8000, n, sr))["pitch"]
        assert math.isclose(soft, 200.0, abs_tol=5.0)
        assert abs(soft - loud) < 5.0

    def test_quiet_frame_is_unvoiced(self) -> None:
        sr = 16000
        ext = ProsodyExtractor(sample_rate=sr)
        n = sr * 30 // 1000
        pitch = ext.extract(_sine(200, 10, n, sr))["pitch"]
        assert pitch == 0.0

    def test_noise_is_unvoiced_despite_energy(self) -> None:
        sr = 16000
        ext = ProsodyExtractor(sample_rate=sr)
        rng = np.random.default_rng(7)
        noise = (rng.standard_normal(480) * 2000).astype(np.int16).tobytes()
        feats = ext.extract(noise)
        assert feats["energy"] > 25.0
        assert feats["pitch"] == 0.0

    def test_empty_chunk_is_safe(self) -> None:
        feats = ProsodyExtractor().extract(b"")
        assert feats == {"pitch": 0.0, "energy": 0.0, "zero_crossing_rate": 0.0}


class TestProsodyAnalyzer:
    def _analyzer(self) -> ProsodyAnalyzer:
        # smoothing=1.0 removes EMA lag so trends track the raw window.
        return ProsodyAnalyzer(smoothing=1.0)

    def test_rising_trajectory(self) -> None:
        an = self._analyzer()
        for i, pitch in enumerate([100, 105, 110, 115, 120, 125]):
            res = an.update({"pitch": float(pitch), "energy": 50.0})
            if i >= 3:
                assert res["pitch_trend"] > 0
        assert an.analyze()["trajectory"] == "rising"

    def test_falling_trajectory(self) -> None:
        an = self._analyzer()
        for pitch in [125, 120, 115, 110, 105, 100]:
            an.update({"pitch": float(pitch), "energy": 50.0})
        assert an.analyze()["trajectory"] == "falling"

    def test_neutral_needs_minimum_voiced_frames(self) -> None:
        an = self._analyzer()
        an.update({"pitch": 100.0, "energy": 50.0})
        an.update({"pitch": 120.0, "energy": 50.0})
        assert an.analyze()["trajectory"] == "neutral"

    def test_silence_does_not_derail_trend(self) -> None:
        an = self._analyzer()
        frames = [{"pitch": 100.0}, {"pitch": 0.0}, {"pitch": 0.0},
                  {"pitch": 108.0}, {"pitch": 0.0}, {"pitch": 116.0},
                  {"pitch": 124.0}, {"pitch": 132.0}]
        for f in frames:
            an.update({"pitch": f.get("pitch", 0.0), "energy": 40.0})
        assert an.analyze()["trajectory"] == "rising"

    def test_reset_clears_state(self) -> None:
        an = self._analyzer()
        for pitch in [100, 145, 110]:
            an.update({"pitch": float(pitch), "energy": 50.0})
        an.reset()
        assert an.analyze()["trajectory"] == "neutral"


class TestProsodySelectorNoiseW:
    def _select(self, text: str, **kw):
        return ProsodySelector().select(text, **kw)

    def test_command_and_positive_raise_noise_w(self) -> None:
        neutral = self._select("Let me help you.", user_sentiment="neutral")
        command = self._select("Let me help you.", intent="command")
        warm = self._select("Let me help you.", user_sentiment="positive")
        assert command.noise_w > neutral.noise_w
        assert warm.noise_w > neutral.noise_w

    def test_negative_sentiment_lowers_noise_w(self) -> None:
        neutral = self._select("Let me help you.", user_sentiment="neutral")
        supportive = self._select(
            "Let me help you.", user_sentiment="negative"
        )
        assert supportive.noise_w < neutral.noise_w

    def test_question_and_emphatic_raise_noise_w(self) -> None:
        statement = self._select("That is fine.")
        question = self._select("What time is it?")
        emphatic = self._select("THAT IS FINE!")
        assert question.noise_w > statement.noise_w
        assert emphatic.noise_w > statement.noise_w

    def test_noise_w_is_clamped(self) -> None:
        sel = ProsodySelector()
        eager = sel.select(
            "Tell me more!", trajectory="rising", engagement=0.9
        )
        floor = sel.select(
            "I see...", trajectory="falling", engagement=0.2,
            user_sentiment="negative",
        )
        assert eager.noise_w <= 0.8
        assert floor.noise_w >= 0.2