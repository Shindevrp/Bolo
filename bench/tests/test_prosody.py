"""Unit tests for offline prosody evaluation (bench.scoring.prosody)."""
from __future__ import annotations

import pytest

from bench.agent.base import Event, EventType
from bench.driver.runner import ScenarioResult, Timeline
from bench.driver.scenario import Scenario
from bench.scoring.prosody import (
    acoustic_features,
    observed_profiles,
    prosody_check,
    prosody_metrics,
    text_cues,
)


def _prosody_result(text: str, label: str = "", emotion: str = "") -> ScenarioResult:
    sc = Scenario(id="p", name="P", category="prosody")
    tl = Timeline()
    tl.add(Event.make(EventType.LLM_TOKEN, text=text, mtime=1.0))
    if label:
        tl.add(Event.make(EventType.PROSODY, text=f"{label}|{emotion}", mtime=2.0))
    res = ScenarioResult(scenario=sc)
    res.timeline = tl
    return res


class TestTextCues:
    def test_question(self):
        assert text_cues("What time?")["has_question"]

    def test_emphasis_tokens(self):
        assert "IMPORTANT" in text_cues("This is IMPORTANT.")["emphasis_tokens"]

    def test_comma_pauses(self):
        assert text_cues("First, confirm; second, pick a time.")["comma_pauses"] == 2

    def test_list_detected(self):
        assert text_cues("1. Do this. 2. Do that.")["numbered_list"]


class TestProsodyCheck:
    def test_question_agrees(self):
        r = prosody_check("Is that right?", responding_to_question=True)
        assert r["score"] == pytest.approx(1.0)

    def test_observed_profile_match(self):
        r = prosody_check("Is that right?", observed={"label": "question"})
        assert r["profile_ok"] == 1

    def test_observed_profile_mismatch(self):
        r = prosody_check("Is that right?", observed={"label": "emphatic"})
        assert r["profile_ok"] == 0
        assert r["score"] < 1.0

    def test_flat_statement_full(self):
        r = prosody_check("Your table is reserved for seven.")
        assert r["score"] == pytest.approx(1.0)

    def test_placeholder_vs_question(self):
        # 'Let me think about that.' never claims a question, so expected
        # has_question is False and the text agrees.
        r = prosody_check("Let me think about that.", responding_to_question=False)
        assert r["score"] == pytest.approx(1.0)


class TestObservedProfiles:
    def test_collects_label_emotion(self):
        res = _prosody_result("That is fine.", label="supportive", emotion="positive")
        assert observed_profiles([res]) == [{"label": "supportive", "emotion": "positive"}]

    def test_no_prosody_events(self):
        res = _prosody_result("That is fine.")
        assert observed_profiles([res]) == []


class TestProsodyMetrics:
    def test_aggregate_with_llm_output(self):
        res = _prosody_result("Your reservation is confirmed at seven.")
        m = prosody_metrics([res])
        assert abs(m["prosody_compliance"].value - 1.0) < 1e-6
        assert m["prosody_compliance"].denominator == 1

    def test_observed_cross_check_counts(self):
        res = _prosody_result("Is that okay?", label="question", emotion="neutral")
        m = prosody_metrics([res])
        # 1 unobserved scoring + 1 observed cross-check
        assert m["prosody_compliance"].denominator == 2

    def test_no_llm_output_skipped(self):
        sc = Scenario(id="p", name="P", category="prosody")
        tl = Timeline()
        tl.add(Event.make(EventType.PROSODY, text="question|neutral", mtime=1.0))
        res = ScenarioResult(scenario=sc)
        res.timeline = tl
        m = prosody_metrics([res])
        assert m["prosody_compliance"].value is None


class TestAcousticFeatures:
    def test_empty(self):
        f = acoustic_features(b"")
        assert f["samples"] == 0
        assert f["pitch_mean"] is None

    def test_silence_no_pitch(self):
        f = acoustic_features(b"\x00" * 32000)
        assert f["voiced_frames"] == 0
        assert f["duration_s"] == pytest.approx(2.0)

    def test_tone_detected_voiced(self):
        import math
        import numpy as np

        sr = 16000
        t = np.arange(sr).astype(np.float64)
        tone = (12000 * np.sin(2 * math.pi * 220 * t / sr)).astype(np.int16)
        f = acoustic_features(tone.tobytes(), sample_rate=sr)
        assert f["voiced_frames"] > 0
        assert f["pitch_mean"] is not None
        assert 150 < f["pitch_mean"] < 300