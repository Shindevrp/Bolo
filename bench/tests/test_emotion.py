"""Unit tests for offline emotion evaluation (bench.scoring.emotion)."""
from __future__ import annotations

import pytest

from bench.agent.base import Event, EventType
from bench.driver.runner import ScenarioResult, Timeline
from bench.driver.scenario import Scenario
from bench.scoring.emotion import (
    confusion,
    emotion_cases,
    emotion_metrics,
    evaluate_emotion,
    classify_default,
)


class TestEvaluateEmotion:
    def test_curated_cases_all_pass(self):
        ev = evaluate_emotion()
        assert ev["accuracy"] == pytest.approx(1.0)
        assert ev["correct"] == ev["n"]

    def test_buckets_covered(self):
        ev = evaluate_emotion()
        for label in ("positive", "negative", "neutral"):
            assert ev["per_class"][label]["tp"] >= 3

    def test_macro_f1(self):
        ev = evaluate_emotion()
        assert ev["macro_f1"] == pytest.approx(1.0)

    def test_custom_classifier(self):
        ev = evaluate_emotion(classify=lambda text: "neutral")
        assert ev["accuracy"] == pytest.approx(5 / 15, abs=1e-3)
        assert ev["per_class"]["positive"]["recall"] == pytest.approx(0.0)

    def test_confusion_counts(self):
        assert confusion(["a"], ["a"])["a"]["a"] == 1


class TestEmotionMetrics:
    def test_no_prosody_falls_back_to_lexicon(self):
        m = emotion_metrics([])
        assert m["emotion_accuracy"].value == pytest.approx(1.0)
        assert m["emotion_accuracy"].denominator == 15

    def test_onwire_agreement_counts(self):
        sc = Scenario(id="e", name="E", category="emotion")
        tl = Timeline()
        tl.add(Event.make(EventType.TRANSCRIPT, text="that is great news", mtime=1.0))
        tl.add(Event.make(EventType.PROSODY, text="positive|positive", mtime=2.0))
        res = ScenarioResult(scenario=sc)
        res.timeline = tl
        m = emotion_metrics([res])
        assert m["emotion_accuracy"].numerator == 1
        assert m["emotion_accuracy"].denominator == 1
        assert m["emotion_accuracy"].value == pytest.approx(1.0)

    def test_onwire_disagreement(self):
        sc = Scenario(id="e", name="E", category="emotion")
        tl = Timeline()
        tl.add(Event.make(EventType.TRANSCRIPT, text="that is bad news", mtime=1.0))
        tl.add(Event.make(EventType.PROSODY, text="neutral|neutral", mtime=2.0))
        res = ScenarioResult(scenario=sc)
        res.timeline = tl
        m = emotion_metrics([res])
        assert m["emotion_accuracy"].value == pytest.approx(0.0)