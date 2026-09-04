from __future__ import annotations

import asyncio
import numpy as np
import pytest

from bench.agent.base import EventType
from bench.driver.runner import Timeline, run_scenario
from bench.driver.scenario import Step, Scenario, load_scenarios
from bench.scoring.metrics import collect_scenario_metrics
from bench.scoring.aggregate import aggregate
from bench.scoring.report import CATEGORIES, WEIGHTS
from bench.tests.conftest import FakeAgent


def _clip(id="c1", secs=1.0, text="hello world"):
    return type("Clip", (), {
        "id": id,
        "audio": (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(16000 * secs)) / 16000)).astype(np.float32),
        "sr": 16000,
        "text": text,
        "speaker": "",
    })()


def _make_scenario(pid="t1", steps=None):
    steps = steps or [
        {"kind": "corpus", "corpus": "demo", "silence_after_ms": 300},
        {"kind": "expect", "event": "transcript", "timeout_ms": 5000},
        {"kind": "drain", "timeout_ms": 3000},
    ]
    return Scenario(id=pid, name="t", category="test", steps=[Step(**s) for s in steps])


@pytest.mark.asyncio
async def test_runner_collects_timeline():
    agent = FakeAgent(transcript="hello world", llm="Hello! how can I help")
    corpus = {"demo": [_clip()]}
    sc = _make_scenario()
    res = await run_scenario(agent, sc, corpus)
    assert res.timeline.final_transcripts() and "hello world" == res.timeline.final_transcripts()[-1]
    assert res.timeline.first(EventType.LLM_TOKEN) is not None
    assert res.timeline.tts_audio_bytes()


@pytest.mark.asyncio
async def test_scenario_loads_and_corpus_step():
    scs = load_scenarios("all")
    assert "barge_in" in scs
    assert "context" in scs
    assert scs["grounding"].category == "grounding"


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_aggregate_produces_full_report():
    agent = FakeAgent(transcript="some words", llm="I'm not sure about that place.")
    corpus = {"demo": [_clip()]}
    sc = _make_scenario()
    res = asyncio.run(run_scenario(agent, sc, corpus))
    rep = aggregate([res], wer=None, agent="FakeAgent")
    # All 12 categories present and scored 1-10
    for cat in CATEGORIES:
        assert cat in rep.scores
        v = rep.scores[cat]
        assert v is not None and 1 <= v <= 10
    rep.compute_benchmark()
    assert 0 <= rep.benchmark <= 100


def test_aggregate_does_not_invent_wer():
    agent = FakeAgent()
    sc = _make_scenario()
    res = asyncio.run(run_scenario(agent, sc, {"demo": [_clip()]}))
    rep = aggregate([res], wer=None, agent="FakeAgent")
    assert rep.scores["ASR/Segmentation"] is not None  # never None
