"""Unit tests for the endpointing/barge-in diagnostics (bench.scoring.endpoint).

These build deterministic in-memory timelines with hand-authored EventType
timestamps and assert the computed metrics, so the arithmetic is exercised
offline without a live agent. Scenarios mirror the real benchmark design where
barge-in and backchannel are separate scenarios, each with a single injection.
"""
from __future__ import annotations

from bench.agent.base import Event, EventType
from bench.driver.runner import ScenarioResult, Timeline
from bench.driver.scenario import Scenario, Step
from bench.scoring.endpoint import (
    collect_endpoint_samples,
    compute_endpoint_metrics,
    reconstruct_user_turns,
)


def _ev(et: EventType, t: float, text=None) -> Event:
    return Event.make(et, text=text, mtime=t)


def _barge_in_result():
    """Real barge-in during playback: TTS is actively playing when the user
    cuts in, the agent detects it, and then returns to a new listening turn."""
    sc = Scenario(id="barge_in", name="Barge-In", category="barge_in",
                  steps=[Step(kind="speak", text="Q"),
                         Step(kind="interrupt_during_playback", text="Wait, no"),
                         Step(kind="drain", timeout_ms=5000)])
    tl = Timeline()
    tl.add(_ev(EventType.SPEECH_START, 1.0))     # original question
    tl.add(_ev(EventType.SPEECH_END, 2.5))
    tl.add(_ev(EventType.TRANSCRIPT, 2.6, "some places?"))
    tl.add(_ev(EventType.LLM_TOKEN, 3.0, "sure "))
    tl.add(_ev(EventType.TTS_CHUNK, 3.4))        # playback starts
    tl.add(_ev(EventType.TTS_CHUNK, 3.6))
    tl.add(_ev(EventType.SPEECH_START, 3.7))     # interruption injection
    tl.add(_ev(EventType.TTS_CHUNK, 3.75))       # still playing
    tl.add(_ev(EventType.TTS_CHUNK, 3.85))
    tl.add(_ev(EventType.INTERRUPT, 3.9))        # agent detects it
    tl.add(_ev(EventType.SPEECH_START, 4.5))     # recovery -> new turn
    tl.add(_ev(EventType.SPEECH_END, 5.4))
    tl.add(_ev(EventType.TRANSCRIPT, 5.5, "okay quieter"))
    res = ScenarioResult(scenario=sc)
    res.timeline = tl
    return res


def _backchannel_result():
    """Harmless backchannel during playback: must NOT surface an INTERRUPT."""
    sc = Scenario(id="backchannel", name="Backchannel", category="barge_in",
                  steps=[Step(kind="speak", text="Q"),
                         Step(kind="interrupt_during_playback", text="yeah uh-huh right"),
                         Step(kind="drain", timeout_ms=5000)])
    tl = Timeline()
    tl.add(_ev(EventType.SPEECH_START, 10.0))
    tl.add(_ev(EventType.SPEECH_END, 11.5))
    tl.add(_ev(EventType.TRANSCRIPT, 11.6, "trails"))
    tl.add(_ev(EventType.LLM_TOKEN, 12.0, "sure "))
    tl.add(_ev(EventType.TTS_CHUNK, 12.4))
    tl.add(_ev(EventType.SPEECH_START, 13.0))    # backchannel injection
    tl.add(_ev(EventType.SPEECH_END, 13.6))
    tl.add(_ev(EventType.TRANSCRIPT, 13.7, "yeah ok"))
    # no INTERRUPT event
    res = ScenarioResult(scenario=sc)
    res.timeline = tl
    return res


def test_barge_in_and_backchannel_metrics():
    samples = collect_endpoint_samples([_barge_in_result(), _backchannel_result()])
    metrics = compute_endpoint_metrics(samples)
    # Barge-in detected 1/1 and recovered 1/1
    assert metrics["barge_in_detection_rate"].numerator == 1
    assert metrics["barge_in_detection_rate"].denominator == 1
    assert metrics["recovery_success_rate"].value == 1.0
    # Backchannel not treated as interrupt
    assert metrics["false_barge_in_rate"].value == 0.0
    # Barge-in stop latency: last TTS during window 3.85, onset 3.7
    assert metrics["barge_in_stop_p50"].value == pytest_close(0.15)


def pytest_close(v):
    from pytest import approx
    return approx(v, abs=1e-6)


def test_false_and_missed_endpoint():
    sc = Scenario(id="tt", name="T", category="turn_taking")
    tl = Timeline()
    # turn A: agent responds BEFORE user finishes -> false endpoint
    tl.add(_ev(EventType.SPEECH_START, 1.0))
    tl.add(_ev(EventType.LLM_TOKEN, 1.6, "cut"))  # before speech_end
    tl.add(_ev(EventType.SPEECH_END, 2.0))
    # turn B: agent responds very late -> missed endpoint
    tl.add(_ev(EventType.SPEECH_START, 3.0))
    tl.add(_ev(EventType.SPEECH_END, 4.0))
    tl.add(_ev(EventType.TTS_CHUNK, 8.0))  # 4s later -> missed
    res = ScenarioResult(scenario=sc)
    res.timeline = tl
    samples = collect_endpoint_samples([res])
    metrics = compute_endpoint_metrics(samples)
    # 1 false out of 2 (turn B is a normal completed turn)
    assert metrics["false_endpoint_rate"].numerator == 1
    assert metrics["false_endpoint_rate"].denominator == 2
    # 1 missed (turn B's 4s delay) of 2 (turn A pre-empted, not missed)
    assert metrics["missed_endpoint_rate"].numerator == 1
    assert metrics["missed_endpoint_rate"].denominator == 2


def test_reconstruct_preserves_timing():
    sc = Scenario(id="x", name="X", category="baseline")
    tl = Timeline()
    tl.add(_ev(EventType.SPEECH_START, 1.0))
    tl.add(_ev(EventType.SPEECH_END, 2.0))
    tl.add(_ev(EventType.TTS_CHUNK, 3.0))
    res = ScenarioResult(scenario=sc)
    res.timeline = tl
    turns = reconstruct_user_turns(tl)
    assert len(turns) == 1
    assert turns[0].speech_end == pytest_close(2.0)
    assert turns[0].response_after_end == pytest_close(3.0)
