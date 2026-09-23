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
    analyze_results,
    collect_endpoint_samples,
    compute_eot_f1,
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


def _eot_result(entries):
    sc = Scenario(id="tt", name="T", category="turn_taking")
    tl = Timeline()
    for et, t, text in entries:
        tl.add(_ev(et, t, text))
    res = ScenarioResult(scenario=sc)
    res.timeline = tl
    return res


def test_eot_f1_all_correct():
    res = _eot_result([
        (EventType.SPEECH_START, 1.0, None),
        (EventType.SPEECH_END, 2.0, None),
        (EventType.TTS_CHUNK, 2.5, None),   # within [2.0, 5.0]
    ])
    m = compute_eot_f1(collect_endpoint_samples([res]))
    assert m["eot_true_positives"].value == 1.0
    assert m["eot_false_positives"].value == 0.0
    assert m["eot_false_negatives"].value == 0.0
    assert m["eot_precision"].value == pytest_close(1.0)
    assert m["eot_recall"].value == pytest_close(1.0)
    assert m["eot_f1"].value == pytest_close(1.0)


def test_eot_f1_cut_off_and_missed():
    res = _eot_result([
        (EventType.SPEECH_START, 1.0, None),
        (EventType.LLM_TOKEN, 1.6, "cut"),  # before speech_end -> FP
        (EventType.SPEECH_END, 2.0, None),
        (EventType.SPEECH_START, 3.0, None),
        (EventType.SPEECH_END, 4.0, None),
        (EventType.TTS_CHUNK, 8.0, None),   # 4s late -> FN
    ])
    m = compute_eot_f1(collect_endpoint_samples([res]))
    assert m["eot_true_positives"].value == 0.0
    assert m["eot_false_positives"].value == 1.0
    assert m["eot_false_negatives"].value == 1.0
    assert m["eot_precision"].value == pytest_close(0.0)
    assert m["eot_recall"].value == pytest_close(0.0)
    # precision + recall == 0 -> F1 is mathematically undefined
    assert m["eot_f1"].value is None


def test_eot_f1_halves_on_one_wrong():
    res = _eot_result([
        (EventType.SPEECH_START, 1.0, None),
        (EventType.SPEECH_END, 2.0, None),
        (EventType.TTS_CHUNK, 2.5, None),   # TP
        (EventType.SPEECH_START, 3.0, None),
        (EventType.SPEECH_END, 4.0, None),  # no response -> FN
    ])
    m = compute_eot_f1(collect_endpoint_samples([res]))
    assert m["eot_true_positives"].value == 1.0
    assert m["eot_false_negatives"].value == 1.0
    assert m["eot_precision"].value == pytest_close(1.0)   # no FPs
    assert m["eot_recall"].value == pytest_close(0.5)
    assert m["eot_f1"].value == pytest_close(2 / 3)


def test_eot_f1_no_turns_undefined():
    res = _eot_result([])
    m = compute_eot_f1(collect_endpoint_samples([res]))
    assert m["eot_precision"].value is None
    assert m["eot_recall"].value is None
    assert m["eot_f1"].value is None
    assert m["eot_true_positives"].denominator == 0


def test_analyze_results_includes_eot():
    res = _eot_result([
        (EventType.SPEECH_START, 1.0, None),
        (EventType.SPEECH_END, 2.0, None),
        (EventType.TTS_CHUNK, 2.5, None),
    ])
    m = analyze_results([res])
    assert m["eot_f1"].value == pytest_close(1.0)
    assert m["completion_capture_rate"] is not None


def test_multi_turn_transcripts_attach_to_own_turn():
    """Transcripts after turn 1's end must NOT attach to turn 1 just because
    its window was previously unbounded."""
    res = _eot_result([
        (EventType.SPEECH_START, 1.0, None),
        (EventType.SPEECH_END, 2.0, None),
        (EventType.TRANSCRIPT, 2.6, "one"),
        (EventType.SPEECH_START, 4.0, None),
        (EventType.SPEECH_END, 5.0, None),
        (EventType.TRANSCRIPT, 5.5, "two"),
    ])
    samples = collect_endpoint_samples([res])
    assert [t.transcript for t in samples.turns] == ["one", "two"]
    assert samples.turns[0].transcript_time == pytest_close(2.6)
    assert samples.turns[1].transcript_time == pytest_close(5.5)


def test_recovery_does_not_bleed_across_scenarios():
    """Recovery is scored against the SAME scenario's later turns only: a
    detected barge-in must not be 'recovered' by an unrelated scenario."""
    sc_a = Scenario(id="sca", name="A", category="barge_in",
                    steps=[Step(kind="speak", text="Q"),
                           Step(kind="interrupt_during_playback", text="Wait"),
                           Step(kind="drain", timeout_ms=1000)])
    tl_a = Timeline()
    tl_a.add(_ev(EventType.SPEECH_START, 1.0))
    tl_a.add(_ev(EventType.SPEECH_END, 2.0))
    tl_a.add(_ev(EventType.TTS_CHUNK, 3.0))
    tl_a.add(_ev(EventType.SPEECH_START, 3.7))     # injection
    tl_a.add(_ev(EventType.INTERRUPT, 3.9))        # detected
    # no later user speech in scenario A -> NOT recovered
    res_a = ScenarioResult(scenario=sc_a)
    res_a.timeline = tl_a

    sc_b = Scenario(id="scb", name="B", category="barge_in",
                    steps=[Step(kind="speak", text="Q"),
                           Step(kind="interrupt_during_playback", text="Wait"),
                           Step(kind="drain", timeout_ms=1000)])
    tl_b = Timeline()
    tl_b.add(_ev(EventType.SPEECH_START, 0.1))     # unrelated clock, restarts near 0
    tl_b.add(_ev(EventType.SPEECH_END, 0.5))
    tl_b.add(_ev(EventType.SPEECH_START, 6.0))     # later speech in another scenario
    res_b = ScenarioResult(scenario=sc_b)
    res_b.timeline = tl_b

    samples = collect_endpoint_samples([res_a, res_b])
    metrics = compute_endpoint_metrics(samples)
    # Before the fix, scenario B's speech_start at 6.0 marked scenario A's
    # barge-in as recovered -> value 1.0. Correctly: no recovery in scenario A.
    assert metrics["recovery_success_rate"].numerator == 0
    assert metrics["recovery_success_rate"].denominator == 1
    assert metrics["recovery_success_rate"].value == 0.0


def test_fragment_count_does_not_bleed_across_scenarios():
    """Repeated transcript text only counts as a split within one scenario."""
    def _one_turn(scenario_id: str):
        res = _eot_result([
            (EventType.SPEECH_START, 1.0, None),
            (EventType.SPEECH_END, 2.0, None),
            (EventType.TRANSCRIPT, 2.5, "same words"),
        ])
        res.scenario = Scenario(id=scenario_id, name=scenario_id, category="x")
        return res

    # One copy of identical text in two DIFFERENT scenarios is not a split.
    samples = collect_endpoint_samples([_one_turn("sca"), _one_turn("scb")])
    metrics = compute_endpoint_metrics(samples)
    assert metrics["utterance_fragment_rate"].numerator == 0

    # Two copies within the SAME scenario are still a split.
    samples = collect_endpoint_samples([_one_turn("sca"), _one_turn("sca")])
    metrics = compute_endpoint_metrics(samples)
    assert metrics["utterance_fragment_rate"].numerator == 1
