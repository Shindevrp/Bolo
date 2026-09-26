"""Phase-3/4 (cadence-1 endpoint/barge) behavior of the System-1 decision layer.

Cadence-1 probes run in the background at the partial-transcript cadence
(~1s). Shadow rows are always recorded; enforcement is opt-in via
``phase3=True`` on the pipeline (or TASA_LAYA_PHASE3=1) and is always clipped
by the deterministic safety floor. Phase 4 made Laya the endpoint/barge
authority once a fresh, confident verdict exists -- the legacy
``is_incomplete`` / ``BackchannelInterrupt`` signals are demoted to the
pre-probe window (no/mismatched/stale verdict) and remain the fail-open
fallback.
"""
from __future__ import annotations

import asyncio
import time

from core.pipeline import FRAME_BYTES, PipelineEvent, StreamingPipeline
from core.state import DialogueState
from providers.laya.client import LayaSystem1
from tests.test_laya_adapters import FakeAgent
from tests.test_laya_phase2 import FixedSTT
from tests.test_streaming import FakeVAD
from tests.test_vad_endpoint import FakeLLM, FakeSTT, FakeTTS, ScriptedVAD


def _c1_answers(
    turn_complete: float = 0.95,
    completion_conf: float = 8.0,
    barge: str = "none",
    barge_conf: float = 0.9,
) -> dict:
    return {
        "turn_complete": {"type": "noul", "noul": turn_complete},
        "completion_conf": {"type": "score", "score": completion_conf},
        "barge_type": {"type": "choice", "choice": barge, "confidence": barge_conf},
    }


async def _wait_for(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met in time")


def _seed(p: StreamingPipeline, sid: str = "sess", **kw) -> dict:
    verdict: dict = {
        "partial": "hello there",
        "generation": p._generation(sid),
        "turn_complete": 0.95,
        "completion_conf": 8.0,
        "barge_type": "none",
        "barge_conf": 0.9,
        "ts": time.monotonic(),
    }
    verdict.update(kw)
    p._s1_cadence1[sid] = verdict
    return verdict


def _endpoint_pipeline(**kwargs) -> StreamingPipeline:
    return StreamingPipeline(
        FakeSTT(),
        FakeLLM(),
        FakeTTS(),
        ScriptedVAD(4),
        endpoint_short_ms=350,
        endpoint_normal_ms=450,
        **kwargs,
    )


async def _run_speech_pause(
    p: StreamingPipeline,
    speech_frames: int,
    silence_frames: int,
    *,
    end_text: str,
    **verdict_kw,
) -> list[PipelineEvent]:
    """Push speech, align the live partial + cadence-1 verdict to ``end_text``,
    then push a silence tail. The verdict timestamp is set just before the
    pause so its freshness window covers the endpoint decision.

    4 speech frames (~512ms, past the 500ms floor) + 3 silence frames (384ms)
    sits inside the 350ms short lane and under the 450ms normal lane, so the
    tests discriminate the Laya lane vs the legacy lanes exactly.
    """
    p._running = True
    loop_task = asyncio.create_task(p._pipeline_loop())
    seen: list[PipelineEvent] = []

    async def collect() -> None:
        async for msg in p.output_stream():
            seen.append(msg.event)

    col_task = asyncio.create_task(collect())
    for _ in range(speech_frames):
        await p.push_audio(b"\x01" * FRAME_BYTES, "sess")
        await asyncio.sleep(0.02)
    p._ctx("sess").last_partial_transcript = end_text
    verdict_kw.setdefault("partial", end_text)
    _seed(p, **verdict_kw)
    for _ in range(silence_frames):
        await p.push_audio(b"\x01" * FRAME_BYTES, "sess")
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.4)
    p._running = False
    loop_task.cancel()
    await asyncio.gather(loop_task, col_task, return_exceptions=True)
    return seen


class TestCadence1Probe:
    def test_probe_stashes_verdict_and_shadow_logs(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(
                FixedSTT("hello there"),
                FakeLLM(),
                FakeTTS(),
                FakeVAD(),
                system1=LayaSystem1(enabled=True, agent=FakeAgent(_c1_answers())),
                # Cadence-1 probes only run for the Phase-3 consumers.
                phase3=True,
            )
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await _wait_for(lambda: "sess" in p._s1_cadence1)
            v = p._s1_cadence1["sess"]
            assert v["turn_complete"] == 0.95
            assert v["completion_conf"] == 8.0
            assert v["barge_type"] == "none"
            report = p._s1.shadow_report()
            assert "c1_turn_complete" in report
            assert "c1_barge_type" in report

        asyncio.run(run())

    def test_low_conf_barge_verdict_stashed_as_none(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(
                FixedSTT("yeah"),
                FakeLLM(),
                FakeTTS(),
                FakeVAD(),
                system1=LayaSystem1(
                    enabled=True,
                    agent=FakeAgent(
                        _c1_answers(barge="backchannel", barge_conf=0.3)
                    ),
                ),
                phase3=True,
            )
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await _wait_for(lambda: "sess" in p._s1_cadence1)
            assert p._s1_cadence1["sess"]["barge_type"] is None

        asyncio.run(run())


class TestCadence1Endpoint:
    def test_confident_complete_shortens_endpoint(self) -> None:
        p = _endpoint_pipeline(phase3=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.97, completion_conf=8.5)
        )
        assert PipelineEvent.SPEECH_END in seen

    def test_weak_verdict_keeps_normal_endpoint(self) -> None:
        p = _endpoint_pipeline(phase3=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.5, completion_conf=3.0)
        )
        assert PipelineEvent.SPEECH_END not in seen

    def test_phase_off_keeps_legacy_endpoint(self) -> None:
        p = _endpoint_pipeline()
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.99, completion_conf=9.5)
        )
        assert PipelineEvent.SPEECH_END not in seen

    def test_stale_verdict_keeps_normal_endpoint(self) -> None:
        p = _endpoint_pipeline(phase3=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.99, completion_conf=9.5,
                              ts=time.monotonic() - 60)
        )
        assert PipelineEvent.SPEECH_END not in seen

    def test_confident_laya_completes_midlist_utterance(self) -> None:
        # Phase-4 demotion: a confident fresh verdict is the endpoint authority
        # even when the legacy incomplete heuristic (trailing "and") would hold
        # the turn open in the hesitation lane (550ms > 384ms pause). The
        # mid-list protection lives in the text-match guard of the next test.
        p = _endpoint_pipeline(phase3=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="red, green, and",
                              turn_complete=0.99, completion_conf=9.0)
        )
        assert PipelineEvent.SPEECH_END in seen

    def test_verdict_text_mismatch_falls_back_to_legacy(self) -> None:
        # The verdict was computed on a shorter partial than the one now live
        # (as after a sync mid-list refresh): it is no longer authoritative, so
        # the legacy incomplete signal holds the turn open (hesitation lane).
        p = _endpoint_pipeline(phase3=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="red, green, and",
                              partial="red, green",
                              turn_complete=0.99, completion_conf=9.0)
        )
        assert PipelineEvent.SPEECH_END not in seen


class TestCadence1Barge:
    @staticmethod
    def _pipeline() -> StreamingPipeline:
        return StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), ScriptedVAD(2), phase3=True)

    def test_backchannel_verdict_suppresses_barge(self) -> None:
        async def run() -> None:
            p = self._pipeline()
            ctx = p._ctx("sess")
            p._barge_pending["sess"] = True
            _seed(p, barge_type="backchannel", barge_conf=0.95)
            await p._maybe_fire_barge("sess", ctx, 0.9)
            assert p._barge_pending.get("sess") is False
            assert ctx.dialogue_state != DialogueState.LISTENING

        asyncio.run(run())

    def test_disagreement_verdict_fires_interrupt_immediately(self) -> None:
        async def run() -> None:
            p = self._pipeline()
            ctx = p._ctx("sess")
            p._barge_pending["sess"] = True
            _seed(p, barge_type="disagreement", barge_conf=0.9)
            await p._maybe_fire_barge("sess", ctx, 0.1)
            assert p._barge_pending.get("sess") is False
            assert ctx.dialogue_state == DialogueState.LISTENING

        asyncio.run(run())

    def test_weak_verdict_falls_back_to_legacy(self) -> None:
        async def run() -> None:
            p = self._pipeline()
            ctx = p._ctx("sess")
            p._barge_pending["sess"] = True
            _seed(p, barge_type=None, barge_conf=0.3)
            await p._maybe_fire_barge("sess", ctx, 0.0)
            # Legacy path: no lexical signal, energy too low -> pending stays.
            assert p._barge_pending.get("sess") is True

        asyncio.run(run())

    def test_stale_verdict_ignored_for_barge(self) -> None:
        async def run() -> None:
            p = self._pipeline()
            ctx = p._ctx("sess")
            p._barge_pending["sess"] = True
            _seed(p, barge_type="disagreement", barge_conf=0.9, ts=time.monotonic() - 60)
            await p._maybe_fire_barge("sess", ctx, 0.0)
            assert p._barge_pending.get("sess") is True

        asyncio.run(run())