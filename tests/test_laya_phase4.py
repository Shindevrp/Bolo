"""Phase-4 (cadence-1 endpoint-only authority) behavior.

Phase 4 is the speculative early-start lane: once a fresh, confident
cadence-1 ``turn_complete`` verdict exists, the turn commits on the short
endpoint WITHOUT Phase-3's barge enforcement. Fail-closed: ``phase4=False``
(default) keeps the legacy endpointing; cadence-1 probes only run for the
Phase-3 and Phase-4 consumers.
"""
from __future__ import annotations

import asyncio
import time

from core.pipeline import PipelineEvent, StreamingPipeline
from providers.laya.client import LayaSystem1
from tests.test_laya_adapters import FakeAgent
from tests.test_laya_phase2 import FixedSTT
from tests.test_laya_phase3 import (
    _c1_answers,
    _endpoint_pipeline,
    _run_speech_pause,
    _seed,
    _wait_for,
)
from tests.test_streaming import FakeVAD


class TestCadence1Endpoint:
    def test_phase4_off_by_default_keeps_legacy_endpoint(self) -> None:
        p = _endpoint_pipeline()
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.99, completion_conf=9.5)
        )
        assert PipelineEvent.SPEECH_END not in seen

    def test_phase4_confident_complete_shortens_endpoint(self) -> None:
        p = _endpoint_pipeline(phase4=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.97, completion_conf=8.5)
        )
        assert PipelineEvent.SPEECH_END in seen

    def test_phase4_weak_verdict_keeps_normal_endpoint(self) -> None:
        p = _endpoint_pipeline(phase4=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.5, completion_conf=3.0)
        )
        assert PipelineEvent.SPEECH_END not in seen

    def test_phase4_stale_verdict_keeps_normal_endpoint(self) -> None:
        p = _endpoint_pipeline(phase4=True)
        seen = asyncio.run(
            _run_speech_pause(p, 4, 3, end_text="hello there",
                              turn_complete=0.99, completion_conf=9.5,
                              ts=time.monotonic() - 60)
        )
        assert PipelineEvent.SPEECH_END not in seen


class TestCadence1Probe:
    def test_phase4_enables_cadence1_probe(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(
                FixedSTT("hello there"),
                None,
                None,
                FakeVAD(),
                system1=LayaSystem1(
                    enabled=True, agent=FakeAgent(_c1_answers())
                ),
                phase4=True,
            )
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await _wait_for(lambda: "sess" in p._s1_cadence1)
            v = p._s1_cadence1["sess"]
            assert v["turn_complete"] == 0.95
            assert v["completion_conf"] == 8.0

        asyncio.run(run())

    def test_no_probe_when_phase4_off(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(
                FixedSTT("hello there"),
                None,
                None,
                FakeVAD(),
                system1=LayaSystem1(
                    enabled=True, agent=FakeAgent(_c1_answers())
                ),
            )
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await asyncio.sleep(0.2)
            assert p._s1_cadence1.get("sess") is None

        asyncio.run(run())


class TestCadence1BargeIsolation:
    def test_phase4_alone_does_not_fire_barge(self) -> None:
        # Phase 4 must not inherit Phase-3's barge authority: a confident
        # disagreement verdict is ignored for interrupting when only phase4 is
        # on (energy too low for the legacy lane -> pending stays).
        async def run() -> None:
            p = _endpoint_pipeline(phase4=True)
            ctx = p._ctx("sess")
            p._barge_pending["sess"] = True
            _seed(p, barge_type="disagreement", barge_conf=0.95)
            await p._maybe_fire_barge("sess", ctx, 0.0)
            assert p._barge_pending.get("sess") is True

        asyncio.run(run())

    def test_phase3_disagreement_fires_interrupt(self) -> None:
        async def run() -> None:
            p = _endpoint_pipeline(phase3=True)
            ctx = p._ctx("sess")
            p._barge_pending["sess"] = True
            _seed(p, barge_type="disagreement", barge_conf=0.9)
            await p._maybe_fire_barge("sess", ctx, 0.1)
            assert p._barge_pending.get("sess") is False

        asyncio.run(run())