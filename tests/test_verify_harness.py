from __future__ import annotations

import asyncio

from core.pipeline import StreamingPipeline
from utils.verify_harness import (
    Utterance,
    AcceptanceTargets,
    run_verification,
    classify_outcome,
)


class FakeSTT:
    async def transcribe(self, audio_blob: bytes) -> str:
        return ""


class FakeLLM:
    async def generate_stream(self, messages):
        if False:
            yield ""


class FakeTTS:
    sample_rate = 16000

    async def synthesize_stream(self, text_chunks, prosody=None):
        async for _ in text_chunks:
            pass
        yield b"\x00\x01\x02\x03"

    async def synthesize(self, text: str, prosody=None) -> bytes:
        return b""


class _BaseVAD:
    sample_rate = 16000

    def is_speech(self, chunk: bytes) -> bool:
        return False

    def reset(self) -> None:
        pass


def _pipeline_factory() -> StreamingPipeline:
    return StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), _BaseVAD())


class TestClassifyOutcome:
    def test_completed_utterance_marks_missed_when_not_ended(self) -> None:
        o = classify_outcome(0, "completed", speech_end_count=0, silence_ms_pushed=600)
        assert o.missed_endpoint is True

    def test_completed_captured_when_ended(self) -> None:
        o = classify_outcome(0, "completed", speech_end_count=1, silence_ms_pushed=600)
        assert o.missed_endpoint is False

    def test_hesitation_not_false_endpoints_when_above_threshold(self) -> None:
        # 640ms of trailing silence is above the 450ms normal threshold, so an
        # end here is a legitimate endpoint, not a premature cut.
        o = classify_outcome(0, "hesitation", speech_end_count=1, silence_ms_pushed=640)
        assert o.false_early_end is False


class TestRunVerification:
    def test_endpoint_metrics_computed(self) -> None:
        utterances = [
            Utterance(speech_frames=4, silence_frames=8, label="completed"),
            Utterance(speech_frames=4, silence_frames=6, label="completed"),
        ]
        report = asyncio.run(run_verification(utterances, _pipeline_factory))
        metrics = report.compute()
        assert metrics["utterances"] == 2
        assert metrics["completion_capture_rate"] == 1.0
        assert metrics["missed_endpoint_rate"] == 0.0
        assert metrics["endpoint_silence_p50_ms"] is not None

    def test_hesitation_silence_does_not_count_as_false_endpoint(self) -> None:
        # Long pause hesitation turn ends at >= normal threshold -> no false EP.
        utterances = [
            Utterance(speech_frames=4, silence_frames=10, label="hesitation"),
        ]
        report = asyncio.run(run_verification(utterances, _pipeline_factory))
        assert report.outcomes[0].false_early_end is False


class TestAcceptance:
    def test_targets_present_in_report(self) -> None:
        t = AcceptanceTargets()
        assert t.false_endpoint_max == 0.03
        assert t.completion_capture_min == 0.97
