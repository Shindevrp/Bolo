from __future__ import annotations

import asyncio

from core.pipeline import StreamingPipeline, PipelineEvent, FRAME_BYTES
from modules.vad.silero_vad import HysteresisVAD


class _FakeProbVAD:
    """Base VAD that yields scripted VAD probabilities per frame."""

    sample_rate = 16000

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self._i = 0

    def speech_probability(self, chunk: bytes) -> float:
        if self._i < len(self._probs):
            p = self._probs[self._i]
        else:
            p = self._probs[-1]
        self._i += 1
        return p

    def is_speech(self, chunk: bytes) -> bool:
        pass

    def reset(self) -> None:
        self._i = 0

    def reset_probs(self) -> None:
        self._i = 0


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


class ScriptedVAD:
    """Returns True for the first `speech_frames` calls, False afterwards."""

    sample_rate = 16000

    def __init__(self, speech_frames: int) -> None:
        self.speech_frames = speech_frames
        self._calls = 0

    def is_speech(self, chunk: bytes) -> bool:
        self._calls += 1
        return self._calls <= self.speech_frames

    def reset(self) -> None:
        pass


def _make_pipeline(vad, **kwargs) -> StreamingPipeline:
    return StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), vad, **kwargs)


async def _run_turn(p, total_frames: int) -> list[PipelineEvent]:
    """Push `total_frames` frames; return the events observed."""
    p._running = True
    loop_task = asyncio.create_task(p._pipeline_loop())
    seen: list[PipelineEvent] = []

    async def collect() -> None:
        async for msg in p.output_stream():
            seen.append(msg.event)

    col_task = asyncio.create_task(collect())
    for _ in range(total_frames):
        await p.push_audio(b"\x01" * FRAME_BYTES, "sess")
    await asyncio.sleep(0.4)
    p._running = False
    loop_task.cancel()
    await asyncio.gather(loop_task, col_task, return_exceptions=True)
    return seen


class TestHysteresisVAD:
    def test_onset_requires_confirmation_frames(self) -> None:
        base = _FakeProbVAD([0.7, 0.7, 0.7])
        v = HysteresisVAD(
            base, onset_threshold=0.6, onset_confirmation=2, ema_alpha=1.0
        )
        # First high-prob frame alone must not enter ON.
        assert v.is_speech(b"x") is False
        # Second frame confirms onset but is still ARMED (not yet ON).
        assert v.is_speech(b"x") is False
        # Third frame carries the armed run past min_speech_frames -> ON.
        assert v.is_speech(b"x") is True
        assert v.state == "ON"

    def test_offset_hangover_keeps_speech_across_a_dip(self) -> None:
        base = _FakeProbVAD([0.9, 0.9, 0.2, 0.9, 0.9])
        v = HysteresisVAD(
            base,
            onset_threshold=0.6,
            offset_threshold=0.35,
            onset_confirmation=1,
            offset_confirmation=2,
            hangover_frames=2,
            ema_alpha=1.0,
        )
        assert v.is_speech(b"x") is False  # OFF -> ARMED
        assert v.is_speech(b"x") is True  # ARMED -> ON
        # A single sub-offset frame is absorbed by the hangover window.
        assert v.is_speech(b"x") is True
        # Above offset threshold keeps us in ON.
        assert v.is_speech(b"x") is True
        assert v.state == "ON"

    def test_confirmed_silence_leaves_on_state(self) -> None:
        base = _FakeProbVAD([0.9, 0.9, 0.0, 0.0, 0.0])
        v = HysteresisVAD(
            base,
            onset_threshold=0.6,
            offset_threshold=0.35,
            onset_confirmation=1,
            offset_confirmation=2,
            hangover_frames=0,
            ema_alpha=1.0,
        )
        assert v.is_speech(b"x") is False  # OFF -> ARMED
        assert v.is_speech(b"x") is True  # ARMED -> ON (min speech reached)
        assert v.is_speech(b"x") is False  # 1st silence frame, not yet confirmed
        assert v.state == "ON"
        assert not v.is_speech(b"x")  # 2nd silence frame confirms offset -> OFF
        assert v.state == "OFF"

    def test_reset(self) -> None:
        base = _FakeProbVAD([0.9, 0.9, 0.9, 0.9])
        v = HysteresisVAD(
            base,
            onset_threshold=0.6,
            offset_threshold=0.35,
            onset_confirmation=1,
            offset_confirmation=2,
            hangover_frames=0,
            ema_alpha=1.0,
        )
        assert v.is_speech(b"x") is False  # OFF -> ARMED
        assert v.is_speech(b"x") is True  # ARMED -> ON
        v.reset()
        assert v.state == "OFF"
        base.reset_probs()
        # After reset it re-arms, then enters ON (confirmation=1 two-step).
        assert v.is_speech(b"x") is False  # OFF -> ARMED
        assert v.is_speech(b"x") is True  # ARMED -> ON
        assert v.state == "ON"


class TestPipelineEndpointing:
    def test_mid_utterance_pause_does_not_end_turn(self) -> None:
        # 4 speech frames (~512ms) then a pause shorter than the 450ms endpoint
        # must not commit the turn.
        p = _make_pipeline(
            ScriptedVAD(4), endpoint_short_ms=350, endpoint_normal_ms=450
        )
        seen = asyncio.run(_run_turn(p, 7))
        assert PipelineEvent.SPEECH_END not in seen

    def test_completed_short_utterance_ends_turn(self) -> None:
        # 4 speech frames followed by enough silence (>450ms) commits the turn.
        p = _make_pipeline(
            ScriptedVAD(4), endpoint_short_ms=350, endpoint_normal_ms=450
        )
        seen = asyncio.run(_run_turn(p, 12))
        assert PipelineEvent.SPEECH_END in seen

    def test_short_blip_does_not_end_turn(self) -> None:
        # A single-frame blip of speech (128ms) is below the min-speech guard,
        # so it must never commit even with a long silence tail.
        p = _make_pipeline(ScriptedVAD(1), min_speech_duration_ms=500)
        seen = asyncio.run(_run_turn(p, 12))
        assert PipelineEvent.SPEECH_END not in seen
