from __future__ import annotations

import asyncio
import itertools

from core.pipeline import StreamingPipeline
from modules.backchannel.generator import BackchannelGenerator, BACKCHANNEL_CANDIDATES


class FakeSTT:
    async def transcribe(self, audio_blob: bytes) -> str:
        return ""


class FakeLLM:
    async def generate_stream(self, messages):
        if False:
            yield ""


class FakeVAD:
    sample_rate = 16000

    def is_speech(self, chunk: bytes) -> bool:
        return False

    def reset(self) -> None:
        pass


class FakeTTS:
    sample_rate = 16000

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    async def synthesize_stream(self, text_chunks):
        buf = ""
        async for chunk in text_chunks:
            buf += chunk
        self.synthesized.append(buf)
        yield b"\x00\x01\x02\x03"

    async def synthesize(self, text: str) -> bytes:
        return b""


def _make_pipeline() -> StreamingPipeline:
    return StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), FakeVAD())


class TestBackchannelGenerator:
    def test_generate_thinking(self) -> None:
        g = BackchannelGenerator()
        assert g.generate_thinking() in BACKCHANNEL_CANDIDATES["thinking"]


class TestTTSWorkerPriority:
    def test_backchannel_priority_before_response(self) -> None:
        async def run() -> list[str]:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Second sentence."))
            q.put_nowait((0, next(seq), "hmm"))
            q.put_nowait((2, next(seq), None))
            await p._tts_worker(q, "sess", asyncio.Event())
            return p.tts.synthesized

        synthesized = asyncio.run(run())
        assert synthesized == ["hmm", "Second sentence."]

    def test_interrupt_drains_without_synthesizing(self) -> None:
        async def run() -> list[str]:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Should be dropped."))
            q.put_nowait((2, next(seq), None))
            int_ev = p._int_event("sess")
            int_ev.set()
            await p._tts_worker(q, "sess", int_ev)
            return p.tts.synthesized

        synthesized = asyncio.run(run())
        assert synthesized == []

    def test_stop_tts_stops_worker(self) -> None:
        async def run() -> list[str]:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Drop me."))
            stop = asyncio.Event()
            stop.set()
            await p._tts_worker(q, "sess", stop)
            return p.tts.synthesized

        synthesized = asyncio.run(run())
        assert synthesized == []
