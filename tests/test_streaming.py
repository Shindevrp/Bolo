from __future__ import annotations

import asyncio
import itertools

import pytest

from core.pipeline import StreamingPipeline, ConversationContext
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


class FakeSTTText:
    async def transcribe(self, audio_blob: bytes) -> str:
        return "hello world"


class FakeLLMText:
    async def generate_stream(self, messages):
        for token in ["Hello ", "there, ", "this ", "is ", "a test."]:
            yield token


class FakeTTSStream:
    sample_rate = 16000

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    async def synthesize_stream(self, text_chunks):
        buf = ""
        async for chunk in text_chunks:
            buf += chunk
        if buf:
            self.synthesized.append(buf)
            yield b"\x00\x00"

    async def synthesize(self, text: str) -> bytes:
        return b""


class TestMultiSessionIsolation:
    def test_signal_interrupt_cancels_only_target_session(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            cancelled: dict[str, bool] = {"A": False, "B": False}
            block_a, block_b = asyncio.Event(), asyncio.Event()

            async def blocker(name: str, block: asyncio.Event) -> None:
                try:
                    await block.wait()
                except asyncio.CancelledError:
                    cancelled[name] = True
                    raise

            task_a = asyncio.create_task(blocker("A", block_a))
            task_b = asyncio.create_task(blocker("B", block_b))
            p._current_tasks["A"] = task_a
            p._current_tasks["B"] = task_b

            await asyncio.sleep(0.01)
            await p.signal_interrupt("A")
            await asyncio.sleep(0.05)

            assert cancelled["A"] is True
            assert cancelled["B"] is False
            assert not task_b.done()

            task_b.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task_b

        asyncio.run(run())

    def test_signal_interrupt_unknown_session_does_not_crash(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            await p.signal_interrupt("unknown_session")
            assert "unknown_session" not in p._current_tasks

        asyncio.run(run())

    def test_process_segment_streams_and_clears_task_map(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = FakeSTTText()
            p.llm = FakeLLMText()
            p.tts = FakeTTSStream()
            ctx = ConversationContext()

            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)

            assert "sess" not in p._current_tasks
            assert p.tts.synthesized, "expected synthesized audio chunks"

        asyncio.run(run())

    def test_new_segment_replaces_previous_task_entry(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = FakeSTTText()
            p.llm = FakeLLMText()
            p.tts = FakeTTSStream()

            await p._process_speech_segment(b"\x00" * 1600, "sess", ConversationContext())
            await p._process_speech_segment(b"\x00" * 1600, "sess", ConversationContext())

            assert "sess" not in p._current_tasks
            assert len(p.tts.synthesized) == 2

        asyncio.run(run())
