"""Mock agent adapter (offline smoke testing).

Emits a scripted event sequence in response to non-silent audio, so the runner,
metrics, aggregation, and report can be exercised end-to-end without a live
agent server. Not a real inference agent — for CI/offline validation only.
"""
from __future__ import annotations

import asyncio

from bench.agent.base import Event, EventType, SpeechAgent


class MockAgent(SpeechAgent):
    sample_rate = 16000

    def __init__(
        self,
        transcript: str = "mock transcript",
        llm: str = "That sounds like a good plan.",
        *,
        emit_interrupt: bool = False,
    ):
        self.transcript = transcript
        self.llm = llm
        self.emit_interrupt = emit_interrupt
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.started = False

    async def connect(self):
        self.started = True

    async def close(self):
        self.started = False

    async def reset_session(self):
        while not self.events.empty():
            try:
                self.events.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def send_audio(self, pcm_bytes: bytes):
        if any(b != 0 for b in pcm_bytes[:128]):
            self._emit(EventType.SPEECH_START)
            await asyncio.sleep(0.01)
            self._emit(EventType.TRANSCRIPT, self.transcript)
            await asyncio.sleep(0.01)
            for tok in self.llm.split(" "):
                self._emit(EventType.LLM_TOKEN, tok + " ")
                await asyncio.sleep(0)
            self._emit(EventType.LLM_DONE, self.llm)
            if self.emit_interrupt:
                self._emit(EventType.INTERRUPT)
            self.events.put_nowait(Event.make(EventType.TTS_CHUNK, audio=b"\x00" * 512))
            self._emit(EventType.TTS_DONE)

    async def send_interrupt(self):
        self._emit(EventType.INTERRUPT)

    def _emit(self, et: EventType, text=None):
        self.events.put_nowait(Event.make(et, text=text))

    async def next_event(self, timeout=60.0):
        try:
            return await asyncio.wait_for(self.events.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
