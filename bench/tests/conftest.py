"""Shared fixtures for bench tests.

FakeAgent simulates a Bolo-like agent: when non-silent audio arrives it emits
the expected normalized event sequence (speech_start, transcript, llm_token,
tts_chunk, tts_done), and honours explicit interrupt signals. This lets the
runner, metrics and aggregation be tested offline and deterministically.
"""
from __future__ import annotations

import asyncio

from bench.agent.base import Event, EventType, SpeechAgent


class FakeAgent(SpeechAgent):
    sample_rate = 16000

    def __init__(self, transcript="hello", llm="Sure! Let me help you."):
        self.transcript = transcript
        self.llm = llm
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.interrupt_pending = False
        self.audio_frames = 0
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
        self.audio_frames += 1
        non_silent = any(b != 0 for b in pcm_bytes[:128])
        if non_silent:
            self._emit(EventType.SPEECH_START)
            await asyncio.sleep(0.01)
            self._emit(EventType.TRANSCRIPT, self.transcript)
            await asyncio.sleep(0.01)
            for tok in self.llm.split(" "):
                self._emit(EventType.LLM_TOKEN, tok + " ")
                await asyncio.sleep(0)
            self._emit(EventType.LLM_DONE, self.llm)
            self.events.put_nowait(Event.make(EventType.TTS_CHUNK, audio=b"\x00" * 1024))
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
