"""TASA WebSocket adapter.

Talks to TASA's /ws/audio endpoint (binary 16kHz PCM in, JSON control in,
JSON events + binary TTS audio out) and normalizes everything to the
bench.agent.base protocol.

Transport (from app/routes/ws.py):
  Client -> server (binary): 16kHz mono 16-bit PCM audio chunks
  Client -> server (text):   {"type":"ping"} | {"type":"interrupt"}
  Server -> client (text):   speech_start, speech_end, partial_transcript,
                             transcript, llm_token, llm_done, tts_done,
                             backchannel, interrupt, status, error, prosody
  Server -> client (binary): headerless/WAV PCM TTS audio
"""
from __future__ import annotations

import asyncio
import json
import time

from bench.agent.base import Event, EventType, SpeechAgent


class TasaWSAdapter(SpeechAgent):
    sample_rate = 16000

    def __init__(self, url: str = "ws://localhost:8000/ws/audio", timeout: float = 60.0):
        self.url = url
        self.timeout = timeout
        self._ws = None
        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._receiver: asyncio.Task | None = None

    async def connect(self) -> None:
        from websockets.asyncio.client import connect

        self._ws = await connect(self.url, max_size=64 * 1024 * 1024)
        self._receiver = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        if self._receiver:
            self._receiver.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def reset_session(self) -> None:
        # TASA keys fresh conversation state per WebSocket connection.
        await self.close()
        await self.connect()

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("adapter not connected")
        await self._ws.send(pcm_bytes)

    async def send_interrupt(self) -> None:
        if self._ws is None:
            raise RuntimeError("adapter not connected")
        await self._ws.send(json.dumps({"type": "interrupt"}))

    async def next_event(self, timeout: float = 60.0) -> Event | None:
        try:
            return await asyncio.wait_for(self._events.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                ev = self._parse(raw)
                if ev is not None:
                    await self._events.put(ev)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await self._events.put(
                Event.make(EventType.ERROR, text=f"adapter receive error: {exc!r}")
            )

    @staticmethod
    def _parse(raw) -> Event | None:
        if isinstance(raw, bytes):
            return Event.make(EventType.TTS_CHUNK, audio=raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        t = data.get("type")
        mapping = {
            "speech_start": EventType.SPEECH_START,
            "speech_end": EventType.SPEECH_END,
            "partial_transcript": EventType.PARTIAL_TRANSCRIPT,
            "transcript": EventType.TRANSCRIPT,
            "llm_token": EventType.LLM_TOKEN,
            "llm_done": EventType.LLM_DONE,
            "tts_done": EventType.TTS_DONE,
            "backchannel": EventType.BACKCHANNEL,
            "interrupt": EventType.INTERRUPT,
            "status": EventType.STATUS,
            "error": EventType.ERROR,
            "prosody": EventType.PROSODY,
        }
        et = mapping.get(t)
        if et is None:
            return None
        if et is EventType.PROSODY:
            label = data.get("label") or "neutral"
            emotion = data.get("emotion") or ""
            text = f"{label}|{emotion}"
            return Event.make(et, text=text)
        text = data.get("text") or None
        return Event.make(et, text=text)
