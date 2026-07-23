from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncGenerator

import time

from providers.stt.base import STTProvider
from providers.llm.base import LLMProvider
from providers.tts.base import TTSProvider
from modules.vad.silero_vad import SileroVAD
from modules.metrics.latency import LatencyTracker
from modules.metrics.logger import MetricsLogger
from utils.logger import logger


class PipelineEvent(Enum):
    AUDIO_CHUNK = auto()
    SPEECH_START = auto()
    SPEECH_END = auto()
    PARTIAL_TRANSCRIPT = auto()
    FINAL_TRANSCRIPT = auto()
    LLM_TOKEN = auto()
    LLM_DONE = auto()
    TTS_CHUNK = auto()
    TTS_DONE = auto()
    BACKCHANNEL = auto()
    INTERRUPT = auto()


@dataclass
class PipelineMessage:
    event: PipelineEvent
    data: str | bytes | None = None
    session_id: str = "default"


class StreamingPipeline:
    def __init__(
        self,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        vad: SileroVAD,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad

        self._latency = LatencyTracker()
        self._metrics = MetricsLogger()
        self._audio_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(256)
        self._output_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(256)
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._pipeline_loop(), name="pipeline"),
        ]
        logger.info("Streaming pipeline started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Streaming pipeline stopped")

    async def push_audio(self, chunk: bytes, session_id: str = "default") -> None:
        await self._audio_queue.put(
            PipelineMessage(PipelineEvent.AUDIO_CHUNK, chunk, session_id)
        )

    async def output_stream(self) -> AsyncGenerator[PipelineMessage, None]:
        while self._running or not self._output_queue.empty():
            try:
                msg = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield msg
            except asyncio.TimeoutError:
                continue

    async def _pipeline_loop(self) -> None:
        speech_buffer = bytearray()
        is_speaking = False

        while self._running:
            msg = await self._audio_queue.get()
            chunk = msg.data
            if not isinstance(chunk, bytes):
                continue

            is_speech = self.vad.is_speech(chunk)

            if is_speech:
                if not is_speaking:
                    is_speaking = True
                    speech_buffer = bytearray(chunk)
                    await self._emit(PipelineEvent.SPEECH_START, session_id=msg.session_id)
                else:
                    speech_buffer.extend(chunk)
            else:
                if is_speaking:
                    speech_buffer.extend(chunk)
                    pause_ms = len(chunk) / (self.vad.sample_rate * 2 / 1000)
                    if pause_ms >= 400:
                        is_speaking = False
                        audio_blob = bytes(speech_buffer)
                        speech_buffer.clear()
                        await self._emit(PipelineEvent.SPEECH_END, session_id=msg.session_id)
                        asyncio.create_task(
                            self._process_speech_segment(audio_blob, msg.session_id)
                        )

    async def _process_speech_segment(
        self, audio_blob: bytes, session_id: str
    ) -> None:
        stt_start = time.perf_counter()
        transcript = await self.stt.transcribe(audio_blob)
        self._latency.measure("stt", stt_start)
        if not transcript:
            return

        await self._emit(PipelineEvent.FINAL_TRANSCRIPT, transcript, session_id)
        self._log_latency("stt → transcript")

        llm_start = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are TASA, a real-time conversational assistant. "
                    "Respond concisely and naturally. Keep responses short, "
                    "conversational, and human-like. Use occasional fillers "
                    "like 'hmm' or 'well' when appropriate."
                ),
            },
            {"role": "user", "content": transcript},
        ]

        full = ""
        first_token = True
        async for token in self.llm.generate_stream(messages):
            if first_token:
                self._latency.measure("llm_first_token", llm_start)
                self._log_latency("audio → first LLM token")
                first_token = False
            full += token
            await self._emit(PipelineEvent.LLM_TOKEN, token, session_id)

        self._latency.measure("llm_full", llm_start)
        await self._emit(PipelineEvent.LLM_DONE, full, session_id)
        asyncio.create_task(self._synthesize_response(full, session_id))
        self._log_latency("llm_full")

    async def _synthesize_response(self, text: str, session_id: str) -> None:
        async def text_gen() -> AsyncGenerator[str, None]:
            import re
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            for s in sentences:
                yield s

        async for audio_chunk in self.tts.synthesize_stream(text_gen()):
            await self._emit(PipelineEvent.TTS_CHUNK, audio_chunk, session_id)
        await self._emit(PipelineEvent.TTS_DONE, session_id=session_id)

    def latency_report(self) -> dict[str, dict[str, float]]:
        return self._latency.report()

    def _log_latency(self, stage: str) -> None:
        report = self._latency.report()
        if stage in report:
            r = report[stage]
            self._metrics.log(stage, {"p50": r["p50"], "p95": r["p95"]})

    async def _emit(
        self, event: PipelineEvent, data: str | bytes | None = None, session_id: str = "default"
    ) -> None:
        await self._output_queue.put(PipelineMessage(event, data, session_id))