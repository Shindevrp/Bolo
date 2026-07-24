from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncGenerator

from providers.stt.base import STTProvider
from providers.llm.base import LLMProvider
from providers.tts.base import TTSProvider
from modules.vad.silero_vad import SileroVAD
from modules.turn.detector import TurnDetector
from modules.turn.interrupt import InterruptHandler
from modules.backchannel.generator import BackchannelGenerator
from modules.backchannel.timing import BackchannelTiming
from modules.metrics.latency import LatencyTracker
from modules.metrics.logger import MetricsLogger
from utils.logger import get_logger

logger = get_logger("pipeline")


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
    ERROR = auto()


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
        turn_detector: TurnDetector | None = None,
        interrupt_handler: InterruptHandler | None = None,
        backchannel_generator: BackchannelGenerator | None = None,
        backchannel_timing: BackchannelTiming | None = None,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector or TurnDetector()
        self.interrupt_handler = interrupt_handler or InterruptHandler()
        self.backchannel_gen = backchannel_generator or BackchannelGenerator()
        self.backchannel_timing = backchannel_timing or BackchannelTiming()

        self._latency = LatencyTracker()
        self._metrics = MetricsLogger()
        self._audio_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        self._output_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        self._interrupt_event = asyncio.Event()
        self._speaking_sessions: dict[str, int] = {}
        self._tasks: list[asyncio.Task] = []
        self._current_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._pipeline_loop(), name="pipeline"),
        ]
        logger.info("pipeline started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("pipeline stopped")

    async def push_audio(self, chunk: bytes, session_id: str = "default") -> None:
        try:
            await self._audio_queue.put(
                PipelineMessage(PipelineEvent.AUDIO_CHUNK, chunk, session_id)
            )
        except asyncio.QueueFull:
            logger.warning("audio queue full, dropping chunk")

    async def signal_interrupt(self, session_id: str = "default") -> None:
        self._interrupt_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        await self._emit(PipelineEvent.INTERRUPT, session_id=session_id)
        logger.info(f"interrupt signaled for session {session_id}")

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
        silence_ms = 0.0
        backchannel_cooldown = 0.0

        while self._running:
            try:
                msg = await self._audio_queue.get()
            except asyncio.CancelledError:
                break
            chunk = msg.data
            if not isinstance(chunk, bytes):
                continue

            try:
                is_speech = self.vad.is_speech(chunk)
            except Exception as e:
                logger.error(f"vad error: {e}")
                continue

            if is_speech:
                if not is_speaking:
                    is_speaking = True
                    silence_ms = 0.0
                    speech_buffer = bytearray(chunk)
                    self.turn_detector.reset()
                    self.vad.reset()
                    await self._emit(PipelineEvent.SPEECH_START, session_id=msg.session_id)
                    logger.debug(f"speech_start session={msg.session_id}")
                else:
                    silence_ms = 0.0
                    speech_buffer.extend(chunk)

                self.turn_detector.process_chunk(chunk, True)

                now = time.time()
                if now - backchannel_cooldown > self.backchannel_gen.cooldown_seconds:
                    speech_dur = len(speech_buffer) / (self.vad.sample_rate * 2 / 1000)
                    if self.backchannel_timing.should_emit(0, speech_dur, 0.5):
                        bc = self.backchannel_gen.generate("", backchannel_cooldown, now)
                        if bc:
                            backchannel_cooldown = now
                            await self._emit(PipelineEvent.BACKCHANNEL, bc, msg.session_id)
            else:
                if is_speaking:
                    speech_buffer.extend(chunk)
                    silence_ms += len(chunk) / (self.vad.sample_rate * 2 / 1000)

                    self.turn_detector.process_chunk(chunk, False)

                    if silence_ms >= 400:
                        is_speaking = False
                        audio_blob = bytes(speech_buffer)
                        speech_buffer.clear()
                        silence_ms = 0.0
                        self.vad.reset()
                        await self._emit(PipelineEvent.SPEECH_END, session_id=msg.session_id)
                        logger.info(f"speech_end session={msg.session_id} dur={len(audio_blob)}")
                        asyncio.create_task(
                            self._process_speech_segment(audio_blob, msg.session_id)
                        )

    async def _process_speech_segment(
        self, audio_blob: bytes, session_id: str
    ) -> None:
        self._interrupt_event.clear()
        self._current_task = asyncio.current_task()

        try:
            stt_start = time.perf_counter()
            transcript = await self.stt.transcribe(audio_blob)
            self._latency.measure("stt", stt_start)
            if not transcript:
                logger.debug(f"empty transcript session={session_id}")
                return

            await self._emit(PipelineEvent.FINAL_TRANSCRIPT, transcript, session_id)
            self._log_latency("stt")
            logger.info(f"transcript session={session_id} text={transcript}")

            system_prompt = (
                "You are TASA, a real-time conversational assistant. "
                "Respond concisely and naturally. Keep responses short, "
                "conversational, and human-like. Use occasional fillers "
                "like 'hmm' or 'well' when appropriate."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ]

            llm_start = time.perf_counter()
            full = ""
            first_token = True

            async for token in self.llm.generate_stream(messages):
                if self._interrupt_event.is_set():
                    logger.info(f"llm interrupted session={session_id}")
                    break
                if first_token:
                    self._latency.measure("llm_first_token", llm_start)
                    self._log_latency("llm_first_token")
                    first_token = False
                full += token
                await self._emit(PipelineEvent.LLM_TOKEN, token, session_id)

            if self._interrupt_event.is_set():
                return

            self._latency.measure("llm_full", llm_start)
            await self._emit(PipelineEvent.LLM_DONE, full, session_id)
            self._log_latency("llm_full")
            logger.info(f"llm_done session={session_id} len={len(full)}")

            asyncio.create_task(self._synthesize_response(full, session_id))

        except asyncio.CancelledError:
            logger.info(f"processing cancelled session={session_id}")
        except Exception as e:
            logger.error(f"processing error session={session_id} error={e}")
            await self._emit(PipelineEvent.ERROR, str(e), session_id)

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_bytes)
        header = bytearray()
        header += b"RIFF"
        header += (36 + data_size).to_bytes(4, "little")
        header += b"WAVE"
        header += b"fmt "
        header += (16).to_bytes(4, "little")
        header += (1).to_bytes(2, "little")
        header += num_channels.to_bytes(2, "little")
        header += sample_rate.to_bytes(4, "little")
        header += byte_rate.to_bytes(4, "little")
        header += block_align.to_bytes(2, "little")
        header += bits_per_sample.to_bytes(2, "little")
        header += b"data"
        header += data_size.to_bytes(4, "little")
        return bytes(header) + pcm_bytes

    async def _synthesize_response(self, text: str, session_id: str) -> None:
        try:
            async def text_gen() -> AsyncGenerator[str, None]:
                import re
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                for s in sentences:
                    if self._interrupt_event.is_set():
                        break
                    yield s

            sr = self.tts.sample_rate
            async for audio_chunk in self.tts.synthesize_stream(text_gen()):
                if self._interrupt_event.is_set():
                    break
                if isinstance(audio_chunk, bytes) and len(audio_chunk) > 0:
                    wav = self._pcm_to_wav(audio_chunk, sr)
                    await self._emit(PipelineEvent.TTS_CHUNK, wav, session_id)
            if not self._interrupt_event.is_set():
                await self._emit(PipelineEvent.TTS_DONE, session_id=session_id)

        except Exception as e:
            logger.error(f"tts error session={session_id} error={e}")

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
        try:
            await self._output_queue.put(PipelineMessage(event, data, session_id))
        except asyncio.QueueFull:
            logger.warning(f"output queue full, dropping {event}")
