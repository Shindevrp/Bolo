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
from modules.turn.timing import TurnTiming
from modules.turn.backchannel import TurnBackchannel
from modules.backchannel.generator import BackchannelGenerator
from modules.backchannel.timing import BackchannelTiming
from modules.memory.session import SessionMemory
from modules.memory.retrieval import RetrievalModule
from modules.dialogue.prompts import build_system_prompt
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
    RESPONSE_DELAY = auto()
    ERROR = auto()


@dataclass
class PipelineMessage:
    event: PipelineEvent
    data: str | bytes | None = None
    session_id: str = "default"


@dataclass
class ConversationContext:
    engagement: float = 0.5
    turn_count: int = 0
    last_turn_duration_ms: float = 0.0
    last_transcript: str = ""
    is_question: bool = False
    rapid_exchange: bool = False
    prosody_trajectory: str = "neutral"


class StreamingPipeline:
    def __init__(
        self,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        vad: SileroVAD,
        turn_detector: TurnDetector | None = None,
        interrupt_handler: InterruptHandler | None = None,
        turn_timing: TurnTiming | None = None,
        turn_backchannel: TurnBackchannel | None = None,
        backchannel_generator: BackchannelGenerator | None = None,
        backchannel_timing: BackchannelTiming | None = None,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector or TurnDetector()
        self.interrupt_handler = interrupt_handler or InterruptHandler()
        self.turn_timing = turn_timing or TurnTiming()
        self.turn_backchannel = turn_backchannel or TurnBackchannel(
            generator=backchannel_generator or BackchannelGenerator(),
            timing=backchannel_timing or BackchannelTiming(),
        )

        self._latency = LatencyTracker()
        self._metrics = MetricsLogger()
        self._audio_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        self._output_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        self._interrupt_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._current_task: asyncio.Task | None = None
        self._running = False
        self._contexts: dict[str, ConversationContext] = {}
        self._memories: dict[str, SessionMemory] = {}
        self._retrievals: dict[str, RetrievalModule] = {}

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

    def register_session(
        self, session_id: str, memory: SessionMemory, retrieval: RetrievalModule
    ) -> None:
        self._memories[session_id] = memory
        self._retrievals[session_id] = retrieval

    def unregister_session(self, session_id: str) -> None:
        self._memories.pop(session_id, None)
        self._retrievals.pop(session_id, None)
        self._contexts.pop(session_id, None)

    async def signal_interrupt(self, session_id: str = "default") -> None:
        self._interrupt_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        logger.info(f"interrupt signaled for session {session_id}")

    async def output_stream(self) -> AsyncGenerator[PipelineMessage, None]:
        while self._running or not self._output_queue.empty():
            try:
                msg = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield msg
            except asyncio.TimeoutError:
                continue

    def _ctx(self, session_id: str) -> ConversationContext:
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext()
        return self._contexts[session_id]

    def _memory(self, session_id: str) -> SessionMemory | None:
        return self._memories.get(session_id)

    def _retrieval(self, session_id: str) -> RetrievalModule | None:
        return self._retrievals.get(session_id)

    def _update_engagement_from_prosody(
        self, ctx: ConversationContext, prosody_result: dict | None
    ) -> None:
        if not prosody_result:
            return
        trajectory = prosody_result.get("trajectory", "neutral")
        ctx.prosody_trajectory = trajectory

        if trajectory == "rising":
            ctx.engagement = min(1.0, ctx.engagement + 0.02)
        elif trajectory == "falling":
            ctx.engagement = max(0.1, ctx.engagement - 0.01)

    async def _pipeline_loop(self) -> None:
        speech_buffer = bytearray()
        is_speaking = False
        silence_ms = 0.0
        chunk_count = 0
        prosody_update_interval = 5

        while self._running:
            try:
                msg = await self._audio_queue.get()
            except asyncio.CancelledError:
                break
            chunk = msg.data
            if not isinstance(chunk, bytes):
                continue

            sid = msg.session_id
            ctx = self._ctx(sid)
            chunk_count += 1

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
                    await self._emit(PipelineEvent.SPEECH_START, session_id=sid)
                else:
                    silence_ms = 0.0
                    speech_buffer.extend(chunk)

                turn_decision = self.turn_detector.process_chunk(chunk, True)

                if chunk_count % prosody_update_interval == 0:
                    self._update_engagement_from_prosody(
                        ctx, self.turn_detector.prosody_analyzer.analyze()
                    )

                speech_dur_ms = len(speech_buffer) / (self.vad.sample_rate * 2 / 1000)
                if self.turn_backchannel.should_emit(
                    0, speech_dur_ms, ctx.engagement
                ):
                    bc = self.turn_backchannel.generate(
                        ctx.last_transcript, time.time(), time.time()
                    )
                    if bc and chunk_count % 10 == 0:
                        await self._emit(PipelineEvent.BACKCHANNEL, bc, sid)
                        logger.debug(f"backchannel session={sid} text={bc}")

            else:
                if is_speaking:
                    speech_buffer.extend(chunk)
                    silence_ms += len(chunk) / (self.vad.sample_rate * 2 / 1000)

                    turn_decision = self.turn_detector.process_chunk(chunk, False)

                    adaptive_threshold = 400.0
                    if turn_decision == "end_turn_force":
                        adaptive_threshold = 200.0
                    elif turn_decision == "end_turn":
                        adaptive_threshold = 300.0
                    elif ctx.engagement > 0.7:
                        adaptive_threshold = 350.0

                    if silence_ms >= adaptive_threshold:
                        is_speaking = False
                        audio_blob = bytes(speech_buffer)
                        speech_buffer.clear()
                        self.vad.reset()
                        self.turn_detector.reset()

                        ctx.last_turn_duration_ms = len(audio_blob) / (
                            self.vad.sample_rate * 2 / 1000
                        )
                        ctx.turn_count += 1

                        await self._emit(PipelineEvent.SPEECH_END, session_id=sid)
                        asyncio.create_task(
                            self._process_speech_segment(audio_blob, sid, ctx)
                        )

    async def _process_speech_segment(
        self, audio_blob: bytes, session_id: str, ctx: ConversationContext
    ) -> None:
        self._interrupt_event.clear()
        self._current_task = asyncio.current_task()

        try:
            stt_start = time.perf_counter()
            transcript = await self.stt.transcribe(audio_blob)
            self._latency.measure("stt", stt_start)
            if not transcript:
                return

            ctx.last_transcript = transcript
            ctx.is_question = transcript.strip().endswith("?")
            self._log_latency("stt")

            await self._emit(PipelineEvent.FINAL_TRANSCRIPT, transcript, session_id)

            if ctx.is_question:
                ctx.engagement = min(1.0, ctx.engagement + 0.05)
            else:
                ctx.engagement = max(0.1, ctx.engagement - 0.02)

            memory = self._memory(session_id)
            retrieval = self._retrieval(session_id)

            messages = await self._build_messages(
                transcript, ctx, memory, retrieval
            )

            llm_start = time.perf_counter()
            full = ""
            first_token = True

            async for token in self.llm.generate_stream(messages):
                if self._interrupt_event.is_set():
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
            self._log_latency("llm_full")

            if memory:
                memory.add("assistant", full)
            if retrieval:
                retrieval.add_to_long_term(full)

            delay = self.turn_timing.compute_delay(
                pause_duration=ctx.last_turn_duration_ms / 1000,
                engagement_score=ctx.engagement,
                turn_duration_ms=ctx.last_turn_duration_ms,
                is_question=ctx.is_question,
                is_backchannel=False,
            )
            if delay > 0.1:
                await self._emit(
                    PipelineEvent.RESPONSE_DELAY, str(round(delay, 2)), session_id
                )
                await asyncio.sleep(delay)

            await self._emit(PipelineEvent.LLM_DONE, full, session_id)

            asyncio.create_task(self._synthesize_response(full, session_id))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"processing error session={session_id} error={e}")
            await self._emit(PipelineEvent.ERROR, str(e), session_id)

    async def _build_messages(
        self,
        transcript: str,
        ctx: ConversationContext,
        memory: SessionMemory | None,
        retrieval: RetrievalModule | None,
    ) -> list[dict[str, str]]:
        has_context = False
        retrieved: list[str] = []

        if retrieval and memory:
            retrieved = retrieval.retrieve_context(transcript, memory, top_k=3)
            if retrieved:
                has_context = True

        system_prompt = build_system_prompt(
            engagement=ctx.engagement,
            turn_count=ctx.turn_count,
            has_context=has_context,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if has_context and retrieved:
            context_block = "\n".join(
                f"- {r}" for r in retrieved[:3]
            )
            messages.append({
                "role": "system",
                "content": f"Relevant context from earlier:\n{context_block}",
            })

        if memory:
            history = memory.get_history(6)
            for entry in history:
                messages.append({
                    "role": entry.role,
                    "content": entry.content,
                })

            if memory.token_estimate() > 3072:
                memory.truncate_to_budget(3072)
                logger.debug(f"truncated memory for session {ctx.turn_count}")

        if memory:
            memory.add("user", transcript)

        messages.append({"role": "user", "content": transcript})

        return messages

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
                sentences = [
                    s.strip()
                    for s in re.split(r"(?<=[.!?])\s+", text)
                    if s.strip()
                ]
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
        self,
        event: PipelineEvent,
        data: str | bytes | None = None,
        session_id: str = "default",
    ) -> None:
        try:
            await self._output_queue.put(PipelineMessage(event, data, session_id))
        except asyncio.QueueFull:
            logger.warning(f"output queue full, dropping {event}")
