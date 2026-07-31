from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncGenerator

from modules.tts.chunker import TTSChunker
from modules.tts.sanitize import sanitize_for_tts
from modules.tts.prosody import ProsodySelector, classify_sentiment
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
from core.state import DialogueState
from modules.dialogue.prompts import build_system_prompt
from modules.metrics.latency import LatencyTracker
from modules.metrics.logger import MetricsLogger
from modules.tools.registry import ToolRegistry
from modules.tools.builtin import get_builtin_tools
from utils.audio import rms_energy
from utils.logger import get_logger

logger = get_logger("pipeline")

ECHO_GRACE_SECONDS = 0.35
ECHO_FLOOR_MARGIN = 1.5


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
    last_partial_transcript: str = ""
    is_question: bool = False
    rapid_exchange: bool = False
    prosody_trajectory: str = "neutral"
    dialogue_state: DialogueState = DialogueState.IDLE
    query_complexity: str = "standard"
    topic_shift: bool = False
    user_sentiment: str = "neutral"
    user_repeated: bool = False


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
        self._prosody = ProsodySelector()

        self._latency = LatencyTracker()
        self._metrics = MetricsLogger()
        self._tool_registry = get_builtin_tools()
        self._audio_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        self._output_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        self._interrupt_events: dict[str, asyncio.Event] = {}
        self._tasks: list[asyncio.Task] = []
        self._current_tasks: dict[str, asyncio.Task] = {}
        self._playback_active: dict[str, bool] = {}
        self._playback_clear_tasks: dict[str, asyncio.Task] = {}
        self._playback_onset: dict[str, float] = {}
        self._echo_floor: dict[str, float] = {}
        self._speaking: dict[str, bool] = {}
        self._last_spoken: dict[str, str] = {}
        self._speech_buffers: dict[str, bytearray] = {}
        self._silence_ms: dict[str, float] = {}
        self._last_partial_time: dict[str, float] = {}
        self._barge_pending: dict[str, bool] = {}
        self._barge_thresholds: dict[str, float] = {}
        self._barge_frames: dict[str, int] = {}
        self._interrupt_handlers: dict[str, InterruptHandler] = {}
        self._running = False
        self._contexts: dict[str, ConversationContext] = {}
        self._memories: dict[str, SessionMemory] = {}
        self._retrievals: dict[str, RetrievalModule] = {}

    def _int_event(self, session_id: str) -> asyncio.Event:
        if session_id not in self._interrupt_events:
            self._interrupt_events[session_id] = asyncio.Event()
        return self._interrupt_events[session_id]

    def _interrupt_handler(self, session_id: str) -> InterruptHandler:
        handler = self._interrupt_handlers.get(session_id)
        if handler is None:
            handler = InterruptHandler(
                speech_energy_threshold=self.interrupt_handler.speech_energy_threshold,
                silence_confidence_threshold=self.interrupt_handler.silence_confidence_threshold,
                consecutive_speech_frames=self.interrupt_handler.consecutive_speech_frames,
                playback_consecutive_speech_frames=self.interrupt_handler.playback_consecutive_speech_frames,
            )
            self._interrupt_handlers[session_id] = handler
        return handler

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
        for task in self._playback_clear_tasks.values():
            task.cancel()
        self._playback_clear_tasks.clear()
        self._playback_active.clear()
        self._playback_onset.clear()
        self._echo_floor.clear()
        self._speaking.clear()
        self._speech_buffers.clear()
        self._silence_ms.clear()
        self._last_partial_time.clear()
        self._barge_pending.clear()
        self._barge_thresholds.clear()
        self._barge_frames.clear()
        self._interrupt_handlers.clear()
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
        try:
            asyncio.create_task(
                asyncio.to_thread(retrieval.warm_up)
            )
        except RuntimeError:
            pass

    def unregister_session(self, session_id: str) -> None:
        self._memories.pop(session_id, None)
        self._retrievals.pop(session_id, None)
        self._contexts.pop(session_id, None)
        self._interrupt_events.pop(session_id, None)
        self._current_tasks.pop(session_id, None)
        self._playback_active.pop(session_id, None)
        self._playback_onset.pop(session_id, None)
        self._echo_floor.pop(session_id, None)
        self._speaking.pop(session_id, None)
        self._speech_buffers.pop(session_id, None)
        self._silence_ms.pop(session_id, None)
        self._last_partial_time.pop(session_id, None)
        self._barge_pending.pop(session_id, None)
        self._barge_thresholds.pop(session_id, None)
        self._barge_frames.pop(session_id, None)
        self._interrupt_handlers.pop(session_id, None)
        clear_task = self._playback_clear_tasks.pop(session_id, None)
        if clear_task and not clear_task.done():
            clear_task.cancel()

    async def signal_interrupt(self, session_id: str = "default") -> None:
        ev = self._int_event(session_id)
        ev.set()
        task = self._current_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        self._playback_active[session_id] = False
        self._playback_onset.pop(session_id, None)
        self._echo_floor.pop(session_id, None)
        clear_task = self._playback_clear_tasks.pop(session_id, None)
        if clear_task and not clear_task.done():
            clear_task.cancel()
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
        chunk_count = 0
        prosody_update_interval = 5
        partial_transcript_interval = 1.0

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
            is_speaking = self._speaking.get(sid, False)
            speech_buffer = self._speech_buffers.get(sid)

            try:
                is_speech = self.vad.is_speech(chunk)
            except Exception as e:
                logger.error(f"vad error: {e}")
                continue

            if is_speech:
                if not is_speaking:
                    playback_on = self._playback_active.get(sid, False)
                    playback_age = time.monotonic() - self._playback_onset.get(
                        sid, 0.0
                    )
                    in_echo_grace = playback_on and (
                        playback_age < ECHO_GRACE_SECONDS
                    )
                    task_alive = (
                        self._current_tasks.get(sid) is not None
                        and not self._current_tasks[sid].done()
                    )
                    barge_pending = (
                        playback_on
                        and not in_echo_grace
                        or (
                            not playback_on
                            and task_alive
                            and ctx.dialogue_state
                            == DialogueState.INTERRUPTIBLE
                        )
                    )
                    self._barge_pending[sid] = barge_pending
                    if barge_pending:
                        threshold = (
                            self.interrupt_handler.speech_energy_threshold
                        )
                        target_frames = (
                            self.interrupt_handler.consecutive_speech_frames
                        )
                        if playback_on:
                            threshold = max(
                                threshold,
                                self._echo_floor.get(sid, 0.0)
                                * ECHO_FLOOR_MARGIN,
                            )
                            target_frames = (
                                self.interrupt_handler.playback_consecutive_speech_frames
                            )
                        self._barge_thresholds[sid] = threshold
                        self._barge_frames[sid] = target_frames
                        self._interrupt_handler(sid).reset()
                    self._speaking[sid] = True
                    ctx.dialogue_state = DialogueState.LISTENING
                    self._silence_ms[sid] = 0.0
                    self._speech_buffers[sid] = bytearray(chunk)
                    speech_buffer = self._speech_buffers[sid]
                    self._last_partial_time[sid] = time.time()
                    self.turn_detector.reset()
                    self.vad.reset()
                    await self._emit(PipelineEvent.SPEECH_START, session_id=sid)
                else:
                    self._silence_ms[sid] = 0.0
                    speech_buffer.extend(chunk)
                    energy = rms_energy(bytes(chunk)) / 32768.0
                    playback_on = self._playback_active.get(sid, False)
                    playback_age = time.monotonic() - self._playback_onset.get(
                        sid, 0.0
                    )
                    in_echo_grace = playback_on and (
                        playback_age < ECHO_GRACE_SECONDS
                    )
                    assistant_responding = (
                        self._current_tasks.get(sid) is not None
                        and not self._current_tasks[sid].done()
                        and ctx.dialogue_state == DialogueState.INTERRUPTIBLE
                    )
                    if (
                        not self._barge_pending.get(sid, False)
                        and not in_echo_grace
                        and assistant_responding
                    ):
                        threshold = (
                            self.interrupt_handler.speech_energy_threshold
                        )
                        target_frames = (
                            self.interrupt_handler.consecutive_speech_frames
                        )
                        if playback_on:
                            threshold = max(
                                threshold,
                                self._echo_floor.get(sid, 0.0)
                                * ECHO_FLOOR_MARGIN,
                            )
                            target_frames = (
                                self.interrupt_handler.playback_consecutive_speech_frames
                            )
                        if energy > threshold:
                            self._barge_pending[sid] = True
                            self._barge_thresholds[sid] = threshold
                            self._barge_frames[sid] = target_frames
                            self._interrupt_handler(sid).reset()
                            logger.debug(
                                f"barge-in candidate session={sid} energy={energy:.3f}"
                                f" threshold={threshold:.3f}"
                            )
                    if self._barge_pending.get(sid, False):
                        handler = self._interrupt_handler(sid)
                        threshold = self._barge_thresholds.get(
                            sid,
                            self.interrupt_handler.speech_energy_threshold,
                        )
                        target_frames = self._barge_frames.get(
                            sid,
                            self.interrupt_handler.consecutive_speech_frames,
                        )
                        if handler.should_interrupt(
                            energy,
                            0.0,
                            True,
                            threshold=threshold,
                            target_frames=target_frames,
                        ):
                            self._barge_pending[sid] = False
                            self._barge_thresholds.pop(sid, None)
                            self._barge_frames.pop(sid, None)
                            await self.signal_interrupt(sid)
                            ctx.dialogue_state = DialogueState.LISTENING
                            await self._emit(
                                PipelineEvent.INTERRUPT, session_id=sid
                            )

                turn_decision = self.turn_detector.process_chunk(chunk, True)

                if chunk_count % prosody_update_interval == 0:
                    self._update_engagement_from_prosody(
                        ctx, self.turn_detector.prosody_analyzer.analyze()
                    )

                now = time.time()
                if (
                    now - self._last_partial_time.get(sid, 0.0)
                    >= partial_transcript_interval
                ):
                    self._last_partial_time[sid] = now
                    partial_blob = bytes(speech_buffer)
                    asyncio.create_task(
                        self._partial_transcribe(partial_blob, sid, ctx)
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
                    silence_ms = (
                        self._silence_ms.get(sid, 0.0)
                        + len(chunk) / (self.vad.sample_rate * 2 / 1000)
                    )
                    self._silence_ms[sid] = silence_ms
                    self._interrupt_handler(sid).reset()

                    turn_decision = self.turn_detector.process_chunk(chunk, False)

                    semantic_score = 0.0
                    if ctx.last_partial_transcript:
                        semantic_score = self.turn_detector.classifier._score_linguistic(
                            ctx.last_partial_transcript
                        )

                    adaptive_threshold = 250.0
                    if semantic_score >= 0.8:
                        adaptive_threshold = 80.0
                    elif semantic_score >= 0.6:
                        adaptive_threshold = 120.0
                    elif semantic_score <= 0.2 and len(ctx.last_partial_transcript.split()) > 2:
                        adaptive_threshold = 300.0
                    elif turn_decision == "end_turn_force":
                        adaptive_threshold = 100.0
                    elif turn_decision == "end_turn":
                        adaptive_threshold = 180.0
                    elif ctx.engagement > 0.7:
                        adaptive_threshold = 200.0

                    if silence_ms >= adaptive_threshold:
                        self._speaking[sid] = False
                        self._barge_pending[sid] = False
                        self._barge_thresholds.pop(sid, None)
                        self._barge_frames.pop(sid, None)
                        self._interrupt_handler(sid).reset()
                        audio_blob = bytes(speech_buffer)
                        self._speech_buffers[sid] = bytearray()
                        self.vad.reset()
                        self.turn_detector.reset()

                        ctx.last_turn_duration_ms = len(audio_blob) / (
                            self.vad.sample_rate * 2 / 1000
                        )
                        ctx.turn_count += 1
                        ctx.dialogue_state = DialogueState.PROCESSING

                        await self._emit(PipelineEvent.SPEECH_END, session_id=sid)
                        asyncio.create_task(
                            self._process_speech_segment(audio_blob, sid, ctx)
                        )

    def _detect_topic_shift(self, prev: str | None, cur: str) -> bool:
        if not prev or not cur:
            return False
        prev_words = {w for w in prev.lower().split() if len(w) > 3}
        cur_words = {w for w in cur.lower().split() if len(w) > 3}
        if not cur_words:
            return False
        overlap = len(prev_words & cur_words) / len(cur_words)
        return overlap < 0.2

    def _detect_repetition(self, prev: str | None, cur: str) -> bool:
        if not prev or not cur:
            return False
        prev_words = {w for w in prev.lower().split() if len(w) > 3}
        cur_words = {w for w in cur.lower().split() if len(w) > 3}
        if len(cur_words) < 3:
            return False
        overlap = len(prev_words & cur_words) / len(cur_words)
        return overlap >= 0.6

    def _looks_like_echo(self, session_id: str, transcript: str) -> bool:
        words = [w for w in transcript.lower().split() if len(w) > 2]
        if len(words) < 3:
            return False
        last_words = {
            w for w in self._last_spoken.get(session_id, "").lower().split()
            if len(w) > 2
        }
        if not last_words:
            return False
        overlap = len(set(words) & last_words) / len(words)
        return overlap >= 0.6

    async def _process_speech_segment(
        self, audio_blob: bytes, session_id: str, ctx: ConversationContext
    ) -> None:
        prev = self._current_tasks.get(session_id)
        if prev and not prev.done() and prev is not asyncio.current_task():
            prev.cancel()
            self._playback_active[session_id] = False
            clear_task = self._playback_clear_tasks.pop(session_id, None)
            if clear_task and not clear_task.done():
                clear_task.cancel()
            await self._emit(PipelineEvent.INTERRUPT, session_id=session_id)
        int_ev = self._int_event(session_id)
        int_ev.clear()
        self._current_tasks[session_id] = asyncio.current_task()
        tts_worker: asyncio.Task | None = None
        bc_timer: asyncio.Task | None = None
        stt_task: asyncio.Task | None = None
        spec_task: asyncio.Task | None = None

        try:
            memory = self._memory(session_id)
            retrieval = self._retrieval(session_id)

            # Start STT concurrently
            stt_start = time.perf_counter()
            stt_task = asyncio.create_task(self.stt.transcribe(audio_blob))

            # Speculative LLM: use last partial transcript to start early
            partial = ctx.last_partial_transcript
            spec_full: str | None = None
            spec_task: asyncio.Task | None = None

            if partial and len(partial.split()) >= 2:
                spec_messages = await self._build_messages(
                    partial, ctx, memory, retrieval
                )

                async def _spec_llm():
                    result = ""
                    async for tok in self.llm.generate_stream(spec_messages):
                        if int_ev.is_set():
                            return None
                        result += tok
                    return result

                spec_task = asyncio.create_task(_spec_llm())

            # Wait for STT to finish
            transcript = await stt_task
            self._latency.measure("stt", stt_start)

            if not transcript:
                if spec_task:
                    spec_task.cancel()
                return

            if self._looks_like_echo(session_id, transcript):
                logger.debug(
                    f"dropped echo-looking transcript session={session_id}"
                    f" transcript={transcript!r}"
                )
                if spec_task:
                    spec_task.cancel()
                return

            ctx.topic_shift = self._detect_topic_shift(
                ctx.last_transcript, transcript
            )
            ctx.user_repeated = self._detect_repetition(
                ctx.last_transcript, transcript
            )
            ctx.user_sentiment = classify_sentiment(transcript)
            ctx.last_transcript = transcript
            ctx.is_question = transcript.strip().endswith("?")
            self._log_latency("stt")

            await self._emit(PipelineEvent.FINAL_TRANSCRIPT, transcript, session_id)

            if ctx.is_question:
                ctx.engagement = min(1.0, ctx.engagement + 0.05)
            else:
                ctx.engagement = max(0.1, ctx.engagement - 0.02)

            # Check if speculative LLM result can be reused
            used_speculation = False
            if spec_task and partial and transcript.startswith(partial):
                try:
                    spec_full = await asyncio.wait_for(
                        asyncio.shield(spec_task), timeout=30.0
                    )
                    if spec_full is not None:
                        used_speculation = True
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            if not used_speculation:
                if spec_task and not spec_task.done():
                    spec_task.cancel()

                messages = await self._build_messages(
                    transcript, ctx, memory, retrieval
                )

            if memory:
                memory.add("user", transcript)

            if int_ev.is_set():
                return

            # Compute response timing delay
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
                if int_ev.is_set():
                    return

            # ---- Parallel LLM → chunker → priority queue → TTS worker ----
            chunker = TTSChunker()
            text_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=128)
            seq = itertools.count()
            stop_tts = asyncio.Event()
            tool_calls: list[dict[str, str]] = []
            first_chunk = True

            def _prosody_for(text: str):
                nonlocal first_chunk
                profile = self._prosody.select(
                    text,
                    trajectory=ctx.prosody_trajectory,
                    engagement=ctx.engagement,
                    turn_count=ctx.turn_count,
                    topic_shift=ctx.topic_shift,
                    first=first_chunk,
                    responding_to_question=ctx.is_question,
                    complexity=ctx.query_complexity,
                    user_sentiment=ctx.user_sentiment,
                    user_repeated=ctx.user_repeated,
                    first_response=ctx.turn_count == 0,
                )
                first_chunk = False
                return profile

            async def push(priority: int, text: str) -> None:
                text = sanitize_for_tts(text)
                if not text.strip():
                    return
                await text_queue.put(
                    (priority, next(seq), text, _prosody_for(text))
                )

            def push_nowait(priority: int, text: str) -> None:
                text = sanitize_for_tts(text)
                if not text.strip():
                    return
                text_queue.put_nowait(
                    (priority, next(seq), text, _prosody_for(text))
                )

            tts_worker = asyncio.create_task(
                self._tts_worker(text_queue, session_id, stop_tts)
            )

            full = ""

            if used_speculation:
                full = spec_full or ""
                ctx.dialogue_state = DialogueState.INTERRUPTIBLE
                await self._emit(PipelineEvent.LLM_TOKEN, full, session_id)
                tool_calls = self._tool_registry.find_calls(full)
                if not tool_calls:
                    for c in chunker.feed(full):
                        push_nowait(1, self._tool_registry.strip_calls(c))
                    tail = chunker.flush()
                    if tail:
                        push_nowait(1, self._tool_registry.strip_calls(tail))
            else:
                llm_start = time.perf_counter()
                first_token = True
                tool_marker_seen = False

                bc_timer = asyncio.create_task(
                    self._backchannel_timer(
                        text_queue, ctx, session_id, int_ev, seq
                    )
                )

                async for token in self.llm.generate_stream(messages):
                    if int_ev.is_set():
                        break
                    if first_token:
                        ctx.dialogue_state = DialogueState.INTERRUPTIBLE
                        if bc_timer:
                            bc_timer.cancel()
                        self._latency.measure("llm_first_token", llm_start)
                        self._log_latency("llm_first_token")
                        first_token = False
                    full += token
                    await self._emit(PipelineEvent.LLM_TOKEN, token, session_id)

                    if not tool_marker_seen and "{tool:" in full:
                        tool_marker_seen = True
                        stop_tts.set()
                        await self._drain_queue(text_queue)
                        chunker.reset()
                        continue
                    if tool_marker_seen:
                        continue

                    for c in chunker.feed(token):
                        await push(1, self._tool_registry.strip_calls(c))

                if first_token and bc_timer:
                    bc_timer.cancel()

                if int_ev.is_set():
                    return

                self._latency.measure("llm_full", llm_start)
                self._log_latency("llm_full")

                tool_calls = self._tool_registry.find_calls(full)

            # ---- Tool call handling: stop speech, execute, stream followup ----
            if tool_calls and not int_ev.is_set():
                stop_tts.set()
                await self._drain_queue(text_queue)
                chunker.reset()
                if tts_worker:
                    tts_worker.cancel()
                    try:
                        await tts_worker
                    except (asyncio.CancelledError, Exception):
                        pass

                tool_results = await asyncio.gather(
                    *[self._tool_registry.execute_call(c) for c in tool_calls]
                )
                followup_messages = messages if not used_speculation else spec_messages
                followup_messages = list(followup_messages)
                followup_messages.append({
                    "role": "assistant",
                    "content": full,
                })
                for tr in tool_results:
                    followup_messages.append({
                        "role": "tool",
                        "content": f"{tr['tool']} result: {tr['result']}",
                    })
                followup_messages.append({
                    "role": "user",
                    "content": "Continue naturally with the tool results.",
                })

                full = ""
                stop_tts = asyncio.Event()
                tts_worker = asyncio.create_task(
                    self._tts_worker(text_queue, session_id, stop_tts)
                )

                llm_start = time.perf_counter()
                first_token = True
                async for token in self.llm.generate_stream(followup_messages):
                    if int_ev.is_set():
                        break
                    if first_token:
                        first_token = False
                    full += token
                    await self._emit(PipelineEvent.LLM_TOKEN, token, session_id)
                    for c in chunker.feed(token):
                        await push(1, self._tool_registry.strip_calls(c))

                if int_ev.is_set():
                    return

                self._latency.measure("llm_full", llm_start)
                self._log_latency("llm_full")

            # Flush any remaining partial chunk
            tail = chunker.flush()
            if tail:
                await push(1, self._tool_registry.strip_calls(tail))

            await self._emit(PipelineEvent.LLM_DONE, full, session_id)

            # End-of-stream sentinel (lowest priority: drained last)
            text_queue.put_nowait((2, next(seq), None, None))

            if tts_worker:
                await tts_worker

            if memory:
                memory.add("assistant", full)
            if retrieval:
                asyncio.create_task(
                    asyncio.to_thread(retrieval.add_to_long_term, full)
                )

            if not int_ev.is_set():
                ctx.dialogue_state = DialogueState.IDLE
                await self._emit(PipelineEvent.TTS_DONE, session_id=session_id)

        except asyncio.CancelledError:
            for child in (tts_worker, bc_timer, stt_task, spec_task):
                if child:
                    child.cancel()
            raise
        except Exception as e:
            logger.error(f"processing error session={session_id} error={e}")
            for child in (tts_worker, bc_timer, stt_task, spec_task):
                if child:
                    child.cancel()
            await self._emit(PipelineEvent.ERROR, str(e), session_id)
        finally:
            if self._current_tasks.get(session_id) is asyncio.current_task():
                self._current_tasks.pop(session_id, None)

    @staticmethod
    async def _drain_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _backchannel_timer(
        self,
        text_queue: asyncio.PriorityQueue,
        ctx: ConversationContext,
        session_id: str,
        int_ev: asyncio.Event,
        seq: itertools.count,
        delay: float = 0.5,
    ) -> None:
        """If the LLM hasn't produced a token within `delay`, speak a filler."""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if int_ev.is_set():
            return
        text = self.turn_backchannel.generator.generate_thinking()
        if text:
            text_queue.put_nowait((0, next(seq), text, None))
            logger.debug(f"thinking backchannel session={session_id} text={text}")

    async def _tts_worker(
        self,
        text_queue: asyncio.PriorityQueue,
        session_id: str,
        stop_tts: asyncio.Event,
    ) -> None:
        """Consume text chunks from the priority queue and synthesize audio."""
        int_ev = self._int_event(session_id)
        sr = self.tts.sample_rate
        total_bytes = 0
        first_emit: float | None = None
        spoken: list[str] = []

        async def _one(text: str) -> AsyncGenerator[str, None]:
            yield text

        try:
            while True:
                try:
                    _, _, text, prosody = await asyncio.wait_for(
                        text_queue.get(), timeout=0.2
                    )
                except asyncio.TimeoutError:
                    if int_ev.is_set() or stop_tts.is_set():
                        await self._drain_queue(text_queue)
                        break
                    continue

                if text is None:
                    break
                if int_ev.is_set() or stop_tts.is_set():
                    await self._drain_queue(text_queue)
                    break
                spoken.append(text)

                try:
                    async for audio_chunk in self.tts.synthesize_stream(
                        _one(text), prosody=prosody
                    ):
                        if int_ev.is_set() or stop_tts.is_set():
                            break
                        if isinstance(audio_chunk, bytes) and len(audio_chunk) > 0:
                            if first_emit is None:
                                first_emit = time.monotonic()
                                self._playback_onset[session_id] = first_emit
                                self._echo_floor[session_id] = 0.0
                            total_bytes += len(audio_chunk)
                            energy = rms_energy(audio_chunk) / 32768.0
                            self._echo_floor[session_id] = max(
                                self._echo_floor.get(session_id, 0.0), energy
                            )
                            self._playback_active[session_id] = True
                            wav = self._pcm_to_wav(audio_chunk, sr)
                            await self._emit(
                                PipelineEvent.TTS_CHUNK, wav, session_id
                            )
                except Exception as e:
                    logger.error(
                        f"tts chunk error session={session_id} error={e}"
                    )
        finally:
            if (
                total_bytes > 0
                and not int_ev.is_set()
                and first_emit is not None
            ):
                self._schedule_playback_clear(
                    session_id,
                    first_emit + total_bytes / (sr * 2) - time.monotonic(),
                )
            if not int_ev.is_set() and spoken:
                self._last_spoken[session_id] = " ".join(spoken)

    def _schedule_playback_clear(self, session_id: str, delay: float) -> None:
        """Mark the client playback window as over after the audio finishes."""
        existing = self._playback_clear_tasks.get(session_id)
        if existing and not existing.done():
            existing.cancel()
        delay = max(0.0, delay) + 0.5

        async def _clear() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            self._playback_active[session_id] = False
            if self._playback_clear_tasks.get(session_id) is asyncio.current_task():
                self._playback_clear_tasks.pop(session_id, None)

        self._playback_clear_tasks[session_id] = asyncio.create_task(_clear())

    async def _partial_transcribe(
        self, audio_blob: bytes, session_id: str, ctx: ConversationContext
    ) -> None:
        text = await self.stt.transcribe(audio_blob)
        if text and text != ctx.last_partial_transcript:
            ctx.last_partial_transcript = text
            await self._emit(PipelineEvent.PARTIAL_TRANSCRIPT, text, session_id)

    def _classify_query_complexity(self, text: str) -> str:
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())

        greeting_words = {"hi", "hello", "hey", "yo", "sup", "good morning",
                          "good afternoon", "good evening", "howdy"}
        if word_count <= 3 and any(g in text_lower for g in greeting_words):
            return "simple"

        if word_count <= 2:
            return "simple"

        code_indicators = {"code", "function", "script", "program", "debug",
                           "error", "exception", "syntax", "algorithm"}
        if any(w in text_lower for w in code_indicators):
            return "complex"

        complex_indicators = {"explain", "compare", "contrast", "analyze",
                              "why", "how does", "summarize", "difference between"}
        if any(w in text_lower for w in complex_indicators):
            return "complex"

        if word_count > 20:
            return "complex"

        return "standard"

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

        complexity = self._classify_query_complexity(transcript)
        ctx.query_complexity = complexity

        system_prompt = build_system_prompt(
            engagement=ctx.engagement,
            turn_count=ctx.turn_count,
            has_context=has_context,
            complexity=complexity,
        )

        tool_block = self._tool_registry.system_prompt_block()
        if tool_block:
            system_prompt += tool_block

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
