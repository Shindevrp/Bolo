from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncGenerator

from modules.tts.chunker import TTSChunker
from modules.tts.sanitize import sanitize_for_tts
from modules.tts.prosody import ProsodySelector, classify_sentiment
from modules.emotion.classifier import EmotionClassifier
from providers.stt.base import STTProvider
from providers.llm.base import LLMProvider
from providers.tts.base import TTSProvider
from modules.vad.silero_vad import SileroVAD
from modules.turn.detector import TurnDetector
from modules.turn.interrupt import InterruptHandler
from modules.turn.timing import TurnTiming
from modules.turn.topic import TopicTracker
from modules.turn.intent import IntentClassifier
from modules.turn.backchannel import TurnBackchannel
from modules.backchannel.generator import BackchannelGenerator
from modules.backchannel.timing import BackchannelTiming
from modules.memory.session import SessionMemory
from modules.memory.retrieval import RetrievalModule
from modules.memory.compression import ContextCompressor
from modules.memory.facts import (
    EXTRACTOR_SYSTEM_PROMPT,
    Fact,
    FactMemory,
)
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

# Fixed pipeline frame: 128ms at 16k mono 16-bit. All incoming audio is
# segmented into these frames so VAD/turn logic sees uniform windows and the
# SileroVAD hidden state decays across trailing-silence frames (otherwise a
# large silence chunk can be misclassified as speech and swallow the turn end).
FRAME_BYTES = 4096

# Energy floor (normalized 0-1) below which a frame is treated as silence even
# if the VAD reports speech. Guards against the VAD RNN carrying its hidden
# state over trailing-silence windows and never firing speech_end. Real piper
# speech frames measure ~0.05-0.28 RMS; digital silence measures 0.0.
SILENCE_ENERGY_FLOOR = 0.003


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
    PROSODY = auto()
    INTERRUPT = auto()
    RESPONSE_DELAY = auto()
    ERROR = auto()


@dataclass
class PipelineMessage:
    event: PipelineEvent
    data: str | bytes | None = None
    session_id: str = "default"
    generation: int = 0


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
    topic: str = ""
    topic_since_turn: int = 0
    intent: str = "statement"
    prev_intent: str = ""
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
        emotion_classifier: EmotionClassifier | None = None,
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
        self._emotion = emotion_classifier or EmotionClassifier(enabled=False)
        self._topic_trackers: dict[str, TopicTracker] = {}
        self._topic_label_tasks: dict[str, asyncio.Task] = {}
        self._compression_tasks: dict[str, asyncio.Task] = {}
        self.compressor = ContextCompressor()
        self.compress_at_tokens = 1500
        self.compress_batch = 8
        self.intent_classifier = IntentClassifier()

        self._latency = LatencyTracker()
        self._metrics = MetricsLogger()
        self._tool_registry = get_builtin_tools()
        self._audio_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        self._output_queue: asyncio.Queue[PipelineMessage] = asyncio.Queue(512)
        # Per-session output queues. The global _output_queue is shared and a
        # single pump_output per client would race other clients for the same
        # messages (dropping events for unrelated sessions). Each websocket /
        # webrtc session consumes its OWN queue so audio and events can never
        # be stolen by another client.
        self._session_outputs: dict[str, asyncio.Queue[PipelineMessage]] = {}
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
        self._frame_buffers: dict[str, bytearray] = {}
        self._silence_ms: dict[str, float] = {}
        self._last_partial_time: dict[str, float] = {}
        self._last_refresh_ms: dict[str, float] = {}
        self._partial_inflight: dict[str, bool] = {}
        self._barge_pending: dict[str, bool] = {}
        self._barge_thresholds: dict[str, float] = {}
        self._barge_frames: dict[str, int] = {}
        self._low_energy_frames: dict[str, int] = {}
        self._interrupt_handlers: dict[str, InterruptHandler] = {}
        self._generations: dict[str, int] = {}
        self._tts_workers: dict[str, asyncio.Task] = {}
        self._prosody_meta: dict[str, dict[str, str]] = {}
        self._running = False
        self._contexts: dict[str, ConversationContext] = {}
        self._memories: dict[str, SessionMemory] = {}
        self._retrievals: dict[str, RetrievalModule] = {}
        self._facts: dict[str, FactMemory] = {}
        self._last_fact_extract: dict[str, float] = {}
        self._facts_llm_enabled = True
        self._facts_llm_timeout = 8.0
        self._facts_min_interval = 30.0
        self._user_profiles: dict[str, Any] = {}

    def _int_event(self, session_id: str) -> asyncio.Event:
        if session_id not in self._interrupt_events:
            self._interrupt_events[session_id] = asyncio.Event()
        return self._interrupt_events[session_id]

    def _generation(self, session_id: str) -> int:
        return self._generations.get(session_id, 0)

    def _bump_generation(self, session_id: str) -> None:
        self._generations[session_id] = self._generation(session_id) + 1

    def stale_event(self, session_id: str, msg: PipelineMessage) -> bool:
        """True when msg belongs to an interrupted, superseded generation."""
        return msg.generation != self._generation(session_id)

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

    def _on_pipeline_loop_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.exception(f"pipeline loop crashed: {exc!r}")

    async def start(self) -> None:
        self._running = True
        loop_task = asyncio.create_task(self._pipeline_loop(), name="pipeline")
        loop_task.add_done_callback(self._on_pipeline_loop_done)
        self._tasks = [loop_task]
        logger.info("pipeline started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for task in self._topic_label_tasks.values():
            task.cancel()
        if self._topic_label_tasks:
            await asyncio.gather(
                *self._topic_label_tasks.values(), return_exceptions=True
            )
        self._topic_label_tasks.clear()
        for task in self._compression_tasks.values():
            task.cancel()
        if self._compression_tasks:
            await asyncio.gather(
                *self._compression_tasks.values(), return_exceptions=True
            )
        self._compression_tasks.clear()
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
        self._low_energy_frames.clear()
        self._interrupt_handlers.clear()
        logger.info("pipeline stopped")

    async def push_audio(self, chunk: bytes, session_id: str = "default") -> None:
        try:
            buf = self._frame_buffers.setdefault(session_id, bytearray())
            buf.extend(chunk)
            while len(buf) >= FRAME_BYTES:
                frame = bytes(buf[:FRAME_BYTES])
                del buf[:FRAME_BYTES]
                await self._audio_queue.put(
                    PipelineMessage(PipelineEvent.AUDIO_CHUNK, frame, session_id)
                )
        except asyncio.QueueFull:
            logger.warning("audio queue full, dropping chunk")

    def register_session(
        self,
        session_id: str,
        memory: SessionMemory,
        retrieval: RetrievalModule,
        facts: FactMemory | None = None,
    ) -> None:
        self._memories[session_id] = memory
        self._retrievals[session_id] = retrieval
        self._facts[session_id] = facts or FactMemory()
        self._session_output(session_id)
        try:
            asyncio.create_task(
                asyncio.to_thread(retrieval.warm_up)
            )
        except RuntimeError:
            pass

    def unregister_session(self, session_id: str) -> None:
        self._memories.pop(session_id, None)
        self._retrievals.pop(session_id, None)
        self._facts.pop(session_id, None)
        # Discard the session's output queue so a dead client's backlog can't
        # leak into anything else. _session_output() lazily recreates it if the
        # session id is reused.
        self._session_outputs.pop(session_id, None)
        self._last_fact_extract.pop(session_id, None)
        self._contexts.pop(session_id, None)
        self._topic_trackers.pop(session_id, None)
        label_task = self._topic_label_tasks.pop(session_id, None)
        if label_task and not label_task.done():
            label_task.cancel()
        compress_task = self._compression_tasks.pop(session_id, None)
        if compress_task and not compress_task.done():
            compress_task.cancel()
        self._interrupt_events.pop(session_id, None)
        self._current_tasks.pop(session_id, None)
        self._generations.pop(session_id, None)
        self._prosody_meta.pop(session_id, None)
        tts_worker = self._tts_workers.pop(session_id, None)
        if tts_worker and not tts_worker.done():
            tts_worker.cancel()
        self._playback_active.pop(session_id, None)
        self._playback_onset.pop(session_id, None)
        self._echo_floor.pop(session_id, None)
        self._speaking.pop(session_id, None)
        self._speech_buffers.pop(session_id, None)
        self._frame_buffers.pop(session_id, None)
        self._silence_ms.pop(session_id, None)
        self._last_partial_time.pop(session_id, None)
        self._last_refresh_ms.pop(session_id, None)
        self._partial_inflight.pop(session_id, None)
        self._barge_pending.pop(session_id, None)
        self._barge_thresholds.pop(session_id, None)
        self._barge_frames.pop(session_id, None)
        self._low_energy_frames.pop(session_id, None)
        self._interrupt_handlers.pop(session_id, None)
        clear_task = self._playback_clear_tasks.pop(session_id, None)
        if clear_task and not clear_task.done():
            clear_task.cancel()

    async def signal_interrupt(self, session_id: str = "default") -> None:
        ev = self._int_event(session_id)
        ev.set()
        self._bump_generation(session_id)

        # Cancel the in-flight generation (LLM stream + its children). We do
        # NOT await it here: it must unwind in the background so that the
        # pipeline loop can return to listening immediately and capture the
        # interruption utterance without stalling.
        task = self._current_tasks.get(session_id)
        if task and not task.done():
            task.cancel()

        # Cancel the TTS worker and let it unwind in the background. Awaiting
        # it would block the pipeline loop for as long as Piper takes to abort
        # the current (CPU-bound, off-thread) synthesis chunk, which would
        # freeze barge-in detection and the capture of the new user speech.
        tts_worker = self._tts_workers.get(session_id)
        if tts_worker and not tts_worker.done():
            tts_worker.cancel()
            self._reap_tts_worker(tts_worker)

        # Clear playback bookkeeping synchronously (plain dict work, no IO).
        self._playback_active[session_id] = False
        self._playback_onset.pop(session_id, None)
        self._echo_floor.pop(session_id, None)
        clear_task = self._playback_clear_tasks.pop(session_id, None)
        if clear_task and not clear_task.done():
            clear_task.cancel()
        logger.info(f"interrupt signaled for session {session_id}")

    def _reap_tts_worker(self, worker: asyncio.Task) -> None:
        """Await a cancelled TTS worker in the background so it can't block
        the pipeline loop, while still collecting its (cancelled) exception."""

        async def _reap() -> None:
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass

        asyncio.create_task(_reap())

    async def output_stream(self) -> AsyncGenerator[PipelineMessage, None]:
        """Legacy global output stream (used by the CLI)."""
        while self._running or not self._output_queue.empty():
            try:
                msg = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield msg
            except asyncio.TimeoutError:
                continue

    def _session_output(self, session_id: str) -> asyncio.Queue[PipelineMessage]:
        q = self._session_outputs.get(session_id)
        if q is None:
            q = asyncio.Queue(maxsize=2048)
            self._session_outputs[session_id] = q
        return q

    async def output_stream_for(
        self, session_id: str
    ) -> AsyncGenerator[PipelineMessage, None]:
        """Per-session output stream — each client gets its own queue so no
        other session can consume (and drop) its events or audio."""
        q = self._session_output(session_id)
        while self._running or not q.empty():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.1)
                yield msg
            except asyncio.TimeoutError:
                continue

    def _ctx(self, session_id: str) -> ConversationContext:
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext()
        return self._contexts[session_id]

    def context(self, session_id: str) -> ConversationContext:
        """Public accessor for a session's live conversation context."""
        return self._ctx(session_id)

    def _memory(self, session_id: str) -> SessionMemory | None:
        return self._memories.get(session_id)

    def _retrieval(self, session_id: str) -> RetrievalModule | None:
        return self._retrievals.get(session_id)

    def _topic_tracker(self, session_id: str) -> TopicTracker:
        if session_id not in self._topic_trackers:
            self._topic_trackers[session_id] = TopicTracker()
        return self._topic_trackers[session_id]

    def _maybe_label_topic(self, session_id: str) -> None:
        tracker = self._topic_tracker(session_id)
        if not tracker.needs_label():
            return
        if session_id in self._topic_label_tasks:
            return
        start_turn = (
            tracker.current.start_turn if tracker.current is not None else None
        )
        if start_turn is None:
            return
        self._topic_label_tasks[session_id] = asyncio.create_task(
            self._label_current_topic(session_id, start_turn)
        )

    def _maybe_compress(self, session_id: str) -> None:
        if session_id in self._compression_tasks:
            return
        memory = self._memory(session_id)
        if not memory:
            return
        if memory.token_estimate() <= self.compress_at_tokens:
            return
        batch = memory.entries_snapshot_oldest(self.compress_batch)
        if not batch:
            return
        self._compression_tasks[session_id] = asyncio.create_task(
            self._compress_history(session_id, memory, batch)
        )

    async def _compress_history(
        self,
        session_id: str,
        memory: SessionMemory,
        batch: list,
    ) -> None:
        try:
            new_summary = await self.compressor.compress(
                self.llm, memory.summary, batch
            )
            if new_summary:
                memory.fold_summary(new_summary, batch)
                logger.debug(
                    f"context compressed session={session_id}"
                    f" turns={len(batch)} chars={len(new_summary)}"
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"context compression failed session={session_id}: {e}")
        finally:
            self._compression_tasks.pop(session_id, None)

    async def _label_current_topic(
        self, session_id: str, start_turn: int
    ) -> None:
        tracker = self._topic_tracker(session_id)
        try:
            raw = tracker.topic
            if not raw:
                return
            label_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You label conversation topics. Given a list of "
                        "keywords, reply with only a short 1-4 word "
                        "human-friendly label. No punctuation or explanation."
                    ),
                },
                {"role": "user", "content": f"Keywords: {raw}"},
            ]
            label = ""
            async for tok in self.llm.generate_stream(label_prompt):
                if len(label) >= 60:
                    break
                label += tok
            label = label.strip()
            if label and tracker.current is not None:
                if tracker.current.start_turn == start_turn:
                    tracker.set_label(label)
                    logger.debug(
                        f"topic labeled session={session_id} label={label!r}"
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"topic labeling failed session={session_id}: {e}")
        finally:
            self._topic_label_tasks.pop(session_id, None)

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
                get_start = time.monotonic()
                msg = await self._audio_queue.get()
                get_wait = time.monotonic() - get_start
            except asyncio.CancelledError:
                break
            if get_wait > 1.0:
                logger.debug(
                    f"loop get() waited {get_wait:.1f}s "
                    f"(qsize={self._audio_queue.qsize()})"
                )
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

            frame_energy = rms_energy(chunk) / 32768.0
            if self._playback_active.get(sid, False):
                floor = self._echo_floor.get(sid, 0.0)
                if not is_speech:
                    if frame_energy > floor:
                        floor = 0.6 * floor + 0.4 * frame_energy
                    else:
                        floor = 0.85 * floor + 0.15 * frame_energy
                    self._echo_floor[sid] = max(0.0, min(floor, 0.25))
            if is_speech and frame_energy < SILENCE_ENERGY_FLOOR:
                low_count = self._low_energy_frames.get(sid, 0) + 1
                self._low_energy_frames[sid] = low_count
                if low_count >= 2:
                    is_speech = False
            else:
                self._low_energy_frames[sid] = 0

            logger.debug(
                f"loop sid={sid} chunk={chunk_count} nbytes={len(chunk)} "
                f"is_speech={is_speech} speaking={is_speaking}"
            )

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
                        await self._maybe_fire_barge(sid, ctx, frame_energy)
                    self._speaking[sid] = True
                    ctx.dialogue_state = DialogueState.LISTENING
                    self._silence_ms[sid] = 0.0
                    self._low_energy_frames[sid] = 0
                    self._speech_buffers[sid] = bytearray(chunk)
                    speech_buffer = self._speech_buffers[sid]
                    self._last_partial_time[sid] = time.time()
                    # Discard the previous utterance's partial transcript so
                    # turn-end detection and speculative LLM for this new
                    # utterance never reuse stale text from the prior turn.
                    ctx.last_partial_transcript = ""
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
                        and (
                            ctx.dialogue_state
                            == DialogueState.INTERRUPTIBLE
                            or playback_on
                        )
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
                        await self._maybe_fire_barge(sid, ctx, energy)

                turn_decision = self.turn_detector.process_chunk(chunk, True)

                if chunk_count % prosody_update_interval == 0:
                    self._update_engagement_from_prosody(
                        ctx, self.turn_detector.prosody_analyzer.analyze()
                    )

                now = time.time()
                if (
                    now - self._last_partial_time.get(sid, 0.0)
                    >= partial_transcript_interval
                    and not self._partial_inflight.get(sid, False)
                ):
                    self._last_partial_time[sid] = now
                    self._partial_inflight[sid] = True
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
                    incomplete = False
                    if ctx.last_partial_transcript:
                        classifier = self.turn_detector.classifier
                        semantic_score = classifier._score_linguistic(
                            ctx.last_partial_transcript
                        )
                        incomplete = classifier.is_incomplete(
                            ctx.last_partial_transcript
                        )

                    # The user has finished a list/enumeration which we suspect
                    # is incomplete, OR the accumulated speech is very short.
                    # Refresh the partial before we commit to ending the turn so
                    # we don't cut off e.g. "Is it color, function," mid-stream.
                    if (
                        silence_ms >= 180.0
                        and incomplete
                        and len(speech_buffer) > 0
                        and silence_ms - self._last_refresh_ms.get(sid, 0.0) >= 150.0
                    ):
                        self._last_refresh_ms[sid] = silence_ms
                        fresh = await self._partial_transcribe(
                            bytes(speech_buffer), sid, ctx, sync=True
                        )
                        if fresh is not None:
                            semantic_score = self.turn_detector.classifier._score_linguistic(
                                fresh
                            )
                            incomplete = self.turn_detector.classifier.is_incomplete(
                                fresh
                            )

                    adaptive_threshold = 250.0
                    if incomplete:
                        # Open list / trailing connective: wait for the speaker
                        # to finish the series before responding.
                        adaptive_threshold = 400.0
                    elif semantic_score >= 0.8:
                        adaptive_threshold = 200.0
                    elif semantic_score >= 0.6:
                        adaptive_threshold = 220.0
                    elif semantic_score <= 0.2 and len(ctx.last_partial_transcript.split()) > 2:
                        adaptive_threshold = 300.0
                    elif turn_decision == "end_turn_force":
                        adaptive_threshold = 200.0
                    elif turn_decision == "end_turn":
                        adaptive_threshold = 220.0
                    elif ctx.engagement > 0.7:
                        adaptive_threshold = 220.0

                    if silence_ms >= adaptive_threshold:
                        self._speaking[sid] = False
                        self._barge_pending[sid] = False
                        self._barge_thresholds.pop(sid, None)
                        self._barge_frames.pop(sid, None)
                        self._interrupt_handler(sid).reset()
                        self._last_refresh_ms.pop(sid, None)
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

    @staticmethod
    def _looks_like_noise(transcript: str) -> bool:
        """True when the transcript contains no letter-bearing word (pure
        punctuation/symbols/digit-only tokens), i.e. STT fired on noise rather
        than real speech and we should not respond to it."""
        if not transcript:
            return True
        words = [w for w in transcript.split() if any(ch.isalpha() for ch in w)]
        return not words

    async def _maybe_fire_barge(
        self, sid: str, ctx: ConversationContext, energy: float
    ) -> None:
        if not self._barge_pending.get(sid, False):
            return
        handler = self._interrupt_handler(sid)
        threshold = self._barge_thresholds.get(
            sid, self.interrupt_handler.speech_energy_threshold
        )
        target_frames = self._barge_frames.get(
            sid, self.interrupt_handler.consecutive_speech_frames
        )
        if self._playback_active.get(sid, False):
            threshold = max(
                threshold, self._echo_floor.get(sid, 0.0) * ECHO_FLOOR_MARGIN
            )
        if handler.should_interrupt(
            energy, 0.0, True, threshold=threshold, target_frames=target_frames
        ):
            self._barge_pending[sid] = False
            self._barge_thresholds.pop(sid, None)
            self._barge_frames.pop(sid, None)
            await self.signal_interrupt(sid)
            ctx.dialogue_state = DialogueState.LISTENING
            await self._emit(PipelineEvent.INTERRUPT, session_id=sid)

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
        self._bump_generation(session_id)
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
            eos_ts = time.perf_counter()
            stt_start = time.perf_counter()
            stt_task = asyncio.create_task(self.stt.transcribe(audio_blob))

            # Speculative LLM: use last partial transcript to start early
            partial = ctx.last_partial_transcript
            spec_full: str | None = None
            spec_task: asyncio.Task | None = None

            if partial and len(partial.split()) >= 2:
                spec_messages = await self._build_messages(
                    partial, ctx, memory, retrieval,
                    facts=self._facts.get(session_id),
                    session_id=session_id,
                    user_repeated=ctx.user_repeated,
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

            if self._looks_like_noise(transcript):
                logger.debug(
                    f"dropped noise-looking transcript session={session_id}"
                    f" transcript={transcript!r}"
                )
                if spec_task:
                    spec_task.cancel()
                return

            topic_change = self._topic_tracker(session_id).update(
                transcript, ctx.turn_count
            )
            ctx.topic_shift = topic_change.shift
            ctx.topic = topic_change.topic
            ctx.topic_since_turn = self._topic_tracker(session_id).since_turn
            self._maybe_label_topic(session_id)
            ctx.user_repeated = self._detect_repetition(
                ctx.last_transcript, transcript
            )
            ctx.user_sentiment = classify_sentiment(transcript)
            ctx.prev_intent = ctx.intent
            ctx.intent = self.intent_classifier.classify(
                transcript, ctx.prev_intent
            )
            emotion_task = asyncio.create_task(
                self._emotion.classify_async(transcript)
            )
            # If we time out below and shield-cancel, swallow any late
            # exception so it never surfaces as "exception never retrieved".
            emotion_task.add_done_callback(lambda t: t.exception())
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
            can_reuse_speculation = ctx.intent != "correction"
            if (
                can_reuse_speculation
                and spec_task
                and partial
                and transcript.startswith(partial)
            ):
                try:
                    spec_full = await asyncio.wait_for(
                        asyncio.shield(spec_task), timeout=30.0
                    )
                    if spec_full is not None:
                        used_speculation = True
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    # This task was cancelled by a new turn starting — propagate
                    # so the old turn aborts instead of continuing and producing
                    # a second response.
                    raise

            if not used_speculation:
                if spec_task and not spec_task.done():
                    spec_task.cancel()

                messages = await self._build_messages(
                    transcript, ctx, memory, retrieval,
                    facts=self._facts.get(session_id),
                    session_id=session_id,
                    user_repeated=ctx.user_repeated,
                )

            if memory:
                memory.add("user", transcript)

            facts = self._facts.get(session_id)
            if facts is not None:
                facts.advance_turn()
                facts.add_all(facts.extract(transcript))
                self._schedule_llm_facts(session_id, transcript, facts)

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

            # Let the emotion classifier finish concurrently (0.5s cap);
            # the lexicon fallback already set above wins on timeout/error.
            if emotion_task:
                try:
                    ctx.user_sentiment = await asyncio.wait_for(
                        asyncio.shield(emotion_task),
                        timeout=self._emotion.timeout,
                    )
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

            # ---- Parallel LLM → chunker → priority queue → TTS worker ----
            chunker = TTSChunker()
            text_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=128)
            seq = itertools.count()
            stop_tts = asyncio.Event()
            tool_calls: list[dict[str, str]] = []
            first_chunk = True
            self._prosody_meta.pop(session_id, None)

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
                    intent=ctx.intent,
                    first_response=ctx.turn_count == 0,
                )
                if session_id not in self._prosody_meta:
                    self._prosody_meta[session_id] = {
                        "label": profile.label,
                        "emotion": ctx.user_sentiment,
                    }
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
                self._tts_worker(text_queue, session_id, stop_tts, eos_ts)
            )
            self._tts_workers[session_id] = tts_worker

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

                # Normal single-speaker streaming path
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

                followup_messages = messages if not used_speculation else spec_messages
                followup_messages = list(followup_messages)
                followup_messages.append({
                    "role": "assistant",
                    "content": full,
                })

                # Bounded tool round-trip loop: execute, feed results back, and
                # stream the LLM's final answer. If the LLM (incorrectly) emits
                # another tool call in the follow-up, execute it too — up to a
                # hard cap so a misbehaving model can't loop forever.
                for _round in range(3):
                    if int_ev.is_set():
                        return

                    tool_results = await asyncio.gather(
                        *[self._tool_registry.execute_call_with_retry(c) for c in tool_calls]
                    )
                    for tr in tool_results:
                        followup_messages.append({
                            "role": "tool",
                            "content": f"{tr['tool']} result: {tr['result']}",
                        })
                    if any(tr.get("failed") for tr in tool_results):
                        followup_messages.append({
                            "role": "user",
                            "content": (
                                "The tool failed and the system already retried. "
                                "Do NOT invent data, do NOT call another tool, and do NOT "
                                "improvise a fallback. Briefly tell the user the "
                                "information is currently unavailable and continue naturally."
                            ),
                        })
                    else:
                        followup_messages.append({
                            "role": "user",
                            "content": "Continue naturally with the tool results.",
                        })

                    full = ""
                    stop_tts = asyncio.Event()
                    tts_worker = asyncio.create_task(
                        self._tts_worker(text_queue, session_id, stop_tts, eos_ts)
                    )
                    self._tts_workers[session_id] = tts_worker

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

                    # If the follow-up still contains a tool call, execute it
                    # next round. Otherwise we're done streaming the answer.
                    next_calls = self._tool_registry.find_calls(full)
                    if not next_calls:
                        break
                    followup_messages.append({
                        "role": "assistant",
                        "content": full,
                    })
                    tool_calls = next_calls
                    stop_tts.set()
                    await self._drain_queue(text_queue)
                    chunker.reset()
                    if tts_worker:
                        tts_worker.cancel()
                        try:
                            await tts_worker
                        except (asyncio.CancelledError, Exception):
                            pass

            # Flush any remaining partial chunk
            tail = chunker.flush()
            if tail:
                await push(1, self._tool_registry.strip_calls(tail))

            # Emit LLM done so clients finalize the turn and memory is recorded.
            await self._emit(PipelineEvent.LLM_DONE, full, session_id)

            # End-of-stream sentinel (lowest priority: drained last)
            text_queue.put_nowait((2, next(seq), None, None))

            if tts_worker:
                await tts_worker

            if memory:
                memory.add("assistant", full)
            if retrieval:
                asyncio.create_task(
                    asyncio.to_thread(
                        retrieval.add_to_long_term, full, ctx.topic
                    )
                )
            self._maybe_compress(session_id)

            if not int_ev.is_set():
                ctx.dialogue_state = DialogueState.IDLE
                if eos_ts is not None:
                    self._latency.measure("tts_done", eos_ts)
                    self._log_latency("tts_done")
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
            self._tts_workers.pop(session_id, None)

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
        eos_ts: float | None = None,
    ) -> None:
        """Consume text chunks from the priority queue and synthesize audio."""
        int_ev = self._int_event(session_id)
        generation = self._generation(session_id)
        sr = self.tts.sample_rate
        total_bytes = 0
        first_emit: float | None = None
        spoken: list[str] = []
        # Set whenever this worker bails due to an interrupt / stop. The
        # session's int_ev can be re-cleared by a later turn before this
        # finally runs, so we track interruption locally to avoid spurious
        # side effects (playback-clear / last_spoken) from stale workers.
        interrupted = False

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
                        interrupted = True
                        await self._drain_queue(text_queue)
                        break
                    continue

                if text is None:
                    break
                if int_ev.is_set() or stop_tts.is_set():
                    interrupted = True
                    await self._drain_queue(text_queue)
                    break
                spoken.append(text)

                try:
                    async for audio_chunk in self.tts.synthesize_stream(
                        _one(text), prosody=prosody
                    ):
                        if int_ev.is_set() or stop_tts.is_set():
                            interrupted = True
                            break
                        if isinstance(audio_chunk, bytes) and len(audio_chunk) > 0:
                            if first_emit is None:
                                first_emit = time.monotonic()
                                if eos_ts is not None:
                                    self._latency.measure("tts_first", eos_ts)
                                    self._log_latency("tts_first")
                                self._playback_onset[session_id] = first_emit
                                self._echo_floor.setdefault(session_id, 0.0)
                                meta = self._prosody_meta.get(
                                    session_id, {}
                                )
                                await self._emit(
                                    PipelineEvent.PROSODY,
                                    json.dumps(meta),
                                    session_id,
                                    generation=generation,
                                )
                            total_bytes += len(audio_chunk)
                            self._playback_active[session_id] = True
                            wav = self._pcm_to_wav(audio_chunk, sr)
                            await self._emit(
                                PipelineEvent.TTS_CHUNK, wav, session_id,
                                generation=generation,
                            )
                except Exception as e:
                    logger.error(
                        f"tts chunk error session={session_id} error={e}"
                    )
        finally:
            if (
                total_bytes > 0
                and not interrupted
                and first_emit is not None
            ):
                self._schedule_playback_clear(
                    session_id,
                    first_emit + total_bytes / (sr * 2) - time.monotonic(),
                )
            if not interrupted and spoken:
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
        self,
        audio_blob: bytes,
        session_id: str,
        ctx: ConversationContext,
        sync: bool = False,
    ) -> str | None:
        try:
            text = await self.stt.transcribe(audio_blob)
            if text and text != ctx.last_partial_transcript:
                ctx.last_partial_transcript = text
                await self._emit(PipelineEvent.PARTIAL_TRANSCRIPT, text, session_id)
                return text
            if sync:
                return text or None
            return None
        finally:
            # Background partials set the in-flight guard; clear it so the
            # next partial can be scheduled once the transcript is ready.
            if not sync:
                self._partial_inflight.pop(session_id, None)

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

    def _schedule_llm_facts(
        self, session_id: str, transcript: str, facts: FactMemory
    ) -> None:
        """Fire-and-forget LLM fact extraction, rate-limited per session.

        vLLM batches concurrent requests, so a background extraction call
        never blocks the main turn. Guarded for LLM providers that lack a
        non-streaming generate().
        """
        if not self._facts_llm_enabled or not hasattr(self.llm, "generate"):
            return
        if transcript.strip().endswith("?"):
            return
        now = time.monotonic()
        if now - self._last_fact_extract.get(session_id, 0.0) < (
            self._facts_min_interval
        ):
            return
        self._last_fact_extract[session_id] = now
        asyncio.create_task(
            self._extract_facts_llm(session_id, transcript, facts)
        )

    async def _extract_facts_llm(
        self, session_id: str, transcript: str, facts: FactMemory
    ) -> None:
        try:
            raw = await asyncio.wait_for(
                asyncio.shield(
                    self.llm.generate(
                        [
                            {
                                "role": "system",
                                "content": EXTRACTOR_SYSTEM_PROMPT,
                            },
                            {"role": "user", "content": transcript},
                        ]
                    )
                ),
                timeout=self._facts_llm_timeout,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed or []:
                key = str(item.get("key", "")).strip().lower()
                value = str(item.get("value", "")).strip()
                if not key or not value or len(value) > 60:
                    continue
                try:
                    confidence = min(1.0, max(0.5, float(item.get("confidence", 0.8))))
                except (TypeError, ValueError):
                    confidence = 0.8
                facts.add(
                    Fact(
                        key=key,
                        value=value,
                        category="personal",
                        source_turn=facts.turn,
                        confidence=confidence,
                        source="llm",
                    )
                )
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    async def _build_messages(
        self,
        transcript: str,
        ctx: ConversationContext,
        memory: SessionMemory | None,
        retrieval: RetrievalModule | None,
        facts: FactMemory | None = None,
        session_id: str = "default",
        user_repeated: bool = False,
    ) -> list[dict[str, str]]:
        has_context = False
        retrieved: list[tuple[str, str | None]] = []

        if retrieval and memory:
            if ctx.topic:
                retrieved = retrieval.retrieve_context_with_topics(
                    transcript, memory, top_k=3, topic=ctx.topic
                )
            else:
                retrieved = [
                    (doc, None)
                    for doc in retrieval.retrieve_context(
                        transcript, memory, top_k=3
                    )
                ]
            if retrieved:
                has_context = True

        complexity = self._classify_query_complexity(transcript)
        ctx.query_complexity = complexity

        system_prompt = build_system_prompt(
            engagement=ctx.engagement,
            turn_count=ctx.turn_count,
            has_context=has_context,
            complexity=complexity,
            user_profile_block=self._user_profiles.get(session_id, ""),
        )

        tool_block = self._tool_registry.system_prompt_block()
        if tool_block:
            system_prompt += tool_block

        if facts is not None:
            facts_block = facts.to_block()
            if facts_block:
                system_prompt += (
                    "\n\nKnown facts about the user "
                    "(treat these as authoritative; do not contradict them):\n"
                    + facts_block
                )

        if ctx.topic and ctx.turn_count > 0:
            system_prompt += f"\n\nCurrent topic: {ctx.topic}."
        if memory and memory.summary:
            system_prompt += (
                f"\n\nConversation summary so far:\n{memory.summary}"
            )
        if ctx.intent == "correction":
            system_prompt += (
                "\n\nThe user just corrected you. Acknowledge the correction "
                "briefly, then respond directly to it. Do not repeat your "
                "previous answer."
            )
        elif ctx.intent == "continuation":
            system_prompt += (
                "\n\nThe user is continuing their previous thought. Respond "
                "fluidly without re-introducing the topic."
            )

        if user_repeated:
            system_prompt += (
                "\n\nThe user is repeating or re-asking something they asked "
                "before. Acknowledge that your previous answer didn't fully "
                "land, then respond again with a fresh, clearer, or more "
                "direct approach. Do not repeat your previous wording."
            )

        if ctx.topic and ctx.topic_shift and ctx.turn_count > 1:
            system_prompt += (
                f"\n\nThe user has switched to a new topic "
                f"('{ctx.topic}'). Move on cleanly and don't keep referring "
                "to the previous topic."
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if memory:
            history = memory.get_history(6)
            history_lower = {e.content.strip().lower() for e in history}
            if has_context and retrieved:
                hits = [
                    r for r in retrieved
                    if r[0].strip().lower() not in history_lower
                ][:3]
                if hits:
                    lines = []
                    for doc, doc_topic in hits:
                        prefix = f"[{doc_topic}] " if doc_topic else ""
                        lines.append(f"- {prefix}{doc}")
                    messages.append({
                        "role": "system",
                        "content": "Relevant context from earlier:\n"
                        + "\n".join(lines),
                    })
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

    def prosody_snapshot(self) -> dict[str, dict[str, str]]:
        """Last selected prosody label + user emotion per live session."""
        return {sid: dict(meta) for sid, meta in self._prosody_meta.items()}

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
        generation: int | None = None,
    ) -> None:
        # Use put_nowait: a slow/dead client must never stall the pipeline for
        # every session. A full queue means the consumer is stuck, so dropping
        # is the right backpressure here — blocking would deadlock all sessions
        # on a single queue. Events route to the session's own queue when one
        # is registered (ws/webrtc), else the global queue (CLI).
        q = self._session_outputs.get(session_id) or self._output_queue
        try:
            q.put_nowait(
                PipelineMessage(
                    event,
                    data,
                    session_id,
                    self._generation(session_id)
                    if generation is None
                    else generation,
                )
            )
        except asyncio.QueueFull:
            logger.warning(f"output queue full, dropping {event}")
