from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncGenerator, ClassVar

from modules.tts.chunker import TTSChunker
from modules.tts.sanitize import sanitize_for_tts
from modules.tts.prosody import ProsodySelector, classify_sentiment
from modules.emotion.classifier import EmotionClassifier
from providers.stt.base import STTProvider
from providers.llm.base import LLMProvider
from providers.tts.base import TTSProvider
from modules.vad.silero_vad import HysteresisVAD, SileroVAD
from modules.turn.detector import TurnDetector
from modules.turn.interrupt import InterruptHandler
from modules.turn.timing import TurnTiming
from modules.turn.topic import TopicTracker
from modules.turn.intent import IntentClassifier
from modules.turn.backchannel import TurnBackchannel
from modules.turn.entity import EntityGate
from modules.turn.backchannel_interrupt import BackchannelInterrupt
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
from modules.tools.registry import ToolMarkerFilter, ToolRegistry
from modules.tools.builtin import get_builtin_tools
from modules.laya.questions import (
    CADENCE1_QUESTIONS,
    PREFETCH_QUESTIONS,
    TURN_QUESTIONS,
)
from providers.laya.client import LayaSystem1
from utils.audio import rms_energy
from utils.logger import get_logger

logger = get_logger("pipeline")

ECHO_GRACE_SECONDS = 0.35

# Prefetchable tools that spend a SerpApi credit per call. They are not run
# on every partial (the transcript is still changing); Laya's vote is kept as
# a candidate and fired once the user pauses, when the partial is most
# likely the final text -- at most once per utterance.
_PAID_PREFETCH_TOOLS = frozenset({"search_web"})
_PAID_PREFETCH_PAUSE_MS = 200.0
# How long a turn waits for an in-flight paid prefetch that matches what the
# user said. Waiting beats the alternative (LLM -> tool call -> same search).
_PAID_PREFETCH_WAIT_S = 4.0
# A pause after one of these is the speaker searching for the next word
# ("find flights to... um... Goa"), not the end of the question -- don't
# spend the utterance's one credit on it. Stricter than the endpointer's
# is_incomplete (which only has to avoid cutting the user off).
# Words that can also end a complete question ("what's the weather like",
# "who said that", "is it over") are deliberately left out.
_DANGLING_WORDS = frozenset({
    "a", "an", "the", "my", "some",
    "to", "from", "in", "at", "for", "of", "with", "near", "about",
    "via", "into", "between", "than",
    "and", "or", "but", "if", "because",
    "um", "uh", "er", "hmm",
})
ECHO_FLOOR_MARGIN = 1.5

# During playback the barge energy gate is raised above the acoustic echo
# floor (the assistant's own voice bleeding into the mic must never trigger a
# self-interrupt). On loud/close-field setups that floor can lift past a real
# user's speaking level, so a genuine interruption then never crosses the gate
# and the whole response plays out. Frames that clear the base speech-energy
# threshold but sit below the raised floor are therefore decided by CONTENT:
# a short probe of the buffered speech that is neither conversational feedback
# nor a strong word-overlap with the sentence currently being spoken triggers
# the interrupt anyway, while echoed assistant audio (which matches that
# sentence) does not.
ECHO_OVERLAP_RATIO = 0.55  # probe-vs-spoken word overlap that means "echo"

# Fixed pipeline frame: 128ms at 16k mono 16-bit. All incoming audio is
# segmented into these frames so VAD/turn logic sees uniform windows and the
# SileroVAD hidden state decays across trailing-silence frames (otherwise a
# large silence chunk can be misclassified as speech and swallow the turn end).
FRAME_BYTES = 4096

# Cadence-1 System-1 decision thresholds (Phase-3/4, opt-in via BOLO_LAYA_PHASE3).
# Mirroring the Phase-2 action bar, base checkpoints are over-confident so these
# are deliberately strict: all three must hold before an endpoint is shortened
# or a barge verdict is acted on, and a probe result is only "fresh" within a
# short window of the ~1s partial cadence.
_C1_ACT_PROB = 0.9    # turn_complete noul probability required
_C1_SCORE_REQ = 7.0   # completion_conf score (0-10) required
_C1_CONF_REQ = 0.8    # barge_type choice confidence required
_C1_FRESH_S = 2.0     # max probe age for the result to be trusted

# Soft-barge content lane (below the echo-raised floor): full STT on the
# growing buffer is expensive, so buffer probes are rate-limited and run in a
# background task instead of stalling the 128ms audio loop on every frame, and
# a single blurred frame must never cancel the response -- the interrupt only
# fires once enough frames have sustained the soft candidate.
_BARGE_PROBE_INTERVAL_S = 0.25   # min gap between background buffer probes
_SOFT_BARGE_MIN_FRAMES = 3       # sustained soft frames before an interrupt
_SOFT_BARGE_PROBE_ATTEMPTS = 3   # rapid background probes before pacing resumes

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
    ENTITY_GUARD = auto()
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
    # Phase-2 (System-1) decisions. Defaults reproduce today's behavior:
    # invoke the LLM, no fast path, primary model tier, no urgency shortcut.
    invoke_llm: bool = True
    urgent: bool = False
    fast_path: bool = False
    model_tier: str = "primary"
    complexity_locked: bool = False
    sentiment_locked: bool = False


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
        system1: LayaSystem1 | None = None,
        llm_fallback: LLMProvider | None = None,
        *,
        use_hysteresis_vad: bool = True,
        endpoint_short_ms: int = 350,
        endpoint_normal_ms: int = 450,
        endpoint_hesitation_ms: int = 550,
        endpoint_safety_cap_ms: int = 900,
        endpoint_short_utterance_ms: int = 1500,
        min_speech_duration_ms: int = 500,
        phase2: bool = False,
        phase3: bool = False,
        # Phase 4: cadence-1 "utterance complete" endpoint authority on its own
        # (the shorter turn-commit lane) without Phase-3's barge authority.
        # Fail-closed: off by default.
        phase4: bool = False,
        # Phase 5: complexity-routing gate (trivial-ack LLM-skip) on its own.
        # Fail-closed: off by default.
        phase5: bool = False,
        # Independent of the Laya phases: the deterministic fast-action table
        # (canned phatics + templated time/date/calc). ``None`` keeps the
        # legacy coupling (fast-path lives under Phase-2); an explicit bool
        # decouples the two so the fast table can serve replies with Laya
        # fully shadowed.
        fast_path: bool | None = None,
        # Latency Step-1: prefetch a confident tool result from the live
        # partial so the LLM's first prompt already carries the data.
        tool_prefetch: bool = False,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.llm_fallback = llm_fallback
        self.tts = tts
        if (
            use_hysteresis_vad
            and not isinstance(vad, HysteresisVAD)
            and isinstance(vad, SileroVAD)
        ):
            vad = HysteresisVAD(vad, sample_rate=getattr(vad, "sample_rate", 16000))
        self.vad = vad
        self.endpoint_short_ms = endpoint_short_ms
        self.endpoint_normal_ms = endpoint_normal_ms
        self.endpoint_hesitation_ms = endpoint_hesitation_ms
        self.endpoint_safety_cap_ms = endpoint_safety_cap_ms
        self.endpoint_short_utterance_ms = endpoint_short_utterance_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.turn_detector = turn_detector or TurnDetector()
        self.interrupt_handler = interrupt_handler or InterruptHandler()
        self.turn_timing = turn_timing or TurnTiming()
        self.turn_backchannel = turn_backchannel or TurnBackchannel(
            generator=backchannel_generator or BackchannelGenerator(),
            timing=backchannel_timing or BackchannelTiming(),
        )
        self._prosody = ProsodySelector()
        self._emotion = emotion_classifier or EmotionClassifier(enabled=False)
        self._s1 = system1 or LayaSystem1(enabled=False)
        # Phase-2 routing switches (see _fast_reply / _llm_for).
        self._phase2 = phase2
        # Phase-3 endpoint/barge switches (see _cadence1_probe consumers).
        self._phase3 = phase3
        # Phase-4 endpoint-only switch: the same signed-off endpoint authority
        # as Phase-3, but without barge enforcement (see _cadence1_probe).
        self._phase4 = phase4
        # Phase-5 complexity-routing gate (trivial-ack LLM-skip, see _ack_phrase).
        self._phase5 = phase5
        # Deterministic fast-path, decoupled from the Laya phases (see above).
        self._fast_path = bool(phase2) if fast_path is None else bool(fast_path)
        # Latency Step-1 live-listening tool prefetch (see _prefetch_probe).
        self._tool_prefetch = tool_prefetch
        self._topic_trackers: dict[str, TopicTracker] = {}
        self._topic_label_tasks: dict[str, asyncio.Task] = {}
        # Shadow-only turn-level Laya passes (held so they aren't GC'd).
        self._s1_bg_tasks: set[asyncio.Task] = set()
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
        self._speaking_text: dict[str, str] = {}
        self._speech_buffers: dict[str, bytearray] = {}
        self._frame_buffers: dict[str, bytearray] = {}
        self._silence_ms: dict[str, float] = {}
        self._speech_ms: dict[str, float] = {}
        self._last_partial_time: dict[str, float] = {}
        self._last_refresh_ms: dict[str, float] = {}
        self._partial_inflight: dict[str, bool] = {}
        # Cadence-1 System-1 state: latest probe result per session and a
        # guard so overlapping partial cadences never stack Laya inferences.
        self._s1_cadence1: dict[str, dict] = {}
        self._s1_c1_inflight: dict[str, bool] = {}
        # Latency Step-1 tool prefetch: latest stashed result per session, its
        # in-flight guard, and the background task (cancelled at speech-end so
        # it never contends with the reply for the GPU).
        self._s1_prefetch: dict[str, dict] = {}
        # Outcome labels for Laya training (what actually happened, not what
        # a classifier guessed): partials heard during the current utterance,
        # the tools each turn really used, and turn rows awaiting that outcome.
        self._utt_partials: dict[str, list[str]] = {}
        self._turn_outcomes: dict[str, dict] = {}
        self._pending_turn_rows: dict[str, dict] = {}
        self._s1_prefetch_inflight: dict[str, bool] = {}
        self._s1_prefetch_tasks: dict[str, asyncio.Task] = {}
        # Paid (SerpApi) prefetch: Laya's latest tool vote on a partial, fired
        # at the next pause; and the generation (utterance) that already
        # spent its one credit -- partials change many times per utterance.
        self._s1_prefetch_candidate: dict[str, dict] = {}
        self._s1_prefetch_paid_gen: dict[str, int] = {}
        # Monotonic timestamps of the last SPEECH_END emit, for the true
        # end-to-end "user stopped talking -> first audio out" metric.
        self._speech_end_ts: dict[str, float] = {}
        self._barge_pending: dict[str, bool] = {}
        self._barge_rejects: dict[str, float] = {}
        self._barge_thresholds: dict[str, float] = {}
        self._barge_frames: dict[str, int] = {}
        # Soft-barge content lane state: cached background probe result, its
        # in-flight guard, and the sustained soft-frame counter.
        self._barge_probe_cache: dict[str, tuple[str, float]] = {}
        self._barge_probe_inflight: dict[str, bool] = {}
        self._barge_soft_frames: dict[str, int] = {}
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
        self._speech_ms.clear()
        self._last_partial_time.clear()
        self._barge_pending.clear()
        self._barge_thresholds.clear()
        self._barge_frames.clear()
        for task in self._s1_bg_tasks:
            task.cancel()
        self._s1_bg_tasks.clear()
        self._barge_probe_cache.clear()
        self._barge_probe_inflight.clear()
        self._barge_soft_frames.clear()
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
        self._speaking_text.pop(session_id, None)
        self._frame_buffers.pop(session_id, None)
        self._silence_ms.pop(session_id, None)
        self._speech_ms.pop(session_id, None)
        self._last_partial_time.pop(session_id, None)
        self._last_refresh_ms.pop(session_id, None)
        self._partial_inflight.pop(session_id, None)
        self._barge_pending.pop(session_id, None)
        self._barge_rejects.pop(session_id, None)
        self._barge_thresholds.pop(session_id, None)
        self._barge_frames.pop(session_id, None)
        self._barge_probe_cache.pop(session_id, None)
        self._barge_probe_inflight.pop(session_id, None)
        self._barge_soft_frames.pop(session_id, None)
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
                    # A fresh utterance: no cadence-1 verdict or soft-barge
                    # context survives from the turn that just ended, otherwise
                    # a stale System-1 result (or an unresolved buffer probe)
                    # could barge the new turn's speech.
                    self._s1_cadence1.pop(sid, None)
                    self._barge_probe_cache.pop(sid, None)
                    self._barge_soft_frames.pop(sid, None)
                    # Same for any tool prefetch stashed for the previous
                    # utterance: its generation tag would be stale and an
                    # in-flight Laya probe only steals GPU in a reply window.
                    self._s1_prefetch.pop(sid, None)
                    self._s1_prefetch_candidate.pop(sid, None)
                    self._s1_prefetch_inflight.pop(sid, None)
                    pf = self._s1_prefetch_tasks.pop(sid, None)
                    if pf and not pf.done():
                        pf.cancel()
                    self._speaking[sid] = True
                    ctx.dialogue_state = DialogueState.LISTENING
                    self._silence_ms[sid] = 0.0
                    self._speech_ms[sid] = len(chunk) / (
                        self.vad.sample_rate * 2 / 1000
                    )
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
                    self._speech_ms[sid] = (
                        self._speech_ms.get(sid, 0.0)
                        + len(chunk) / (self.vad.sample_rate * 2 / 1000)
                    )
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
                        elif (
                            playback_on
                            and energy
                            > self.interrupt_handler.speech_energy_threshold
                            and time.time()
                            - self._barge_rejects.get(sid, 0.0)
                            > 1.0
                        ):
                            # Below the echo-raised floor but above a real
                            # speaking level: hold a content-checked candidate
                            # so _maybe_fire_barge can decide by probing the
                            # buffered speech instead of by raw gain alone.
                            self._barge_pending[sid] = True
                            self._barge_thresholds[sid] = threshold
                            self._barge_frames[sid] = target_frames
                            self._interrupt_handler(sid).reset()
                            logger.debug(
                                f"barge-in soft candidate session={sid}"
                                f" energy={energy:.3f} raised={threshold:.3f}"
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

                    # Evidence-based adaptive endpointing. The commit decision
                    # is gated on BOTH a minimum speech duration (so a blip or
                    # very short sound is never cut off) AND an adaptive silence
                    # threshold that reflects how complete the utterance sounds.
                    speech_dur_ms = self._speech_ms.get(sid, 0.0)
                    trajectory = self.turn_detector.prosody_analyzer.analyze().get(
                        "trajectory", "neutral"
                    )

                    # Paid prefetch fires on a pause that sounds like the end
                    # of the question (after the refresh above, so it sees
                    # the freshest partial), never on a mid-sentence one.
                    if (
                        silence_ms >= _PAID_PREFETCH_PAUSE_MS
                        and not incomplete
                        and trajectory != "rising"
                        and not self._looks_unfinished(ctx.last_partial_transcript)
                    ):
                        self._start_paid_prefetch(sid, ctx)

                    endpoint_ms = self.endpoint_normal_ms
                    if incomplete or trajectory == "rising":
                        # Speaker is mid-flow / building toward more; give them
                        # room to finish the thought or enumeration.
                        endpoint_ms = self.endpoint_hesitation_ms
                    elif (
                        turn_decision == "end_turn_force"
                        and semantic_score >= 0.8
                        and speech_dur_ms <= self.endpoint_short_utterance_ms
                        and trajectory == "falling"
                    ):
                        # Clearly completed short utterance - respond briskly.
                        endpoint_ms = self.endpoint_short_ms

                    # Phase-4 cadence-1 endpoint authority: once a fresh,
                    # confident probe exists, Laya's verdict decides the
                    # "utterance already complete -> respond briskly" lane and
                    # the legacy mid-list/rising-trajectory signals no longer
                    # veto it (they remain the primary path only before a
                    # verdict exists / in the pre-probe window). The verdict is
                    # only honored while ``ctx's`` partial still matches the
                    # text it was computed on -- a sync refresh that extends
                    # the partial (e.g. list continuation) demotes to the
                    # legacy lanes until the next probe lands. The min-speech +
                    # safety-cap guards below remain hard bounds.
                    c1 = self._s1_cadence1.get(sid)
                    if (
                        (self._phase3 or self._phase4)
                        and c1 is not None
                        and c1.get("generation") == self._generation(sid)
                        and (time.monotonic() - c1.get("ts", 0.0)) <= _C1_FRESH_S
                        and c1.get("turn_complete") is not None
                        and c1.get("turn_complete") >= _C1_ACT_PROB
                        and c1.get("completion_conf") is not None
                        and c1.get("completion_conf") >= _C1_SCORE_REQ
                        and c1.get("partial") == ctx.last_partial_transcript
                    ):
                        endpoint_ms = self.endpoint_short_ms

                    # Safety cap: never wait longer than this so long pauses
                    # don't feel sluggish.
                    endpoint_ms = min(endpoint_ms, self.endpoint_safety_cap_ms)

                    speech_long_enough = (
                        speech_dur_ms >= self.min_speech_duration_ms
                    )

                    if speech_long_enough and silence_ms >= endpoint_ms:
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

                        self._speech_end_ts[sid] = time.perf_counter()
                        await self._emit(PipelineEvent.SPEECH_END, session_id=sid)
                        # The latency Step-1 prefetch fought for the GPU during
                        # listening; at speech-end it must never keep holding it
                        # while the reply needs it. Cancel the in-flight probe:
                        # whatever it already stashed is consumed by the turn.
                        pf = self._s1_prefetch_tasks.pop(sid, None)
                        if pf and not pf.done():
                            pf.cancel()
                        self._s1_prefetch_inflight.pop(sid, None)
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

    def _barge_echo_match(self, probe: str, session_id: str) -> bool:
        """True when a barge probe transcript looks like the sentence the
        assistant is currently speaking, i.e. speaker bleed bleeding back into
        the mic rather than a genuine user interruption."""
        words = [w for w in probe.lower().split() if len(w) >= 2]
        if len(words) < 2:
            return False
        spoken_words = {
            w for w in self._speaking_text.get(session_id, "").lower().split()
            if len(w) >= 2
        }
        if not spoken_words:
            return False
        overlap = len(set(words) & spoken_words) / len(words)
        return overlap >= ECHO_OVERLAP_RATIO

    async def _maybe_fire_barge(
        self, sid: str, ctx: ConversationContext, energy: float
    ) -> None:
        if not self._barge_pending.get(sid, False):
            self._barge_soft_frames.pop(sid, None)
            return

        # Phase-3 cadence-1 gate: a fresh, sufficiently-confident System-1
        # verdict is the primary barge authority (backchannel keeps the
        # response going, disagreement cancels immediately). The legacy lexical
        # + energy-floor paths below still apply when the probe is absent or
        # stale, or when this flag is off. The verdict is only trusted when it
        # was computed for the CURRENT utterance: the generation tag is stamped
        # by _cadence1_probe, so a leftover from the user's previous turn can
        # never gate this one.
        c1 = self._s1_cadence1.get(sid)
        if (
            self._phase3
            and c1 is not None
            and c1.get("generation") == self._generation(sid)
            and (time.monotonic() - c1.get("ts", 0.0)) <= _C1_FRESH_S
        ):
            bt = c1.get("barge_type")
            if bt == "backchannel":
                # Conversational feedback: keep the response going regardless
                # of accumulated speech frames/energy.
                self._keep_barge(sid)
                return
            if bt == "disagreement":
                # Firm interrupt: cancel immediately. Matches the legacy lexical
                # disagreement behavior but without waiting for a partial.
                self._barge_soft_frames.pop(sid, None)
                await self._do_interrupt(sid, ctx)
                return

        # Lexical gate: if a partial transcript is already available, decide
        # from its content whether this speech is a genuine interruption.
        partial = (ctx.last_partial_transcript or "").strip()
        if partial:
            lexical = BackchannelInterrupt.classify(partial)
            if lexical == "backchannel":
                # Conversational feedback ("yeah", "uh-huh", "right", "okay")
                # should NOT cancel our response. Keep the response going.
                self._keep_barge(sid)
                return
            if lexical == "disagreement":
                # Firm interruption ("no", "stop", "wait", "that's wrong"):
                # cancel immediately regardless of accumulated speech frames.
                self._barge_soft_frames.pop(sid, None)
                await self._do_interrupt(sid, ctx)
                return

        handler = self._interrupt_handler(sid)
        threshold = self._barge_thresholds.get(
            sid, self.interrupt_handler.speech_energy_threshold
        )
        target_frames = self._barge_frames.get(
            sid, self.interrupt_handler.consecutive_speech_frames
        )
        base_threshold = self.interrupt_handler.speech_energy_threshold
        if self._playback_active.get(sid, False):
            threshold = max(
                threshold, self._echo_floor.get(sid, 0.0) * ECHO_FLOOR_MARGIN
            )
        fast = handler.should_interrupt(
            energy, 0.0, True, threshold=threshold, target_frames=target_frames
        )
        # Soft candidate: below the raised (echo-aware) floor but above a real
        # speaking level while we're playing back. Raw gain can't separate the
        # user's voice from speaker bleed, so this path decides by CONTENT:
        # it probes the buffered speech and only interrupts on novel,
        # non-backchannel words.
        soft = (
            self._playback_active.get(sid, False)
            and not fast
            and energy > base_threshold
        )
        if not (fast or soft):
            # A frame below both lanes breaks the run: soft frames must be
            # consecutive, not accumulated across scattered noise.
            self._barge_soft_frames.pop(sid, None)
            return

        # Sustained-frame gate for the soft lane: a single blurred frame must
        # not cancel the response, so at least _SOFT_BARGE_MIN_FRAMES frames
        # have to exist at soft level before any content check can fire an
        # interrupt. (The fast lane is loud enough to be authoritative on one
        # frame, as before.)
        if soft:
            self._barge_soft_frames[sid] = self._barge_soft_frames.get(sid, 0) + 1
            if self._barge_soft_frames[sid] < _SOFT_BARGE_MIN_FRAMES:
                return
        else:
            self._barge_soft_frames.pop(sid, None)

        # Before committing to an interrupt during playback, classify the
        # current speech. The periodic partial transcript is rate-limited
        # (~1s), so short backchannels reach this point before any partial
        # exists; probe the buffer directly so conversational feedback
        # ("yeah", "uh-huh", "right") doesn't cancel our response. Full STT
        # on the growing buffer is expensive, so the probe runs in a
        # background task (never in the 128ms audio loop), rate-limited and
        # cached; the frame path only ever reads that cache.
        if self._playback_active.get(sid, False):
            now = time.monotonic()
            probe, probe_ts = self._barge_probe_cache.get(sid, ("", 0.0))
            if (now - probe_ts) >= _BARGE_PROBE_INTERVAL_S:
                if not self._barge_probe_inflight.get(sid, False):
                    self._barge_probe_inflight[sid] = True
                    asyncio.create_task(
                        self._barge_probe_and_decide(sid, ctx, soft=soft)
                    )
                # The soft lane's verdict belongs to the content probe: until
                # it has resolved, hold the interrupt rather than deciding on
                # blurred energy alone.
                if soft:
                    return
                probe, probe_ts = self._barge_probe_cache.get(sid, ("", 0.0))
            if probe:
                if BackchannelInterrupt.is_backchannel(probe):
                    self._keep_barge(sid)
                    logger.info(f"barge kept (backchannel {probe!r}) session={sid}")
                    return
                if soft and self._barge_echo_match(probe, sid):
                    self._keep_barge(sid)
                    logger.info(
                        f"barge kept (echo of spoken text {probe!r}) session={sid}"
                    )
                    return
            elif soft:
                # Empty transcript: nothing to classify yet. Only the
                # background probe cycle may decide this lane; a frame path
                # interrupted on silence would cancel the assistant for echo.
                return
        await self._do_interrupt(sid, ctx)

    async def _probe_barge_buffer(
        self, sid: str, ctx: ConversationContext
    ) -> str:
        """Transcribe the current speech buffer for a barge content probe."""
        buf = self._speech_buffers.get(sid)
        if not buf:
            return ""
        try:
            text = await self._partial_transcribe(bytes(buf), sid, ctx, sync=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"barge classify probe failed: {e}")
            return ""
        return text or ""

    async def _barge_probe_and_decide(
        self,
        sid: str,
        ctx: ConversationContext,
        *,
        soft: bool = False,
    ) -> None:
        """Background full-STT probe of the growing speech buffer, cached so
        the audio loop never blocks on transcription. After caching it resolves
        the pending barge from content; a soft lane with nothing to classify
        yet re-probes the (grown) buffer a few times before handing back so the
        frame path can pace the next cycle."""
        probe = await self._probe_barge_buffer(sid, ctx)
        self._barge_probe_cache[sid] = (probe, time.monotonic())
        try:
            for _ in range(_SOFT_BARGE_PROBE_ATTEMPTS):
                if not await self._decide_contents_barge(sid, ctx, probe, soft=soft):
                    return
                probe = await self._probe_barge_buffer(sid, ctx)
                self._barge_probe_cache[sid] = (probe, time.monotonic())
        finally:
            self._barge_probe_inflight.pop(sid, None)

    async def _decide_contents_barge(
        self,
        sid: str,
        ctx: ConversationContext,
        probe: str,
        *,
        soft: bool = False,
    ) -> bool:
        """Content decision for a barge candidate once a buffer probe exists.
        Backchannels and speaker bleed keep the response; genuinely novel
        speech interrupts. The soft lane (below the echo-raised floor) only
        ever interrupts on content: until the transcript is non-empty (and
        enough frames have sustained the candidate) it returns True so the
        caller re-probes the grown buffer instead of committing.
        """
        if not self._barge_pending.get(sid, False):
            return False
        if soft:
            if self._barge_soft_frames.get(sid, 0) < _SOFT_BARGE_MIN_FRAMES:
                return True
            if not probe:
                return True
        if self._playback_active.get(sid, False) and probe:
            if BackchannelInterrupt.is_backchannel(probe):
                self._keep_barge(sid)
                logger.info(f"barge kept (backchannel {probe!r}) session={sid}")
                return False
            if soft and self._barge_echo_match(probe, sid):
                self._keep_barge(sid)
                logger.info(
                    f"barge kept (echo of spoken text {probe!r}) session={sid}"
                )
                return False
        await self._do_interrupt(sid, ctx)
        return False

    def _keep_barge(self, sid: str) -> None:
        """Non-interrupt resolution: backchannel/echo keeps the response."""
        self._barge_rejects[sid] = time.time()
        self._barge_pending[sid] = False
        self._barge_thresholds.pop(sid, None)
        self._barge_frames.pop(sid, None)
        self._barge_soft_frames.pop(sid, None)
        self._interrupt_handler(sid).reset()

    async def _do_interrupt(self, sid: str, ctx: ConversationContext) -> None:
        """Barge decision committed: cancel playback and return to listening."""
        self._barge_pending[sid] = False
        self._barge_thresholds.pop(sid, None)
        self._barge_frames.pop(sid, None)
        self._barge_soft_frames.pop(sid, None)
        self._interrupt_handler(sid).reset()
        await self.signal_interrupt(sid)
        ctx.dialogue_state = DialogueState.LISTENING
        await self._emit(PipelineEvent.INTERRUPT, session_id=sid)

    async def _process_speech_segment(
        self, audio_blob: bytes, session_id: str, ctx: ConversationContext
    ) -> None:
        prev = self._current_tasks.get(session_id)
        # Latency Step-1: take any tool prefetch that was stashed while the
        # user was speaking. Must run BEFORE the generation bump below so the
        # probe's generation tag (stamped during listening) still matches this
        # turn; anything stale or absent is simply dropped. The cache is always
        # cleared so a future turn can never reuse an old tool answer.
        prefetch = self._consume_prefetch(session_id)
        # Per-turn context is only valid for the segment that produced it.
        # Without this reset, an urgent/escalated/locked decision from one turn
        # silently bleeds into every later turn: the assistant stays on the
        # fallback model, skips turn delays, and refuses to re-evaluate query
        # complexity hours after the original request.
        ctx.urgent = False
        ctx.model_tier = "primary"
        ctx.complexity_locked = False
        ctx.sentiment_locked = False
        # During playback, classify the incoming segment so conversational
        # backchannels ("yeah", "uh-huh", "right") don't cancel the assistant's
        # response mid-playback.
        if (
            prev is not None
            and not prev.done()
            and prev is not asyncio.current_task()
            and self._playback_active.get(session_id, False)
        ):
            try:
                classify_text = await self.stt.transcribe(audio_blob) or ""
            except Exception as e:  # noqa: BLE001
                logger.warning(f"backchannel classify failed: {e}")
                classify_text = ""
            if classify_text and BackchannelInterrupt.is_backchannel(classify_text):
                # Passive feedback: keep the current response flowing, don't
                # cancel playback or emit an interrupt.
                self._barge_pending.pop(session_id, None)
                self._barge_thresholds.pop(session_id, None)
                self._barge_frames.pop(session_id, None)
                self._interrupt_handler(session_id).reset()
                logger.info(
                    f"backchannel {session_id} ignored during playback: {classify_text}"
                )
                return
        if prev and not prev.done() and prev is not asyncio.current_task():
            prev.cancel()
            self._playback_active[session_id] = False
            clear_task = self._playback_clear_tasks.pop(session_id, None)
            if clear_task and not clear_task.done():
                clear_task.cancel()
            await self._emit(PipelineEvent.INTERRUPT, session_id=session_id)
        self._bump_generation(session_id)
        turn_gen = self._generation(session_id)
        utt_partials = self._utt_partials.pop(session_id, [])
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
                prefetch = await self._resolve_prefetch(prefetch, partial, stt_task)
                spec_messages = await self._build_messages(
                    partial, ctx, memory, retrieval,
                    facts=self._facts.get(session_id),
                    session_id=session_id,
                    user_repeated=ctx.user_repeated,
                    prefetch=self._prefetch_for(prefetch, partial),
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

            self._log_endpoint_labels(
                session_id, utt_partials, transcript, ctx.turn_count
            )

            # Phase-2/5: turn-level System-1 pass. Rows are always shadow-logged;
            # context overrides are applied only when BOLO_LAYA_PHASE2=1 -- or
            # when BOLO_LAYA_PHASE5=1 AND the utterance is shaped like a bare
            # acknowledgment (a routing decision can't wait on a background
            # verdict, so the gate's synchronous pass only ever runs for those
            # few ack-shaped turns). This runs after every legacy classifier has
            # populated ctx, so a weak or missing Laya answer silently keeps the
            # legacy value (fail-open).
            phase5_sync = self._phase5 and self._ack_phrase(transcript) is not None
            if self._phase2 or phase5_sync:
                await self._shadow_cadence2(session_id, transcript, ctx)
            elif self._s1.enabled:
                # Shadow-only: answers are just logged, so the reply must not
                # wait for them.
                task = asyncio.create_task(
                    self._shadow_cadence2(session_id, transcript, ctx)
                )
                self._s1_bg_tasks.add(task)
                task.add_done_callback(self._s1_bg_tasks.discard)

            # Conservative LLM-skip: only when the deterministic fast-action
            # table yields a ready answer. A miss forces invoke_llm=True, so an
            # "LLM-skip" can never mean "no reply". The table is pure rule
            # matching with no GPU cost, so it is decoupled from the Laya
            # phases -- if the phase flags are off but BOLO_FAST_PATH is on
            # (the normal deployed configuration), canned phatics and templated
            # tool replies still answer instantly.
            fast_reply: str | None = None
            fast_tools: list[str] = []
            if self._fast_path:
                fast_reply = await self._fast_reply(transcript, ctx, fast_tools)
            # Phase-5 complexity-routing gate (fail-open, off by default): a
            # bare acknowledgment is answered deterministically instead of the
            # LLM only when a confident Laya verdict marks it trivial (simple +
            # not a question), it isn't urgent, and no escalation routed to the
            # fallback model. Any weak/missing verdict keeps the LLM.
            if (
                fast_reply is None
                and self._phase5
                and ctx.complexity_locked
                and ctx.query_complexity == "simple"
                and not ctx.is_question
                and not ctx.urgent
                and ctx.model_tier == "primary"
            ):
                ack = self._ack_phrase(transcript)
                if ack is not None:
                    fast_reply = ack
            if fast_reply is not None:
                self._record_turn_tools(session_id, turn_gen, fast_tools)
            ctx.fast_path = fast_reply is not None
            ctx.invoke_llm = fast_reply is None

            # Check if speculative LLM result can be reused. The shadow
            # (escalated) model can quickly vote to route this utterance off
            # the fast path, so never reuse speculation when it does: the
            # fallback model must produce this answer itself.
            used_speculation = False
            can_reuse_speculation = (
                ctx.intent != "correction"
                and fast_reply is None
                and ctx.model_tier != "fallback"
            )
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

                if fast_reply is None:
                    prefetch = await self._resolve_prefetch(prefetch, transcript)
                    messages = await self._build_messages(
                        transcript, ctx, memory, retrieval,
                        facts=self._facts.get(session_id),
                        session_id=session_id,
                        user_repeated=ctx.user_repeated,
                        prefetch=self._prefetch_for(prefetch, transcript),
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

            # Compute response timing delay. Fast replies and urgent turns
            # answer immediately (urgency is a latency modifier, not a
            # business decision).
            if fast_reply is None and not ctx.urgent:
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
            # Fast replies skip this wait: enjoy the low latency. A confident
            # Laya sentiment (Phase-2) already decided, so don't wait to
            # overwrite it either.
            if emotion_task and fast_reply is None and not ctx.sentiment_locked:
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

            # Evidence the model may legitimately name things from this turn:
            # what the user said, the recent conversation (assistant turns in
            # memory already passed this guard), a prefetched result, and
            # anything a tool returns below. Whole texts, not just extracted
            # names, so "Paradise" is backed by "Paradise Biryani".
            supported_entities: set[str] = {transcript}
            if memory:
                supported_entities.update(
                    e.content for e in memory.get_history(max_turns=6)
                )
            if prefetch and prefetch.get("result"):
                supported_entities.add(str(prefetch["result"]))

            if fast_reply is not None:
                # Deterministic fast reply: no LLM call, straight to TTS.
                full = fast_reply
                ctx.dialogue_state = DialogueState.INTERRUPTIBLE
                await self._emit(PipelineEvent.LLM_TOKEN, full, session_id)
                for c in chunker.feed(full):
                    push_nowait(1, self._tool_registry.strip_calls(c))
                tail = chunker.flush()
                if tail:
                    push_nowait(1, self._tool_registry.strip_calls(tail))
            elif used_speculation:
                full = spec_full or ""
                ctx.dialogue_state = DialogueState.INTERRUPTIBLE
                stripped = self._tool_registry.strip_calls(full)
                if stripped:
                    await self._emit(PipelineEvent.LLM_TOKEN, stripped, session_id)
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
                marker_filter = ToolMarkerFilter()

                bc_timer = None
                if not ctx.urgent:
                    bc_timer = asyncio.create_task(
                        self._backchannel_timer(
                            text_queue, ctx, session_id, int_ev, seq
                        )
                    )

                async for token in self._llm_for(ctx).generate_stream(messages):
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
                    safe = marker_filter.feed(token)

                    if not tool_marker_seen and marker_filter.seen:
                        tool_marker_seen = True
                        stop_tts.set()
                        await self._drain_queue(text_queue)
                        chunker.reset()
                        continue
                    if tool_marker_seen:
                        if self._tool_registry.find_calls(full):
                            break
                        continue

                    if safe:
                        await self._emit(PipelineEvent.LLM_TOKEN, safe, session_id)
                        for c in chunker.feed(safe):
                            await push(1, self._tool_registry.strip_calls(c))

                if first_token and bc_timer:
                    bc_timer.cancel()

                if int_ev.is_set():
                    return

                self._latency.measure("llm_full", llm_start)
                self._log_latency("llm_full")

                held = marker_filter.flush()
                if held:
                    await self._emit(PipelineEvent.LLM_TOKEN, held, session_id)
                    for c in chunker.feed(held):
                        await push(1, self._tool_registry.strip_calls(c))

                tool_calls = self._tool_registry.find_calls(full)

            if fast_reply is None and not int_ev.is_set():
                self._record_turn_tools(
                    session_id, turn_gen, [c.get("name", "") for c in tool_calls]
                )

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

                    tool_start = time.perf_counter()
                    tool_results = await asyncio.gather(
                        *[self._tool_registry.execute_call_with_retry(c) for c in tool_calls]
                    )
                    self._latency.measure("tool_exec", tool_start)
                    self._log_latency("tool_exec")
                    for c, tr in zip(tool_calls, tool_results):
                        logger.info(
                            f"tool session={session_id} name={c.get('name')} "
                            f"args={c.get('args')} kwargs={c.get('kwargs', {})} "
                            f"failed={bool(tr.get('failed'))} "
                            f"result={str(tr.get('result', ''))[:160]!r}"
                        )
                    for tr in tool_results:
                        followup_messages.append({
                            "role": "tool",
                            "content": f"{tr['tool']} result: {tr['result']}",
                        })
                        if not tr.get("failed"):
                            supported_entities.add(str(tr["result"]))
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
                    followup_filter = ToolMarkerFilter()
                    async for token in self._llm_for(ctx).generate_stream(
                        followup_messages
                    ):
                        if int_ev.is_set():
                            break
                        if first_token:
                            first_token = False
                        full += token
                        # A follow-up that calls another tool must not show or
                        # speak the call; the next round executes it.
                        safe = followup_filter.feed(token)
                        if not safe:
                            continue
                        await self._emit(PipelineEvent.LLM_TOKEN, safe, session_id)
                        for c in chunker.feed(safe):
                            await push(1, self._tool_registry.strip_calls(c))

                    if int_ev.is_set():
                        return
                    held = followup_filter.flush()
                    if held:
                        await self._emit(PipelineEvent.LLM_TOKEN, held, session_id)
                        for c in chunker.feed(held):
                            await push(1, self._tool_registry.strip_calls(c))

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

            # Layer-2 entity guard: if this was an entity-seeking query and the
            # final answer asserted a specific place/business name that was not
            # provided by the user, a tool result, or retrieval, treat it as a
            # possible hallucination. Speak an honest correction, surface an
            # ENTITY_GUARD event, and keep the fabrication out of memory so it
            # can't later be re-injected as authoritative.
            fabricated = self._verify_entity_assertions(
                transcript, full, supported_entities
            )
            if fabricated:
                safe = EntityGate.safety_response(fabricated)
                logger.warning(
                    f"entity guard blocked fabricated names session={session_id}"
                    f" names={fabricated}"
                )
                await self._emit(
                    PipelineEvent.ENTITY_GUARD, "\n".join(fabricated), session_id
                )
                if not int_ev.is_set():
                    for c in chunker.feed(safe):
                        await push(1, self._tool_registry.strip_calls(c))
                    tail = chunker.flush()
                    if tail:
                        await push(1, self._tool_registry.strip_calls(tail))
                full = safe

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
            pending = self._pending_turn_rows.get(session_id)
            if pending and pending["gen"] == turn_gen:
                self._pending_turn_rows.pop(session_id, None)
                self._s1.log_shadow(**pending["log"])

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
                self._speaking_text[session_id] = " ".join(spoken)

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
                                anchor = self._speech_end_ts.get(session_id) or eos_ts
                                if anchor is not None:
                                    self._latency.measure("e2e_reply", anchor)
                                    self._log_latency("e2e_reply")
                                self._speech_end_ts.pop(session_id, None)
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
                if self._s1.enabled:
                    heard = self._utt_partials.setdefault(session_id, [])
                    if len(heard) < 30:
                        heard.append(text)
                await self._emit(PipelineEvent.PARTIAL_TRANSCRIPT, text, session_id)
                if not sync:
                    # Cadence-1 live-listening verdicts are consumed ONLY by the
                    # Phase-3 endpoint/barge paths and the Phase-4 endpoint-only
                    # path, so skip the per-partial Laya round-trip entirely when
                    # both are off -- otherwise every partial shares the GPU with
                    # a reply for a verdict nobody reads (the single largest Laya
                    # latency cost in the common shadow deployment).
                    if (
                        (self._phase3 or self._phase4)
                        and not self._s1_c1_inflight.get(session_id, False)
                    ):
                        self._s1_c1_inflight[session_id] = True
                        asyncio.create_task(
                            self._cadence1_probe(text, session_id)
                        )

                    # Latency Step-1: while the user is still talking, ask Laya
                    # only the tool decision and (on a confident hit) execute
                    # the tool once, stashing the result so the LLM's first
                    # prompt already carries it. Runs at most once per partial
                    # change and never in parallel with a cadence-1 probe; the
                    # task is cancelled at speech-end.
                    if (
                        self._tool_prefetch
                        and not self._s1_prefetch_inflight.get(session_id, False)
                        and not self._s1_c1_inflight.get(session_id, False)
                    ):
                        self._s1_prefetch_inflight[session_id] = True
                        task = asyncio.create_task(
                            self._prefetch_probe(text, session_id)
                        )
                        self._s1_prefetch_tasks[session_id] = task
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

    async def _shadow_cadence2(
        self, session_id: str, transcript: str, ctx: ConversationContext
    ) -> None:
        """Run the cadence-2 (turn-level) Laya pass.

        Always shadow-logs Laya-vs-legacy rows for telemetry/calibration. When
        Phase-2 is enabled (``BOLO_LAYA_PHASE2=1``) the confident answers also
        override legacy classifier output directly on ``ctx``. Fails open: any
        error, timeout, or low-confidence answer keeps today's behavior.
        """
        s1 = self._s1
        if not s1.enabled:
            return
        gen = self._generation(session_id)

        state = {
            "transcript": transcript,
            "prev_intent": ctx.prev_intent,
            "turn_count": ctx.turn_count,
            "topic": ctx.topic,
            "last_transcript": ctx.last_transcript,
        }
        # Snapshot legacy values first: in shadow-only mode this runs in the
        # background while the turn keeps mutating ctx.
        legacy_intent = ctx.intent
        legacy_is_question = ctx.is_question
        legacy_sentiment = ctx.user_sentiment
        legacy_complexity = self._classify_query_complexity(transcript)
        legacy_topic_shift = ctx.topic_shift
        legacy_verify = EntityGate.query_needs_verification(transcript)

        if self._phase2 or self._phase5:
            # Enforcement (Phase-2 routing / Phase-5 ack-gate): synchronous
            # path -- we are already inside the reply path and must not wait
            # for the turn task to finish.
            pass
        else:
            # Shadow-only telemetry: the reply owns the GPU right now, so never
            # steal it for a pass that only logs rows. Wait until the current
            # turn task has fully wrapped up before running the Laya forward
            # pass. The state + legacy snapshot above was taken at call time, so
            # a later turn mutating ctx cannot skew these rows.
            current = self._current_tasks.get(session_id)
            if (
                current is not None
                and current is not asyncio.current_task()
                and not current.done()
            ):
                try:
                    await asyncio.shield(current)
                except (asyncio.CancelledError, Exception):
                    pass

        start = time.perf_counter()
        answers = await s1.predict(state, TURN_QUESTIONS)
        self._latency.measure("laya_s1", start)
        if not answers:
            return
        self._log_latency("laya_s1")

        def _noul(q: str) -> float | None:
            return s1.noul_prob(answers, q)

        rows: list[dict] = []

        def _cmp(q, laya_value, legacy, use_prob: bool = False):
            entry = answers.get(q) if isinstance(answers.get(q), dict) else {}
            if "noul" in entry:
                # Confidence of the answer given: a confident "no" (P=0.05)
                # is 0.95, not 0.05.
                p_true = float(entry["noul"])
                laya_conf = float(entry.get("confidence", max(p_true, 1 - p_true)))
            elif "choice" in entry:
                laya_conf = float(entry.get("confidence", 0.0))
            else:
                laya_conf = 0.0
            if use_prob:
                match = laya_value is not None and (
                    (laya_value >= 0.5) == bool(legacy)
                )
                shown_match = bool(laya_value is not None and laya_value >= 0.5)
            else:
                match = laya_value is not None and laya_value == legacy
                shown_match = bool(match)
            rows.append({
                "question": q,
                "laya": laya_value if not use_prob else shown_match,
                "laya_conf": laya_conf,
                "legacy": bool(legacy) if use_prob else legacy,
                "match": bool(match),
            })

        # UNDERSTANDING
        _cmp("intent", s1.choice(answers, "intent", None), legacy_intent)
        _cmp("is_question", _noul("is_question"), legacy_is_question, use_prob=True)
        _cmp("query_complexity", s1.choice(answers, "query_complexity", None), legacy_complexity)
        _cmp("topic_changed", _noul("topic_changed"), legacy_topic_shift, use_prob=True)
        _cmp("needs_verify", _noul("needs_verify"), legacy_verify, use_prob=True)

        # ACTION (no legacy counterpart yet -- logged for Phase-2 design)
        for q in ("tool_needed", "invoke_llm", "escalate"):
            _cmp(q, _noul(q), False, use_prob=True)
        _cmp("tool", s1.choice(answers, "tool", None), "none")

        # SIGNAL (no legacy counterpart)
        _cmp("sentiment", s1.choice(answers, "sentiment", None), legacy_sentiment)
        for q in ("urgent", "high_stakes"):
            _cmp(q, _noul(q), False, use_prob=True)

        self._emit_turn_rows(
            session_id,
            gen,
            {
                "session_id": session_id,
                "transcript": transcript,
                "state": state,
                "rows": rows,
            },
        )

        if not (self._phase2 or self._phase5):
            return

        # ---- Phase-2/5 enforcement (fail-open) -------------------------
        # Every override below requires a present, sufficiently-confident
        # answer; anything weak or missing keeps the legacy value, and an
        # empty ``answers`` dict already returned above. Choice-based fields
        # use ``conf_threshold``; routing flags use the stricter 0.9 action
        # threshold because base checkpoints are over-confident.
        laya_intent = s1.choice(answers, "intent", None)
        if laya_intent is not None:
            ctx.intent = laya_intent

        is_question_prob = _noul("is_question")
        if is_question_prob is not None:
            ctx.is_question = is_question_prob >= 0.5

        laya_sentiment = s1.choice(answers, "sentiment", None)
        if laya_sentiment is not None:
            ctx.user_sentiment = laya_sentiment
            ctx.sentiment_locked = True

        laya_complexity = s1.choice(answers, "query_complexity", None)
        if laya_complexity is not None:
            ctx.query_complexity = laya_complexity
            ctx.complexity_locked = True

        if s1.action_noul(answers, "urgent", default=False):
            ctx.urgent = True
        if s1.action_noul(answers, "escalate", default=False):
            ctx.model_tier = "fallback"

    async def _cadence1_probe(
        self, partial: str, session_id: str
    ) -> None:
        """Cadence-1 (live-listening) System-1 probe.

        Runs as a background task at the partial-transcript cadence (~1s) on
        the freshest partial text. It stashes the verdict on
        ``self._s1_cadence1[session_id]`` for the endpoint/barge paths to
        consume and always shadow-logs Laya-vs-legacy rows (the consumers gate
        actual Enforcement behind ``self._phase3`` / ``self._phase4``). Fails
        open: any error or timeout simply leaves the last good verdict (or none)
        in place.
        """
        s1 = self._s1
        try:
            if not s1.enabled or not partial.strip():
                return
            try:
                ctx = self._ctx(session_id)
                state = {"transcript": partial, "turn_count": ctx.turn_count}
                # Stamp the turn now: if the utterance ends while Laya runs,
                # the verdict must not count for the next turn.
                generation = self._generation(session_id)
                start = time.perf_counter()
                # Low priority: skipped while the model is busy or a
                # turn-level call is waiting, so it never delays a reply.
                answers = await s1.predict(
                    state, CADENCE1_QUESTIONS, priority=False
                )
                self._latency.measure("laya_c1", start)
                if not answers:
                    return
                self._log_latency("laya_c1")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"laya cadence-1 probe failed: {e!r}")
                return

            turn_complete = s1.noul_prob(answers, "turn_complete")
            completion_conf = s1.score(answers, "completion_conf", default=None)
            barge, barge_conf, _ = s1.choice_of(answers, "barge_type")
            self._s1_cadence1[session_id] = {
                "partial": partial,
                "generation": generation,
                "turn_complete": turn_complete,
                "completion_conf": completion_conf,
                "barge_type": barge if barge_conf >= _C1_CONF_REQ else None,
                "barge_conf": barge_conf,
                "ts": time.monotonic(),
            }

            rows: list[dict] = []
            incomplete = self.turn_detector.classifier.is_incomplete(partial)
            rows.append({
                "question": "c1_turn_complete",
                "laya": bool(turn_complete is not None and turn_complete >= 0.5),
                "laya_conf": float(
                    (answers.get("turn_complete") or {}).get("confidence", 0.0)
                ),
                "legacy": not incomplete,
                "match": bool(
                    turn_complete is not None
                    and (turn_complete >= 0.5) == (not incomplete)
                ),
            })
            legacy_barge = BackchannelInterrupt.classify(partial)
            rows.append({
                "question": "c1_barge_type",
                "laya": barge,
                "laya_conf": barge_conf,
                "legacy": legacy_barge,
                "match": bool(barge is not None and barge == legacy_barge),
            })
            s1.log_shadow(
                session_id=session_id,
                transcript=partial,
                state=state,
                rows=rows,
            )
        finally:
            self._s1_c1_inflight.pop(session_id, None)

    # ------------------------------------------------------------------
    # Outcome labels for Laya training
    # ------------------------------------------------------------------

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _log_endpoint_labels(
        self, session_id: str, partials: list[str], final: str, turn_count: int
    ) -> None:
        """Label each partial heard during the utterance by what happened
        next: the user had finished (final adds no words) or kept talking
        (final adds 2+ words). One extra word is ambiguous (STT jitter) and
        skipped. Needs no Laya call, so it costs no GPU time."""
        if not self._s1.enabled or not partials:
            return
        final_len = len(self._words(final))
        seen: set[str] = set()
        for part in partials:
            if part in seen:
                continue
            seen.add(part)
            part_len = len(self._words(part))
            if not part_len:
                continue
            extra = final_len - part_len
            if extra <= 0:
                label = True
            elif extra >= 2:
                label = False
            else:
                continue
            self._s1.log_shadow(
                session_id=session_id,
                transcript=part,
                state={"transcript": part, "turn_count": turn_count},
                rows=[{
                    "question": "turn_complete",
                    "laya": None,
                    "laya_conf": None,
                    "legacy": None,
                    "match": False,
                    "outcome": label,
                }],
            )

    _TOOL_LABELS: ClassVar[frozenset[str]] = frozenset(
        TURN_QUESTIONS["tool"]["criteria"]
    )

    def _apply_tool_outcome(self, rows: list[dict], tools: list[str]) -> None:
        for row in rows:
            if row.get("question") == "tool_needed":
                row["outcome"] = bool(tools)
            elif row.get("question") == "tool":
                if not tools:
                    row["outcome"] = "none"
                elif tools[0] in self._TOOL_LABELS:
                    row["outcome"] = tools[0]

    def _record_turn_tools(
        self, session_id: str, gen: int, tools: list[str]
    ) -> None:
        """The tools this turn really ran (empty = answered without one)."""
        tools = [t for t in tools if t]
        self._turn_outcomes[session_id] = {"gen": gen, "tools": tools}
        pending = self._pending_turn_rows.get(session_id)
        if pending and pending["gen"] == gen:
            self._pending_turn_rows.pop(session_id, None)
            self._apply_tool_outcome(pending["log"]["rows"], tools)
            self._s1.log_shadow(**pending["log"])

    def _emit_turn_rows(self, session_id: str, gen: int, log: dict) -> None:
        """Log turn rows with the turn's tool outcome attached. Shadow-only
        passes run after the turn, so the outcome is usually known; a Phase-2
        pass runs before the LLM, so its rows wait for ``_record_turn_tools``
        (or the turn's end, if it is interrupted first)."""
        outcome = self._turn_outcomes.get(session_id)
        if outcome is not None and outcome["gen"] == gen:
            self._apply_tool_outcome(log["rows"], outcome["tools"])
            self._s1.log_shadow(**log)
            return
        if gen == self._generation(session_id) and session_id in self._current_tasks:
            self._pending_turn_rows[session_id] = {"gen": gen, "log": log}
            return
        self._s1.log_shadow(**log)

    def _prefetch_for(self, entry: dict | None, text: str) -> dict | None:
        """A prefetched tool result, only if it still answers *text*. Time,
        date and dice don't depend on the wording; a web search or a
        calculation must match what the user actually said."""
        if not entry or "task" in entry:
            return None
        tool = entry.get("tool")
        if tool == "search_web":
            if self._words(entry.get("partial", "")) != self._words(text):
                return None
        elif tool == "calculate":
            expr = "".join(str((entry.get("args") or [""])[0]).split())
            if not expr or expr not in "".join(text.split()):
                return None
        return entry

    # ------------------------------------------------------------------
    # Latency Step-1: live-listening tool prefetch (see _prefetch_probe).
    # ------------------------------------------------------------------

    def _consume_prefetch(self, session_id: str) -> dict | None:
        """Take (and clear) a stashed live-listening tool prefetch.

        Returns the entry only when it was computed for THIS utterance (the
        generation stamped while the user spoke still matches -- this runs
        before ``_bump_generation``). The cache is always cleared so a later
        turn can never reuse a stale tool answer.
        """
        entry = self._s1_prefetch.pop(session_id, None)
        if not entry:
            return None
        if entry.get("gen") != self._generation(session_id):
            return None
        return entry

    def _start_paid_prefetch(self, session_id: str, ctx: ConversationContext) -> None:
        """Fire the paid-tool candidate once the user pauses.

        Only when Laya's vote was on the partial the user paused on (so the
        search is for what they actually said), and only once per utterance.
        The search runs as its own task: speech-end does not cancel it (it is
        network I/O, not GPU), and the turn awaits it if the final
        transcript still matches (see ``_resolve_prefetch``).
        """
        cand = self._s1_prefetch_candidate.get(session_id)
        generation = self._generation(session_id)
        if (
            not cand
            or cand["gen"] != generation
            or self._s1_prefetch_paid_gen.get(session_id) == generation
        ):
            return
        current = ctx.last_partial_transcript
        if current != cand["partial"]:
            # The partial moved on since Laya voted (a newer partial, or the
            # endpointer's sync refresh, which runs no probe). A pure
            # extension ("who is nikola" -> "who is nikola tesla") keeps the
            # vote; search the full text. Anything else is a stale vote.
            voted = self._words(cand["partial"])
            if self._words(current)[: len(voted)] != voted:
                return
            args = self._prefetch_args(current, cand["tool"])
            if args is None:
                return
            cand = {**cand, "partial": current, "args": args}
        self._s1_prefetch_paid_gen[session_id] = generation
        self._s1_prefetch_candidate.pop(session_id, None)

        async def _run() -> dict | None:
            try:
                out = await self._tool_registry.execute_call(
                    {"name": cand["tool"], "args": cand["args"]}
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"paid prefetch failed: {e!r}")
                return None
            result = str(out.get("result", ""))
            if ToolRegistry._result_is_bad(result):
                return None
            return {**cand, "result": result}

        self._s1_prefetch[session_id] = {**cand, "task": asyncio.create_task(_run())}
        logger.debug(
            f"paid prefetch session={session_id} tool={cand['tool']} "
            f"partial={cand['partial']!r}"
        )

    @staticmethod
    def _looks_unfinished(text: str) -> bool:
        """True when *text* trails off mid-phrase ("flights to", "the")."""
        words = text.strip().rstrip(",.").split()
        if not words or text.strip().endswith((",", "...")):
            return True
        return words[-1].lower().strip("?!") in _DANGLING_WORDS

    async def _resolve_prefetch(
        self,
        entry: dict | None,
        text: str,
        stt_task: asyncio.Task | None = None,
    ) -> dict | None:
        """Settle an in-flight paid prefetch for *text*.

        A finished (or free-tool) entry is returned as is. A pending one is
        awaited -- bounded by ``_PAID_PREFETCH_WAIT_S`` -- only when it was
        started for these exact words; otherwise it stays pending and
        ``_prefetch_for`` ignores it. With *stt_task* (the speculative path,
        before the final transcript exists), waiting stops early if the final
        transcript lands first and differs from *text*.
        """
        if not entry or "task" not in entry:
            return entry
        if self._words(entry.get("partial", "")) != self._words(text):
            return entry
        task: asyncio.Task = entry["task"]
        start = time.perf_counter()
        try:
            if stt_task is not None and not task.done():
                await asyncio.wait(
                    {task, stt_task},
                    timeout=_PAID_PREFETCH_WAIT_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not task.done() and stt_task.done():
                    heard = None
                    if not stt_task.cancelled() and stt_task.exception() is None:
                        heard = stt_task.result()
                    if heard is not None and self._words(heard) != self._words(text):
                        return entry
            remaining = _PAID_PREFETCH_WAIT_S - (time.perf_counter() - start)
            return await asyncio.wait_for(asyncio.shield(task), max(0.0, remaining))
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            return None
        finally:
            self._latency.measure("prefetch_wait", start)

    @staticmethod
    def _prefetch_args(partial: str, tool: str) -> list[str] | None:
        """Conservative args for a prefetched tool call.

        Only tools whose arguments can be derived directly from the raw
        partial are prefetched. Structured tools (weather city, reminder text)
        return ``None`` so a prefetch never guesses a nonsense argument.
        """
        text = partial.strip().strip("?.!")
        if tool == "roll_dice":
            return ["6"]
        if tool in ("get_time", "get_date"):
            return []
        if tool == "calculate":
            m = StreamingPipeline._FAST_EXPR_RE.search(text)
            return [m.group(0)] if m else None
        if tool == "search_web":
            return [text] if text else None
        return None

    async def _prefetch_probe(self, partial: str, session_id: str) -> None:
        """Latency Step-1: prefetch a confident tool result during speech.

        Runs as a background task on fresh partials. A confident
        ``tool_needed -> tool`` answer (strict 0.9 action threshold) whose args
        can be derived from the partial is executed once and the result stashed
        (with the utterance's generation tag) for ``_consume_prefetch`` to feed
        the LLM's first prompt. ``priority=False`` keeps the call out of the
        model's face while the user speaks; the task is cancelled at
        speech-end. A paid (SerpApi) tool is only recorded as a candidate
        here -- see ``_start_paid_prefetch`` -- so a worst-case miss costs
        exactly one wasted credit per utterance.
        """
        s1 = self._s1
        try:
            if not s1.enabled or len(partial.split()) < 3:
                return
            try:
                generation = self._generation(session_id)
                start = time.perf_counter()
                answers = await s1.predict(
                    {
                        "transcript": partial,
                        "turn_count": self._ctx(session_id).turn_count,
                    },
                    PREFETCH_QUESTIONS,
                    priority=False,
                )
                self._latency.measure("laya_prefetch", start)
                if not answers:
                    return
            except Exception as e:  # noqa: BLE001
                logger.debug(f"laya prefetch probe failed: {e!r}")
                return

            # Gate on `tool` itself (the benchmark's most accurate question,
            # at the normal choice bar). tool_needed never reaches the strict
            # 0.9 action bar, and a wrong guess only costs one lookup.
            tool = s1.choice(answers, "tool", None)
            if not tool or tool == "none":
                return
            args = self._prefetch_args(partial, tool)
            if args is None:
                return
            if tool in _PAID_PREFETCH_TOOLS:
                # Don't spend a credit on a transcript that is still moving:
                # remember the vote; _start_paid_prefetch fires it on a pause.
                self._s1_prefetch_candidate[session_id] = {
                    "gen": generation,
                    "partial": partial,
                    "tool": tool,
                    "args": args,
                }
                return
            out = await self._tool_registry.execute_call({"name": tool, "args": args})
            result = str(out.get("result", ""))
            if not result.strip() or result.strip().lower().startswith("error"):
                return
            self._s1_prefetch[session_id] = {
                "gen": generation,
                "partial": partial,
                "tool": tool,
                "args": args,
                "result": result,
            }
            logger.debug(
                f"laya prefetch session={session_id} tool={tool} "
                f"partial={partial!r}"
            )
        finally:
            self._s1_prefetch_inflight.pop(session_id, None)

    # ------------------------------------------------------------------
    # Phase-2 deterministic routing (used only when BOLO_LAYA_PHASE2=1)
    # ------------------------------------------------------------------

    _FAST_CANNED_REPLIES: ClassVar[dict[str, str]] = {
        "greeting": "Hi! How can I help you today?",
        "farewell": "Goodbye! Have a great day.",
        "backchannel": "Got it.",
    }

    # Whole-word phatic cues. The legacy intent classifier matches fragments
    # ("ok" inside "book", "so" inside "I think so"), so a canned
    # greeting/farewell/backchannel must only fire when the utterance is a
    # briefly-phrased phatic exchange that actually contains the cue -- a stray
    # label on "show me this" or "I think so" must never get "Hi! How can I
    # help?".
    # Whole-word phatic cues per intent, plus a shared filler set. The canned
    # reply is only correct when the WHOLE utterance is the phatic phrase:
    # "hello there" is a greeting, but "ok book a table", "sure cancel my
    # order", "show me this", or "hello world" all mix in real content words
    # and must go to the LLM. A whole-word cue anywhere is therefore NOT enough
    # -- every token must be a known phatic word or an allowed filler (see
    # ``_phatic_gate`` below).
    _FAST_PHATIC_WORDS: ClassVar[dict[str, frozenset[str]]] = {
        "greeting": frozenset({
            "hi", "hello", "hey", "hiya", "yo", "howdy", "greetings",
            "namaste", "sup", "welcome", "hola", "good", "morning",
            "afternoon", "evening", "day", "there",
        }),
        "farewell": frozenset({
            "bye", "goodbye", "good-bye", "bye-bye", "farewell", "goodnight",
            "good-night", "cheerio", "ciao", "adios", "later", "good",
            "night", "see", "ya", "you", "there",
        }),
        "backchannel": frozenset({
            "ok", "okay", "yeah", "yep", "yes", "yup", "sure", "uh", "huh",
            "mhm", "mm", "alright", "right", "got", "it",
            "thanks", "thank", "sounds", "good", "great", "fine", "perfect",
            "cool", "understood", "roger", "there",
        }),
    }

    # Fillers may legally surround the cue in any lane ("hello there",
    # "ok thanks", "thanks so much") without turning the canned reply off.
    _FAST_PHATIC_FILLERS: ClassVar[frozenset[str]] = frozenset({
        "please", "thanks", "thank", "you", "there", "very", "much",
    })

    # Templated-tool detectors. Deliberately narrow so a mismatch can never
    # swallow a real question: time queries reject an "in <place>" modifier,
    # date queries reject possessive/weather continuations, and calc requires
    # the WHOLE utterance to be a calculable expression. A matched template
    # must also be _bare_tail-clean: "what time is it" + nothing (or only
    # "now"/"today") fires, while "... in paris", "... difference between
    # london and tokyo", or "what day is christmas" dial past the template
    # back to the LLM.
    _FAST_TIME_RE = re.compile(
        r"\b(?:what(?:'s| is)? the time|what time is it|current time|"
        r"tell me the time)\b(?!\s+in\b)",
        re.IGNORECASE,
    )
    _FAST_DATE_RE = re.compile(
        r"\b(?:what(?:'s| is)? (?:the |today's )?(?:date|day)|"
        r"what day is it|today's date|what is today)\b(?!'s|\s+\w*weather)",
        re.IGNORECASE,
    )
    _FAST_CALC_RE = re.compile(
        r"^(?:please\s+)?(?:calculate|compute)\s+(.+?)\??\s*$",
        re.IGNORECASE,
    )
    _FAST_EXPR_RE = re.compile(
        r"-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+"
    )

    # Phase-5 complexity-routing gate: whole-utterance acknowledgment phrases
    # that get a deterministic reply INSTEAD of the LLM when a confident Laya
    # complexity verdict marks them trivial. Multi-word only, so they can never
    # collide with the single-word phatic cues above; the exact whole-utterance
    # match (below) means a phrase with extra content words always misses and
    # falls through to the LLM.
    _ACK_REPLIES: ClassVar[dict[str, str]] = {
        "sounds good": "Sounds good!",
        "makes sense": "Makes sense.",
        "got it": "Got it!",
        "no problem": "No problem!",
        "sure thing": "Sure thing!",
        "you bet": "You bet.",
        "that works": "That works!",
        "roger that": "Roger that.",
        "all good": "All good.",
        "no worries": "No worries!",
        "works for me": "Works for me!",
        "fine by me": "Fine by me!",
    }

    @staticmethod
    def _ack_phrase(text: str) -> str | None:
        """Whole-utterance match against the Phase-5 acknowledgment table.

        Case/punct-insensitive but exact otherwise: ``"sounds good"`` and
        ``"sounds good okay"`` differ, so only a genuinely trivial bare
        acknowledgment is ever routed off the LLM."""
        norm = re.sub(r"[^a-z ]", "", text.lower()).strip()
        return StreamingPipeline._ACK_REPLIES.get(norm)

    @staticmethod
    def _bare_tail(
        text: str, start: int, end: int, allowed: tuple[str, ...] = ()
    ) -> bool:
        """True when nothing but trailing punctuation or one of *allowed*
        phrases follows a regex match. Keeps broad fast-path templates from
        swallowing follow-up clauses ("what time is it in Paris", "what's the
        time difference between London and Tokyo", "what day is Christmas")."""
        tail = text[end:].strip(" .?!,")
        return not tail or tail in allowed

    @staticmethod
    def _phatic_gate(text: str, vocab: frozenset[str]) -> bool:
        """True when the WHOLE utterance is a phatic phrase: every content word
        must be a phatic cue for that intent or an allowed filler, and at least
        one real cue must be present. "hello there", "ok thanks", "hi", "bye"
        can, while "hello world", "ok book a table", "sure cancel my order",
        "show me this", or "I think so" all carry real content words and must
        go to the LLM."""
        toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'-]*", text)]
        if not toks or len(toks) > 6:
            return False
        fillers = StreamingPipeline._FAST_PHATIC_FILLERS
        return any(t in vocab for t in toks) and all(
            t in vocab or t in fillers for t in toks
        )

    @staticmethod
    def _fast_ok(result: str) -> bool:
        return bool(result) and not result.strip().lower().startswith("error")

    async def _fast_tool_reply(
        self, transcript: str, tools: list[str] | None = None
    ) -> str | None:
        """Deterministic templated-tool replies; None means: use the LLM.
        The tool actually run is appended to *tools* (outcome labels)."""
        tools = [] if tools is None else tools
        text = transcript.strip()
        if not text:
            return None

        m = self._FAST_TIME_RE.search(text)
        if m and self._bare_tail(text, m.start(), m.end(), ("now", "right now", "today")):
            out = await self._tool_registry.execute_call(
                {"name": "get_time", "args": []}
            )
            if self._fast_ok(str(out["result"])):
                tools.append("get_time")
                return f"It is {out['result']}."
            return None

        m = self._FAST_DATE_RE.search(text)
        if m and self._bare_tail(text, m.start(), m.end(), ("today", "right now")):
            out = await self._tool_registry.execute_call(
                {"name": "get_date", "args": []}
            )
            if self._fast_ok(str(out["result"])):
                tools.append("get_date")
                return f"Today is {out['result']}."
            return None

        m = self._FAST_CALC_RE.search(text)
        if m:
            expr = m.group(1).strip().replace("\u00d7", "*").replace("\u00f7", "/")
            if self._FAST_EXPR_RE.fullmatch(expr.strip()):
                out = await self._tool_registry.execute_call(
                    {"name": "calculate", "args": [expr]}
                )
                if self._fast_ok(str(out["result"])):
                    tools.append("calculate")
                    return f"The answer is {out['result']}."
            return None

        return None

    async def _fast_reply(
        self,
        transcript: str,
        ctx: ConversationContext,
        tools: list[str] | None = None,
    ) -> str | None:
        """Conservative deterministic fast-action table (Phase-2 LLM-skip).

        Returns a ready-to-speak reply or ``None`` to run the LLM as usual.
        ``invoke_llm=False`` is only ever true when this returns a reply; any
        miss (unknown intent, tool failure, bare expression) falls through to
        the LLM. Tools are checked before canned phatic replies so that
        "hi, what time is it" answers the time, not a fixed greeting. Canned
        phatics additionally require a whole-word cue (see ``_phatic_gate``),
        so a fragment-matching intent label can never short-circuit a real
        request.
        """
        tool_reply = await self._fast_tool_reply(transcript, tools)
        if tool_reply is not None:
            return tool_reply
        canned = self._FAST_CANNED_REPLIES.get(ctx.intent)
        if canned is None:
            return None
        phat = self._FAST_PHATIC_WORDS.get(ctx.intent)
        if phat is None or not self._phatic_gate(transcript, phat):
            return None
        return canned

    def _llm_for(self, ctx: ConversationContext) -> LLMProvider:
        """Primary or escalated (fallback) LLM for this turn's answer."""
        if ctx.model_tier == "fallback" and self.llm_fallback is not None:
            return self.llm_fallback
        return self.llm

    def shadow_report(self) -> dict:
        """Phase-1 telemetry: Laya-vs-legacy agreement per question type."""
        return self._s1.shadow_report()

    def system1(self) -> LayaSystem1 | None:
        """Public accessor for the System-1 decision layer (shadow mode)."""
        return self._s1

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

    def _verify_entity_assertions(
        self,
        transcript: str,
        final_text: str,
        supported: set[str],
    ) -> list[str]:
        """Return fabricated (unsupported) entities asserted in `final_text`.

        Runs only for entity-seeking queries. `supported` holds names that came
        from tool results, retrieval, or the user's own words — anything else
        that looks like a specific place/business name is treated as a possible
        hallucination so the pipeline can refuse to let it stand.
        """
        if not EntityGate.query_needs_verification(transcript):
            return []
        return EntityGate.unsupported_entities(
            final_text,
            supported,
            user_input_entities={
                e for e in EntityGate.extract_entities(transcript)
            },
        )

    async def _build_messages(
        self,
        transcript: str,
        ctx: ConversationContext,
        memory: SessionMemory | None,
        retrieval: RetrievalModule | None,
        facts: FactMemory | None = None,
        session_id: str = "default",
        user_repeated: bool = False,
        prefetch: dict | None = None,
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

        if ctx.complexity_locked:
            # Phase-2: Laya supplied a confident complexity label; don't let
            # the legacy heuristic clobber it.
            complexity = ctx.query_complexity
        else:
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

        # Entity queries (specific place/business/entity) MUST be verified via
        # a search round-trip before any concrete name is spoken. Without
        # this hard upstream gate the model can fabricate plausible-looking
        # restaurant/place names that then stream straight to TTS.
        if EntityGate.query_needs_verification(transcript):
            system_prompt += (
                "\n\nENTITY VERIFICATION (mandatory):\n"
                "The user is asking about a specific place, business, or named "
                "entity. You MUST output exactly one tool call -- "
                "{tool:search_places(query=..., location=...)} for somewhere to "
                "eat, stay, shop or visit, otherwise {tool:search_web(query)} -- "
                "and wait for its result before naming any specific place, "
                "restaurant, shop, or business.\n"
                "  - NEVER name, recommend, or describe a specific place, "
                "restaurant, shop, or business unless its name came from a "
                "search result in this turn.\n"
                "  - If you do not have a verified name, say you're not sure "
                "and offer to look it up — never invent one.\n"
                "  - Do not produce any other text before the tool call."
            )

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
                "\n\nConversation summary so far (attributed to speakers, "
                "'user' = the other participant, 'assistant' = you):\n"
                + memory.summary
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

        # Latency Step-1: a tool lookup already ran while the user spoke. Seed
        # the answer with it so the LLM can reply without paying the tool
        # round-trip; it is clearly labeled as a pre-query lookup so the model
        # still calls a tool itself when the data does not cover the question.
        if prefetch and str(prefetch.get("result", "")).strip():
            messages.append({
                "role": "system",
                "content": (
                    "A live-listening lookup already ran and returned:\n"
                    f"{prefetch.get('tool')}: {prefetch.get('result')}\n"
                    "Use it to answer; call the tool again only if it does "
                    "not cover what the user is asking."
                ),
            })

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
