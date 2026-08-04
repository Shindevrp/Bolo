from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from modules.speaker.profile import SpeakerProfile
from modules.speaker.queue import SpeakerQueue
from modules.speaker.tts_worker import SpeakerTTSWorker
from modules.speaker.interrupt_policy import InterruptPolicy, InterruptStyle, InterruptType
from modules.speaker.mixer import AudioMixer
from modules.speaker.stream_state import StreamState
from modules.speaker.speaker_predictor import SpeakerPredictor
from modules.speaker.resume import ResumeManager
from modules.speaker.metrics import SpeakerMetrics
from modules.speaker.prosody_events import AdaptiveOverlapCalculator, ProsodyEventGenerator
from utils.logger import get_logger

logger = get_logger("speaker_coordinator")


@dataclass
class SpeakerSegment:
    speaker: str
    text: str
    is_urgent: bool = False
    urgency_tag: str = ""


class SpeakerCoordinator:
    """Manages multi-speaker conversation flow with full intelligence.

    Features:
    - Token-level streaming with per-speaker queues
    - Multi-factor interrupt decisions
    - Incremental intent detection via StreamState
    - Speaker prediction before tags appear
    - Resume-after-interrupt with checkpoints
    - Adaptive audio overlap
    - Metrics tracking
    """

    def __init__(
        self,
        speakers: list[SpeakerProfile],
        interrupt_policy: InterruptPolicy | None = None,
        overlap_ms: int = 200,
    ) -> None:
        self.speakers = {s.name: s for s in speakers}
        self.speaker_order = [s.name for s in speakers]
        self.interrupt_policy = interrupt_policy or InterruptPolicy()
        self.overlap_ms = overlap_ms

        # Per-speaker state
        self._turn_counts: dict[str, int] = {name: 0 for name in self.speaker_order}
        self._consecutive: dict[str, int] = {name: 0 for name in self.speaker_order}
        self.last_speaker: str | None = None

        # Intelligence components
        self._stream_states: dict[str, StreamState] = {
            name: StreamState() for name in self.speaker_order
        }
        self._predictor = SpeakerPredictor(known_speakers=self.speaker_order)
        self._resume_manager = ResumeManager()
        self._metrics = SpeakerMetrics()
        self._overlap_calc = AdaptiveOverlapCalculator()
        self._prosody_gen = ProsodyEventGenerator()

        # Infrastructure (set up by init_workers)
        self._queues: dict[str, SpeakerQueue] = {}
        self._workers: dict[str, SpeakerTTSWorker] = {}
        self._mixer = AudioMixer(overlap_ms=overlap_ms)
        self._initialized = False

        # Wire metrics into interrupt policy
        self.interrupt_policy.set_metrics(self._metrics)

    @property
    def enabled(self) -> bool:
        return len(self.speakers) >= 2

    def init_workers(
        self,
        tts_provider,
        session_id: str,
    ) -> None:
        """Initialize per-speaker queues and TTS workers."""
        if self._initialized:
            return

        for name in self.speaker_order:
            queue = SpeakerQueue(speaker=name)
            self._queues[name] = queue

            voice = tts_provider.get_voice(name)
            if voice is None:
                logger.warning(f"no voice found for speaker {name}")
                continue

            worker = SpeakerTTSWorker(
                speaker=name,
                tts=voice,
                queue=queue,
                output_queue=asyncio.Queue(maxsize=512),
                session_id=session_id,
            )
            self._workers[name] = worker

        self._initialized = True
        logger.info(
            f"speaker coordinator initialized: {list(self._workers.keys())}"
        )

    def start_workers(self) -> None:
        """Start all TTS workers."""
        for worker in self._workers.values():
            worker.start()

    async def stop_workers(self) -> None:
        """Stop all TTS workers."""
        for worker in self._workers.values():
            await worker.stop()
        self._workers.clear()
        self._queues.clear()
        self._initialized = False

    def select_next_speaker(
        self,
        user_transcript: str = "",
        addressed_name: str | None = None,
    ) -> str:
        """Decide which speaker goes first after user input."""
        if addressed_name and addressed_name in self.speakers:
            return addressed_name

        lower = user_transcript.lower()
        for name in self.speaker_order:
            if name.lower() in lower:
                return name

        # Use predictor if available
        predicted, confidence = self._predictor.predict(
            current_speaker=self.last_speaker or "",
            recent_tokens=[],
            stream_state=self._stream_states[self.speaker_order[0]],
        )
        if predicted and confidence > 0.5:
            return predicted

        # Force switch if one speaker has dominated
        if self.last_speaker and self._consecutive.get(self.last_speaker, 0) >= 2:
            for name in self.speaker_order:
                if name != self.last_speaker:
                    return name

        # Default: alternate
        if self.last_speaker:
            idx = self.speaker_order.index(self.last_speaker)
            next_idx = (idx + 1) % len(self.speaker_order)
            return self.speaker_order[next_idx]

        return self.speaker_order[0]

    def should_interrupt(
        self,
        incoming_speaker: str,
        incoming_text: str,
    ) -> tuple[InterruptStyle, InterruptType, int]:
        """Check if the incoming speaker should interrupt.

        Returns (style, type, overlap_ms).
        """
        if not self.last_speaker or self.last_speaker == incoming_speaker:
            return InterruptStyle.NONE, InterruptType.ENHANCEMENT, 0

        current_turns = self._consecutive.get(self.last_speaker, 0)

        # Check StreamState for incremental intent
        stream_state = self._stream_states.get(incoming_speaker)
        should_preempt, preempt_reason = stream_state.should_trigger_interrupt() if stream_state else (False, "")

        decision = self.interrupt_policy.evaluate(
            incoming_text=incoming_text,
            current_speaker=self.last_speaker,
            incoming_speaker=incoming_speaker,
            current_speaker_turns=current_turns,
            stream_state=stream_state,
        )

        # Calculate adaptive overlap
        overlap = self._overlap_calc.calculate(
            interrupt_type=decision.interrupt_type.value,
            urgency=decision.confidence,
            claim_strength=stream_state.claim_strength if stream_state else 0.5,
            is_clause_boundary=stream_state.is_at_clause_boundary() if stream_state else False,
        )

        if decision.style != InterruptStyle.NONE:
            logger.info(
                f"interrupt: {incoming_speaker} → {self.last_speaker} "
                f"type={decision.interrupt_type.value} "
                f"style={decision.style.name} "
                f"confidence={decision.confidence:.2f} "
                f"overlap={overlap}ms "
                f"reason={decision.reason}"
            )

        return decision.style, decision.interrupt_type, overlap

    def update_stream_state(self, speaker: str, token: str) -> None:
        """Update the real-time stream state for a speaker."""
        state = self._stream_states.get(speaker)
        if state:
            state.update(token)

    def get_stream_state(self, speaker: str) -> StreamState | None:
        return self._stream_states.get(speaker)

    def predict_next_speaker(self, current_speaker: str) -> str | None:
        """Predict who speaks next (before tag appears)."""
        state = self._stream_states.get(current_speaker)
        predicted, confidence = self._predictor.predict(
            current_speaker=current_speaker,
            recent_tokens=[],
            stream_state=state,
        )
        if confidence > 0.5:
            return predicted
        return None

    def checkpoint_on_interrupt(
        self,
        speaker: str,
        current_response: str,
        topic_context: str = "",
    ) -> None:
        """Create a checkpoint when a speaker is interrupted."""
        self._resume_manager.checkpoint(
            speaker=speaker,
            full_response=current_response,
            topic_context=topic_context,
            turn_count=self._turn_counts.get(speaker, 0),
        )

    def should_resume(self, speaker: str) -> bool:
        return self._resume_manager.should_resume(speaker)

    def get_resume_messages(self, speaker: str) -> list[dict[str, str]] | None:
        """Get LLM messages for resuming after interrupt."""
        partner = self._other_than(speaker)
        return self._resume_manager.get_resume_messages(
            speaker, f"Your partner {partner} just finished speaking."
        )

    def record_resume(self, speaker: str) -> None:
        cp = self._resume_manager.get_checkpoint(speaker)
        if cp:
            self._resume_manager.record_resume(speaker, cp.last_complete_clause)

    async def push_chunk(
        self,
        speaker: str,
        text: str,
        prosody=None,
        is_urgent: bool = False,
        urgency_tag: str = "",
    ) -> None:
        """Push a text chunk to a speaker's queue."""
        queue = self._queues.get(speaker)
        if not queue:
            return

        priority = 0 if is_urgent else 1
        await queue.push(
            text=text,
            priority=priority,
            prosody=prosody,
            is_urgent=is_urgent,
            urgency_tag=urgency_tag,
        )

    def pause_speaker(self, speaker: str) -> None:
        """Tell a speaker's worker to finish current sentence and pause."""
        worker = self._workers.get(speaker)
        if worker:
            worker.pause_at_sentence_end()

    def interrupt_speaker(self, speaker: str) -> None:
        """Immediately stop a speaker's TTS."""
        worker = self._workers.get(speaker)
        if worker:
            worker.hard_interrupt()

    def record_turn(self, speaker: str) -> None:
        """Record that a speaker took a turn."""
        if self.last_speaker and self.last_speaker != speaker:
            self._consecutive[self.last_speaker] = 0
        self._consecutive[speaker] = self._consecutive.get(speaker, 0) + 1
        self._turn_counts[speaker] = self._turn_counts.get(speaker, 0) + 1
        self.last_speaker = speaker
        self._predictor.record_turn(speaker, "")

    def get_worker_output(self, speaker: str) -> asyncio.Queue | None:
        worker = self._workers.get(speaker)
        if worker:
            return worker._output_queue
        return None

    def get_all_worker_outputs(self) -> dict[str, asyncio.Queue]:
        return {
            name: worker._output_queue
            for name, worker in self._workers.items()
        }

    def get_prosody_events(self, speaker: str) -> ProsodyEventGenerator:
        return self._prosody_gen

    def inter_speaker_pause(self, prev_speaker: str, next_speaker: str) -> float:
        if prev_speaker == next_speaker:
            return 0.0
        return 0.4

    def speaker_names(self) -> list[str]:
        return list(self.speaker_order)

    def _other_than(self, name: str | None) -> str:
        if name and name in self.speakers:
            for s in self.speaker_order:
                if s != name:
                    return s
        return self.speaker_order[0]

    def get_metrics_summary(self) -> dict:
        return self._metrics.get_summary()

    def get_status(self) -> dict:
        return {
            "speakers": self.speaker_order,
            "last_speaker": self.last_speaker,
            "turn_counts": dict(self._turn_counts),
            "consecutive": dict(self._consecutive),
            "metrics": self._metrics.get_summary(),
            "workers": {
                name: worker.get_status()
                for name, worker in self._workers.items()
            },
        }
