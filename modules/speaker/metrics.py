from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger

logger = get_logger("speaker_metrics")


@dataclass
class InterruptEvent:
    timestamp: float
    interrupter: str
    interrupted: str
    interrupt_type: str  # "correction", "enhancement", "skepticism", "completion"
    latency_ms: float
    overlap_ms: float
    confidence: float
    was_natural: bool = True  # Will be determined by post-analysis


@dataclass
class TurnEvent:
    timestamp: float
    speaker: str
    duration_ms: float
    was_interrupted: bool = False
    was_resume: bool = False


class SpeakerMetrics:
    """Tracks multi-speaker conversation metrics for analysis and improvement."""

    def __init__(self) -> None:
        self._interrupts: list[InterruptEvent] = []
        self._turns: list[TurnEvent] = []
        self._session_start: float = time.monotonic()

    def record_interrupt(
        self,
        interrupter: str,
        interrupted: str,
        interrupt_type: str,
        latency_ms: float,
        overlap_ms: float,
        confidence: float,
    ) -> None:
        event = InterruptEvent(
            timestamp=time.monotonic(),
            interrupter=interrupter,
            interrupted=interrupted,
            interrupt_type=interrupt_type,
            latency_ms=latency_ms,
            overlap_ms=overlap_ms,
            confidence=confidence,
        )
        self._interrupts.append(event)
        logger.info(
            f"interrupt: {interrupter} interrupted {interrupted} "
            f"type={interrupt_type} latency={latency_ms:.0f}ms "
            f"overlap={overlap_ms:.0f}ms confidence={confidence:.2f}"
        )

    def record_turn(
        self,
        speaker: str,
        duration_ms: float,
        was_interrupted: bool = False,
        was_resume: bool = False,
    ) -> None:
        event = TurnEvent(
            timestamp=time.monotonic(),
            speaker=speaker,
            duration_ms=duration_ms,
            was_interrupted=was_interrupted,
            was_resume=was_resume,
        )
        self._turns.append(event)

    def get_summary(self) -> dict[str, Any]:
        total_turns = len(self._turns)
        total_interrupts = len(self._interrupts)

        # Per-speaker stats
        speaker_turns: dict[str, int] = {}
        speaker_interrupts: dict[str, int] = {}
        for t in self._turns:
            speaker_turns[t.speaker] = speaker_turns.get(t.speaker, 0) + 1
        for i in self._interrupts:
            speaker_interrupts[i.interrupter] = speaker_interrupts.get(i.interrupter, 0) + 1

        # Timing stats
        avg_latency = 0.0
        avg_overlap = 0.0
        if self._interrupts:
            avg_latency = sum(i.latency_ms for i in self._interrupts) / len(self._interrupts)
            avg_overlap = sum(i.overlap_ms for i in self._interrupts) / len(self._interrupts)

        # Interrupt type distribution
        type_counts: dict[str, int] = {}
        for i in self._interrupts:
            type_counts[i.interrupt_type] = type_counts.get(i.interrupt_type, 0) + 1

        # Naturalness (simple heuristic: shorter latency = more natural)
        natural_count = sum(1 for i in self._interrupts if i.latency_ms < 500)

        return {
            "session_duration_s": time.monotonic() - self._session_start,
            "total_turns": total_turns,
            "total_interrupts": total_interrupts,
            "speaker_turns": speaker_turns,
            "speaker_interrupts": speaker_interrupts,
            "avg_interrupt_latency_ms": round(avg_latency, 1),
            "avg_overlap_ms": round(avg_overlap, 1),
            "interrupt_type_distribution": type_counts,
            "natural_interrupt_ratio": (
                round(natural_count / total_interrupts, 2)
                if total_interrupts > 0 else 0.0
            ),
        }

    def to_debug_log(self) -> str:
        """Format metrics as a debug log string."""
        summary = self.get_summary()
        lines = [
            f"Session: {summary['session_duration_s']:.0f}s, "
            f"{summary['total_turns']} turns, "
            f"{summary['total_interrupts']} interrupts",
        ]
        for speaker, count in summary["speaker_turns"].items():
            lines.append(f"  {speaker}: {count} turns")
        if summary["total_interrupts"] > 0:
            lines.append(
                f"  Avg latency: {summary['avg_interrupt_latency_ms']:.0f}ms, "
                f"Avg overlap: {summary['avg_overlap_ms']:.0f}ms"
            )
            lines.append(
                f"  Natural ratio: {summary['natural_interrupt_ratio']:.0%}"
            )
        return "\n".join(lines)
