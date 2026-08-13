from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass
from typing import AsyncGenerator

from utils.logger import get_logger

logger = get_logger("prosody_events")


# Pre-generated breath sound (60ms of filtered noise at 22050 Hz)
# This creates a subtle "inhale" effect before speech
_BREATH_DURATION_MS = 60
_SAMPLE_RATE = 22050


def _generate_breath_audio(duration_ms: int = _BREATH_DURATION_MS) -> bytes:
    """Generate a subtle breath/inhale sound.

    Creates soft filtered noise that sounds like a quick inhale.
    """
    import random
    samples = int(_SAMPLE_RATE * duration_ms / 1000)
    audio = bytearray(samples * 2)  # 16-bit mono

    # Generate soft noise with amplitude envelope
    for i in range(samples):
        t = i / samples
        # Envelope: quick rise, slower fall (inhale shape)
        envelope = t * (1.0 - t) * 4.0
        envelope = min(1.0, envelope)
        # Low amplitude (barely audible)
        amplitude = int(random.gauss(0, 1500) * envelope)
        amplitude = max(-32768, min(32767, amplitude))
        struct.pack_into("<h", audio, i * 2, amplitude)

    return bytes(audio)


def _generate_micro_pause(duration_ms: int = 50) -> bytes:
    """Generate a micro-pause (brief silence)."""
    samples = int(_SAMPLE_RATE * duration_ms / 1000)
    return b"\x00\x00" * samples


# Pre-generate and cache
_BREATH_AUDIO = None


def _get_breath() -> bytes:
    global _BREATH_AUDIO
    if _BREATH_AUDIO is None:
        _BREATH_AUDIO = _generate_breath_audio()
    return _BREATH_AUDIO


@dataclass
class ProsodyEvent:
    event_type: str  # "breath", "micro_pause", "silence", "emphasis"
    audio: bytes
    duration_ms: int
    speaker: str = ""


class ProsodyEventGenerator:
    """Generates micro-prosody events for natural conversation.

    Adds subtle audio cues that make speech feel more human:
    - Breath sounds before speech onset
    - Micro-pauses at clause boundaries
    - Emphasis pauses for dramatic effect
    """

    def __init__(self) -> None:
        self._breath = _get_breath()

    def before_speech(self, speaker: str) -> ProsodyEvent:
        """Generate a pre-speech breath sound."""
        return ProsodyEvent(
            event_type="breath",
            audio=self._breath,
            duration_ms=_BREATH_DURATION_MS,
            speaker=speaker,
        )

    def micro_pause(self, duration_ms: int = 50, speaker: str = "") -> ProsodyEvent:
        """Generate a micro-pause."""
        return ProsodyEvent(
            event_type="micro_pause",
            audio=_generate_micro_pause(duration_ms),
            duration_ms=duration_ms,
            speaker=speaker,
        )

    def clause_boundary_pause(self, speaker: str = "") -> ProsodyEvent:
        """Generate a pause at a clause boundary (comma, semicolon)."""
        return self.micro_pause(duration_ms=80, speaker=speaker)

    def sentence_boundary_pause(self, speaker: str = "") -> ProsodyEvent:
        """Generate a longer pause at sentence end."""
        return self.micro_pause(duration_ms=150, speaker=speaker)

    def interrupt_prep(self, speaker: str) -> ProsodyEvent:
        """Generate the prep sequence before an interrupt.

        Returns a breath + micro-pause combo.
        """
        return self.before_speech(speaker)


class AdaptiveOverlapCalculator:
    """Calculates overlap duration based on interrupt type and context."""

    _BASE_OVERLAPS = {
        "soft_interrupt": 120,
        "excited_interrupt": 300,
        "correction": 180,
        "enhancement": 150,
        "skepticism": 100,
        "completion": 200,
        "default": 200,
    }

    def calculate(
        self,
        interrupt_type: str,
        urgency: float,
        claim_strength: float,
        is_clause_boundary: bool,
    ) -> int:
        """Calculate adaptive overlap in milliseconds.

        Factors:
        - Base overlap for interrupt type
        - Urgency modifier (higher = more overlap)
        - Claim strength modifier (stronger claims get less overlap)
        - Clause boundary (natural pause = less overlap needed)
        """
        base = self._BASE_OVERLAPS.get(interrupt_type, 200)

        # Urgency modifier: +50ms at high urgency, -30ms at low
        urgency_mod = (urgency - 0.5) * 60

        # Claim strength modifier: strong claims get shorter overlap
        claim_mod = -(claim_strength * 40)

        # Clause boundary: less overlap needed (natural pause exists)
        boundary_mod = -40 if is_clause_boundary else 0

        overlap = base + urgency_mod + claim_mod + boundary_mod
        return max(80, min(500, int(overlap)))
