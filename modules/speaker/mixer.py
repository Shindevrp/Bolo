from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator

from utils.logger import get_logger

logger = get_logger("audio_mixer")


@dataclass
class MixerEvent:
    speaker: str
    audio: bytes  # Empty = sentence end signal
    is_urgent: bool
    is_interrupt: bool = False


class AudioMixer:
    """Handles dual-speaker audio output with overlap support.

    Scenarios:
    1. Sequential: Sh finishes → Ti starts (current behavior)
    2. Overlap: Sh's audio plays while Ti starts synthesizing
    3. Interrupt: Ti cuts off Sh mid-utterance (crossfade)

    The mixer receives tagged audio from per-speaker workers and
    emits a single stream to the client, with speaker metadata.
    """

    def __init__(
        self,
        overlap_ms: int = 200,
        crossfade_ms: int = 100,
    ) -> None:
        self.overlap_ms = overlap_ms
        self.crossfade_ms = crossfade_ms
        self._output_queue: asyncio.Queue[MixerEvent] = asyncio.Queue(maxsize=512)

    @property
    def output_queue(self) -> asyncio.Queue[MixerEvent]:
        return self._output_queue

    async def emit(
        self,
        speaker: str,
        audio: bytes,
        is_urgent: bool = False,
        is_interrupt: bool = False,
    ) -> None:
        """Emit a tagged audio event to the output queue."""
        event = MixerEvent(
            speaker=speaker,
            audio=audio,
            is_urgent=is_urgent,
            is_interrupt=is_interrupt,
        )
        await self._output_queue.put(event)

    async def emit_sentence_end(self, speaker: str) -> None:
        """Signal that a speaker finished a sentence."""
        await self._output_queue.put(
            MixerEvent(speaker=speaker, audio=b"", is_urgent=False)
        )

    async def emit_interrupt(self, speaker: str) -> None:
        """Signal that a speaker was interrupted."""
        await self._output_queue.put(
            MixerEvent(speaker=speaker, audio=b"", is_urgent=False, is_interrupt=True)
        )
