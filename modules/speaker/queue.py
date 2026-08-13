from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from modules.tts.prosody import ProsodyProfile


@dataclass(order=True)
class SpeakerQueueItem:
    """A single item in a per-speaker priority queue."""

    priority: int  # 0=urgent/backchannel, 1=normal
    seq: int  # Monotonic sequence for FIFO within same priority
    text: str = field(compare=False)
    prosody: ProsodyProfile | None = field(default=None, compare=False)
    speaker: str = field(default="", compare=False)
    is_urgent: bool = field(default=False, compare=False)
    urgency_tag: str = field(default="", compare=False)


class SpeakerQueue:
    """Per-speaker priority queue with interrupt support."""

    def __init__(self, speaker: str, maxsize: int = 128) -> None:
        self.speaker = speaker
        self._queue: asyncio.PriorityQueue[SpeakerQueueItem] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._seq = 0
        self._busy = False  # True when TTS is actively synthesizing

    @property
    def busy(self) -> bool:
        return self._busy

    @busy.setter
    def busy(self, value: bool) -> None:
        self._busy = value

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    async def push(
        self,
        text: str,
        priority: int = 1,
        prosody: ProsodyProfile | None = None,
        is_urgent: bool = False,
        urgency_tag: str = "",
    ) -> None:
        self._seq += 1
        item = SpeakerQueueItem(
            priority=priority,
            seq=self._seq,
            text=text,
            prosody=prosody,
            speaker=self.speaker,
            is_urgent=is_urgent,
            urgency_tag=urgency_tag,
        )
        await self._queue.put(item)

    def push_nowait(
        self,
        text: str,
        priority: int = 1,
        prosody: ProsodyProfile | None = None,
        is_urgent: bool = False,
        urgency_tag: str = "",
    ) -> None:
        self._seq += 1
        item = SpeakerQueueItem(
            priority=priority,
            seq=self._seq,
            text=text,
            prosody=prosody,
            speaker=self.speaker,
            is_urgent=is_urgent,
            urgency_tag=urgency_tag,
        )
        self._queue.put_nowait(item)

    async def get(self, timeout: float = 0.2) -> SpeakerQueueItem | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def put_back(self, item: SpeakerQueueItem) -> None:
        """Put an item back at the front of the queue."""
        self._queue.put_nowait(item)

    async def drain(self) -> list[SpeakerQueueItem]:
        """Remove all items and return them."""
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    async def drain_urgent(self) -> list[SpeakerQueueItem]:
        """Remove all non-urgent items (keep urgent ones)."""
        kept: list[SpeakerQueueItem] = []
        removed: list[SpeakerQueueItem] = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item.is_urgent:
                    kept.append(item)
                else:
                    removed.append(item)
            except asyncio.QueueEmpty:
                break
        # Re-add urgent items
        for item in kept:
            self._queue.put_nowait(item)
        return removed

    async def clear(self) -> None:
        """Remove all items."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def clear_nowait(self) -> None:
        """Remove all items synchronously (for sync callers)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @staticmethod
    async def _drain_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
