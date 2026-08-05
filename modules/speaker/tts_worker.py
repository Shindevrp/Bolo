from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from modules.speaker.queue import SpeakerQueue, SpeakerQueueItem
from modules.tts.prosody import ProsodyProfile
from providers.tts.base import TTSProvider
from utils.logger import get_logger

logger = get_logger("speaker_tts_worker")

_SENTINEL = None  # End-of-stream marker


class SpeakerTTSWorker:
    """Per-speaker TTS consumer that runs concurrently.

    Each worker owns a TTSProvider (one voice model) and consumes from
    a dedicated SpeakerQueue. Supports pause-at-sentence-end for
    speaker-to-speaker interrupts.
    """

    def __init__(
        self,
        speaker: str,
        tts: TTSProvider,
        queue: SpeakerQueue,
        output_queue: asyncio.Queue,
        session_id: str,
    ) -> None:
        self.speaker = speaker
        self.tts = tts
        self.queue = queue
        self._output_queue = output_queue
        self._session_id = session_id
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()  # Set = pause after current sentence
        self._interrupt_event = asyncio.Event()  # Set = hard interrupt
        self._total_bytes = 0
        self._first_emit: float | None = None
        self._spoken: list[str] = []

    @property
    def busy(self) -> bool:
        return self.queue.busy

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._interrupt_event.clear()
        self._total_bytes = 0
        self._first_emit = None
        self._spoken = []
        self._task = asyncio.create_task(self._run(), name=f"tts_worker_{self.speaker}")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    def pause_at_sentence_end(self) -> None:
        """Signal the worker to finish the current sentence and then pause."""
        self._pause_event.set()

    def hard_interrupt(self) -> None:
        """Immediately stop synthesis."""
        self._interrupt_event.set()
        self._stop_event.set()

    async def wait_done(self) -> None:
        if self._task:
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def wait_idle(self, timeout: float = 5.0) -> None:
        """Wait until the worker has consumed all queued items and gone idle.

        Workers persist across turns (they loop until stopped), so we can't
        await the task itself; we wait for the queue to drain and synthesis
        to finish so the end-of-turn bookkeeping can proceed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.queue.empty and not self.queue.busy:
                return
            await asyncio.sleep(0.05)

    async def _run(self) -> None:
        sr = self.tts.sample_rate

        async def _one(text: str) -> AsyncGenerator[str, None]:
            yield text

        try:
            while not self._stop_event.is_set():
                item = await self.queue.get(timeout=0.1)
                if item is None:
                    continue

                if item.text is _SENTINEL:
                    break

                self.queue.busy = True
                self._spoken.append(item.text)

                # If pause was requested, finish this chunk then stop
                should_pause = self._pause_event.is_set()
                should_interrupt = self._interrupt_event.is_set()

                if should_interrupt:
                    break

                try:
                    async for audio_chunk in self.tts.synthesize_stream(
                        _one(item.text), prosody=item.prosody
                    ):
                        if self._stop_event.is_set() or self._interrupt_event.is_set():
                            break
                        if isinstance(audio_chunk, bytes) and len(audio_chunk) > 0:
                            if self._first_emit is None:
                                self._first_emit = time.monotonic()
                            self._total_bytes += len(audio_chunk)
                            await self._output_queue.put(
                                (self.speaker, audio_chunk, item.is_urgent)
                            )
                except Exception as e:
                    logger.error(
                        f"tts error speaker={self.speaker} error={e}"
                    )

                self.queue.busy = False

                if should_pause:
                    # Emit a sentence-end signal so the next speaker can start
                    await self._output_queue.put(
                        (self.speaker, b"", False)  # Empty = sentence end
                    )
                    self._pause_event.clear()
                    break

        except asyncio.CancelledError:
            pass
        finally:
            self.queue.busy = False
            if self._interrupt_event.is_set():
                await self.queue.clear()

    def get_status(self) -> dict:
        return {
            "speaker": self.speaker,
            "busy": self.busy,
            "running": self.is_running,
            "total_bytes": self._total_bytes,
            "spoken": len(self._spoken),
        }
