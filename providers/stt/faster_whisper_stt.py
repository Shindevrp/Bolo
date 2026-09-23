from __future__ import annotations

import asyncio
import numpy as np
from typing import AsyncGenerator

from faster_whisper import WhisperModel

from providers.stt.base import STTProvider


class FasterWhisperSTT(STTProvider):
    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ) -> None:
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            num_workers=1,
        )
        self.language = language

    async def transcribe_stream(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)
            if len(buffer) < 16000:
                continue
            chunk_to_process = bytes(buffer)
            buffer.clear()
            result = await asyncio.to_thread(
                self._transcribe_segment, chunk_to_process
            )
            if result:
                yield result

    def _transcribe_segment(self, audio_bytes: bytes) -> str | None:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return None
        # Whisper is sensitive to overall level: boost quiet mic captures up to
        # a sane ceiling so clipped/tiny signal doesn't degrade into garbage.
        peak = float(np.max(np.abs(audio)))
        if 0.0 < peak < 0.25:
            audio = audio * min(1.0 / peak, 4.0)
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            # Skip pure-silence / noise tails inside the capture so Whisper
            # isn't nudged into hallucinating tokens on near-silent audio.
            vad_filter=True,
            # Conditioning on the previous text makes Whisper lock onto (and
            # repeat) its own prior output on noisy/short audio — the classic
            # source of trailing-token hallucinations. Turn it off.
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text for seg in segments)
        return self._trim_repetition(text).strip() or None

    @staticmethod
    def _trim_repetition(text: str) -> str:
        
        if not text:
            return text
        words = text.split()
        if len(words) < 5:
            return text
        tail = words[-1].lower().strip(".,!?;:'\"")
        if not tail:
            return text
        run = 0
        for w in reversed(words):
            if w.lower().strip(".,!?;:'\"") == tail:
                run += 1
            else:
                break
        return " ".join(words[:-run]).strip() if run >= 3 else text

    async def transcribe(self, audio_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_segment, audio_bytes) or ""