from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from faster_whisper import WhisperModel

from providers.stt.base import STTProvider


class FasterWhisperSTT(STTProvider):
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        vad_threshold: float = 0.5,
    ) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        self.vad_threshold = vad_threshold

    async def transcribe_stream(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)
            if len(buffer) < 16000:  # ~1s of audio at 16kHz
                continue
            chunk_to_process = bytes(buffer)
            buffer.clear()
            result = await asyncio.to_thread(
                self._transcribe_segment, chunk_to_process
            )
            if result:
                yield result

    def _transcribe_segment(self, audio_bytes: bytes) -> str | None:
        import io
        import wave

        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            wav_buffer.seek(0)
            segments, _ = self.model.transcribe(
                wav_buffer,
                language=self.language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    threshold=self.vad_threshold,
                ),
            )
            text = " ".join(seg.text for seg in segments)
            return text.strip() or None

    async def transcribe(self, audio_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_segment, audio_bytes) or ""