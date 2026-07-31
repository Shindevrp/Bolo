from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

import piper
import numpy as np

from providers.tts.base import TTSProvider


class PiperTTS(TTSProvider):
    def __init__(
        self,
        model_path: str,
        model_config_path: str | None = None,
        sentence_silence: float = 0.02,
        length_scale: float = 0.65,
        noise_scale: float = 0.4,
        noise_w: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_config_path = (
            Path(model_config_path) if model_config_path else None
        )
        self.sentence_silence = sentence_silence
        self._syn_config = piper.SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )
        self._voice = None
        self._sample_rate: int = 22050
        self._warm_up()

    def _warm_up(self) -> None:
        _ = self.voice

    @property
    def voice(self) -> piper.PiperVoice:
        if self._voice is None:
            self._voice = piper.PiperVoice.load(
                self.model_path,
                config_path=self.model_config_path,
                use_cuda=True,
            )
            self._sample_rate = self._voice.config.sample_rate
        return self._voice

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _split_sentences(self, text: str) -> list[str]:
        import re
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _synthesize_sentence_chunks(self, sentence: str) -> list[bytes]:
        chunks = []
        for chunk in self.voice.synthesize(sentence, syn_config=self._syn_config):
            chunks.append(bytes(chunk.audio_int16_bytes))
        return chunks

    def _synthesize_to_bytes(self, sentence: str) -> bytes:
        audio = bytearray()
        for chunk in self.voice.synthesize(sentence, syn_config=self._syn_config):
            audio.extend(chunk.audio_int16_bytes)
        return bytes(audio)

    async def synthesize_stream(
        self, text_chunks: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        buffer = ""
        async for chunk in text_chunks:
            buffer += chunk
            sentences = self._split_sentences(buffer)
            if not sentences:
                continue
            for sentence in sentences[:-1]:
                chunks = await asyncio.to_thread(self._synthesize_sentence_chunks, sentence)
                for audio_chunk in chunks:
                    yield audio_chunk
                    await asyncio.sleep(0)
            buffer = sentences[-1]

        if buffer.strip():
            chunks = await asyncio.to_thread(
                self._synthesize_sentence_chunks, buffer.strip()
            )
            for audio_chunk in chunks:
                yield audio_chunk

    async def synthesize(self, text: str) -> bytes:
        audio = bytearray()
        sentences = self._split_sentences(text)
        for sentence in sentences:
            chunk = await asyncio.to_thread(self._synthesize_to_bytes, sentence)
            audio.extend(chunk)
        return bytes(audio)