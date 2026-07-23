from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

import piper

from providers.tts.base import TTSProvider


class PiperTTS(TTSProvider):
    def __init__(
        self,
        model_path: str,
        model_config_path: str | None = None,
        sentence_silence: float = 0.15,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_config_path = (
            Path(model_config_path) if model_config_path else None
        )
        self.sentence_silence = sentence_silence
        self._syn_config = piper.SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
        )
        self._voice = None
        self._sample_rate = 22050

    @property
    def voice(self) -> piper.PiperVoice:
        if self._voice is None:
            self._voice = piper.PiperVoice.load(
                self.model_path, config_path=self.model_config_path
            )
            self._sample_rate = self._voice.config.sample_rate
        return self._voice

    def _split_sentences(self, text: str) -> list[str]:
        import re
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _synthesize_sentence(self, sentence: str) -> bytes:
        audio_chunks = list(
            self.voice.synthesize(sentence, syn_config=self._syn_config)
        )
        audio = bytearray()
        for chunk in audio_chunks:
            audio.extend(chunk.audio.tobytes())
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
                audio = await asyncio.to_thread(self._synthesize_sentence, sentence)
                if audio:
                    yield audio
                    await asyncio.sleep(self.sentence_silence)
            buffer = sentences[-1]

        if buffer.strip():
            audio = await asyncio.to_thread(
                self._synthesize_sentence, buffer.strip()
            )
            if audio:
                yield audio

    async def synthesize(self, text: str) -> bytes:
        audio = bytearray()
        sentences = self._split_sentences(text)
        for sentence in sentences:
            chunk = await asyncio.to_thread(self._synthesize_sentence, sentence)
            audio.extend(chunk)
        return bytes(audio)