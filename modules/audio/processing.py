from __future__ import annotations


class AudioProcessing:
    """Chunking and resampling utilities for streaming audio."""

    def chunk(self, audio: bytes, chunk_size: int = 1600) -> list[bytes]:
        return [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]
