from __future__ import annotations


class TTSWorker:
    """Worker responsible for text-to-speech synthesis."""

    def run(self, text: str) -> bytes:
        return text.encode("utf-8")
