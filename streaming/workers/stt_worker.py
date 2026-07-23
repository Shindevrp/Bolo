from __future__ import annotations


class STTWorker:
    """Worker responsible for speech-to-text processing."""

    def run(self, payload: bytes) -> str:
        return payload.decode("utf-8", errors="ignore")
