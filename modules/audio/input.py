from __future__ import annotations


class AudioInput:
    """Audio input handler for microphone or websocket streams."""

    def read(self, chunk: bytes) -> bytes:
        return chunk
