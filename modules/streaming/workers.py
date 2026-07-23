from __future__ import annotations


class StreamWorkers:
    """Container class for streaming workers."""

    def __init__(self) -> None:
        self.stt = None
        self.llm = None
        self.tts = None
        self.playback = None
