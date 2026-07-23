from __future__ import annotations


class DialogueMemory:
    """Session memory for the dialogue subsystem."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def remember(self, entry: str) -> None:
        self.entries.append(entry)
