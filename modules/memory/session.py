from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    role: str
    content: str
    turn_index: int


class SessionMemory:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.entries: deque[MemoryEntry] = deque(maxlen=max_turns)
        self._turn_counter = 0

    def add(self, role: str, content: str) -> None:
        self.entries.append(MemoryEntry(role, content, self._turn_counter))
        self._turn_counter += 1

    def get_history(self, max_turns: int | None = None) -> list[MemoryEntry]:
        k = max_turns if max_turns else self.max_turns
        return list(self.entries)[-k:]

    def context_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for entry in self.entries:
            messages.append({"role": entry.role, "content": entry.content})
        return messages

    def token_estimate(self) -> int:
        return sum(len(e.content.split()) + 4 for e in self.entries)

    def truncate_to_budget(self, max_tokens: int = 4096) -> None:
        while self.token_estimate() > max_tokens and len(self.entries) > 1:
            self.entries.popleft()