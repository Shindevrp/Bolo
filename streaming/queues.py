from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueueA:
    items: deque[Any] = field(default_factory=deque)

    def push(self, item: Any) -> None:
        self.items.append(item)

    def pop(self) -> Any:
        return self.items.popleft()


@dataclass
class QueueB:
    items: deque[Any] = field(default_factory=deque)

    def push(self, item: Any) -> None:
        self.items.append(item)

    def pop(self) -> Any:
        return self.items.popleft()


@dataclass
class QueueC:
    items: deque[Any] = field(default_factory=deque)

    def push(self, item: Any) -> None:
        self.items.append(item)

    def pop(self) -> Any:
        return self.items.popleft()


@dataclass
class QueueD:
    items: deque[Any] = field(default_factory=deque)

    def push(self, item: Any) -> None:
        self.items.append(item)

    def pop(self) -> Any:
        return self.items.popleft()
