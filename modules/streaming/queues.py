from __future__ import annotations

from collections import deque
from typing import Any


class StreamQueues:
    """Simple in-memory queues for streaming stages."""

    def __init__(self) -> None:
        self.a = deque()
        self.b = deque()
        self.c = deque()
        self.d = deque()

    def enqueue(self, queue_name: str, item: Any) -> None:
        getattr(self, queue_name).append(item)

    def dequeue(self, queue_name: str) -> Any:
        return getattr(self, queue_name).popleft()
