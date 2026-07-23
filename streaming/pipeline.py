from __future__ import annotations

import asyncio
from typing import Any, Callable


class AsyncPipeline:
    """Simple async pipeline executor."""

    def __init__(self) -> None:
        self._steps: list[Callable[[Any], Any]] = []

    def add_step(self, step: Callable[[Any], Any]) -> None:
        self._steps.append(step)

    async def run(self, initial_value: Any) -> Any:
        value = initial_value
        for step in self._steps:
            value = await asyncio.to_thread(step, value)
        return value
