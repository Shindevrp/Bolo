from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Reminder:
    text: str
    trigger_at: float
    created_at: float = field(default_factory=time.time)
    fired: bool = False


class ReminderStore:
    """In-memory reminder store with background checker."""

    def __init__(self) -> None:
        self._reminders: list[Reminder] = []
        self._checker_task: asyncio.Task | None = None
        self._callbacks: list[asyncio.Queue[str]] = []

    def register_callback(self, queue: asyncio.Queue[str]) -> None:
        self._callbacks.append(queue)

    def start_checker(self) -> None:
        if self._checker_task and not self._checker_task.done():
            return
        self._checker_task = asyncio.create_task(self._check_loop())

    async def _check_loop(self) -> None:
        while True:
            now = time.time()
            for r in self._reminders:
                if not r.fired and now >= r.trigger_at:
                    r.fired = True
                    for q in self._callbacks:
                        try:
                            q.put_nowait(r.text)
                        except asyncio.QueueFull:
                            pass
            self._reminders = [r for r in self._reminders if not r.fired]
            await asyncio.sleep(1)

    def add(self, text: str, delay_seconds: float) -> Reminder:
        r = Reminder(text=text, trigger_at=time.time() + delay_seconds)
        self._reminders.append(r)
        self.start_checker()
        return r

    def list_active(self) -> list[str]:
        return [r.text for r in self._reminders if not r.fired]


_store = ReminderStore()


async def set_reminder(text: str, minutes: str = "5") -> str:
    """Set a reminder that fires after the given number of minutes."""
    try:
        mins = max(0.1, float(minutes))
    except ValueError:
        mins = 5.0
    delay = mins * 60
    _store.add(text, delay)
    return f"Reminder set: '{text}' in {mins:.0f} minutes"
