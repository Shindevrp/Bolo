from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from utils.logger import get_logger

logger = get_logger("resume")


@dataclass
class SpeakerCheckpoint:
    """Stores a checkpoint for resume-after-interrupt."""

    speaker: str
    last_complete_clause: str
    pending_text: str
    topic_context: str
    turn_count: int
    full_response_so_far: str = ""

    @property
    def resume_prompt(self) -> str:
        """Generate a prompt to resume from where the speaker left off."""
        parts = []
        if self.last_complete_clause:
            parts.append(f'You were saying: "{self.last_complete_clause}"')
        if self.pending_text:
            parts.append(f'You were about to say: "{self.pending_text}"')
        parts.append("Continue naturally from where you left off.")
        return " ".join(parts)


class ResumeManager:
    """Manages speaker checkpoints for resume-after-interrupt.

    When Speaker A is interrupted by Speaker B, this stores A's
    context so A can resume naturally after B finishes.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, SpeakerCheckpoint] = {}
        self._resume_history: list[tuple[str, str]] = []  # (speaker, resumed_from)

    def checkpoint(
        self,
        speaker: str,
        full_response: str,
        topic_context: str = "",
        turn_count: int = 0,
    ) -> SpeakerCheckpoint:
        """Create a checkpoint from the current response text.

        Splits the response at the last clause boundary to find
        the last complete thought.
        """
        # Find the last clause boundary
        last_clause_idx = -1
        for i, ch in enumerate(full_response):
            if ch in ".!?,;—":
                last_clause_idx = i

        if last_clause_idx > 0:
            last_complete = full_response[:last_clause_idx + 1].strip()
            pending = full_response[last_clause_idx + 1:].strip()
        else:
            last_complete = ""
            pending = full_response.strip()

        cp = SpeakerCheckpoint(
            speaker=speaker,
            last_complete_clause=last_complete,
            pending_text=pending,
            topic_context=topic_context,
            turn_count=turn_count,
            full_response_so_far=full_response,
        )
        self._checkpoints[speaker] = cp
        return cp

    def get_checkpoint(self, speaker: str) -> SpeakerCheckpoint | None:
        return self._checkpoints.get(speaker)

    def clear_checkpoint(self, speaker: str) -> None:
        self._checkpoints.pop(speaker, None)

    def should_resume(self, speaker: str) -> bool:
        """Check if a speaker has an unfinished checkpoint."""
        cp = self._checkpoints.get(speaker)
        return cp is not None and bool(cp.pending_text)

    def get_resume_prompt(self, speaker: str) -> str | None:
        """Get a prompt for the LLM to resume from checkpoint."""
        cp = self._checkpoints.get(speaker)
        if not cp or not cp.pending_text:
            return None
        return cp.resume_prompt

    def record_resume(self, speaker: str, resumed_from: str) -> None:
        """Record that a speaker resumed from a checkpoint."""
        self._resume_history.append((speaker, resumed_from))
        if len(self._resume_history) > 10:
            self._resume_history = self._resume_history[-10:]
        self.clear_checkpoint(speaker)

    def get_resume_messages(
        self,
        speaker: str,
        partner_context: str,
    ) -> list[dict[str, str]] | None:
        """Get the message list for a resume LLM call.

        Returns messages that tell the LLM to continue from checkpoint.
        """
        cp = self._checkpoints.get(speaker)
        if not cp:
            return None

        resume_prompt = cp.resume_prompt
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {speaker}. Your partner just finished speaking. "
                    f"{partner_context}\n\n"
                    f"{resume_prompt}"
                ),
            },
            {"role": "user", "content": "Continue."},
        ]
        return messages
