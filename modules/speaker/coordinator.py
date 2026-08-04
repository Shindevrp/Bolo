from __future__ import annotations

import re
from dataclasses import dataclass

from modules.speaker.profile import SpeakerProfile

_TAG_RE = re.compile(r"\[([^\]]+)\]\s*")


@dataclass
class SpeakerSegment:
    speaker: str
    text: str


class SpeakerCoordinator:
    """Parses multi-speaker tagged LLM output and manages turn-taking."""

    def __init__(self, speakers: list[SpeakerProfile]) -> None:
        self.speakers = {s.name: s for s in speakers}
        self.speaker_order = [s.name for s in speakers]
        self.last_speaker: str | None = None
        self._turn_index = 0

    @property
    def enabled(self) -> bool:
        return len(self.speakers) >= 2

    def parse_response(self, llm_output: str) -> list[SpeakerSegment]:
        """Parse '[SpeakerName] text' tagged output into segments.

        Falls back to assigning the full text to the next expected speaker
        if no tags are found.
        """
        if not llm_output.strip():
            return []

        segments: list[SpeakerSegment] = []
        parts = _TAG_RE.split(llm_output)

        # split produces: [pre-tag, name1, post-tag1, name2, post-tag2, ...]
        # Skip any leading text before the first tag
        i = 1
        while i < len(parts) - 1:
            name = parts[i].strip()
            text = parts[i + 1].strip()
            if name and text:
                segments.append(SpeakerSegment(speaker=name, text=text))
            i += 2

        if not segments:
            fallback = self.select_next_speaker("")
            text = llm_output.strip()
            if text:
                segments.append(SpeakerSegment(speaker=fallback, text=text))
            return segments

        # Validate speaker names exist
        valid: list[SpeakerSegment] = []
        for seg in segments:
            if seg.speaker in self.speakers:
                valid.append(seg)
            else:
                # Try fuzzy match (case-insensitive)
                for real_name in self.speaker_order:
                    if real_name.lower() == seg.speaker.lower():
                        valid.append(SpeakerSegment(speaker=real_name, text=seg.text))
                        break
                else:
                    # Unknown speaker — assign to the other speaker
                    other = self._other_than(self.last_speaker)
                    valid.append(SpeakerSegment(speaker=other, text=seg.text))

        if valid:
            self.last_speaker = valid[-1].speaker

        return valid

    def select_next_speaker(
        self,
        user_transcript: str,
        addressed_name: str | None = None,
    ) -> str:
        """Decide which speaker goes first after user input."""
        if addressed_name and addressed_name in self.speakers:
            self._turn_index = self.speaker_order.index(addressed_name)
            return addressed_name

        # Check if user mentions a speaker name
        lower = user_transcript.lower()
        for name in self.speaker_order:
            if name.lower() in lower:
                self._turn_index = self.speaker_order.index(name)
                return name

        # Alternate
        self._turn_index = (self._turn_index + 1) % len(self.speaker_order)
        return self.speaker_order[self._turn_index]

    def inter_speaker_pause(self, prev_speaker: str, next_speaker: str) -> float:
        """Compute natural pause between two speakers in seconds."""
        if prev_speaker == next_speaker:
            return 0.0
        return 0.4

    def speaker_names(self) -> list[str]:
        return list(self.speaker_order)

    def _other_than(self, name: str | None) -> str:
        if name and name in self.speakers:
            for s in self.speaker_order:
                if s != name:
                    return s
        return self.speaker_order[0]
