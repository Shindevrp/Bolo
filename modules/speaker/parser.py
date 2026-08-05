from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

_TAG_RE = re.compile(r"\[([^\]]+)\]")


@dataclass
class ParsedChunk:
    speaker: str
    text: str
    is_urgent: bool = False
    urgency_tag: str = ""


class SpeakerTokenParser:
    """Parses LLM tokens in real-time, detecting speaker tag boundaries.

    Buffers tokens until a complete [SpeakerName] tag is detected, then
    yields (speaker, text) tuples as the LLM streams output.

    Handles edge cases:
    - Partial tags across token boundaries: token "[S" + token "h] hello"
    - Multiple tags in one token: "[Sh] hi [Ti] hey"
    - Unknown speaker names: falls back to last known speaker
    - Urgency tags: "[Ti! excited]" or "[Sh! disagree]"
    """

    def __init__(self, known_speakers: list[str]) -> None:
        self._speaker_order = list(known_speakers)
        self.known_speakers = set(known_speakers)
        self._buffer = ""
        self._current_speaker: str | None = None
        self._pending_text = ""

    def feed_token(self, token: str) -> list[ParsedChunk]:
        """Process a single LLM token, return any complete chunks."""
        self._buffer += token
        return self._extract_chunks()

    def flush(self) -> ParsedChunk | None:
        """Flush remaining buffer as a chunk for the current speaker."""
        text = self._buffer.strip()
        self._buffer = ""
        if not text:
            return None
        speaker = self._current_speaker or (
            self._speaker_order[0] if self._speaker_order else "assistant"
        )
        return ParsedChunk(speaker=speaker, text=text)

    def reset(self) -> None:
        self._buffer = ""
        self._current_speaker = None
        self._pending_text = ""

    def _extract_chunks(self) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []

        while True:
            match = _TAG_RE.search(self._buffer)
            if not match:
                # No complete tag found. Check for partial tag at end.
                if self._buffer.endswith("[") or (
                    "[" in self._buffer
                    and self._buffer.rfind("[") > self._buffer.rfind("]")
                ):
                    # Partial tag — hold in buffer
                    break
                # No tag at all — emit as current speaker's text
                text = self._buffer.strip()
                if text and self._current_speaker:
                    chunks.append(ParsedChunk(speaker=self._current_speaker, text=text))
                elif text:
                    # No speaker yet — assign to first known speaker
                    speaker = self._speaker_order[0] if self._speaker_order else "assistant"
                    chunks.append(ParsedChunk(speaker=speaker, text=text))
                self._buffer = ""
                break

            tag_start = match.start()
            tag_end = match.end()
            tag_content = match.group(1).strip()

            # Text before the tag belongs to the previous speaker
            before = self._buffer[:tag_start].strip()
            if before and self._current_speaker:
                chunks.append(ParsedChunk(speaker=self._current_speaker, text=before))

            # Parse the tag: may contain urgency marker like "Ti! excited"
            speaker, is_urgent, urgency_tag = self._parse_tag(tag_content)
            self._current_speaker = speaker

            # Text after the tag is the start of this speaker's segment
            self._buffer = self._buffer[tag_end:]

            # If there's already text after the tag and another tag follows,
            # it will be handled in the next iteration
            if not _TAG_RE.search(self._buffer):
                # No more tags — the rest is this speaker's text
                break

        return chunks

    def _parse_tag(self, tag_content: str) -> tuple[str, bool, str]:
        """Parse a tag like 'Ti! excited' into (speaker, is_urgent, urgency_tag)."""
        parts = tag_content.split("!", 1)
        speaker = parts[0].strip()
        urgency_tag = parts[1].strip() if len(parts) > 1 else ""
        is_urgent = bool(urgency_tag)

        # Validate speaker name (fuzzy match)
        matched = self._match_speaker(speaker)
        return matched, is_urgent, urgency_tag

    def _match_speaker(self, name: str) -> str:
        """Match a speaker name against known speakers (case-insensitive)."""
        for known in self.known_speakers:
            if known.lower() == name.lower():
                return known
        # Fuzzy: check if name starts with or contains a known speaker
        for known in self.known_speakers:
            if known.lower().startswith(name.lower()) or name.lower().startswith(
                known.lower()
            ):
                return known
        return name  # Return as-is if no match
