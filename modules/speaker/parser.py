from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

_TAG_RE = re.compile(r"\[([^\]]+)\]")
_SENTENCE_TERMINATORS = frozenset(".!?…\n")
_CLAUSE_CHARS = frozenset(",;:\u2014-")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "dept",
    "e.g", "i.e", "etc", "vs", "cf", "al", "fig", "no", "vol",
    "approx", "inc", "ltd", "co", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
}


@dataclass
class ParsedChunk:
    speaker: str
    text: str
    is_urgent: bool = False
    urgency_tag: str = ""


class SpeakerTokenParser:
    """Parses LLM tokens into whole-sentence chunks for smooth TTS.

    Buffers tokens until a complete [SpeakerName] tag is detected, then
    tracks the current speaker. Text is accumulated raw (preserving the
    LLM's own token spacing, so word fragments like "Today" + "'s" join
    correctly) and emitted as a ParsedChunk only at natural boundaries:

    - a sentence terminator (. ! ? … newline),
    - a clause boundary once the buffer grows large,
    - or a hard character cap.

    This keeps each TTS synthesis a full, smooth utterance instead of one
    synthesis per token (which caused broken, choppy speech).

    Also handles:
    - Partial tags across token boundaries: token "[S" + token "h] hello"
    - Multiple tags in one token: "[Sh] hi [Ti] hey"
    - Unknown speaker names: falls back to last known speaker
    - Urgency tags: "[Ti! excited]" or "[Sh! disagree]"
    """

    def __init__(
        self,
        known_speakers: list[str],
        max_chars: int = 150,
        clause_threshold: int = 60,
    ) -> None:
        self._speaker_order = list(known_speakers)
        self.known_speakers = set(known_speakers)
        self._buffer = ""
        self._current_speaker: str | None = None
        self._speech = ""  # raw accumulated text for the current speaker
        self._speech_urgent = False
        self._speech_tag = ""
        self.max_chars = max_chars
        self.clause_threshold = clause_threshold

    def feed_token(self, token: str) -> list[ParsedChunk]:
        """Process a single LLM token, return any complete chunks."""
        self._buffer += token
        return self._extract_chunks()

    def flush(self) -> ParsedChunk | None:
        """Flush remaining buffer as a chunk for the current speaker."""
        text = (self._buffer + self._speech).strip()
        self._buffer = ""
        self._speech = ""
        if not text:
            return None
        speaker = self._current_speaker or (
            self._speaker_order[0] if self._speaker_order else "assistant"
        )
        return ParsedChunk(
            speaker=speaker,
            text=text,
            is_urgent=self._speech_urgent,
            urgency_tag=self._speech_tag,
        )

    def reset(self) -> None:
        self._buffer = ""
        self._current_speaker = None
        self._speech = ""
        self._speech_urgent = False
        self._speech_tag = ""

    def _extract_chunks(self) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []

        while True:
            match = _TAG_RE.search(self._buffer)
            if not match:
                # No complete tag. Check for a partial tag at the end.
                if self._buffer.endswith("[") or (
                    "[" in self._buffer
                    and self._buffer.rfind("[") > self._buffer.rfind("]")
                ):
                    break
                # No tag at all — everything is the current speaker's speech.
                self._speech += self._buffer
                self._buffer = ""
                self._emit_ready(chunks)
                break

            # Text before the tag belongs to the current speaker.
            before = self._buffer[: match.start()]
            self._speech += before
            self._buffer = self._buffer[match.end():]
            self._emit_ready(chunks)

            # Parse the tag: may contain urgency marker like "Ti! excited".
            speaker, is_urgent, urgency_tag = self._parse_tag(match.group(1).strip())

            if speaker != self._current_speaker:
                # Flush any remaining speech from the previous speaker.
                if self._speech.strip():
                    chunks.append(ParsedChunk(
                        speaker=self._current_speaker or speaker,
                        text=self._speech.strip(),
                        is_urgent=self._speech_urgent,
                        urgency_tag=self._speech_tag,
                    ))
                self._current_speaker = speaker
                self._speech = ""
                self._speech_urgent = is_urgent
                self._speech_tag = urgency_tag

            # Text after the tag accumulates for this speaker in the next loop.

        return chunks

    def _emit_ready(self, chunks: list[ParsedChunk]) -> None:
        """Emit the accumulated speech once it reaches a natural boundary.

        Only splits at the END of the buffer (a completed sentence/clause),
        never mid-sentence, so no word is ever fragmented across syntheses.
        """
        while self._speech:
            text = self._speech
            cut = self._last_boundary(text)
            if cut is None and len(text) < self.max_chars:
                break
            if cut is None:
                cut = self._last_clause(text)
            if cut is None:
                space = text.rfind(" ", 1, self.max_chars)
                cut = space if space > 0 else self.max_chars
            ready = text[:cut]
            self._speech = text[cut:]
            ready_stripped = ready.strip()
            if ready_stripped and self._current_speaker:
                chunks.append(ParsedChunk(
                    speaker=self._current_speaker,
                    text=ready_stripped,
                    is_urgent=self._speech_urgent,
                    urgency_tag=self._speech_tag,
                ))

    def _last_boundary(self, text: str) -> int | None:
        """Index just past the final sentence terminator, or None."""
        best: int | None = None
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in _SENTENCE_TERMINATORS:
                if ch == "." and self._is_abbreviation(text, i):
                    i += 1
                    continue
                j = i + 1
                while j < n and text[j] in "\"')\u201d\u2019]":
                    j += 1
                if j >= n:
                    best = j
                elif text[j].isspace():
                    best = j + 1
            i += 1
        return best

    def _last_clause(self, text: str) -> int | None:
        positions = [i for i, ch in enumerate(text) if ch in _CLAUSE_CHARS]
        if not positions:
            return None
        for pos in reversed(positions):
            if pos >= self.clause_threshold:
                return pos + 1
        return None

    def _is_abbreviation(self, text: str, idx: int) -> bool:
        if text[idx] != ".":
            return False
        # Decimal / number like "3.5" or "1."
        start = idx - 1
        while start >= 0 and (text[start].isdigit() or text[start] == ","):
            start -= 1
        if text[start + 1:idx] and text[start + 1:idx].isdigit():
            return True
        # Word ending at the period.
        start = idx - 1
        while start >= 0 and (text[start].isalnum() or text[start] in "'._"):
            start -= 1
        word = text[start + 1:idx].strip().rstrip(".").lower()
        if not word:
            return False
        if word in _ABBREVIATIONS:
            return True
        if len(word) <= 2 and word.isalpha():
            return True
        return False

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
