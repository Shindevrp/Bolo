"""Distinguish conversational backchannels from genuine interruptions.

During playback, the barge-in detector fires on speech energy alone, so short
conversational feedback like "yeah", "uh-huh", "right", or "okay" can
incorrectly cancel the assistant's ongoing response. Here we classify the
(partial) transcript so the pipeline can:

  - suppress interruption for backchannels, and
  - force interruption for disagreement / correction ("stop", "no", "wait").

Classification is lexicon-based, offline, and tolerant of punctuation and
short multi-word variants.
"""
from __future__ import annotations

import re

# Normative conversational feedback: agreement/acknowledgement/continuers.
# These should NOT cancel the assistant's response.
_BACKCHANNEL_WORDS = {
    "uhhuh", "uh-huh", "mhmm", "mm-hmm", "hmm", "hm", "mmm", "mm",
    "yeah", "yea", "yep", "yup", "right", "okay", "ok", "k", "sure",
    "alright", "alrighty", "gotcha", "got it", "understood", "i see",
    "aha", "oh", "cool", "nice", "great", "good", "yes", "mhm",
}
_BACKCHANNEL_PHRASES = {
    "i see", "got it", "yeah yeah", "okay okay", "right right",
    "uh huh", "mm hm", "mm hmm", "yeah sure", "sure thing", "no problem",
    "makes sense", "fair enough", "sounds good", "okay cool", "yeah okay",
}

# Genuine interruption intent: disagreement / stop / correction. These SHOULD
# cancel the assistant's response even if short.
_DISAGREEMENT_WORDS = {
    "stop", "wait", "hold on", "hold up", "no", "nope", "that's wrong",
    "that is wrong", "not that", "wrong", "actually", "i didn't mean that",
    "i meant", "no wait", "that's not", "that is not", "stop that", "cut",
    "cancel", "never mind", "nevermind", "forget it", "forget that",
    "that's incorrect", "not right", "i disagree", "correction", "mistake",
    "you're wrong", "you are wrong", "let me stop", "no stop",
}

_WORD_RE = re.compile(r"[a-z]+")


class BackchannelInterrupt:
    """Classify user speech during playback as backchannel/interrupt/neutral."""

    # Default: "neutral" keeps existing energy-based behavior.
    @staticmethod
    def classify(transcript: str) -> str:
        """Return 'backchannel', 'disagreement', or 'neutral'.

        'backchannel'  -> conversational feedback, should NOT interrupt.
        'disagreement' -> user wants to stop/correct, SHOULD interrupt.
        'neutral'      -> not clearly either; keep default behavior.
        """
        if not transcript:
            return "neutral"
        t = transcript.strip().lower().strip(" .,!?;:")
        words = _WORD_RE.findall(t)
        compact = re.sub(r"[.,!?;:]+", " ", t).strip()

        # Exact multi-word phrase matches take precedence over single-word
        # checks so "no problem" (backchannel) isn't read as "no" (a stop).
        if t in _DISAGREEMENT_WORDS or compact in _DISAGREEMENT_WORDS:
            return "disagreement"
        if t in _BACKCHANNEL_PHRASES or compact in _BACKCHANNEL_PHRASES:
            return "backchannel"

        # Disagreement is strong: any disagreement token means stop/cancel,
        # even when the user flows into it (e.g. "no, wait", "stop that").
        if any(w in _DISAGREEMENT_WORDS for w in words):
            return "disagreement"

        # A short utterance whose every word is a backchannel filler is just
        # feedback, e.g. "yeah sure", "right okay", "oh nice".
        if 0 < len(words) <= 3 and all(w in _BACKCHANNEL_WORDS for w in words):
            return "backchannel"

        # Single-word / single-token feedback like "uh-huh" or "yep".
        if t in _BACKCHANNEL_WORDS:
            return "backchannel"

        return "neutral"

    @staticmethod
    def is_backchannel(transcript: str) -> bool:
        return BackchannelInterrupt.classify(transcript) == "backchannel"

    @staticmethod
    def is_disagreement(transcript: str) -> bool:
        return BackchannelInterrupt.classify(transcript) == "disagreement"
