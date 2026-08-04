from __future__ import annotations

import re


# Urgency signals mapped to scores
_URGENCY_SIGNALS: list[tuple[str, float]] = [
    # Disagreement (high urgency)
    (r"\bactually\b", 0.4),
    (r"\bbut\b", 0.3),
    (r"\bhowever\b", 0.3),
    (r"\bi disagree\b", 0.6),
    (r"\bnot sure about\b", 0.4),
    (r"\bwait\b", 0.5),
    (r"\bhold on\b", 0.5),
    (r"\bthat'?s not quite right\b", 0.5),
    (r"\blet me push back\b", 0.6),
    # Excitement (high urgency)
    (r"\bwow\b", 0.5),
    (r"\bamazing\b", 0.4),
    (r"\bexactly\b", 0.3),
    (r"\bthat'?s it\b", 0.4),
    (r"\bbrilliant\b", 0.4),
    (r"\bperfect\b", 0.3),
    (r"\bi just realized\b", 0.5),
    (r"\boh wait\b", 0.5),
    # Topic shift (moderate urgency)
    (r"\banyway\b", 0.3),
    (r"\bby the way\b", 0.3),
    (r"\bspeaking of\b", 0.2),
    (r"\bthat reminds me\b", 0.3),
    # Question (moderate urgency — questions invite response)
    (r"\?$", 0.2),
    # Exclamation (moderate urgency)
    (r"!$", 0.3),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), s) for p, s in _URGENCY_SIGNALS]


def score_urgency(text: str) -> float:
    """Score the urgency of a text segment (0.0-1.0).

    Higher scores indicate the speaker is more eager to interject.
    Based on linguistic markers: disagreement, excitement, questions.
    """
    if not text.strip():
        return 0.0

    score = 0.0
    for pattern, weight in _COMPILED:
        if pattern.search(text):
            score += weight

    # Length factor: very short utterances are more likely urgent interjections
    word_count = len(text.split())
    if word_count <= 3:
        score += 0.1
    elif word_count > 20:
        score -= 0.1

    return min(1.0, max(0.0, score))
