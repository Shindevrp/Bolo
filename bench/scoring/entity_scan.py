"""Deterministic entity/uncertainty scanning for grounding metrics.

Provides honest, black-box proxies for hallucination risk:
  - extract capitalized proper-noun-like runs from a response,
  - determine which entities the user themselves supplied,
  - list common uncertainty phrases.

This is a heuristic, not a ground-truth existence check — that is the honest
limit of a black-box harness, and it mirrors TASA's own EntityGate.
"""
from __future__ import annotations

import re

_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z]+\b")

_CAP_FUNCTION_WORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "Its", "I",
    "You", "Your", "Youre", "My", "Mine", "Me", "We", "Our", "He", "His",
    "She", "Her", "They", "Their", "Them", "What", "When", "Where", "Who",
    "Which", "Why", "How", "So", "And", "But", "Or", "Nor", "Then", "Well",
    "Also", "There", "Here", "Try", "Go", "Yes", "No", "Not", "Just", "Very",
    "Really", "Only", "Even", "Still", "Already", "Please", "Now", "Firstly",
    "Sure", "Great", "Okay", "Thanks",
    "Near", "Nearby", "Best", "Good", "Inside", "Outside", "Around",
    "At", "In", "On", "From", "For", "To", "With", "By", "Up", "Down",
}

UNCERTAINTY_PHRASES = [
    "i'm not sure", "i'm not certain", "i don't know", "not sure",
    "not certain", "i can't verify", "i cannot verify", "i couldn't verify",
    "i'm not aware", "i'd have to check", "i would need to check",
    "i can look it up", "hard to say", "i don't have that information",
    "i'm not confident", "i can't confirm",
]


def extract_capitalized_entities(text: str) -> list[str]:
    """Return capitalized runs that look like proper-noun entities.

    Filters sentence-opening function words. Multi-word runs (e.g. "Grand
    Canyon") are merged.
    """
    text = text or ""
    tokens = re.findall(r"[A-Z][a-zA-Z]+|\S+", text)
    runs: list[str] = []
    out: list[str] = []
    prev_cap = False
    for tok in tokens:
        tok_clean = tok.rstrip(".,!?;:")
        if (
            tok_clean in _CAP_FUNCTION_WORDS
            or not re.match(r"^[A-Z][a-zA-Z]+$", tok_clean)
        ):
            prev_cap = False
            if runs:
                out.append(" ".join(runs))
                runs = []
            continue
        if prev_cap:
            runs.append(tok_clean)
        else:
            if runs:
                out.append(" ".join(runs))
            runs = [tok_clean]
        prev_cap = True
    if runs:
        out.append(" ".join(runs))
    return out


def user_provided_entities(text: str) -> list[str]:
    return extract_capitalized_entities(text)


def has_uncertainty(response: str) -> bool:
    r = (response or "").lower()
    return any(p in r for p in UNCERTAINTY_PHRASES)
