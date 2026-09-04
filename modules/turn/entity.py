"""Entity-verification gate.

Guards against the assistant asserting a specific place / business / proper
noun that it fabricated. Pure, testable functions used by the pipeline to:

  1. Detect when a query is asking about specific named entities and therefore
     requires tool verification before the model names anything concrete.
  2. Detect capitalized entity-like tokens in a response and decide whether
     they are supported by evidence (tool results, retrieval, conversation
     history) or are unverified/fabricated assertions.
"""
from __future__ import annotations

import re

_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z]+\b")

# Common capitalized function words / pronouns that often open a clause or
# sentence and must never be treated as the start of a proper-noun entity.
_CAP_FUNCTION_WORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "Its", "I",
    "You", "Your", "Youre", "My", "Mine", "Me", "We", "Our", "He", "His",
    "She", "Her", "They", "Their", "Them", "What", "When", "Where", "Who",
    "Which", "Why", "How", "So", "And", "But", "Or", "Nor", "Then", "Well",
    "Also", "There", "Here", "Try", "Go", "Yes", "No", "Not", "Just", "Very",
    "Really", "Only", "Even", "Still", "Already", "Please", "Now", "Firstly",
}

# Words that signal a query is looking for a concrete named thing (place,
# business, restaurant, or a specific verifiable entity) rather than general
# chat. When present, the answer must go through tool verification.
ENTITY_QUERY_WORDS = {
    "restaurant", "cafe", "café", "hotel", "shop", "store", "mall", "market",
    "bazaar", "place", "spot", "location", "address", "near", "nearby",
    "recommend", "suggest", "good", "best", "open", "operating", "timings",
    "menu", "famous", "known", "popular", "directions", "route", "business",
    "company", "brand", "booking", "ticket", "hospital", "school", "park",
    "temple", "bank",
}

# Cue nouns/verbs that often precede a specific name being asserted, used to
# locate entity mentions inside a response (e.g. "the restaurant called X",
# "I recommend X", "try X", "there's X in Y").
_ASSERT_CUES = re.compile(
    r"\b(?:called|named|recommend|try|visit|go to|check out|suggest)\b",
    re.IGNORECASE,
)


class EntityGate:
    """Deterministic entity-provenance checks (no LLM call)."""

    @staticmethod
    def query_needs_verification(text: str) -> bool:
        """True if a user utterance appears to seek a specific named entity.

        Conservative: only flags genuine entity-seeking queries whose answer
        would require verifiable facts (places, businesses, etc.).
        """
        tl = text.lower()
        return any(w in tl for w in ENTITY_QUERY_WORDS)

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        """Return capitalized looks-like-proper-noun runs (place/business names).

        A run is at least one capitalized word following a capitalized word, or
        a capitalized word directly after an assertion cue. Sentence-opening
        function words (Try, Go, Yes, The, This, ...) are never treated as
        entity starts.
        """
        words = re.findall(r"[A-Z][a-zA-Z]+|\S+", text)
        candidates: list[str] = []
        runs: list[str] = []
        prev_capitalized_word = False
        for w in words:
            if w not in _CAP_FUNCTION_WORDS and re.match(r"^[A-Z][a-zA-Z]+$", w):
                if prev_capitalized_word:
                    runs.append(w)
                else:
                    if len(runs) >= 1:
                        candidates.append(" ".join(runs))
                    runs = [w]
                prev_capitalized_word = True
            else:
                prev_capitalized_word = False
                if len(runs) >= 1:
                    candidates.append(" ".join(runs))
                    runs = []
        if len(runs) >= 1:
            candidates.append(" ".join(runs))
        return candidates

    @staticmethod
    def unsupported_entities(
        response: str,
        supported: set[str],
        *,
        user_input_entities: set[str] | None = None,
    ) -> list[str]:
        """Entities asserted in `response` with no support in `supported`.

        An entity is 'supported' if it appeared in a tool result, in
        retrieval/context, or was originally provided by the user (so the
        model is allowed to reference/echo what the user said). Unsupported
        capitalized runs are treated as potential fabrications.
        """
        user_entities = {e.lower() for e in (user_input_entities or set())}
        supported_low = {e.lower() for e in supported}
        blocked: list[str] = []
        for ent in EntityGate.extract_entities(response):
            low = ent.lower()
            if low in supported_low or low in user_entities:
                continue
            blocked.append(ent)
        return blocked

    @staticmethod
    def safety_response(fabricated: list[str]) -> str:
        """Canonical fallback when the response asserted unverified entities."""
        if len(fabricated) == 1:
            return (
                "I'm not certain that place actually exists, so I don't want "
                "to make it up. I can look it up if you'd like."
            )
        return (
            "I'm not sure all of those places actually exist, and I don't "
            "want to make them up. I can verify one for you if you'd like."
        )
