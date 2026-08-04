from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StreamState:
    """Tracks semantic trajectory of LLM output in real-time.

    Maintains a rolling window of tokens and computes:
    - intent_confidence: how confident the speaker is (0-1)
    - claim_strength: how strong/definitive the claim is (0-1)
    - uncertainty_score: how uncertain/hedging the language is (0-1)
    - topic_vector: embedding of current topic direction
    """

    intent_confidence: float = 0.5
    claim_strength: float = 0.0
    uncertainty_score: float = 0.0
    topic_vector: np.ndarray | None = None
    last_clause_boundary: int = 0
    token_count: int = 0
    _tokens: list[str] = field(default_factory=list)
    _claim_markers: int = 0
    _uncertainty_markers: int = 0
    _last_update: float = field(default_factory=time.monotonic)

    # Linguistic markers
    _CLAIM_WORDS = frozenset({
        "is", "are", "will", "should", "must", "definitely", "certainly",
        "absolutely", "clearly", "obviously", "undoubtedly", "always",
        "never", "impossible", "essential", "critical", "fundamental",
    })
    _HEDGE_WORDS = frozenset({
        "think", "maybe", "perhaps", "possibly", "might", "could",
        "somewhat", "sort of", "kind of", "not sure", "unclear",
        "debatable", "arguably", "presumably", "apparently",
        "i guess", "i suppose", "i feel like",
    })
    _CLAUSE_BOUNDARIES = frozenset({",", ";", "—", "but", "and", "or", "so", "yet"})
    _HARD_BOUNDARIES = frozenset({".", "!", "?", "..."})

    def update(self, token: str) -> None:
        """Process a new token and update state."""
        self._tokens.append(token)
        self.token_count += 1
        self._last_update = time.monotonic()

        lower = token.lower().strip()

        # Update claim/uncertainty markers
        if lower in self._CLAIM_WORDS:
            self._claim_markers += 1
        if lower in self._HEDGE_WORDS:
            self._uncertainty_markers += 1

        # Update scores with decay
        total = max(1, self.token_count)
        claim_ratio = self._claim_markers / total
        hedge_ratio = self._uncertainty_markers / total

        # Claim strength: rises with definitive language, falls with hedging
        self.claim_strength = min(1.0, claim_ratio * 3.0)

        # Uncertainty: rises with hedging language
        self.uncertainty_score = min(1.0, hedge_ratio * 4.0)

        # Intent confidence: high when claims are strong and uncertainty is low
        self.intent_confidence = max(0.0, min(1.0,
            self.claim_strength * 0.7 + (1.0 - self.uncertainty_score) * 0.3
        ))

        # Track clause boundaries
        if lower in self._CLAUSE_BOUNDARIES:
            self.last_clause_boundary = self.token_count

        # Update topic vector (simple rolling average approach)
        self._update_topic_vector(token)

    def should_trigger_interrupt(self) -> tuple[bool, str]:
        """Check if the current state warrants an interrupt.

        Returns (should_interrupt, reason).
        """
        # Weak claim + high uncertainty = speaker is hedging, good time to jump in
        if self.claim_strength > 0.3 and self.uncertainty_score > 0.4:
            return True, "weak_claim"

        # Strong claim forming but not yet delivered = interrupt before delivery
        if self.claim_strength > 0.6 and self.token_count - self.last_clause_boundary > 5:
            return True, "pre_delivery"

        # Too much hedging = speaker is lost, good time to redirect
        if self.uncertainty_score > 0.6:
            return True, "excessive_hedging"

        return False, ""

    def is_at_clause_boundary(self) -> bool:
        """Check if we're near a clause boundary (good interrupt point)."""
        if not self._tokens:
            return False
        last = self._tokens[-1].lower().strip()
        return last in self._CLAUSE_BOUNDARIES or last in self._HARD_BOUNDARIES

    def tokens_since_boundary(self) -> int:
        """How many tokens since last clause boundary."""
        return self.token_count - self.last_clause_boundary

    def _update_topic_vector(self, token: str) -> None:
        """Simple topic tracking via character-level hashing."""
        # Use a lightweight approach: hash the token and accumulate
        if self.topic_vector is None:
            self.topic_vector = np.zeros(32, dtype=np.float32)

        # Simple character n-gram hash
        for i in range(len(token)):
            idx = hash(token[i:i+2]) % 32
            self.topic_vector[idx] += 1.0

        # Normalize periodically
        if self.token_count % 20 == 0:
            norm = np.linalg.norm(self.topic_vector)
            if norm > 0:
                self.topic_vector /= norm

    def reset(self) -> None:
        self.intent_confidence = 0.5
        self.claim_strength = 0.0
        self.uncertainty_score = 0.0
        self.topic_vector = None
        self.last_clause_boundary = 0
        self.token_count = 0
        self._tokens.clear()
        self._claim_markers = 0
        self._uncertainty_markers = 0

    def cosine_similarity_with(self, other: StreamState) -> float:
        """Compute topic similarity with another stream state."""
        if self.topic_vector is None or other.topic_vector is None:
            return 0.0
        dot = np.dot(self.topic_vector, other.topic_vector)
        norm_a = np.linalg.norm(self.topic_vector)
        norm_b = np.linalg.norm(other.topic_vector)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
