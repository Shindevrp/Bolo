from __future__ import annotations


class TurnBackchannel:
    """Generate short listener responses within the turn module."""

    def generate(self, context: str) -> str:
        return "uh-huh" if context else "hmm"
