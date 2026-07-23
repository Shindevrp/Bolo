from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Orchestrator:
    """High-level coordinator for the TASA pipeline."""

    session_state: dict[str, Any] = field(default_factory=dict)

    def initialize_session(self, session_id: str) -> dict[str, Any]:
        self.session_state[session_id] = {
            "turn": None,
            "transcript": [],
            "memory": [],
        }
        return self.session_state[session_id]

    def handle_turn(self, session_id: str, user_input: str) -> dict[str, Any]:
        state = self.session_state.setdefault(
            session_id,
            {
                "turn": None,
                "transcript": [],
                "memory": [],
            },
        )

        state["transcript"].append(user_input)
        state["turn"] = {
            "user_input": user_input,
            "status": "processed",
        }

        return {
            "session_id": session_id,
            "turn": state["turn"],
            "transcript": state["transcript"],
        }

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        return self.session_state.get(session_id, {})
