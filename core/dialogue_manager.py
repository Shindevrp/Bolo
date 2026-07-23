from __future__ import annotations

from core.state import SessionState, DialogueState
from modules.turn.detector import TurnDetector
from modules.backchannel.generator import BackchannelGenerator
from utils.logger import logger


class DialogueManager:
    def __init__(
        self,
        turn_detector: TurnDetector | None = None,
        backchannel_generator: BackchannelGenerator | None = None,
    ) -> None:
        self.turn_detector = turn_detector or TurnDetector()
        self.backchannel = backchannel_generator or BackchannelGenerator()

    def on_speech_start(self, session: SessionState) -> SessionState:
        session.set_state(DialogueState.LISTENING)
        return session

    def on_speech_end(self, session: SessionState) -> SessionState:
        if session.state == DialogueState.LISTENING:
            session.set_state(DialogueState.PROCESSING)
        return session

    def on_transcript(self, session: SessionState, text: str) -> SessionState:
        session.add_user_turn(text)

        if self._is_question(text):
            session.set_state(DialogueState.PROCESSING)
        elif self._is_statement(text):
            session.set_state(DialogueState.PROCESSING)
        else:
            return session

        return session

    def on_response_start(self, session: SessionState) -> SessionState:
        session.set_state(DialogueState.RESPONDING)
        return session

    def on_response_token(
        self, session: SessionState, token: str
    ) -> SessionState:
        if session.state == DialogueState.RESPONDING:
            session.set_state(DialogueState.INTERRUPTIBLE)
        return session

    def on_response_done(
        self, session: SessionState, full: str
    ) -> SessionState:
        session.add_ai_turn(full)
        session.set_state(DialogueState.IDLE)
        return session

    def on_interrupt(self, session: SessionState) -> SessionState:
        logger.info(f"Interrupt detected for session {session.session_id}")
        session.set_state(DialogueState.LISTENING)
        return session

    def should_backchannel(
        self, session: SessionState, pause_duration: float
    ) -> bool:
        if session.state != DialogueState.LISTENING:
            return False
        if pause_duration < 0.3:
            return False
        if session.total_user_turns < 1:
            return False
        if session.consecutive_silence > 2.0:
            return False
        return pause_duration >= 0.4 and session.engagement_score >= 0.3

    def _is_question(self, text: str) -> bool:
        return text.strip().endswith("?")

    def _is_statement(self, text: str) -> bool:
        return len(text.strip()) > 3 and not text.strip().endswith("?")