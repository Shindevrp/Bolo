from __future__ import annotations

import pytest

from modules.turn.detector import TurnDetector
from modules.turn.classifier import TurnClassifier
from modules.turn.interrupt import InterruptHandler


class TestTurnDetector:
    def test_initial_decision_is_continue(self) -> None:
        d = TurnDetector(silence_threshold=0.6)
        decision = d.process_chunk(b"\x00" * 1600, is_speech=True, partial_transcript="")
        assert decision == "continue"

    def test_reset(self) -> None:
        d = TurnDetector()
        d.process_chunk(b"\x00" * 1600, is_speech=True)
        d.reset()
        decision = d.process_chunk(b"\x00" * 1600, is_speech=True)
        assert decision == "continue"


class TestTurnClassifier:
    def test_question_ends_turn(self) -> None:
        c = TurnClassifier()
        feat = {"silence_duration": 0.4, "speech_duration": 2.0}
        decision = c.classify(
            feat,
            partial_transcript="what is the weather?",
            prosody={"trajectory": "falling", "pitch_trend": -30, "energy_trend": -5},
        )
        assert decision == "end_turn"

    def test_short_speech_continues(self) -> None:
        c = TurnClassifier()
        feat = {"silence_duration": 0.1, "speech_duration": 0.2}
        decision = c.classify(feat, partial_transcript="hi")
        assert decision == "continue"

    def test_linguistic_falling_trajectory(self) -> None:
        c = TurnClassifier()
        feat = {"silence_duration": 0.3, "speech_duration": 2.0}
        decision = c.classify(feat, partial_transcript="I think so.", prosody={"trajectory": "falling"})
        assert decision == "end_turn"

    def test_trailing_conjunction_continues(self) -> None:
        c = TurnClassifier()
        feat = {"silence_duration": 0.3, "speech_duration": 2.0}
        decision = c.classify(feat, partial_transcript="I was going to say and")
        assert decision == "continue"


class TestTurnClassifierIncompleteUtterances:
    def test_enumeration_ending_comma_continues(self) -> None:
        c = TurnClassifier()
        assert c.is_incomplete("Is it color, function,")
        feat = {"silence_duration": 0.3, "speech_duration": 2.0}
        decision = c.classify(feat, partial_transcript="Is it color, function,")
        assert decision == "continue"

    def test_serial_comma_without_terminal_continues(self) -> None:
        c = TurnClassifier()
        assert c.is_incomplete("color, function, placement")

    def test_open_list_marker_continues(self) -> None:
        c = TurnClassifier()
        assert c.is_incomplete("what do you think about or")
        assert c.is_incomplete("maybe something like")

    def test_completed_question_not_incomplete(self) -> None:
        c = TurnClassifier()
        assert not c.is_incomplete("what is the weather?")
        assert not c.is_incomplete("is it color, function, placement, or something else?")

    def test_completed_statement_not_incomplete(self) -> None:
        c = TurnClassifier()
        assert not c.is_incomplete("I think it's the red one.")
        assert not c.is_incomplete("nothing")
        assert not c.is_incomplete("")

    def test_discourse_marker_comma_not_incomplete(self) -> None:
        c = TurnClassifier()
        assert not c.is_incomplete("well, I think so")
        assert not c.is_incomplete("well, I think so.")

    def test_question_word_incomplete_gets_low_score(self) -> None:
        # "Is it color, function," starts like a question but is an open list,
        # so it must NOT get the 0.9 question score that triggers the 80ms cut.
        c = TurnClassifier()
        assert c._score_linguistic("Is it color, function,") == 0.3

    def test_completed_enumeration_question_gets_high_score(self) -> None:
        c = TurnClassifier()
        assert c._score_linguistic(
            "Is it color, function, placement, or something else?"
        ) == 0.9


class TestInterruptHandler:
    def test_no_interrupt_when_not_speaking(self) -> None:
        h = InterruptHandler()
        assert not h.should_interrupt(energy=0.05, silence_duration=0.0, is_speaking=False)

    def test_interrupt_on_consecutive_speech(self) -> None:
        h = InterruptHandler(consecutive_speech_frames=3, speech_energy_threshold=0.01)
        assert not h.should_interrupt(0.05, 0.0, is_speaking=True)
        assert not h.should_interrupt(0.05, 0.0, is_speaking=True)
        assert h.should_interrupt(0.05, 0.0, is_speaking=True)

    def test_low_energy_no_interrupt(self) -> None:
        h = InterruptHandler(speech_energy_threshold=0.1, consecutive_speech_frames=1)
        assert not h.should_interrupt(0.05, 0.0, is_speaking=True)