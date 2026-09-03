from __future__ import annotations

from modules.turn.backchannel_interrupt import BackchannelInterrupt


class TestBackchannelClassification:
    def test_acknowledging_feedback_is_backchannel(self) -> None:
        for t in ["Yeah.", "Uh-huh.", "Right.", "Okay.", "uh huh", "mm-hmm", "yep"]:
            assert BackchannelInterrupt.classify(t) == "backchannel", t

    def test_agreeing_feedback_is_backchannel(self) -> None:
        for t in ["Yeah sure.", "right okay", "okay cool"]:
            assert BackchannelInterrupt.classify(t) == "backchannel", t

    def test_disagreement_is_interruption(self) -> None:
        for t in ["Wait.", "No.", "Stop.", "That's wrong.", "no wait", "hold on"]:
            assert BackchannelInterrupt.classify(t) == "disagreement", t

    def test_disagreement_wins_over_filler(self) -> None:
        # "no" turns conversational filler into a genuine interruption.
        assert BackchannelInterrupt.classify("no, no") == "disagreement"

    def test_normal_utterance_is_neutral(self) -> None:
        assert BackchannelInterrupt.classify("can you tell me more about that") == "neutral"

    def test_empty_is_neutral(self) -> None:
        assert BackchannelInterrupt.classify("") == "neutral"

    def test_is_backchannel_helper(self) -> None:
        assert BackchannelInterrupt.is_backchannel("yes")
        assert not BackchannelInterrupt.is_backchannel("stop")

    def test_is_disagreement_helper(self) -> None:
        assert BackchannelInterrupt.is_disagreement("stop")
        assert not BackchannelInterrupt.is_disagreement("okay")

    def test_no_problem_is_backchannel_not_disagreement(self) -> None:
        # "no" alone is a stop signal, but "no problem" is polite feedback.
        assert BackchannelInterrupt.is_backchannel("no problem")
        assert BackchannelInterrupt.classify("no problem") == "backchannel"

    def test_no_alone_is_disagreement(self) -> None:
        assert BackchannelInterrupt.classify("No.") == "disagreement"
