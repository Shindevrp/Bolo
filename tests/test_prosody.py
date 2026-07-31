import asyncio

from core.pipeline import ConversationContext, StreamingPipeline
from modules.tts.prosody import ProsodySelector
from tests.test_streaming import (
    FakeLLMText,
    FakeSTTText,
    FakeTTSStream,
    FakeVAD,
)


class _FakeSTTQuestion:
    async def transcribe(self, audio_blob: bytes) -> str:
        return "what time is it?"


class _FakeSTTEcho:
    async def transcribe(self, audio_blob: bytes) -> str:
        return "hello world and welcome back"


def _make_pipeline() -> StreamingPipeline:
    return StreamingPipeline(FakeSTTText(), FakeLLMText(), FakeTTSStream(), FakeVAD())


class TestProsodySelector:
    def test_question_is_slower_than_statement(self) -> None:
        sel = ProsodySelector()
        question = sel.select("What time is it?")
        statement = sel.select("It is noon.")
        assert question.length_scale > statement.length_scale
        assert question.label == "question"

    def test_emphatic_is_faster(self) -> None:
        sel = ProsodySelector()
        emphatic = sel.select("That is amazing!")
        conversational = sel.select("That is fine.")
        assert emphatic.length_scale < conversational.length_scale
        assert emphatic.label == "emphatic"

    def test_ellipsis_adds_pause(self) -> None:
        sel = ProsodySelector()
        thoughtful = sel.select("Well... let me think.")
        plain = sel.select("Let me think.")
        assert thoughtful.sentence_silence > plain.sentence_silence
        assert thoughtful.label == "thoughtful"

    def test_rising_engagement_is_eager_and_fast(self) -> None:
        sel = ProsodySelector()
        eager = sel.select("Tell me more about it.", trajectory="rising", engagement=0.8)
        calm = sel.select("Tell me more about it.", trajectory="falling", engagement=0.3)
        assert eager.label == "eager"
        assert eager.length_scale < calm.length_scale

    def test_falling_is_calm_and_slow(self) -> None:
        sel = ProsodySelector()
        calm = sel.select("I see.", trajectory="falling", engagement=0.3)
        assert calm.label == "calm"
        assert calm.length_scale > 1.0

    def test_topic_shift_intro_slower_only_on_first_chunk(self) -> None:
        sel = ProsodySelector()
        first = sel.select(
            "Let us talk about space.", topic_shift=True, first=True
        )
        later = sel.select(
            "Let us talk about space.", topic_shift=True, first=False
        )
        assert first.length_scale > later.length_scale
        assert first.label == "conversational-intro"

    def test_numbered_item_is_steady(self) -> None:
        sel = ProsodySelector()
        item = sel.select("1. First step")
        assert item.label == "list"
        assert item.length_scale == 1.0


class TestProsodyPipelineIntegration:
    def test_statement_uses_conversational_profile(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized
            assert "conversational" in p.tts.prosody_labels

        asyncio.run(run())

    def test_question_uses_answer_profile(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = _FakeSTTQuestion()
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized
            assert "answer" in p.tts.prosody_labels

        asyncio.run(run())


class TestEchoGating:
    def test_looks_like_echo_positive(self) -> None:
        p = _make_pipeline()
        p._last_spoken["sess"] = (
            "let me explain the Hadamard matrix and its uses"
        )
        assert p._looks_like_echo(
            "sess", "explain the Hadamard matrix"
        )

    def test_looks_like_echo_negative_on_distinct_topic(self) -> None:
        p = _make_pipeline()
        p._last_spoken["sess"] = (
            "the weather is sunny and warm today"
        )
        assert not p._looks_like_echo("sess", "what is the capital of France")

    def test_segment_dropped_when_transcript_matches_last_spoken(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p._last_spoken["sess"] = "hello world and welcome back everyone"
            p.stt = _FakeSTTEcho()

            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized == []

        asyncio.run(run())

    def test_segment_kept_when_transcript_is_new(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p._last_spoken["sess"] = "the weather is sunny and warm"
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized

        asyncio.run(run())
