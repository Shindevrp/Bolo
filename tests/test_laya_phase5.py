"""Phase-5 (complexity-routing gate) behavior of the System-1 decision layer.

Phase 5 is opt-in via ``phase5=True`` (or TASA_LAYA_PHASE5=1). A bare
multi-word acknowledgment ("sounds good", "that works") is answered with a
deterministic reply instead of the LLM ONLY when a confident Laya capstone
verdict marks it trivial: complex is simple, not a question, not urgent, and
the turn did not escalate to the fallback model. Everything weak/missing
keeps the LLM (fail-open), and phase5 off never changes routing.
"""
from __future__ import annotations

import asyncio

from core.pipeline import ConversationContext, StreamingPipeline
from providers.laya.client import LayaSystem1
from tests.test_laya_adapters import FakeAgent, _answers
from tests.test_laya_phase2 import CountingLLM, FixedSTT
from tests.test_streaming import FakeTTSStream, FakeVAD


def _pipeline(
    text: str,
    *,
    phase5: bool = True,
    answers: dict | None = None,
):
    llm = CountingLLM()
    s1 = LayaSystem1(
        enabled=True,
        agent=FakeAgent({} if answers is None else answers),
    )
    p = StreamingPipeline(
        FixedSTT(text),
        llm,
        FakeTTSStream(),
        FakeVAD(),
        system1=s1,
        phase5=phase5,
    )
    return p, llm


async def _run_turn(p: StreamingPipeline, text: str) -> ConversationContext:
    ctx = ConversationContext()
    seg = asyncio.create_task(p._process_speech_segment(b"\x80" * 1600, "sess", ctx))
    await seg
    return ctx


class TestAckTable:
    def test_ack_phrase_exact_match_only(self) -> None:
        assert StreamingPipeline._ack_phrase("Sounds good!") == "Sounds good!"
        assert StreamingPipeline._ack_phrase("trahns aaux") is None
        assert StreamingPipeline._ack_phrase("sounds good okay") is None
        assert StreamingPipeline._ack_phrase("that works") == "That works!"
        assert StreamingPipeline._ack_phrase("book a table") is None


class TestComplexityGate:
    def test_phase5_off_never_routes_ack_off_llm(self) -> None:
        async def run() -> None:
            answers = dict(
                _answers(
                    complexity="simple", is_question=False, urgent=False
                )
            )
            p, llm = _pipeline("that works", phase5=False, answers=answers)
            ctx = await _run_turn(p, "that works")
            assert ctx.fast_path is False
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())

    def test_phase5_confident_simple_serves_ack(self) -> None:
        async def run() -> None:
            answers = dict(
                _answers(
                    complexity="simple", is_question=False, urgent=False
                )
            )
            p, llm = _pipeline("sounds good", answers=answers)
            ctx = await _run_turn(p, "sounds good")
            assert ctx.fast_path is True
            assert ctx.invoke_llm is False
            assert llm.calls == 0

        asyncio.run(run())

    def test_weak_or_missing_complexity_keeps_llm(self) -> None:
        async def run() -> None:
            answers = dict(_answers(complexity="complex", is_question=False, urgent=False))
            answers["query_complexity"] = {
                "type": "choice", "choice": "simple", "confidence": 0.1,
            }
            p, llm = _pipeline("makes sense", answers=answers)
            ctx = await _run_turn(p, "makes sense")
            assert ctx.complexity_locked is False
            assert ctx.fast_path is False
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())

    def test_question_shaped_utterance_keeps_llm(self) -> None:
        async def run() -> None:
            answers = dict(
                _answers(
                    complexity="simple", is_question=True, urgent=False
                )
            )
            p, llm = _pipeline("sounds good?", answers=answers)
            ctx = await _run_turn(p, "sounds good?")
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())

    def test_urgent_utterance_keeps_llm(self) -> None:
        async def run() -> None:
            answers = dict(
                _answers(
                    complexity="simple", is_question=False, urgent=True
                )
            )
            p, llm = _pipeline("no problem", answers=answers)
            ctx = await _run_turn(p, "no problem")
            assert ctx.urgent is True
            assert ctx.fast_path is False
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())

    def test_escalated_turn_keeps_llm(self) -> None:
        async def run() -> None:
            answers = dict(
                _answers(
                    complexity="simple", is_question=False, urgent=False,
                    escalate=True,
                )
            )
            p, llm = _pipeline("got it", answers=answers)
            ctx = await _run_turn(p, "got it")
            assert ctx.model_tier == "fallback"
            assert ctx.invoke_llm is True
            assert llm.calls >= 1

        asyncio.run(run())

    def test_non_ack_utterance_never_served(self) -> None:
        async def run() -> None:
            answers = dict(
                _answers(
                    complexity="simple", is_question=False, urgent=False
                )
            )
            p, llm = _pipeline("sounds good but check the news", answers=answers)
            ctx = await _run_turn(p, "sounds good but check the news")
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())