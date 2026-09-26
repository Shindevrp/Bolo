from __future__ import annotations

import asyncio

import pytest

from core.pipeline import ConversationContext, StreamingPipeline
from modules.laya.questions import CADENCE1_QUESTIONS, TURN_QUESTIONS
from providers.laya.client import LayaSystem1
from tests.test_streaming import FakeLLMText, FakeSTTText, FakeTTSStream, FakeVAD


class FakeAgent:
    """Sync stand-in for the Laya agent (returns canned answers)."""

    def __init__(self, answers=None) -> None:
        self._answers = answers or {}
        self.calls: list[tuple[dict, dict]] = []

    def predict(self, state, questions):
        self.calls.append((dict(state), dict(questions)))
        return {"answers": self._answers}


def _answers(
    intent="command",
    is_question=True,
    sentiment="neutral",
    complexity="standard",
    topic_changed=False,
    needs_verify=True,
    tool_needed=True,
    invoke_llm=True,
    escalate=False,
    urgent=True,
    high_stakes=False,
    tool="search_web",
) -> dict:
    return {
        "intent": {"type": "choice", "choice": intent, "confidence": 0.93},
        "is_question": {"type": "noul", "noul": 1.0 if is_question else 0.0},
        "query_complexity": {"type": "choice", "choice": complexity, "confidence": 0.9},
        "topic_changed": {"type": "noul", "noul": 1.0 if topic_changed else 0.0},
        "needs_verify": {"type": "noul", "noul": 1.0 if needs_verify else 0.0},
        "tool_needed": {"type": "noul", "noul": 1.0 if tool_needed else 0.0},
        "tool": {"type": "choice", "choice": tool, "confidence": 0.97},
        "invoke_llm": {"type": "noul", "noul": 1.0 if invoke_llm else 0.0},
        "escalate": {"type": "noul", "noul": 1.0 if escalate else 0.0},
        "sentiment": {"type": "choice", "choice": sentiment, "confidence": 0.85},
        "urgent": {"type": "noul", "noul": 1.0 if urgent else 0.0},
        "high_stakes": {"type": "noul", "noul": 1.0 if high_stakes else 0.0},
    }


def _client(answers=None, **kwargs) -> LayaSystem1:
    return LayaSystem1(enabled=True, agent=FakeAgent(answers), **kwargs)


class TestLayaSystem1:
    def test_disabled_client_returns_empty(self) -> None:
        c = LayaSystem1(enabled=False, agent=FakeAgent(_answers()))

        async def run() -> dict:
            return await c.predict({"transcript": "hi"}, TURN_QUESTIONS)

        assert asyncio.run(run()) == {}
        assert c.available is False
        assert c.latency_report_ms() == {}

    def test_predict_records_latency(self) -> None:
        c = _client(_answers())

        async def run() -> dict:
            a = await c.predict({"transcript": "hi"}, TURN_QUESTIONS)
            return a

        answers = asyncio.run(run())
        assert answers["intent"]["choice"] == "command"
        assert c.latency_report_ms()["count"] == 1

    def test_typed_extractors(self) -> None:
        c = _client(_answers(), conf_threshold=0.8)
        answers = _answers()

        assert c.choice(answers, "intent") == "command"
        assert c.choice(answers, "intent", default="statement") == "command"
        # below-threshold confidence -> default (fail-open)
        weak = {"intent": {"type": "choice", "choice": "statement", "confidence": 0.4}}
        assert c.choice(weak, "intent", default="fallback") == "fallback"

        assert c.noul_prob(answers, "is_question") == 1.0
        assert c.noul(answers, "urgent") is True
        low = {"urgent": {"type": "noul", "noul": 0.1}}
        assert c.noul(low, "urgent", default=False) is False

        score_answers = {"urgency": {"type": "score", "score": 2.0}}
        assert c.score(score_answers, "urgency") == 2.0
        assert c.score(score_answers, "missing") is None

    def test_shadow_report_aggregates_agreement(self) -> None:
        c = _client()
        c.log_shadow(session_id="s1", transcript="a", rows=[
            {"question": "intent", "laya": "question", "laya_conf": 0.9, "legacy": "question", "match": True},
            {"question": "urgency", "laya": True, "laya_conf": 0.8, "legacy": False, "match": False},
            {"question": "intent", "laya": None, "laya_conf": 0.0, "legacy": "question", "match": False},
        ])
        report = c.shadow_report()
        assert report["intent"]["n"] == 2
        assert report["intent"]["agreement"] == 0.5
        assert report["urgency"]["agreement"] == 0.0
        assert report["urgency"]["n"] == 1


class TestShadowIntegration:
    def _pipeline(self, answers=None) -> tuple[StreamingPipeline, FakeAgent]:
        agent = FakeAgent(answers or _answers())
        s1 = LayaSystem1(enabled=True, agent=agent)
        p = StreamingPipeline(
            FakeSTTText(), FakeLLMText(), FakeTTSStream(), FakeVAD(), system1=s1
        )
        return p, agent

    def test_shadow_does_not_override_legacy_context(self) -> None:
        async def run() -> None:
            p, _ = self._pipeline()
            ctx = ConversationContext()
            seg = asyncio.create_task(
                p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            )
            await seg
            # Shadow-only passes run AFTER the reply finishes (never stealing
            # its GPU) -- flush the deferred task before reading telemetry.
            await asyncio.gather(
                *list(p._s1_bg_tasks), return_exceptions=True
            )
            # legacy classifier wins on "hello world" -> greeting; Laya said command
            assert ctx.intent == "greeting"
            assert ctx.user_sentiment == "neutral"
            assert not ctx.is_question  # "hello world" has no '?'
            report = p.shadow_report()
            assert report  # shadow rows were recorded
            intent_row = report.get("intent")
            assert intent_row is not None and intent_row["n"] == 1
            assert intent_row["agreement"] == 0.0  # command != greeting

        asyncio.run(run())

    def test_shadow_schema_received_full_turn_questions(self) -> None:
        async def run() -> None:
            p, agent = self._pipeline()
            ctx = ConversationContext()
            seg = asyncio.create_task(
                p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            )
            await seg
            await asyncio.gather(
                *list(p._s1_bg_tasks), return_exceptions=True
            )
            assert agent.calls
            _, questions = agent.calls[0]
            # cadence-2 batch covers the whole action surface
            assert set(TURN_QUESTIONS) == set(questions)

        asyncio.run(run())


class TestLayaSchemaContract:
    """Guard the typed-question schemas against the installed laya runtime.

    Laya rejects malformed question definitions inside ``predict`` -- the
    whole batched forward pass fails, so one bad question (e.g. a ``score``
    without its ``criteria`` level list) silently disables the entire
    System-1 surface. These tests run the exact definitions through laya's
    own validator (``_check_question``) and option-head budget check, so a
    schema mistake fails in CI instead of in shadow telemetry.
    """

    def test_all_question_definitions_valid(self) -> None:
        pytest.importorskip("laya")
        from laya.agent import Agent as LayaAgent

        for qid, q in {**TURN_QUESTIONS, **CADENCE1_QUESTIONS}.items():
            LayaAgent._check_question(qid, q)  # raises ValueError when malformed

    def test_option_heads_within_model_budget(self) -> None:
        pytest.importorskip("laya")
        from laya import common

        # head_max_len=192; each option renders as [MASK] + text, so keep the
        # per-question option count (the strictest, simplest bound) far below it.
        for qid, q in {**TURN_QUESTIONS, **CADENCE1_QUESTIONS}.items():
            internal = {"t": q["type"], "ins": q.get("instructions"), "crit": q.get("criteria")}
            assert len(common.render_options(internal)) <= 64, (
                f"{qid!r} renders {len(common.render_options(internal))} options, "
                f"too close to the 192-token option head"
            )