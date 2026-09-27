"""Phase-2 (enforcement) behavior of the System-1 decision layer.

Phase-2 is opt-in via ``phase2=True`` on the pipeline (or BOLO_LAYA_PHASE2=1).
All checks here either prove the fail-open fallback (weak/missing Laya answers
keep legacy behavior) or the conservative routing guarantees agreed at design
time: urgent = latency modifier only, LLM-skip only when a deterministic
fast-action exists, escalate = route to the fallback model.
"""
from __future__ import annotations

import asyncio
import re
import time

from core.pipeline import ConversationContext, StreamingPipeline
from providers.laya.client import LayaSystem1
from tests.test_laya_adapters import FakeAgent, _answers
from tests.test_streaming import FakeTTSStream, FakeVAD, _is_label_call


class FixedSTT:
    def __init__(self, text: str) -> None:
        self._text = text

    async def transcribe(self, audio_blob: bytes) -> str:
        return self._text


class CountingLLM:
    """Counts real turns; background topic-label prompts are filtered."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[list[dict]] = []

    async def generate_stream(self, messages):
        if _is_label_call(messages):
            for tok in ["label"]:
                yield tok
            return
        self.calls += 1
        self.seen.append(messages)
        for tok in ["Sure", ", ", "here ", "you ", "go."]:
            yield tok


class SpyTiming:
    def __init__(self) -> None:
        self.calls = 0

    def compute_delay(self, **kwargs) -> float:
        self.calls += 1
        return 0.05


def _pipeline(
    text: str,
    *,
    phase2: bool = True,
    answers: dict | None = None,
    fallback: CountingLLM | None = None,
    timing=None,
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
        llm_fallback=fallback,
        turn_timing=timing,
        phase2=phase2,
    )
    return p, llm


async def _run_turn(p, text: str) -> ConversationContext:
    ctx = ConversationContext()
    # Prod runs the turn as its own task (pipeline.py) -- the deferred
    # shadow pass waits on it, so tests must mirror that or a child that
    # awaits the inline test task can deadlock with a later gather.
    seg = asyncio.create_task(
        p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
    )
    await seg
    return ctx


class TestPhase2Override:
    def test_weak_or_missing_laya_keeps_legacy(self) -> None:
        """Empty answers (Laya down/timeout) -> legacy intent wins, still fast."""

        async def run() -> None:
            p, llm = _pipeline("hello there", answers={})
            ctx = await _run_turn(p, "hello there")
            assert ctx.intent == "greeting"  # legacy classifier
            assert ctx.fast_path is True
            assert ctx.invoke_llm is False
            assert llm.calls == 0

        asyncio.run(run())

    def test_weak_confidence_keeps_legacy_but_fast_path_uses_it(self) -> None:
        """Low-confidence intent answer cannot override; fast table still works."""

        async def run() -> None:
            answers = dict(_answers(tool_needed=False, tool="none", urgent=False))
            answers["intent"] = {
                "type": "choice", "choice": "command", "confidence": 0.1,
            }
            p, llm = _pipeline("hello there", answers=answers)
            ctx = await _run_turn(p, "hello there")
            assert ctx.intent == "greeting"  # override refused (< threshold)
            assert ctx.fast_path is True
            assert llm.calls == 0

        asyncio.run(run())

    def test_confident_intent_override_routes_off_fast_path(self) -> None:
        """High-confidence Laya intent overrides legacy and clears fast path."""

        async def run() -> None:
            answers = _answers(
                intent="command",
                is_question=False,
                urgent=False,
                tool_needed=False,
                tool="none",
            )
            p, llm = _pipeline("hello world", answers=answers)
            ctx = await _run_turn(p, "hello world")
            assert ctx.intent == "command"  # Laya beat the legacy "greeting"
            assert ctx.is_question is False
            assert ctx.fast_path is False
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())

    def test_is_question_override(self) -> None:
        """A confident noul override changes ctx.is_question even without '?'."""

        async def run() -> None:
            answers = _answers(
                intent="question",
                is_question=True,
                urgent=False,
                tool_needed=False,
                tool="none",
            )
            p, _ = _pipeline("hello world", answers=answers)
            ctx = await _run_turn(p, "hello world")
            assert ctx.intent == "question"
            assert ctx.is_question is True  # legacy: no '?' -> False

        asyncio.run(run())

    def test_complexity_lock_survives_build_messages(self) -> None:
        """Locked Laya complexity is not clobbered by the legacy heuristic."""

        async def run() -> None:
            answers = _answers(
                intent="command",
                is_question=False,
                urgent=False,
                tool_needed=False,
                tool="none",
                complexity="complex",
            )
            p, _ = _pipeline("hello world", answers=answers)
            ctx = await _run_turn(p, "hello world")
            assert ctx.complexity_locked is True
            assert ctx.query_complexity == "complex"  # legacy would say "simple"

        asyncio.run(run())


class TestFastPath:
    def test_greeting_canned(self) -> None:
        async def run() -> None:
            p, llm = _pipeline("hello there", answers={})
            ctx = await _run_turn(p, "hello there")
            assert p.tts.synthesized, "fast reply must be spoken"
            assert p.tts.synthesized[0].startswith("Hi! How can I")
            assert llm.calls == 0
            assert ctx.fast_path is True

        asyncio.run(run())

    def test_farewell_canned(self) -> None:
        async def run() -> None:
            p, _ = _pipeline("goodbye", answers={})
            await _run_turn(p, "goodbye")
            assert p.tts.synthesized
            assert p.tts.synthesized[0].startswith("Goodbye!")

        asyncio.run(run())

    def test_time_tool_without_llm(self) -> None:
        async def run() -> None:
            p, llm = _pipeline("what time is it", answers={})
            ctx = await _run_turn(p, "what time is it")
            assert llm.calls == 0
            assert ctx.fast_path is True
            assert p.tts.synthesized
            assert re.match(r"It is \d{1,2}:\d{2} [AP]M\.", p.tts.synthesized[0])

        asyncio.run(run())

    def test_time_in_place_falls_back_to_llm(self) -> None:
        """A place modifier must NOT trigger the local-time fast path."""

        async def run() -> None:
            p, llm = _pipeline("what time is it in paris", answers={})
            ctx = await _run_turn(p, "what time is it in paris")
            assert ctx.fast_path is False
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())

    def test_calculate_whole_utterance_only(self) -> None:
        async def run() -> None:
            p, llm = _pipeline("calculate 2 + 3", answers={})
            ctx = await _run_turn(p, "calculate 2 + 3")
            assert ctx.fast_path is True
            assert llm.calls == 0
            assert p.tts.synthesized and p.tts.synthesized[0] == "The answer is 5."

        asyncio.run(run())

    def test_calculate_with_context_falls_back_to_llm(self) -> None:
        """'calculate the tip for 20 plus 30' has no standalone expression."""

        async def run() -> None:
            p, llm = _pipeline(
                "calculate the tip for 20 plus 30", answers={}
            )
            ctx = await _run_turn(p, "calculate the tip for 20 plus 30")
            assert ctx.fast_path is False
            assert llm.calls == 1

        asyncio.run(run())

    def test_no_fast_action_runs_llm(self) -> None:
        async def run() -> None:
            p, llm = _pipeline("tell me a story about dragons", answers={})
            ctx = await _run_turn(p, "tell me a story about dragons")
            assert ctx.fast_path is False
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())


class TestUrgentLatency:
    def test_urgent_skips_turn_delay(self) -> None:
        async def run() -> None:
            spy = SpyTiming()
            answers = _answers(
                intent="statement",
                urgent=True,
                is_question=False,
                tool_needed=False,
                tool="none",
            )
            p, _ = _pipeline(
                "tell me a story about dragons", answers=answers, timing=spy
            )
            ctx = await _run_turn(p, "tell me a story about dragons")
            assert ctx.urgent is True
            assert spy.calls == 0  # latency modifier: delay block skipped

        asyncio.run(run())

    def test_non_urgent_still_delays(self) -> None:
        async def run() -> None:
            spy = SpyTiming()
            answers = _answers(
                intent="statement",
                urgent=False,
                is_question=False,
                tool_needed=False,
                tool="none",
            )
            p, _ = _pipeline(
                "tell me a story about dragons", answers=answers, timing=spy
            )
            await _run_turn(p, "tell me a story about dragons")
            assert spy.calls == 1

        asyncio.run(run())

    def test_weak_urgent_vote_is_ignored(self) -> None:
        """A sub-0.9 urgent vote must not be trusted (fail-open)."""

        async def run() -> None:
            spy = SpyTiming()
            answers = _answers(
                intent="statement",
                urgent=True,
                is_question=False,
                tool_needed=False,
                tool="none",
            )
            answers["urgent"] = {"type": "noul", "noul": 0.4}
            p, _ = _pipeline(
                "tell me a story about dragons", answers=answers, timing=spy
            )
            ctx = await _run_turn(p, "tell me a story about dragons")
            assert ctx.urgent is False
            assert spy.calls == 1  # delay not skipped

        asyncio.run(run())


class TestEscalation:
    def test_escalate_routes_to_fallback_llm(self) -> None:
        async def run() -> None:
            fb = CountingLLM()
            answers = _answers(
                intent="statement",
                urgent=False,
                is_question=False,
                tool_needed=False,
                tool="none",
                escalate=True,
            )
            p, primary = _pipeline(
                "tell me a story about dragons", answers=answers, fallback=fb
            )
            ctx = await _run_turn(p, "tell me a story about dragons")
            assert ctx.model_tier == "fallback"
            assert primary.calls == 0
            assert fb.calls == 1

        asyncio.run(run())

    def test_escalate_without_fallback_degrades_to_primary(self) -> None:
        async def run() -> None:
            answers = _answers(
                intent="statement",
                urgent=False,
                is_question=False,
                tool_needed=False,
                tool="none",
                escalate=True,
            )
            p, llm = _pipeline("tell me a story about dragons", answers=answers)
            ctx = await _run_turn(p, "tell me a story about dragons")
            assert ctx.model_tier == "fallback"
            assert llm.calls == 1  # graceful degrade, never a dead turn

        asyncio.run(run())

    def test_no_escalate_keeps_primary(self) -> None:
        async def run() -> None:
            fb = CountingLLM()
            answers = _answers(
                intent="statement",
                urgent=False,
                is_question=False,
                tool_needed=False,
                tool="none",
                escalate=False,
            )
            p, llm = _pipeline(
                "tell me a story about dragons", answers=answers, fallback=fb
            )
            await _run_turn(p, "tell me a story about dragons")
            assert llm.calls == 1
            assert fb.calls == 0

        asyncio.run(run())


class TestPhase2Disabled:
    def test_phase2_off_is_prior_behavior(self) -> None:
        """phase2=False: shadow rows only, no overrides, no fast path."""

        async def run() -> None:
            answers = _answers(
                intent="command", is_question=True, urgent=False, tool_needed=False,
                tool="none", complexity="complex",
            )
            p, llm = _pipeline("hello world", phase2=False, answers=answers)
            ctx = await _run_turn(p, "hello world")
            # Shadow-only passes run AFTER the reply finishes (never stealing
            # its GPU) -- flush the deferred task before reading telemetry.
            await asyncio.gather(
                *list(p._s1_bg_tasks), return_exceptions=True
            )
            assert ctx.intent == "greeting"       # legacy classifier
            assert ctx.fast_path is False         # fast table off (legacy coupling)
            assert ctx.invoke_llm is True
            assert ctx.complexity_locked is False
            assert llm.calls == 1
            assert p.shadow_report()              # shadow rows still recorded

        asyncio.run(run())


class SlowAgent:
    """Sync agent that blocks the worker thread for ``delay`` seconds."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.calls = 0

    def predict(self, state, questions):
        self.calls += 1
        time.sleep(self._delay)
        return {"answers": _answers(tool_needed=False, tool="none", urgent=False)}


class TestShadowLock:
    def test_overlapping_calls_serialize_and_free_lock_once(self) -> None:
        """3 concurrent predicts on a slow model: none overlap, the lock is
        back to 1 once the model finishes, and every call returns within about
        the timeout (the loser never double-frees the counter)."""

        async def run() -> None:
            agent = SlowAgent(delay=0.4)
            s1 = LayaSystem1(enabled=True, agent=agent, shadow_timeout=0.15)
            t0 = time.perf_counter()
            results = await asyncio.gather(
                *(s1.predict({}, {}) for _ in range(3))
            )
            elapsed = time.perf_counter() - t0
            # Losers give up at the acquire timeout; the winner only waits its
            # own post-acquire remaining budget, so the gather is bounded --
            # not 3x, and not acquire+model in series.
            assert elapsed < 0.9, elapsed
            # The model is slower than the timeout, so every caller gives up
            # fail-open -- and exactly one inference actually ran: the losers
            # never entered the model, they never touched the lock.
            assert results == [{}, {}, {}]
            assert agent.calls == 1
            # The winner's thread is still running (lock held by its bg task);
            # once it finishes, the lock is back to exactly 1 -- a double-free
            # would leave it at 2, a lost release would strand it at 0.
            assert s1._semaphore._value == 0
            await asyncio.sleep(0.5)
            assert s1._semaphore._value == 1

        asyncio.run(run())

    def test_fast_model_returns_answers_to_holder(self) -> None:
        """A model that finishes inside the budget reaches the foreground via
        the holder dict (bg writes flat answers, fg reads the holder)."""

        async def run() -> None:
            agent = SlowAgent(delay=0.0)
            s1 = LayaSystem1(enabled=True, agent=agent, shadow_timeout=1.0)
            answers = await s1.predict({}, {})
            assert isinstance(answers, dict) and answers
            assert "intent" in answers
            assert s1._semaphore._value == 1

        asyncio.run(run())


class TestCannedReplies:
    """Whole-utterance phatics gate: a canned lane fires only when EVERY
    token belongs to that lane (or is a filler), so a real request phrased
    around a backchannel word still reaches the LLM."""

    def _assert_llm(self, text: str) -> None:
        async def run() -> None:
            p, llm = _pipeline(text, answers={})
            ctx = await _run_turn(p, text)
            assert ctx.fast_path is False, text
            assert ctx.invoke_llm is True, text
            assert llm.calls == 1, text

        asyncio.run(run())

    def test_backchannel_cue_plus_request_reaches_llm(self) -> None:
        for text in ("ok book a table", "sure cancel my order", "hello world"):
            self._assert_llm(text)

    def test_bare_backchannel_cue_is_canned(self) -> None:
        async def run() -> None:
            for text in ("ok", "hello there", "see you"):
                p, llm = _pipeline(text, answers={})
                ctx = await _run_turn(p, text)
                assert ctx.fast_path is True, text
                assert llm.calls == 0, text
                assert p.tts.synthesized, text

        asyncio.run(run())

    def test_no_and_problem_are_not_cues(self) -> None:
        """'no'/'problem' were dropped from the backchannel vocab, so these
        carry no cue at all and must reach the LLM."""

        for text in ("no thanks", "no problem"):
            self._assert_llm(text)
