"""Regression pins for the Laya decision-layer review fixes.

- Laya shadow lock: never drifts above 1 and bounds the caller's wait.
- Canned phatic replies: only whole-utterance phatics skip the LLM.
- Soft-barge frame gate: frames must be consecutive, not accumulated.
"""
from __future__ import annotations

import asyncio
import time

from core.pipeline import StreamingPipeline
from providers.laya.client import LayaSystem1
from tests.test_laya_adapters import FakeAgent
from tests.test_laya_phase2 import _pipeline, _run_turn
from tests.test_streaming import _make_pipeline


class SlowAgent:
    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.calls = 0

    def predict(self, state, questions):
        self.calls += 1
        time.sleep(self._delay)
        return {"answers": {"intent": {"choice": "command", "confidence": 0.99}}}


class TestLayaLock:
    def test_contended_slow_calls_do_not_inflate_lock(self) -> None:
        async def run() -> None:
            agent = SlowAgent(0.6)
            s1 = LayaSystem1(enabled=True, agent=agent, shadow_timeout=0.2)
            t0 = time.perf_counter()
            results = await asyncio.gather(*[s1.predict({}, {}) for _ in range(3)])
            elapsed = time.perf_counter() - t0
            assert results == [{}, {}, {}]
            # Caller-side wait is bounded by one timeout (plus slack).
            assert elapsed < 0.4
            # Only one inference ever started; the others never got the lock.
            assert agent.calls == 1
            # Once the slow thread finishes the lock is back to exactly 1.
            await asyncio.sleep(0.6)
            assert s1._semaphore._value == 1

        asyncio.run(run())

    def test_fast_call_returns_answers_and_releases(self) -> None:
        async def run() -> None:
            s1 = LayaSystem1(enabled=True, agent=SlowAgent(0.0), shadow_timeout=0.5)
            answers = await s1.predict({}, {})
            assert answers["intent"]["choice"] == "command"
            assert s1._semaphore._value == 1

        asyncio.run(run())


class TestPhaticGate:
    def test_requests_with_a_leading_cue_go_to_llm(self) -> None:
        async def run() -> None:
            for text in ("ok book a table", "sure cancel my order", "hello world"):
                p, llm = _pipeline(text, answers={})
                ctx = await _run_turn(p, text)
                assert ctx.invoke_llm is True, text
                assert llm.calls == 1, text

        asyncio.run(run())

    def test_pure_phatics_use_canned_reply(self) -> None:
        words = StreamingPipeline._FAST_PHATIC_WORDS
        gate = StreamingPipeline._phatic_gate
        assert gate("ok", words["backchannel"])
        assert gate("hello there", words["greeting"])
        assert gate("see you", words["farewell"])
        assert not gate("no thanks", words["backchannel"])
        assert not gate("ok book a table", words["backchannel"])


class TestSoftBargeFrames:
    def test_neutral_frame_resets_soft_run(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            sid = "sess"
            ctx = p._ctx(sid)
            p._playback_active[sid] = True
            p._echo_floor[sid] = 0.15
            p._barge_pending[sid] = True
            interrupted: list[str] = []

            async def fake_interrupt(s, c) -> None:
                interrupted.append(s)

            p._do_interrupt = fake_interrupt  # type: ignore[method-assign]
            soft, neutral = 0.122, 0.0
            await p._maybe_fire_barge(sid, ctx, soft)
            await p._maybe_fire_barge(sid, ctx, soft)
            await p._maybe_fire_barge(sid, ctx, neutral)
            await p._maybe_fire_barge(sid, ctx, soft)
            assert p._barge_soft_frames.get(sid) == 1
            assert interrupted == []

        asyncio.run(run())


class TestLayaScheduling:
    def test_low_priority_call_skips_while_turn_call_waits(self) -> None:
        async def run() -> None:
            agent = SlowAgent(0.3)
            s1 = LayaSystem1(enabled=True, agent=agent, shadow_timeout=1.0)
            first = asyncio.create_task(s1.predict({}, {}))  # holds the model
            await asyncio.sleep(0.05)
            turn = asyncio.create_task(s1.predict({}, {}))  # waits for it
            await asyncio.sleep(0.05)
            assert await s1.predict({}, {}, priority=False) == {}
            assert (await first)["intent"]["choice"] == "command"
            assert (await turn)["intent"]["choice"] == "command"
            assert agent.calls == 2  # the skipped probe never ran

        asyncio.run(run())

    def test_no_gpu_disables_real_model_unless_forced(self, monkeypatch) -> None:
        from providers.laya import client

        monkeypatch.setattr(client, "_resolve_device", lambda d: "cpu")
        monkeypatch.setattr(client, "_laya", object())
        assert not LayaSystem1(enabled=True).enabled
        assert LayaSystem1(enabled=True, device="cpu").enabled
        assert LayaSystem1(enabled=True, allow_cpu=True).enabled

    def test_shadow_only_turn_does_not_wait_for_laya(self) -> None:
        async def run() -> None:
            p, llm = _pipeline("hello world", phase2=False)
            p._s1 = LayaSystem1(enabled=True, agent=SlowAgent(1.0))
            t0 = time.perf_counter()
            await _run_turn(p, "hello world")
            assert time.perf_counter() - t0 < 0.8
            assert llm.calls == 1
            await asyncio.gather(*p._s1_bg_tasks)
            assert p.shadow_report()  # rows still recorded afterwards

        asyncio.run(run())


def _rows(p: StreamingPipeline, question: str) -> list[dict]:
    return [
        r for e in p._s1._shadow for r in e["rows"] if r["question"] == question
    ]


class TestOutcomeLabels:
    """Training labels come from what actually happened, not from agreeing
    with a placeholder "legacy" value."""

    def _pipeline(self, text: str, answers: dict, **kw) -> StreamingPipeline:
        from tests.test_laya_phase2 import FixedSTT
        from tests.test_streaming import FakeLLMText, FakeTTSStream, FakeVAD

        return StreamingPipeline(
            FixedSTT(text),
            FakeLLMText(),
            FakeTTSStream(),
            FakeVAD(),
            system1=LayaSystem1(enabled=True, agent=FakeAgent(answers)),
            **kw,
        )

    def test_tool_outcome_labels_shadow_and_phase2(self) -> None:
        from tests.test_laya_adapters import _answers
        from tests.test_laya_speed import _run_turn as run_task_turn

        async def run(phase2: bool) -> None:
            # Laya guesses "no tool"; the turn really ran get_time.
            answers = _answers(tool="none", tool_needed=False)
            p = self._pipeline(
                "what time is it", answers, fast_path=True, phase2=phase2
            )
            await run_task_turn(p)
            tool, needed = _rows(p, "tool")[0], _rows(p, "tool_needed")[0]
            assert tool["self_label"] == "get_time"
            assert tool["label_source"] == "outcome"
            assert needed["self_label"] is True

        asyncio.run(run(False))
        asyncio.run(run(True))

    def test_turn_without_tool_labels_none(self) -> None:
        from tests.test_laya_adapters import _answers
        from tests.test_laya_speed import _run_turn as run_task_turn

        async def run() -> None:
            p = self._pipeline("tell me a story", _answers(tool="search_web"))
            await run_task_turn(p)
            assert _rows(p, "tool")[0]["self_label"] == "none"
            assert _rows(p, "tool_needed")[0]["self_label"] is False

        asyncio.run(run())

    def test_confident_no_is_a_label(self) -> None:
        from tests.test_laya_adapters import _answers
        from tests.test_laya_speed import _run_turn as run_task_turn

        async def run() -> None:
            p = self._pipeline("hello world", _answers(is_question=False))
            await run_task_turn(p)
            row = _rows(p, "is_question")[0]
            assert row["laya_conf"] == 1.0  # P(yes)=0 is a certain "no"
            assert row["self_label"] is False

        asyncio.run(run())

    def test_endpoint_labels_from_partials(self) -> None:
        p = self._pipeline("x", {})
        final = "what is the weather in paris"
        p._log_endpoint_labels(
            "s",
            ["what is the", "what is the weather in", final, final],
            final,
            1,
        )
        labels = {
            e["transcript"]: e["rows"][0]["self_label"] for e in p._s1._shadow
        }
        # 3 words short -> kept talking; 1 short -> ambiguous, skipped;
        # same length -> finished. Duplicates are logged once.
        assert labels == {"what is the": False, final: True}


class TestPrefetchGate:
    def test_confident_tool_prefetches_without_tool_needed(self) -> None:
        from tests.test_laya_phase2 import FixedSTT
        from tests.test_laya_speed import TestToolPrefetch
        from tests.test_streaming import FakeLLMText, FakeTTSStream, FakeVAD

        async def run() -> None:
            agent = FakeAgent({
                "tool_needed": {"type": "noul", "noul": 0.3},
                "tool": {"type": "choice", "choice": "get_time", "confidence": 0.8},
            })
            p = StreamingPipeline(
                FixedSTT("what time is it now"),
                FakeLLMText(),
                FakeTTSStream(),
                FakeVAD(),
                system1=LayaSystem1(enabled=True, agent=agent),
                tool_prefetch=True,
            )
            registry = TestToolPrefetch.FakeRegistry()
            p._tool_registry = registry  # type: ignore[assignment]
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await asyncio.sleep(0.05)
            assert registry.calls == [{"name": "get_time", "args": []}]

        asyncio.run(run())

    def test_prefetch_used_only_when_it_still_answers(self) -> None:
        p = _make_pipeline()
        search = {"tool": "search_web", "partial": "who is nikola", "args": ["who is nikola"]}
        assert p._prefetch_for(search, "who is nikola tesla") is None
        assert p._prefetch_for(search, "Who is Nikola?") is search
        calc = {"tool": "calculate", "partial": "calculate 2 + 2", "args": ["2 + 2"]}
        assert p._prefetch_for(calc, "calculate 2 + 2 please") is calc
        assert p._prefetch_for(calc, "calculate 2 + 25") is not None  # substring
        assert p._prefetch_for(calc, "calculate 3 + 3") is None
        clock = {"tool": "get_time", "partial": "what time", "args": []}
        assert p._prefetch_for(clock, "what time is it in rome") is clock
