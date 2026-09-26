"""Latency speed-up plan (approved): Steps 0 and 1 behavior.

Step 0.1: cadence-1 live-listening probes only fire under Phase-3 -- otherwise
  the per-partial Laya round-trip shares the GPU for a verdict nobody reads.
Step 0.2: shadow-only turn-level (cadence-2) passes run AFTER the reply
  finishes so they never steal GPU time from the LLM streaming the answer.
Step 0.3: prefetch schema (PREFETCH_QUESTIONS) + fp16 hook on the client.
Step 0.4: the deterministic fast-path table is decoupled from the Laya phases
  (``fast_path=`` / TASA_FAST_PATH).
Step 0.5: ``e2e_reply`` latency = SPEECH_END -> first TTS chunk.
Step 1: a confident live-listening tool prefetch seeds the LLM's first prompt
  so a tool round-trip is saved on a hit (and costs one lookup on a miss).
"""
from __future__ import annotations

import asyncio
import time

from core.pipeline import ConversationContext, StreamingPipeline
from modules.tools.registry import ToolRegistry
from providers.laya.client import LayaSystem1
from tests.test_laya_adapters import FakeAgent, _answers
from tests.test_laya_phase2 import FixedSTT
from tests.test_laya_phase3 import _c1_answers
from tests.test_streaming import FakeLLMText, FakeTTSStream, FakeVAD, _is_label_call


async def _run_turn(
    p: StreamingPipeline,
    blob: bytes = b"\x00" * 1600,
    session_id: str = "sess",
    ctx: ConversationContext | None = None,
) -> ConversationContext:
    """Drive one turn the way production does -- as its own task -- so the
    deferred shadow pass waits on that task and can finish once the reply
    does. Awaiting the segment inline would make the child await the whole
    test task (deadlock with the gather below)."""
    ctx = ctx or ConversationContext()
    seg = asyncio.create_task(p._process_speech_segment(blob, session_id, ctx))
    await seg
    await asyncio.gather(*list(p._s1_bg_tasks), return_exceptions=True)
    return ctx


class RecordingLLM:
    """Records the real turn's messages and when its stream finished."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.calls = 0
        self.finished_at: float | None = None

    async def generate_stream(self, messages):
        if _is_label_call(messages):
            for _ in ["label"]:
                yield "label"
            return
        self.calls += 1
        self.messages = list(messages)
        for tok in ["Sure", ", ", "here ", "you ", "go."]:
            yield tok
        self.finished_at = time.monotonic()


class RecordingAgent:
    def __init__(self, answers: dict | None = None) -> None:
        self._answers = answers or _answers()
        self.predict_at: float | None = None

    def predict(self, state, questions):
        if self.predict_at is None:
            self.predict_at = time.monotonic()
        return {"answers": self._answers}


class TestCadence1Gating:
    """Step 0.1: cadence-1 probes are a Phase-3-only cost."""

    def _pipeline(self, phase3: bool) -> StreamingPipeline:
        s1 = LayaSystem1(enabled=True, agent=FakeAgent(_c1_answers()))
        return StreamingPipeline(
            FixedSTT("who is nikola tesla"),
            FakeLLMText(),
            FakeTTSStream(),
            FakeVAD(),
            system1=s1,
            phase3=phase3,
        )

    def test_no_probe_when_phase3_off(self) -> None:
        async def run() -> None:
            p = self._pipeline(phase3=False)
            agent: FakeAgent = p._s1._agent  # type: ignore[assignment]
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await asyncio.sleep(0.05)
            assert agent.calls == []
            assert "sess" not in p._s1_cadence1

        asyncio.run(run())

    def test_probe_runs_when_phase3_on(self) -> None:
        async def run() -> None:
            p = self._pipeline(phase3=True)
            agent: FakeAgent = p._s1._agent  # type: ignore[assignment]
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            for _ in range(50):
                if "sess" in p._s1_cadence1:
                    break
                await asyncio.sleep(0.02)
            assert p._s1_cadence1["sess"]["turn_complete"] == 0.95
            assert agent.calls  # a cadence-1 pass ran
            await asyncio.gather(*list(p._s1_bg_tasks), return_exceptions=True)

        asyncio.run(run())


class TestCadence2Deferral:
    """Step 0.2: shadow-only cadence-2 runs after the reply, never over it."""

    def _pipeline(
        self, *, phase2: bool, agent=None
    ) -> tuple[StreamingPipeline, RecordingLLM]:
        llm = RecordingLLM()
        s1 = LayaSystem1(enabled=True, agent=agent or FakeAgent(_answers()))
        return (
            StreamingPipeline(
                FixedSTT("tell me a story"),
                llm,
                FakeTTSStream(),
                FakeVAD(),
                system1=s1,
                phase2=phase2,
            ),
            llm,
        )

    def test_shadow_pass_runs_after_reply_finished(self) -> None:
        async def run() -> None:
            agent = RecordingAgent(_answers())
            p, llm = self._pipeline(phase2=False, agent=agent)
            await _run_turn(p)
            assert llm.finished_at is not None
            assert agent.predict_at is not None
            assert agent.predict_at >= llm.finished_at
            assert p.shadow_report()

        asyncio.run(run())

    def test_phase2_pass_stays_synchronous(self) -> None:
        async def run() -> None:
            agent = RecordingAgent(_answers())
            p, llm = self._pipeline(phase2=True, agent=agent)
            await _run_turn(p)
            assert agent.predict_at is not None
            assert llm.finished_at is None or agent.predict_at <= llm.finished_at

        asyncio.run(run())


class TestFastPathDecoupling:
    """Step 0.4: fast-path table is independent of the Laya phases."""

    def _pipeline(self, *, fast_path):
        llm = RecordingLLM()
        s1 = LayaSystem1(enabled=True, agent=FakeAgent({}))
        p = StreamingPipeline(
            FixedSTT("hello there"),
            llm,
            FakeTTSStream(),
            FakeVAD(),
            system1=s1,
            phase2=False,
            fast_path=fast_path,
        )
        return p, llm

    def test_fast_path_on_without_phase2(self) -> None:
        async def run() -> None:
            p, llm = self._pipeline(fast_path=True)
            ctx = await _run_turn(p)
            assert ctx.fast_path is True
            assert ctx.invoke_llm is False
            assert llm.calls == 0
            assert p.tts.synthesized[0].startswith("Hi! How can I")

        asyncio.run(run())

    def test_fast_path_off_keeps_legacy_coupling(self) -> None:
        async def run() -> None:
            p, llm = self._pipeline(fast_path=False)
            ctx = await _run_turn(p)
            assert ctx.fast_path is False
            assert ctx.invoke_llm is True
            assert llm.calls == 1

        asyncio.run(run())


class TestE2ELatency:
    """Step 0.5: SPEECH_END -> first TTS chunk is tracked as e2e_reply."""

    def test_e2e_reply_recorded(self) -> None:
        async def run() -> None:
            s1 = LayaSystem1(enabled=False)
            p = StreamingPipeline(
                FixedSTT("tell me a story"),
                FakeLLMText(),
                FakeTTSStream(),
                FakeVAD(),
                system1=s1,
            )
            await _run_turn(p, ctx=ConversationContext())
            await asyncio.sleep(0.05)
            report = p.latency_report()
            assert "e2e_reply" in report
            assert report["e2e_reply"]["count"] >= 1
            assert report["e2e_reply"]["p50"] >= 0.0

        asyncio.run(run())


class TestToolPrefetch:
    """Step 1: a confident live-listening tool prefetch seeds the LLM prompt."""

    class FakeRegistry(ToolRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict] = []

        async def execute_call(self, call) -> dict:
            self.calls.append(call)
            return {
                "tool": call["name"],
                "result": "Prefetched lookup result",
            }

    def _pipeline(self) -> tuple[StreamingPipeline, RecordingLLM, TestToolPrefetch.FakeRegistry]:
        agent = FakeAgent(
            {
                "tool_needed": {"type": "noul", "noul": 0.95},
                "tool": {"type": "choice", "choice": "search_web", "confidence": 0.97},
            }
        )
        llm = RecordingLLM()
        s1 = LayaSystem1(enabled=True, agent=agent)
        p = StreamingPipeline(
            FixedSTT("who is nikola tesla"),
            llm,
            FakeTTSStream(),
            FakeVAD(),
            system1=s1,
            tool_prefetch=True,
        )
        registry = self.FakeRegistry()
        p._tool_registry = registry  # type: ignore[assignment]
        return p, llm, registry

    def test_prefetch_stashes_and_seeds_prompt(self) -> None:
        async def run() -> None:
            p, llm, registry = self._pipeline()
            ctx = p._ctx("sess")
            await p._partial_transcribe(b"\x00" * 1600, "sess", ctx)
            await asyncio.sleep(0.05)
            entry = p._s1_prefetch.get("sess")
            assert entry is not None
            assert entry["tool"] == "search_web"
            assert entry["result"] == "Prefetched lookup result"
            assert registry.calls == [{"name": "search_web", "args": ["who is nikola tesla"]}]

            await _run_turn(p, ctx=ctx)
            # Consumed and cleared by the turn.
            assert "sess" not in p._s1_prefetch
            assert any(
                "Prefetched lookup result" in m.get("content", "")
                for m in llm.messages
            )

        asyncio.run(run())

    def test_weak_tool_vote_skips_prefetch(self) -> None:
        async def run() -> None:
            s1 = LayaSystem1(
                enabled=True,
                agent=FakeAgent(
                    {
                        "tool_needed": {"type": "noul", "noul": 0.4},
                        "tool": {"type": "choice", "choice": "search_web", "confidence": 0.2},
                    }
                ),
            )
            p = StreamingPipeline(
                FixedSTT("who is nikola tesla"),
                FakeLLMText(),
                FakeTTSStream(),
                FakeVAD(),
                system1=s1,
                tool_prefetch=True,
            )
            registry = self.FakeRegistry()
            p._tool_registry = registry  # type: ignore[assignment]
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await asyncio.sleep(0.05)
            assert "sess" not in p._s1_prefetch
            assert registry.calls == []

        asyncio.run(run())

    def test_prefetch_turned_off_schedules_nothing(self) -> None:
        async def run() -> None:
            agent = FakeAgent(
                {
                    "tool_needed": {"type": "noul", "noul": 0.95},
                    "tool": {"type": "choice", "choice": "search_web", "confidence": 0.97},
                }
            )
            s1 = LayaSystem1(enabled=True, agent=agent)
            p = StreamingPipeline(
                FixedSTT("who is nikola tesla"),
                FakeLLMText(),
                FakeTTSStream(),
                FakeVAD(),
                system1=s1,
                tool_prefetch=False,
            )
            registry = self.FakeRegistry()
            p._tool_registry = registry  # type: ignore[assignment]
            await p._partial_transcribe(b"\x00" * 1600, "sess", p._ctx("sess"))
            await asyncio.sleep(0.05)
            assert "sess" not in p._s1_prefetch
            assert registry.calls == []

        asyncio.run(run())


class TestFp16Flag:
    """Step 0.3: fp16 hook is wired and fails open."""

    def test_flag_wires_through(self) -> None:
        assert LayaSystem1(enabled=True, agent=FakeAgent({}))._use_fp16 is True
        assert LayaSystem1(enabled=True, agent=FakeAgent({}), use_fp16=False)._use_fp16 is False