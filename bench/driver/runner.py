"""Scenario runner.

Executes a Scenario's ordered steps against a SpeechAgent adapter, capturing a
normalized timeline of events with monotonic timestamps, then produces a
ScenarioResult for the scoring layer.

Timing is always client-observed from the event stream (never internal), so it
stays comparable across agents and honest.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from bench.agent.base import (
    Event,
    EventType,
    SpeechAgent,
    make_silence,
    wav_to_pcm16,
)
from bench.driver.scenario import Scenario

FRAME_BYTES = 4096  # 128ms of 16k mono 16-bit
EVENT_WAIT = 60.0


@dataclass
class Timeline:
    events: list[Event] = field(default_factory=list)

    def add(self, ev: Event) -> None:
        self.events.append(ev)

    def seen(self, et: EventType) -> bool:
        return any(e.type == et for e in self.events)

    def first(self, et: EventType) -> Event | None:
        return next((e for e in self.events if e.type == et), None)

    def last(self, et: EventType) -> Event | None:
        return next((e for e in reversed(self.events) if e.type == et), None)

    def count(self, et: EventType) -> int:
        return sum(1 for e in self.events if e.type == et)

    def tts_audio_bytes(self) -> bytes:
        return b"".join((e.audio or b"") for e in self.events if e.type == EventType.TTS_CHUNK)

    def final_transcripts(self) -> list[str]:
        return [e.text or "" for e in self.events if e.type == EventType.TRANSCRIPT]

    def llm_output(self) -> str:
        return "".join((e.text or "") for e in self.events if e.type == EventType.LLM_TOKEN)

    def llm_done_text(self) -> str:
        e = self.last(EventType.LLM_DONE)
        return e.text or "" if e else ""

    def token_count(self) -> int:
        return sum(1 for e in self.events if e.type == EventType.LLM_TOKEN)


@dataclass
class StepResult:
    kind: str
    ok: bool = True
    detail: str = ""
    measured_ms: float | None = None
    data: dict = field(default_factory=dict)


@dataclass
class ScenarioResult:
    scenario: Scenario
    timeline: Timeline = field(default_factory=Timeline)
    step_results: list[StepResult] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.step_results) and not self.error

    def add_step(self, sr: StepResult) -> None:
        self.step_results.append(sr)

    def step(self, kind: str) -> StepResult | None:
        return next((s for s in self.step_results if s.kind == kind), None)

    def as_dict(self) -> dict:
        return {
            "id": self.scenario.id,
            "name": self.scenario.name,
            "category": self.scenario.category,
            "ok": self.ok,
            "error": self.error,
            "steps": [
                {
                    "kind": s.kind,
                    "ok": s.ok,
                    "detail": s.detail,
                    "measured_ms": s.measured_ms,
                    "data": s.data,
                }
                for s in self.step_results
            ],
        }


class ScenarioRunner:
    def __init__(self, agent: SpeechAgent, corpus: dict[str, list] | None = None):
        self.agent = agent
        self.corpus = corpus or {}

    async def _collect(self, tl: Timeline, timeout: float = EVENT_WAIT) -> None:
        ev = await self.agent.next_event(timeout=timeout)
        if ev is not None:
            tl.add(ev)

    async def _push_pcm(self, pcm: bytes) -> None:
        # Pace delivery at real-time (FRAME_BYTES = 128ms) so the agent's
        # silero VAD / endpointing sees naturally-spaced speech + trailing
        # silence. Bursting all frames at once skips real trailing silence and
        # prevents the turn from ever committing (no speech_end -> no final
        # transcript -> no LLM turn).
        for i in range(0, len(pcm), FRAME_BYTES):
            await self.agent.send_audio(pcm[i : i + FRAME_BYTES])
            await asyncio.sleep(FRAME_BYTES / self.agent.sample_rate)

    async def _push_silence(self, ms: float) -> None:
        if ms <= 0:
            return
        chunks = max(1, int(ms / 128.0))
        for _ in range(chunks):
            await self.agent.send_audio(make_silence(128))
            await asyncio.sleep(0)

    async def run(self, scenario: Scenario, timeout: float = 120.0) -> ScenarioResult:
        res = ScenarioResult(scenario=scenario)
        tl = res.timeline
        try:
            for idx, step in enumerate(scenario.steps):
                await self._do_step(step, tl, res)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
        return res

    async def _do_step(self, step, tl: Timeline, res: ScenarioResult) -> None:
        kind = step.kind
        t0 = time.monotonic()
        ok = True
        detail = "ok"
        data: dict = {}

        if kind == "speak":
            await self._utterance(step, tl)
        elif kind == "corpus":
            await self._corpus_utterance(step, tl)
        elif kind == "silence":
            await self._push_silence(step.ms if step.ms else step.silence_after_ms)
        elif kind == "wait_for":
            ev = await self._wait_for_event(tl, step.event, step.timeout_ms / 1000.0)
            data["saw_event"] = ev is not None
            ok = ev is not None
            detail = f"event {step.event} {'seen' if ev else 'timeout'}"
        elif kind == "expect":
            if step.absent:
                await self._push_silence(step.timeout_ms if step.timeout_ms else 3000)
                ev = await self._wait_for_event(tl, step.event, 1.0)
                ok = ev is None
                data["saw_event"] = ev is not None
                detail = f"expect ABSENT {step.event} -> {'clear' if ok else 'UNEXPECTED'}"
            else:
                ev = await self._wait_for_event(tl, step.event, step.timeout_ms / 1000.0)
                data["saw_event"] = ev is not None
                ok = ev is not None
                detail = f"expect {step.event} {'seen' if ev else 'MISSING'}"
        elif kind == "interrupt_during_playback":
            await self._interrupt_during_playback(step, tl)
        elif kind == "interrupt_during_llm":
            await self._interrupt_during_llm(step, tl)
        elif kind == "send_interrupt":
            await self.agent.send_interrupt()
            ev = await self._wait_for_event(tl, EventType.INTERRUPT, 3.0)
            data["saw_interrupt"] = ev is not None
            ok = ev is not None
        elif kind == "verify_text":
            await self._verify_text(step, tl)
        elif kind == "drain":
            await self._drain(tl, step.timeout_ms / 1000.0 if step.timeout_ms else 5.0)
        else:
            ok = False
            detail = f"unknown step kind: {kind}"

        res.add_step(
            StepResult(
                kind=kind,
                ok=ok,
                detail=detail,
                measured_ms=(time.monotonic() - t0) * 1000.0,
                data=data,
            )
        )

    async def _utterance(self, step, tl: Timeline) -> None:
        from bench.driver.audio_gen import (
            synthesize_pcm16_async,
            interleave_silence,
        )

        pcm = await synthesize_pcm16_async(step.text or "")
        if step.silence_after_ms > 0:
            pcm = interleave_silence(pcm, step.silence_after_ms)
        if step.text:
            data = {"spoken": step.text, "pcm_bytes": len(pcm)}
        await self._push_pcm(pcm)
        # Guarantee a minimum trailing-silence window after a spoken turn. The
        # Bolo pipeline only finalizes a turn once it observes silence >= its
        # adaptive endpoint threshold (~350-900ms nominal, but in practice up to
        # ~2.5s under real-time frame pacing). Scenario pauses of 600-900ms are
        # often too short, so the turn never commits and the subsequent
        # event/assert steps fail for harness reasons, not agent reasons.
        if step.pause_after_ms > 0:
            await self._push_silence(step.pause_after_ms)
        need_extra = self.MIN_TRAIL_SILENCE_MS - step.pause_after_ms
        if need_extra > 0:
            await self._push_silence(need_extra)

    MIN_TRAIL_SILENCE_MS = 2500.0

    async def _corpus_utterance(self, step, tl: Timeline) -> None:
        clips = self.corpus.get(step.corpus or "", [])
        if not clips:
            raise RuntimeError(f"corpus scope '{step.corpus}' empty")
        clip = clips[0]
        pcm = wav_to_pcm16(clip.audio, clip.sr, self.agent.sample_rate)
        await self._push_pcm(pcm)
        # Real clips often carry little/no trailing silence (the dummy
        # LibriSpeech clip has ~46ms), but the Bolo pipeline only finalizes a
        # turn once it sees silence >= endpoint (~350-900ms). Without a
        # trailing-silence window the turn never commits -> no transcript ->
        # empty hypothesis -> WER 100%. Always guarantee a silence tail so the
        # turn can finalize and be measured honestly.
        need = max(self.MIN_TRAIL_SILENCE_MS, step.silence_after_ms)
        await self._push_silence(need)

    async def _drain(self, tl: Timeline, timeout: float) -> None:
        # Wait for the agent's spoken reply to finish (TTS_DONE) or the full
        # window, whichever comes first. Do NOT bail on a brief event gap: after
        # the final user transcript the STT->LLM->TTS pipeline can take 1-5s to
        # produce the response, and bailing on 200ms of silence hid the agent's
        # reply from the judge transcript (depressing Error recovery /
        # Naturalness / Dialogue flow / etc.).
        deadline = time.monotonic() + timeout
        last_seen = time.monotonic()
        while time.monotonic() < deadline:
            ev = await self.agent.next_event(timeout=0.5)
            if ev is None:
                # Give the pipeline a generous silence allowance; only stop
                # early once the agent has clearly finished its spoken turn.
                if tl.seen(EventType.TTS_DONE) and time.monotonic() - last_seen >= 1.0:
                    break
                continue
            tl.add(ev)
            last_seen = time.monotonic()
            if ev.type == EventType.TTS_DONE and time.monotonic() - last_seen >= 0.0:
                # Keep draining a touch more so trailing events land.
                pass

    async def _wait_for_event(self, tl: Timeline, event: str, timeout: float):
        et = self._event_type(event)
        if et is None:
            return None
        # First check already-collected timeline.
        for e in tl.events:
            if e.type == et:
                return e
        deadline = time.monotonic() + (timeout if timeout > 0 else 1.0)
        while time.monotonic() < deadline:
            ev = await self.agent.next_event(timeout=max(0.05, deadline - time.monotonic()))
            if ev is None:
                continue
            tl.add(ev)
            if ev.type == et:
                return ev
        return None

    async def _interrupt_during_playback(self, step, tl: Timeline) -> None:
        # Wait until agent starts TTS output (playback active).
        await self._wait_for_event(tl, "tts_chunk", step.timeout_ms / 1000.0 if step.timeout_ms else 15.0)
        from bench.driver.audio_gen import synthesize_pcm16_async

        pcm = await synthesize_pcm16_async(step.text or "wait no")
        await self._push_pcm(pcm)

    async def _interrupt_during_llm(self, step, tl: Timeline) -> None:
        await self._wait_for_event(tl, "llm_token", 15.0)
        from bench.driver.audio_gen import synthesize_pcm16_async

        pcm = await synthesize_pcm16_async(step.text or "actually no")
        await self._push_pcm(pcm)

    async def _verify_text(self, step, tl: Timeline) -> None:
        text = tl.llm_done_text() if step.text_field == "llm_done" else tl.llm_output()
        missing = [c for c in (step.contains or []) if c.lower() not in text.lower()]
        unwanted = [c for c in (step.not_contains or []) if c.lower() in text.lower()]
        if missing:
            raise AssertionError(f"missing expected text: {missing}")
        if unwanted:
            raise AssertionError(f"contains unwanted text: {unwanted}")

    @staticmethod
    def _event_type(name: str) -> EventType | None:
        mapping = {
            "speech_start": EventType.SPEECH_START,
            "speech_end": EventType.SPEECH_END,
            "partial_transcript": EventType.PARTIAL_TRANSCRIPT,
            "transcript": EventType.TRANSCRIPT,
            "llm_token": EventType.LLM_TOKEN,
            "llm_done": EventType.LLM_DONE,
            "tts_chunk": EventType.TTS_CHUNK,
            "tts_done": EventType.TTS_DONE,
            "backchannel": EventType.BACKCHANNEL,
            "interrupt": EventType.INTERRUPT,
            "llm_token_or_done": EventType.LLM_TOKEN,
        }
        return mapping.get(name)


async def run_scenario(
    agent: SpeechAgent,
    scenario: Scenario,
    corpus: dict[str, list] | None = None,
    timeout: float = 120.0,
) -> ScenarioResult:
    runner = ScenarioRunner(agent, corpus)
    return await asyncio.wait_for(runner.run(scenario, timeout), timeout=timeout + 10)
