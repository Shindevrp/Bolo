"""Verification harness for TASA turn-end / VAD / endpointing metrics.

Drives a :class:`StreamingPipeline` with scripted speech/silence frames (and
optionally real providers for the live GPT-vs-TASA rerun, Phase 4), observes the
emitted :class:`~core.pipeline.PipelineEvent` stream, and computes the endpoint /
barge-in quality metrics and their acceptance targets.

The harness is deliberately dry-run-able with fake providers so CI can exercise
the measurement path; real-provider/real-audio runs plug in via ``build_pipeline``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from core.pipeline import StreamingPipeline, PipelineEvent, FRAME_BYTES


class _ScriptedVAD:
    """VAD that reports speech for the first ``speech_frames`` calls then
    silence, mirroring one scripted utterance."""

    sample_rate = 16000

    def __init__(self, speech_frames: int) -> None:
        self.speech_frames = speech_frames
        self._calls = 0

    def is_speech(self, chunk: bytes) -> bool:
        self._calls += 1
        return self._calls <= self.speech_frames

    def reset(self) -> None:
        self._calls = 0


@dataclass
class Utterance:
    """One scripted user turn to push through the pipeline.

    ``speech_frames`` frames report speech followed by ``silence_frames`` frames
    of silence; the pair represents one utterance + its trailing pause.
    """
    speech_frames: int
    silence_frames: int
    label: str = "completed"  # completed | hesitation | backchannel
    expected_text: str = ""


@dataclass
class AcceptanceTargets:
    false_endpoint_max: float = 0.03
    missed_endpoint_max: float = 0.05
    completion_capture_min: float = 0.97
    fragmentation_max: float = 0.02
    endpoint_p50_max_ms: float = 500.0
    endpoint_p95_max_ms: float = 700.0


@dataclass
class UtteranceOutcome:
    index: int
    label: str
    speech_end_count: int
    false_early_end: bool = False
    missed_endpoint: bool = False
    endpoint_silence_ms: float | None = None


@dataclass
class EvaluationReport:
    targets: AcceptanceTargets
    outcomes: list[UtteranceOutcome] = field(default_factory=list)

    def compute(self) -> dict[str, float | str]:
        completed = [o for o in self.outcomes if o.label == "completed"]
        hesitations = [o for o in self.outcomes if o.label == "hesitation"]
        total = len(self.outcomes) or 1

        false_endpoint = sum(1 for o in hesitations if o.false_early_end) / (
            len(hesitations) or 1
        )
        missed_endpoint = sum(1 for o in completed if o.missed_endpoint) / (
            len(completed) or 1
        )
        completion_capture = sum(
            1 for o in completed if o.speech_end_count == 1 and not o.false_early_end
        ) / (len(completed) or 1)
        fragmentation = sum(1 for o in self.outcomes if o.speech_end_count > 1) / total

        sil = [
            o.endpoint_silence_ms
            for o in self.outcomes
            if o.endpoint_silence_ms is not None
        ]
        if sil:
            sil_sorted = sorted(sil)
            endpoint_p50 = statistics.median(sil)
            endpoint_p95 = sil_sorted[int(len(sil_sorted) * 0.95)]
        else:
            endpoint_p50 = endpoint_p95 = None

        return {
            "false_endpoint_rate": round(false_endpoint, 4),
            "missed_endpoint_rate": round(missed_endpoint, 4),
            "completion_capture_rate": round(completion_capture, 4),
            "fragmentation_rate": round(fragmentation, 4),
            "endpoint_silence_p50_ms": (
                round(endpoint_p50, 1) if endpoint_p50 is not None else None
            ),
            "endpoint_silence_p95_ms": (
                round(endpoint_p95, 1) if endpoint_p95 is not None else None
            ),
            "utterances": len(self.outcomes),
        }

    def passes(self) -> dict[str, bool]:
        m = self.compute()
        t = self.targets
        return {
            "false_endpoint": m["false_endpoint_rate"] <= t.false_endpoint_max,
            "missed_endpoint": m["missed_endpoint_rate"] <= t.missed_endpoint_max,
            "completion_capture": m["completion_capture_rate"] >= t.completion_capture_min,
            "fragmentation": m["fragmentation_rate"] <= t.fragmentation_max,
            "endpoint_p50": (
                m["endpoint_silence_p50_ms"] is None
                or m["endpoint_silence_p50_ms"] <= t.endpoint_p50_max_ms
            ),
            "endpoint_p95": (
                m["endpoint_silence_p95_ms"] is None
                or m["endpoint_silence_p95_ms"] <= t.endpoint_p95_max_ms
            ),
        }


# Frame duration for the fixed 128ms pipeline frame.
_FRAME_MS = 1000.0 * FRAME_BYTES / (16000 * 2)


def classify_outcome(
    index: int,
    label: str,
    speech_end_count: int,
    silence_ms_pushed: float,
    hesitation_threshold_ms: float = 450.0,
) -> UtteranceOutcome:
    """Decide pass/fail for one scripted utterance outcome.

    For ``hesitation`` utterances a premature end at or below the supplied
    silence threshold is a false endpoint. For ``completed`` utterances the turn
    must end (a missing end is a missed endpoint).
    """
    occ = UtteranceOutcome(
        index=index,
        label=label,
        speech_end_count=speech_end_count,
        endpoint_silence_ms=silence_ms_pushed if speech_end_count >= 1 else None,
    )
    if label == "hesitation":
        occ.false_early_end = speech_end_count >= 1 and (
            silence_ms_pushed <= hesitation_threshold_ms
        )
    elif label == "completed":
        occ.missed_endpoint = speech_end_count == 0
    return occ


async def run_scripted_utterance(
    pipeline: StreamingPipeline,
    sid: str,
    speech_frames: int,
    silence_frames: int,
    *,
    collect_ms: float = 0.3,
) -> tuple[int, list[PipelineEvent]]:
    """Push one scripted utterance through the pipeline and count SPEECH_END."""
    script = _ScriptedVAD(speech_frames)
    pipeline.vad = script
    pipeline._running = True
    loop_task = asyncio.create_task(pipeline._pipeline_loop())
    seen: list[PipelineEvent] = []

    async def collect() -> None:
        async for msg in pipeline.output_stream():
            if msg.session_id == sid:
                seen.append(msg.event)

    col_task = asyncio.create_task(collect())
    for _ in range(speech_frames + silence_frames):
        await pipeline.push_audio(b"\x01" * FRAME_BYTES, sid)
    await asyncio.sleep(collect_ms)
    pipeline._running = False
    loop_task.cancel()
    await asyncio.gather(loop_task, col_task, return_exceptions=True)
    end_count = seen.count(PipelineEvent.SPEECH_END)
    return end_count, seen


async def run_verification(
    utterances: list[Utterance],
    pipeline_factory: Callable[[], StreamingPipeline],
    targets: AcceptanceTargets | None = None,
    sid: str = "verify",
) -> EvaluationReport:
    """Run a scripted conversation and produce an evaluation report."""
    targets = targets or AcceptanceTargets()
    report = EvaluationReport(targets=targets)
    idx = 0
    for utt in utterances:
        p = pipeline_factory()
        end_count, _seen = await run_scripted_utterance(
            p, sid, utt.speech_frames, utt.silence_frames
        )
        silence_ms_pushed = utt.silence_frames * _FRAME_MS
        report.outcomes.append(
            classify_outcome(
                idx,
                utt.label,
                end_count,
                silence_ms_pushed,
                hesitation_threshold_ms=p.endpoint_normal_ms,
            )
        )
        idx += 1
    return report


def render_report(report: EvaluationReport) -> str:
    """Render an evaluation report as a compact markdown table."""
    metrics = report.compute()
    passes = report.passes()
    lines = ["| Metric | Value | Target | Pass |", "|---|---|---|---|"]
    mm = {
        "false_endpoint": ("false_endpoint_rate", "<=3%"),
        "missed_endpoint": ("missed_endpoint_rate", "<=5%"),
        "completion_capture": ("completion_capture_rate", ">=97%"),
        "fragmentation": ("fragmentation_rate", "<=2%"),
        "endpoint_p50": ("endpoint_silence_p50_ms", "<=500ms"),
        "endpoint_p95": ("endpoint_silence_p95_ms", "<=700ms"),
    }
    for key, (metric, target) in mm.items():
        value = metrics.get(metric)
        if value is None:
            value = "n/a"
        lines.append(
            f"| {key} | {value} | {target} | {'PASS' if passes[key] else 'FAIL'} |"
        )
    return "\n".join(lines)
