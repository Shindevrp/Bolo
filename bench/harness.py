"""Benchmark harness.

Orchestrates: corpus loading -> scenario selection -> agent connection ->
per-scenario run -> metric collection -> optional judge LLM -> report.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from bench.agent.base import SpeechAgent
from bench.driver.audio_gen import synthesize_pcm16
from bench.driver.corpus import load_corpus
from bench.driver.runner import ScenarioResult, run_scenario
from bench.driver.scenario import load_scenarios
from bench.scoring.aggregate import aggregate
from bench.scoring.judge import score_timeline
from bench.scoring.report import Report
from bench.scoring.wer import cer_stats, wer_stats


def _agent_factory(kind: str, base_url: str) -> SpeechAgent:
    if kind == "tasa-ws":
        from bench.agent.tasa_ws import TasaWSAdapter
        return TasaWSAdapter(url=base_url)
    if kind in ("mock", "fake"):
        from bench.agent.mock import MockAgent
        return MockAgent()
    raise ValueError(f"unknown agent kind: {kind} (expected 'tasa-ws' or 'mock')")


def _synthetic_clips(sc_map: dict, limits: dict[str, int]) -> dict[str, list]:
    """Placeholder clips (tone audio + scripted text) for offline/mock runs."""
    import numpy as np

    out: dict[str, list] = {}
    for sc in sc_map.values():
        scope = sc.corpus
        if not scope:
            continue
        if scope in out:
            continue
        n = limits.get(scope, limits.get("all", 10))
        sr = 16000
        clips = []
        for i in range(n):
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            audio = (0.3 * np.sin(2 * np.pi * (200 + i * 40) * t)).astype(np.float32)
            clips.append(
                _make_clip(
                    id=f"{scope}-{i}",
                    audio=audio,
                    sr=sr,
                    text=f"mock reference sentence number {i + 1}",
                )
            )
        out[scope] = clips
    return out


def _make_clip(id="c", audio=None, sr=16000, text=""):
    from bench.driver.corpus import Clip
    return Clip(id=id, audio=audio, sr=sr, text=text, speaker="")


def build_corpus(scenario_corpus_map: dict[str, str], limits: dict[str, int]) -> dict[str, list]:
    """Load corpus clips needed by the selected scenarios."""
    out: dict[str, list] = {}
    for scope, corpus_name in scenario_corpus_map.items():
        if scope in out:
            continue
        limit = limits.get(scope, limits.get("all", 10))
        out[scope] = load_corpus(corpus_name, limit=limit)
    return out


def dump_event_timelines(
    results: list[ScenarioResult],
    path: str | Path,
    refs_by_scenario: dict[str, list[str]] | None = None,
) -> Path:
    """Persist the raw per-scenario event timelines to a JSON file so the
    endpointing/barge-in diagnostics can be recomputed offline without
    re-running the whole benchmark. Ground-truth reference texts (used by the
    offline CER path) are persisted alongside when provided."""
    refs_by_scenario = refs_by_scenario or {}
    out = []
    for r in results:
        out.append(
            {
                "scenario": r.scenario.id,
                "name": r.scenario.name,
                "refs": refs_by_scenario.get(r.scenario.id, []),
                "ok": r.ok,
                "error": r.error,
                "events": [
                    {
                        "type": e.type.name,
                        "text": e.text,
                        "audio_bytes": len(e.audio) if e.audio is not None else None,
                        "t": round(e.mtime, 3),
                    }
                    for e in r.timeline.events
                ],
            }
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return p


def make_transcript(results: list[ScenarioResult]) -> str:
    """Build a human-readable timed transcript for the judge LLM."""
    lines: list[str] = []
    for r in results:
        lines.append(f"--- scenario: {r.scenario.id} ({r.scenario.name}) ---")
        lines.append(f"steps_exit_ok: {r.ok}  error: {r.error or 'none'}")
        for e in r.timeline.events:
            label = e.type.name
            if e.text:
                lines.append(f"  [{label}] {e.text}")
            elif e.audio:
                lines.append(f"  [{label}] <{len(e.audio)} bytes audio>")
        lines.append("")
    return "\n".join(lines)


async def _run_all(agent: SpeechAgent, corpus: dict[str, list], scenarios: dict, timeout: float) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for sc in scenarios.values():
        # Fresh session per scenario so STT transcripts, partials and dialogue
        # state from one scenario never leak into the next.
        await agent.reset_session()
        res = await run_scenario(agent, sc, corpus, timeout=timeout)
        results.append(res)
    return results


def run_benchmark(
    *,
    agent_kind: str = "tasa-ws",
    base_url: str = "ws://localhost:8000/ws/audio",
    scenarios: str | None = None,
    corpus_limits: dict[str, int] | None = None,
    n_repeats: int = 1,
    judge_url: str | None = None,
    judge_model: str = "qwen2.5:3b",
    timeout: float = 120.0,
    out_path: str | None = None,
    events_out: str | None = None,
    label: str = "TASA",
) -> Report:
    """High-level entry: connect, run, score, report."""
    return asyncio.run(
        _run_benchmark_async(
            agent_kind=agent_kind,
            base_url=base_url,
            scenarios=scenarios,
            corpus_limits=corpus_limits,
            n_repeats=n_repeats,
            judge_url=judge_url,
            judge_model=judge_model,
            timeout=timeout,
            out_path=out_path,
            events_out=events_out,
            label=label,
        )
    )


async def _run_benchmark_async(
    *,
    agent_kind: str,
    base_url: str,
    scenarios: str | None,
    corpus_limits: dict[str, int] | None,
    n_repeats: int,
    judge_url: str | None,
    judge_model: str,
    timeout: float,
    out_path: str | None,
    events_out: str | None,
    label: str,
) -> Report:
    sc_map = load_scenarios(scenarios)
    if not sc_map:
        raise SystemExit("no scenarios matched")

    # Determine corpus scopes needed
    needed: dict[str, str] = {}
    for sc in sc_map.values():
        scope = sc.corpus
        if scope:
            needed[scope] = scope  # scope name == corpus name shorthand handled by load_corpus
    corpus_loader: dict[str, str] = {}
    for scope in needed:
        corpus_loader[scope] = scope

    if agent_kind in ("mock", "fake"):
        corpus = _synthetic_clips(sc_map, corpus_limits or {})
        corpus_sources = {k: "synthetic (mock)" for k in corpus}
    else:
        corpus = {}
        corpus_sources: dict[str, str] = {}
        try:
            corpus = build_corpus(corpus_loader, corpus_limits or {})
            for scope in corpus:
                corpus_sources[scope] = corpus_loader[scope]
        except Exception as exc:  # offline fallback
            print(f"[harness] corpus load failed ({exc!r}); using synthetic clips")
            corpus = _synthetic_clips(sc_map, corpus_limits or {})
            corpus_sources = {k: "synthetic (load failed)" for k in corpus}

    aggregations: list[ScenarioResult] = []
    wer_items_refs: list[str] = []
    wer_items_hyp: list[str] = []

    agent = _agent_factory(agent_kind, base_url)
    try:
        for repeat in range(n_repeats):
            await agent.reset_session()
            results = await _run_all(agent, corpus, sc_map, timeout)
            for res in results:
                aggregations.append(res)
                if res.scenario.id in ("asr_wer", "robustness"):
                    hyp = ""
                    transcripts = res.timeline.final_transcripts()
                    if transcripts:
                        hyp = transcripts[-1]
                    scope = res.scenario.corpus
                    refs = corpus_ground_truth(corpus, scope)
                    if refs:
                        wer_items_refs.append(refs[0])
                        wer_items_hyp.append(hyp)
            if n_repeats > 1:
                await asyncio.sleep(0.5)
    finally:
        await agent.close()

    wer_result = wer_stats(wer_items_refs, wer_items_hyp) if wer_items_refs else {}
    cer_result = cer_stats(wer_items_refs, wer_items_hyp) if wer_items_refs else {}

    # Offline deterministic diagnostics, always computable from captured data.
    try:
        from bench.scoring.emotion import emotion_metrics
        from bench.scoring.prosody import prosody_metrics

        offline = {
            "cer": cer_result or None,
            "prosody": prosody_metrics(aggregations),
            "emotion": emotion_metrics(aggregations),
        }
    except Exception as exc:  # pragma: no cover - defensive; offline layer is optional
        offline = {"cer": cer_result or None, "warning": f"offline metrics unavailable: {exc!r}"}

    judge = None
    if judge_url:
        transcript = make_transcript(aggregations)
        # Keep transcript bounded
        transcript = transcript[:30000]
        judge = score_timeline(transcript, judge_url=judge_url, judge_model=judge_model)

    rep = aggregate(
        aggregations,
        wer=wer_result.get("wer"),
        cer=cer_result.get("cer"),
        judge=judge,
        agent=label,
    )
    rep.metadata = {
        "agent_kind": agent_kind,
        "base_url": base_url,
        "scenarios": list(sc_map.keys()),
        "n_repeats": n_repeats,
        "wer": wer_result if wer_result else None,
        "offline": offline,
        "judge_active": bool(judge and judge.get("scored")),
        "corpus_sources": corpus_sources,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            data = rep.to_dict()
            data["markdown"] = rep.render_markdown()
            json.dump(data, f, indent=2)

    if events_out:
        refs_by_scenario: dict[str, list[str]] = {}
        for res in aggregations:
            if res.scenario.id in ("asr_wer", "robustness"):
                refs = corpus_ground_truth(corpus, res.scenario.corpus)
                if refs:
                    refs_by_scenario[res.scenario.id] = refs
        dump_event_timelines(aggregations, events_out, refs_by_scenario=refs_by_scenario)

    return rep


def corpus_ground_truth(corpus: dict[str, list], scope: str) -> list[str]:
    """Ground-truth texts for the first clip of a corpus scope."""
    for key, clips in corpus.items():
        if scope in key or key in scope:
            return [c.text for c in clips if c.text]
    return []
