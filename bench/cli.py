"""Benchmark CLI: run / compare / list.

Usage:
  python -m bench.cli list
  python -m bench.cli run --agent tasa-ws --base-url ws://localhost:8000/ws/audio \
      --scenarios all --out report.json --judge-llm-url http://localhost:11434/v1
  python -m bench.cli compare --a gpt_report.json --b tasa_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.driver.scenario import load_scenarios


def cmd_endpoint(args) -> int:
    """Run the benchmark, persist raw event timelines, and report
    endpointing/barge-in diagnostics computed from the captured events."""
    from bench.harness import run_benchmark
    from bench.scoring.endpoint import EndpointSample, analyze_results, render_metrics

    run_benchmark(
        agent_kind=args.agent,
        base_url=args.base_url,
        scenarios=args.scenarios or "all",
        corpus_limits={} if args.limit is None else {"all": args.limit},
        n_repeats=args.repeats,
        judge_url=args.judge_llm_url,
        judge_model=args.judge_model,
        timeout=args.timeout,
        out_path=args.out,
        events_out=args.events,
        label=args.label,
    )
    metrics = analyze_results(_load_results(args.events), _spoken_ground_truth(args.scenarios or "all"))
    print(render_metrics(metrics))
    return 0


def cmd_offline(args) -> int:
    """Compute the deterministic offline metrics (CER, EOT F1, prosody,
    emotion) from captured events and/or curated offline corpora.

    Uses dumped timelines when `--events` is given; otherwise evaluates the
    offline-curated corpora only (emotion classifier accuracy, prosody cue
    coverage). No live agent is required.
    """
    metrics = {}
    if args.events and Path(args.events).exists():
        results = _load_results(args.events)

        from bench.scoring.emotion import emotion_metrics
        from bench.scoring.endpoint import analyze_results, render_metrics
        from bench.scoring.prosody import prosody_metrics

        ep = analyze_results(results, _spoken_ground_truth(args.scenarios or "all"))
        metrics["endpointing_eot"] = ep
        metrics["prosody"] = prosody_metrics(results)
        metrics["emotion"] = emotion_metrics(results)
        print(render_metrics(ep))
    else:
        from bench.scoring.emotion import evaluate_emotion
        from bench.scoring.prosody import prosody_check

        metrics["emotion"] = evaluate_emotion()
        metrics["prosody"] = prosody_check("Let me think about that.", responding_to_question=False)
        print("Offline metric corpora: no event timeline provided (--events).")

    # CER: character error is offline by construction — measured straight on
    # the harness ASR corpus rather than reconstructed events.
    cer_value = _offline_cer(args)
    if cer_value is not None:
        metrics["cer"] = cer_value
        print(f"\nCharacter Error Rate (CER): {cer_value['cer'] * 100:.2f}% "
              f"over {cer_value['items']} item(s)")
    else:
        print("\nCharacter Error Rate (CER): n/a (no ASR corpus reference vs hypothesis)")

    for label, m in metrics.items():
        if isinstance(m, dict):
            print(f"\n[{label}]")
            for k, v in m.items():
                if isinstance(v, dict):
                    print(f"  {k}: {v}")
                else:
                    print(f"  {k}: {v}")
    return 0


def _offline_cer(args) -> dict | None:
    """CER from an ASR corpus + captured hypotheses when both are available."""
    from bench.scoring.wer import cer_stats

    if not (args.events and Path(args.events).exists()):
        return None
    results = _load_results(args.events)
    refs: list[str] = []
    hyps: list[str] = []
    spoken = _spoken_ground_truth(args.scenarios or "all")
    for r in results:
        if r.scenario.id not in ("asr_wer", "robustness"):
            continue
        transcripts = r.timeline.final_transcripts()
        hyp = transcripts[-1] if transcripts else ""
        ref = spoken.get(r.scenario.id, [("", 0.0)])[0][0]
        if ref:
            refs.append(ref)
            hyps.append(hyp)
    if not refs:
        return None
    st = cer_stats(refs, hyps)
    return st or None


def _spoken_ground_truth(scenarios_spec: str) -> dict:
    """Extract the scripted user-utterance reference texts from scenario speak
    steps, keyed by scenario id, so Completion Capture can score against the
    intended text rather than a transcript-produced proxy."""
    from bench.driver.scenario import load_scenarios

    sc_map = load_scenarios(scenarios_spec)
    out: dict = {}
    offset = 0.0
    for sc in sc_map.values():
        refs = []
        for step in sc.steps:
            if step.kind in ("speak", "corpus") and step.text:
                refs.append((step.text, offset))
                offset += 0.1
        out[sc.id] = refs
    return out


def _load_results(events_path: str) -> "list":
    """Reconstruct lightweight ScenarioResult-like objects from a dumped event
    timeline so the deterministic endpoint metrics can be recomputed offline.
    Real event timestamps (t) are preserved as mtime for latency analysis."""
    import json
    from pathlib import Path

    from bench.agent.base import Event, EventType
    from bench.driver.runner import ScenarioResult, Timeline
    from bench.driver.scenario import load_scenarios

    scenarios = load_scenarios()
    data = json.loads(Path(events_path).read_text())
    out = []
    for item in data:
        tl = Timeline()
        for e in item["events"]:
            try:
                et = EventType[e["type"]]
            except KeyError:
                continue
            tl.add(Event.make(et, text=e.get("text"), mtime=float(e.get("t", 0.0))))
        sc = scenarios.get(item["scenario"])
        if sc is None:
            from bench.driver.scenario import Scenario
            sc = Scenario(id=item["scenario"], name=item["name"], category="")
        res = ScenarioResult(scenario=sc)
        res.timeline = tl
        res.error = item.get("error") or ""
        out.append(res)
    return out


def cmd_list(_args) -> int:
    scs = load_scenarios("all")
    for sc_id, sc in sorted(scs.items()):
        print(f"{sc_id:<16} {sc.name}  [{sc.category}]")
    return 0


def cmd_run(args) -> int:
    from bench.harness import run_benchmark

    rep = run_benchmark(
        agent_kind=args.agent,
        base_url=args.base_url,
        scenarios=args.scenarios or "all",
        corpus_limits={} if args.limit is None else {"all": args.limit},
        n_repeats=args.repeats,
        judge_url=args.judge_llm_url,
        judge_model=args.judge_model,
        timeout=args.timeout,
        out_path=args.out,
        label=args.label,
    )
    print(rep.render_markdown())
    return 0


def cmd_compare(args) -> int:
    def load(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    a = load(args.a)
    b = load(args.b)
    print(f"Comparison: {a.get('agent','A')} vs {b.get('agent','B')}\n")
    print(f"{'Category':<24}{'A':>6}{'B':>6}{'Δ':>8}")
    print("-" * 46)
    scores_a = a.get("scores", {})
    scores_b = b.get("scores", {})
    for cat in [
        "Turn-taking", "Barge-in", "ASR/Segmentation", "Context/State",
        "Intent understanding", "Dialogue flow", "Grounding", "Error recovery",
        "TTS quality", "Latency", "Naturalness", "Backchannel handling",
    ]:
        sa = scores_a.get(cat)
        sb = scores_b.get(cat)
        sa_s = f"{sa:.1f}" if sa is not None else "n/a"
        sb_s = f"{sb:.1f}" if sb is not None else "n/a"
        d = ""
        if sa is not None and sb is not None:
            d = f"{sb - sa:+.1f}"
        print(f"{cat:<24}{sa_s:>6}{sb_s:>6}{d:>8}")
    print("-" * 46)
    ba = a.get("benchmark")
    bb = b.get("benchmark")
    print(f"Benchmark (100): {ba} vs {bb}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tasa-bench", description="Speech-to-speech benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available scenarios")

    p_run = sub.add_parser("run", help="Run the benchmark")
    p_run.add_argument("--agent", default="tasa-ws", help="agent adapter kind: tasa-ws (live) or mock (offline smoke)")
    p_run.add_argument("--base-url", default="ws://localhost:8000/ws/audio")
    p_run.add_argument("--scenarios", default="all", help="comma list or 'all'")
    p_run.add_argument("--repeats", type=int, default=1, help="repeat content scenarios")
    p_run.add_argument("--limit", type=int, default=10, help="max clips per corpus scope")
    p_run.add_argument("--judge-llm-url", default=None, help="optional judge LLM base URL")
    p_run.add_argument("--judge-model", default="qwen2.5:3b")
    p_run.add_argument("--timeout", type=float, default=120.0)
    p_run.add_argument("--out", default=None, help="write report JSON+markdown")
    p_run.add_argument("--label", default="TASA")

    p_end = sub.add_parser("endpoint", help="Run benchmark and report endpointing/barge-in diagnostics")
    p_end.add_argument("--agent", default="tasa-ws")
    p_end.add_argument("--base-url", default="ws://localhost:8000/ws/audio")
    p_end.add_argument("--scenarios", default="all")
    p_end.add_argument("--repeats", type=int, default=1)
    p_end.add_argument("--limit", type=int, default=10)
    p_end.add_argument("--judge-llm-url", default=None)
    p_end.add_argument("--judge-model", default="qwen2.5:3b")
    p_end.add_argument("--timeout", type=float, default=120.0)
    p_end.add_argument("--out", default=None, help="write report JSON+markdown")
    p_end.add_argument("--events", default="bench/events/latest.json", help="write raw event timelines JSON")
    p_end.add_argument("--label", default="TASA")

    p_off = sub.add_parser("offline", help="Run offline deterministic metrics (CER, EOT F1, prosody, emotion)")
    p_off.add_argument("--scenarios", default="all")
    p_off.add_argument("--events", default=None, help="use a captured event timeline JSON for CER/EOT/prosody/emotion")
    p_off.add_argument("--limit", type=int, default=10)

    p_cmp = sub.add_parser("compare", help="Compare two benchmark reports")
    p_cmp.add_argument("--a", required=True)
    p_cmp.add_argument("--b", required=True)

    args = parser.parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "endpoint":
        return cmd_endpoint(args)
    if args.command == "offline":
        return cmd_offline(args)
    if args.command == "compare":
        return cmd_compare(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
