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

    p_cmp = sub.add_parser("compare", help="Compare two benchmark reports")
    p_cmp.add_argument("--a", required=True)
    p_cmp.add_argument("--b", required=True)

    args = parser.parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "compare":
        return cmd_compare(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
