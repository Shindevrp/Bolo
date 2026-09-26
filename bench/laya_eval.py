"""Laya System-1 eval runner (`python -m bench.cli laya`).

Runs Laya's typed-question schemas (cadence-2 turn questions and cadence-1
partial questions, exactly as production asks them) over the golden-label
corpus in ``bench/gold/laya_gold.json`` and reports per-question
accuracy / legacy agreement / calibration.

Two modes:

  - default (fake / deterministic): an ideal agent answers each case with its
    exact gold labels, so a perfect report (accuracy ~1.0) proves the harness,
    gating, and scoring are correct -- no model required, CI-safe.
  - ``--real``: load ``convaiinnovations/laya`` and score its actual answers;
    this is the report that gates Phase-2/3 rollout decisions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from bench.scoring.laya import gated, legacy_prediction, summarize
from modules.laya.questions import CADENCE1_QUESTIONS, TURN_QUESTIONS
from providers.laya.client import LayaSystem1

GOLD_PATH = Path("bench/gold/laya_gold.json")
DEFAULT_OUT = "report.laya.json"


def load_cases(path: str | Path, limit: int = 0) -> list[dict]:
    data = json.loads(Path(path).read_text())
    cases = list(data["cases"])
    if limit:
        cases = cases[:limit]
    for c in cases:
        c.setdefault("prev_intent", "")
        c.setdefault("prev_topic", "")
        c.setdefault("turn_count", 0)
        c.setdefault("partial", True)
    return cases


def cadence2_labels(rec: dict) -> dict:
    """Production-identical cadence-2 payload shape."""
    return {
        "transcript": rec["text"],
        "prev_intent": rec["prev_intent"],
        "turn_count": rec["turn_count"],
        "topic": rec["prev_topic"],
        "last_transcript": rec.get("last_transcript", ""),
    }


def cadence1_labels(rec: dict) -> dict:
    return {"transcript": rec["text"], "turn_count": rec["turn_count"]}


class _IdealAgent:
    """Sync stand-in that answers each case with its exact gold labels."""

    def __init__(self, by_text: dict[str, dict]) -> None:
        self._by_text = by_text

    def predict(self, state: dict, questions: dict) -> dict:
        labels = self._by_text.get(state.get("transcript", ""), {})
        answers: dict = {}
        for qid, qdef in questions.items():
            if qid not in labels:
                continue
            gold = labels[qid]
            if qdef["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.97 if gold else 0.06}
            elif qdef["type"] == "score":
                answers[qid] = {"type": "score", "score": 9.0 if bool(gold) else 1.0}
            else:
                answers[qid] = {"type": "choice", "choice": gold, "confidence": 0.98}
        return {"answers": answers}


class _BoundSystem1:
    """Minimal async predict() wrapper for the deterministic (fake) mode so it
    never depends on the optional ``laya`` package."""

    def __init__(self, agent) -> None:
        self._agent = agent

    async def predict(self, state: dict, questions: dict) -> dict:
        return (await asyncio.to_thread(self._agent.predict, state, questions)).get(
            "answers", {}
        )

    def latency_report_ms(self) -> dict:
        return {}


async def _run(s1, cases: list[dict]) -> list[dict]:
    out: list[dict] = []
    for rec in cases:
        c2 = await s1.predict(cadence2_labels(rec), TURN_QUESTIONS)
        c1 = await s1.predict(cadence1_labels(rec), CADENCE1_QUESTIONS)
        pred = {**c2, **c1}
        gold = rec["labels"]
        legacy = {
            q: legacy_prediction(rec["text"], q, rec.get("prev_intent", ""))
            for q in gold
        }
        out.append({"id": rec["id"], "text": rec["text"], "gold": gold,
                    "pred": pred, "legacy": legacy})
    return out


async def _run_async(s1, cases: list[dict], source: str) -> dict:
    cases_and_preds = await _run(s1, cases)
    report = summarize(cases_and_preds)
    latency = s1.latency_report_ms() if hasattr(s1, "latency_report_ms") else {}
    decisions = []
    for c in cases_and_preds:
        row: dict = {}
        for qid, expected in sorted(c["gold"].items()):
            entry = c["pred"].get(qid)
            row[qid] = {"expected": expected, "got": gated(qid, entry)}
        decisions.append({"id": c["id"], "text": c["text"], "decisions": row})
    report.update({"source": source, "caseload": len(cases_and_preds),
                   "latency_ms": latency, "cases": decisions})
    return report


def _render(report: dict) -> str:
    q = report["questions"]
    header = (f"{'question':<20}{'n':>4}{'acc':>8}{'raw':>8}"
              f"{'agree':>9}{'skip':>6}")
    rows = [header, "-" * len(header)]
    for qid in sorted(q):
        b = q[qid]
        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.3f}"
        rows.append(
            f"{qid:<20}{b['n']:>4}{fmt(b['accuracy']):>8}{fmt(b.get('raw_accuracy')):>8}"
            f"{fmt(b.get('agreement')):>9}{b['skipped']:>6}"
        )
    crit = report.get("critical", {})
    if crit:
        rows.append("")
        rows.append("critical: " + ", ".join(
            f"{q}={a}" for q, a in sorted(crit.items())
        ))
    macro = report.get("macro_accuracy")
    if macro is not None:
        rows.append(f"macro accuracy: {macro}")
    return "\n".join(rows)


def run_laya_cli(args: argparse.Namespace) -> int:
    cases = load_cases(args.gold, args.limit)
    if len(cases) != len({c["id"] for c in cases}):
        print("error: duplicate case ids in gold dataset")
        return 2

    if args.real:
        s1 = LayaSystem1(
            enabled=True, model=args.model, device=args.device, allow_cpu=True,
            # Scoring wants every answer, however slow the device.
            shadow_timeout=120.0,
        )
    else:
        by_text = {c["text"]: c["labels"] for c in cases}
        s1 = _BoundSystem1(_IdealAgent(by_text))

    report = asyncio.run(_run_async(s1, cases, source="fake" if not args.real else "real"))
    report["model"] = args.model if args.real else "fake(gold-ideal)"

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(_render(report))
    if report["source"] == "real":
        print(f"\nreal model report written to {args.out}")
    else:
        print(f"\nfake (gold-ideal) harness report written to {args.out}")
    return 0