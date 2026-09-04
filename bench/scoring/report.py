"""Report assembly: category scores 1-10, weighted 100 benchmark, issues,
and a regression-test table mirroring the GPT evaluation format.

This is the final, human-and-JSON-readable output the harness produces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# Weights mirror the evaluator's suggested weighting (sum = 100).
WEIGHTS = {
    "Turn-taking": 10,
    "Barge-in": 15,
    "ASR/Segmentation": 10,
    "Context/State": 10,
    "Intent understanding": 8,
    "Dialogue flow": 8,
    "Grounding": 10,
    "Error recovery": 7,
    "TTS quality": 7,
    "Latency": 5,
    "Naturalness": 5,
    "Backchannel handling": 5,
}

CATEGORIES = list(WEIGHTS.keys())


def _cap(v: float, lo: float = 1.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, v))


def _score_map() -> dict[str, float]:
    return {c: None for c in CATEGORIES}


@dataclass
class RegressionRow:
    test_id: str
    scenario: str
    utterance: str
    expected: str
    observed: str
    passed: bool
    severity: str  # critical / major / minor
    category: str = ""


@dataclass
class Report:
    agent: str
    scores: dict[str, float | None] = field(default_factory=_score_map)
    benchmark: float | None = None
    issues: dict = field(default_factory=dict)  # critical/major/minor lists
    positives: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    regression: list[RegressionRow] = field(default_factory=list)
    scenario_details: dict = field(default_factory=dict)
    judge_active: bool = False
    metadata: dict = field(default_factory=dict)

    def compute_benchmark(self) -> float:
        total = 0.0
        for cat, w in WEIGHTS.items():
            s = self.scores.get(cat)
            if s is not None:
                total += s * w
        # Scores are 1-10, weights sum to 100 -> max = 1000. Scale to /100.
        self.benchmark = round(total / 10.0, 1)
        return self.benchmark

    def to_dict(self) -> dict:
        self.compute_benchmark()
        return {
            "agent": self.agent,
            "benchmark": self.benchmark,
            "scores": self.scores,
            "weighted": WEIGHTS,
            "issues": self.issues,
            "positives": self.positives,
            "repeated_patterns": self.patterns,
            "priorities": self.priorities,
            "judge_active": self.judge_active,
            "regression_tests": [
                {
                    "id": r.test_id,
                    "scenario": r.scenario,
                    "utterance": r.utterance,
                    "expected": r.expected,
                    "observed": r.observed,
                    "pass": r.passed,
                    "severity": r.severity,
                    "category": r.category,
                }
                for r in self.regression
            ],
            "scenario_details": self.scenario_details,
            "metadata": self.metadata,
        }

    def render_markdown(self) -> str:
        self.compute_benchmark()
        lines: list[str] = []
        lines.append(f"# Benchmark Report — {self.agent}")
        lines.append("")
        lines.append(f"**Overall benchmark score: {self.benchmark} / 100**")
        lines.append("")
        lines.append("## Category Scores (1-10)")
        lines.append("")
        lines.append("| Category | Score | Weight | Weighted |")
        lines.append("|---|---|---|---|")
        for cat in CATEGORIES:
            s = self.scores.get(cat)
            w = WEIGHTS[cat]
            ws = f"**{round(s * w, 1)} / {w}**" if s is not None else (f"{w * 5 / 10:.1f} / {w} (unscored)")
            sc = f"{s:.1f}" if s is not None else "n/a"
            lines.append(f"| {cat} | {sc} | {w} | {ws} |")
        lines.append("")

        for key, label in [
            ("critical", "Critical Failures"),
            ("major", "Major Issues"),
            ("minor", "Minor Issues"),
        ]:
            items = self.issues.get(key, [])
            lines.append(f"## {label}")
            lines.append("")
            if not items:
                lines.append("_None observed._")
            else:
                for it in items:
                    lines.append(f"- {it}")
            lines.append("")

        lines.append("## Positive Behaviors")
        lines.append("")
        if self.positives:
            for p in self.positives:
                lines.append(f"- {p}")
        else:
            lines.append("_None captured._")
        lines.append("")

        lines.append("## Repeated Failure Patterns")
        lines.append("")
        if self.patterns:
            for p in self.patterns:
                lines.append(f"- {p}")
        else:
            lines.append("_None._")
        lines.append("")

        lines.append("## Recommended Engineering Priorities")
        lines.append("")
        for i, p in enumerate(self.priorities, 1):
            lines.append(f"{i}. {p}")
        lines.append("")

        lines.append("## Regression Tests")
        lines.append("")
        if self.regression:
            lines.append("| ID | Scenario | Utterance | Expected | Observed | Pass | Severity |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in self.regression:
                utt = (r.utterance[:48] + "…") if len(r.utterance) > 50 else r.utterance
                lines.append(
                    f"| **{r.test_id}** | {r.scenario} | _{utt}_ | {r.expected} | {r.observed} | "
                    f"{'✅ PASS' if r.passed else '❌ FAIL'} | {r.severity} |"
                )
        else:
            lines.append("_No regression tests captured._")
        lines.append("")
        return "\n".join(lines)
