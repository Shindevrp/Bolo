"""Golden-corpus + Laya eval harness tests.

Covers: (1) the gold dataset is well-formed and every label is in the typed
question vocabulary, (2) the deterministic fake (gold-ideal) mode produces a
perfect report -- proving the runner + gating + scoring math -- and (3) the
production gate thresholds translate raw answers into the expected decisions.
"""
from __future__ import annotations

import asyncio

from bench.laya_eval import _BoundSystem1, _IdealAgent, _run_async, load_cases
from bench.scoring.laya import gated
from modules.laya.questions import CADENCE1_QUESTIONS, TURN_QUESTIONS

GOLD = "bench/gold/laya_gold.json"

ALL_QUESTIONS = set(TURN_QUESTIONS) | set(CADENCE1_QUESTIONS)


def _vocab(question: str) -> set | None:
    for defs in (TURN_QUESTIONS, CADENCE1_QUESTIONS):
        q = defs.get(question)
        if q is not None and q["type"] == "choice":
            return set(q.get("criteria") or {})
    return None


class TestGoldDataset:
    def test_cases_well_formed(self) -> None:
        cases = load_cases(GOLD)
        assert len(cases) > 40
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids)), "duplicate case ids"
        for c in cases:
            assert c["text"].strip(), f"{c['id']}: empty text"
            assert isinstance(c["labels"], dict), f"{c['id']}: missing labels"
            assert set(c["labels"]) <= ALL_QUESTIONS, f"{c['id']}: unknown question keys"
            for q, v in c["labels"].items():
                if q in ("is_question", "topic_changed", "needs_verify",
                         "tool_needed", "invoke_llm", "escalate", "urgent",
                         "high_stakes", "turn_complete", "completion_conf"):
                    assert isinstance(v, bool), f"{c['id']}.{q}: expected bool"
                else:
                    vocab = _vocab(q)
                    assert vocab is not None, f"{c['id']}.{q}: no vocab for {q}"
                    assert v in vocab, f"{c['id']}.{q}: {v!r} not in {sorted(vocab)}"

    def test_every_choice_label_has_a_case(self) -> None:
        seen: dict[str, set] = {}
        for c in load_cases(GOLD):
            for q, v in c["labels"].items():
                if _vocab(q):
                    seen.setdefault(q, set()).add(v)
        for q in ("intent", "query_complexity", "sentiment", "tool",
                  "barge_type"):
            assert seen[q] >= (set(TURN_QUESTIONS[q].get("criteria"))
                               if q in TURN_QUESTIONS
                               else set(CADENCE1_QUESTIONS[q].get("criteria"))), (
                f"{q}: gold corpus does not cover every label"
            )


class TestEvalHarness:
    def test_fake_gold_ideal_is_perfect(self) -> None:
        cases = load_cases(GOLD)
        by_text = {c["text"]: c["labels"] for c in cases}
        s1 = _BoundSystem1(_IdealAgent(by_text))
        report = asyncio.run(_run_async(s1, cases, source="fake"))
        assert report["source"] == "fake"
        assert report["caseload"] == len(cases)
        for qid, b in report["questions"].items():
            assert b["accuracy"] == 1.0, f"{qid}: {b}"
        assert report["macro_accuracy"] == 1.0
        assert report["critical"]

    def test_gating_thresholds(self) -> None:
        # action noul bar (P>=0.9), cadence-1 noul bar, score confidence
        # threshold, and barge confidence threshold all flow to decisions.
        assert gated("invoke_llm", {"type": "noul", "noul": 0.89}) is False
        assert gated("invoke_llm", {"type": "noul", "noul": 0.9}) is True
        assert gated("turn_complete", {"type": "noul", "noul": 0.89}) is False
        assert gated("turn_complete", {"type": "noul", "noul": 0.9}) is True
        assert gated("completion_conf", {"type": "score", "score": 6.99}) is False
        assert gated("completion_conf", {"type": "score", "score": 7.0}) is True
        assert gated("barge_type", {"type": "choice", "choice": "disagreement",
                                    "confidence": 0.79}) is None
        assert gated("barge_type", {"type": "choice", "choice": "disagreement",
                                    "confidence": 0.8}) == "disagreement"
        assert gated("sentiment", {"type": "choice", "choice": "positive",
                                   "confidence": 0.49}) is None
        assert gated("is_question", {"type": "noul", "noul": 0.5}) is True