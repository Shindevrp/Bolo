"""Offline scoring for the Laya System-1 decision-layer eval.

The eval (`bench.cli laya`) runs Laya's typed questions over a golden-label
corpus (`bench/scenarios/laya_gold.json`) and scores three things per question:

  - **accuracy**: predicted decision vs the curated gold label, where the
    predicted decision is produced by the exact production gates
    (``choice`` at conf >= 0.5, routing ``noul`` at P >= 0.9, cadence-1
    gating) so the numbers transfer directly to Phase-2/3 rollout.
  - **agreement**: predicted decision vs the deterministic legacy classifier
    for the same question (only where a legacy counterpart exists).
  - **calibration**: mean raw probability (``noul``) / score level per gold
    class, so an over-confident head shows up as a skewed mean rather than a
    binary miss.

The gate thresholds are imported from ``core.pipeline`` (_C1_*) to keep this
module a single source of truth with runtime enforcement.
"""
from __future__ import annotations

from core.pipeline import _C1_ACT_PROB, _C1_CONF_REQ, _C1_SCORE_REQ

#: Questions whose routing/action semantics use the strict P>=0.9 action bar.
ACTION_NOUL = {"tool_needed", "invoke_llm", "escalate", "urgent"}

#: Questions with a legacy deterministic counterpart for agreement rows.
LEGACY_SOURCES = {
    "intent", "is_question", "query_complexity", "needs_verify", "sentiment",
    "turn_complete", "barge_type",
}

#: Decision-critical questions that gate Phase-2/3 rollout go/no-go.
CRITICAL = [
    "intent", "is_question", "query_complexity", "needs_verify",
    "tool", "urgent", "invoke_llm", "escalate",
    "turn_complete", "completion_conf", "barge_type",
]


def gated(question: str, entry: dict | None):
    """Map a raw Laya answer entry to the production decision.

    Returns a comparable value: bool for ``noul``/``completion_conf``, the
    choice label for ``choice`` (``None`` when confidence is below threshold),
    or ``None`` for anything unknown/absent (fail-open).
    """
    if not isinstance(entry, dict):
        return None
    entry_type = entry.get("type")
    if question in ACTION_NOUL:
        return bool(entry.get("noul", 0.0) >= _C1_ACT_PROB)
    if question == "turn_complete":
        return bool(entry.get("noul", 0.0) >= _C1_ACT_PROB)
    if question == "completion_conf":
        return bool(entry.get("score", 0.0) >= _C1_SCORE_REQ)
    if entry_type == "noul":
        return bool(entry.get("noul", 0.0) >= 0.5)
    if entry_type == "choice" and question == "barge_type":
        conf = entry.get("confidence", 0.0)
        return entry.get("choice") if conf >= _C1_CONF_REQ else None
    if entry_type == "choice":
        conf = entry.get("confidence", 0.0)
        return entry.get("choice") if conf >= 0.5 else None
    if entry_type == "score":
        return entry.get("score")
    return None


def raw_value(question: str, entry: dict | None):
    """Un-thresholded value for calibration stats (noul P or score level)."""
    if not isinstance(entry, dict):
        return None
    if entry.get("type") == "noul":
        return entry.get("noul")
    if entry.get("type") == "score":
        return entry.get("score")
    return entry.get("choice")


def raw_decision(question: str, entry: dict | None):
    """Threshold-free argmax decision, to separate head quality from
    calibration: noul at P>=0.5, completion_conf at score>=7, choice labels
    taken as-is regardless of confidence."""
    if not isinstance(entry, dict):
        return None
    entry_type = entry.get("type")
    if question == "completion_conf":
        return bool(entry.get("score", 0.0) >= _C1_SCORE_REQ)
    if entry_type == "noul":
        noul = entry.get("noul", 0.0)
        return bool(noul >= 0.5 if question not in ACTION_NOUL else noul >= 0.9)
    if entry_type == "choice":
        label = entry.get("choice")
        return label if label is not None else None
    return None


def legacy_prediction(text: str, question: str, prev_intent: str = ""):
    """Deterministic legacy classifier value for the agreement row, or None
    when the question has no legacy counterpart."""
    if question == "intent":
        from modules.turn.intent import IntentClassifier

        return IntentClassifier().classify(text, prev_intent or "")
    if question == "is_question":
        return text.rstrip().endswith("?")
    if question == "query_complexity":
        from core.pipeline import StreamingPipeline

        return StreamingPipeline._classify_query_complexity(None, text)
    if question == "needs_verify":
        from modules.turn.entity import EntityGate

        return EntityGate.query_needs_verification(text)
    if question == "sentiment":
        from modules.tts.prosody import classify_sentiment

        return classify_sentiment(text)
    if question == "turn_complete":
        from modules.turn.classifier import TurnClassifier

        return not TurnClassifier().is_incomplete(text)
    if question == "barge_type":
        from modules.turn.backchannel_interrupt import BackchannelInterrupt

        return BackchannelInterrupt.classify(text)
    return None


def summarize(cases_and_preds: list[dict]) -> dict:
    """Aggregate per-case {id, gold, pred, legacy} records into a report.

    ``gold`` = {qid: expected}, ``pred`` = {qid: answers-entry},
    ``legacy`` = {qid: legacy value}. Only questions present in a case's gold
    contribute to that question's accuracy; agreements need a legacy value.
    """
    acc: dict[str, dict] = {}
    for case in cases_and_preds:
        gold = case.get("gold", {})
        pred = case.get("pred", {})
        legacy = case.get("legacy", {})
        for qid, expected in gold.items():
            entry = pred.get(qid)
            bucket = acc.setdefault(
                qid, {"n": 0, "correct": 0, "raw_correct": 0, "skipped": 0,
                      "agree": 0, "agree_n": 0, "cal": {}, "confusion": {}}
            )
            bucket["n"] += 1
            decision = gated(qid, entry)
            if decision is None and entry is not None:
                bucket["skipped"] += 1
            if decision == expected:
                bucket["correct"] += 1
            if raw_decision(qid, entry) == expected:
                bucket["raw_correct"] += 1
            leg = legacy.get(qid)
            if decision is not None and leg is not None:
                if decision == leg:
                    bucket["agree"] += 1
                bucket["agree_n"] += 1
            # calibration + confusion buckets for choice/score
            if qid in ("completion_conf",) or (entry or {}).get("type") == "noul":
                rv = raw_value(qid, entry)
                if rv is not None:
                    bucket["cal"].setdefault(str(expected), []).append(rv)
            if (entry or {}).get("type") == "choice" and decision is not None:
                row = bucket["confusion"].setdefault(str(expected), {})
                row[str(decision)] = row.get(str(decision), 0) + 1

    out: dict[str, dict] = {}
    for qid, b in acc.items():
        b["accuracy"] = round(b["correct"] / b["n"], 4) if b["n"] else None
        b["raw_accuracy"] = (
            round(b["raw_correct"] / b["n"], 4) if b["n"] else None
        )
        b["agreement"] = (
            round(b["agree"] / b["agree_n"], 4) if b["agree_n"] else None
        )
        if b["cal"]:
            b["calibration"] = {
                cls: round(sum(vals) / len(vals), 4) for cls, vals in b["cal"].items()
            }
        b.pop("agree_n")
        b.pop("agree")
        b.pop("cal")
        b.pop("raw_correct")
        if not b["confusion"]:
            b.pop("confusion")
        out[qid] = b

    scored = [q for q, b in out.items() if b.get("n", 0)]
    critical_acc = {
        q: out[q]["accuracy"] for q in CRITICAL if out.get(q, {}).get("n", 0)
    }
    return {
        "questions": out,
        "critical": critical_acc,
        "macro_accuracy": (
            round(
                sum(out[q]["accuracy"] for q in scored) / len(scored), 4,
            )
            if scored
            else None
        ),
    }