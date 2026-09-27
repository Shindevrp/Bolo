"""Offline emotion classification evaluation.

Bolo's `EmotionClassifier` maps user transcripts to a sentiment bucket
(`positive` | `negative` | `neutral`) using a transformer model with a
deterministic lexicon fallback (`modules.tts.prosody.classify_sentiment`). This
module scores that decision contract offline:

  - `emotion_cases()`          curated (utterance, expected-bucket) pairs drawn
                               from the lexicon's own vocabulary so the eval is
                               a faithful, deterministic check of the fallback
  - `confusion()`              per-class counts over (predicted, expected) pairs
  - `evaluate_emotion()`       accuracy / recall / macro-F1 over those pairs
  - `emotion_metrics(results)` consistency between on-wire `prosody` emotion
                               labels and the lexicon prediction for the same
                               user turn (live + mock runs)

Everything is computed from observable text; we never claim transformer-model
accuracy that an offline harness cannot measure.
"""
from __future__ import annotations

from bench.scoring.endpoint import MetricResult


def emotion_cases() -> list[tuple[str, str]]:
    """Curated labelled utterances for the sentiment lexicon.

    Every expected bucket is reachable by `classify_sentiment` using only its
    positive/negative vocabulary, so a perfect classifier scores 1.0 and any
    regression is immediately visible.
    """
    return [
        ("that is great news", "positive"),
        ("that is perfect, thank you", "positive"),
        ("thanks so much", "positive"),
        ("awesome, I love it", "positive"),
        ("sure, sounds good", "positive"),
        ("that is bad news", "negative"),
        ("this is broken again", "negative"),
        ("I am so frustrated", "negative"),
        ("no, stop", "negative"),
        ("that is wrong", "negative"),
        ("let me think about it", "neutral"),
        ("what time is it", "neutral"),
        ("give me a moment", "neutral"),
        ("the appointment is at three", "neutral"),
        ("tell me more", "neutral"),
    ]


def classify_default(text: str) -> str:
    """Deterministic lexicon bucket — the offline-verifiable classifier."""
    from modules.tts.prosody import classify_sentiment

    return classify_sentiment(text)


def confusion(predictions: list[str], expected: list[str]) -> dict[str, dict[str, int]]:
    """Per-bucket confusion counts: {expected: {predicted: count}}."""
    out: dict[str, dict[str, int]] = {}
    for p, e in zip(predictions, expected):
        row = out.setdefault(e, {})
        row[p] = row.get(p, 0) + 1
        row.setdefault("_total", 0)
        row["_total"] += 1
    return out


def _f1_precision_recall(conf: dict[str, dict[str, int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    labels = ("positive", "negative", "neutral")
    for c in labels:
        tp = conf.get(c, {}).get(c, 0)
        fp = sum(row.get(c, 0) for e, row in conf.items() if e != c)
        fn = sum(v for e, row in conf.items() for k, v in row.items() if k != "_total" and e == c and k != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
        }
    return out


def evaluate_emotion(
    cases: list[tuple[str, str]] | None = None,
    classify=None,
) -> dict:
    """Accuracy + per-class recall/macro-F1 over labelled utterances."""
    cases = cases if cases is not None else emotion_cases()
    classify = classify or classify_default
    texts = [t for t, _ in cases]
    expected = [e for _, e in cases]
    predictions = [classify(t) for t in texts]
    conf = confusion(predictions, expected)
    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
    per_class = _f1_precision_recall(conf)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "accuracy": round(correct / len(cases), 4) if cases else None,
        "n": len(cases),
        "correct": correct,
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "confusion": conf,
    }


def emotion_metrics(results) -> dict[str, MetricResult]:
    """Consistency between on-wire `prosody` emotion labels and the lexicon.

    For each captured `prosody` event we re-predict the last user transcript
    with the deterministic lexicon; agreement is a measured, reproducible proxy
    for classifier-correct output when the transformer model is unavailable.
    When no on-wire labels exist we fall back to pure lexicon accuracy over the
    curated case set (still deterministic, still honest).
    """
    from bench.agent.base import EventType

    observed: list[tuple[str, str]] = []
    for r in results:
        last_user = ""
        for e in r.timeline.events:
            if e.type is EventType.TRANSCRIPT:
                last_user = (e.text or "").strip()
        for e in r.timeline.events:
            if e.type is EventType.PROSODY:
                _, _, emotion = (e.text or "||").partition("|")
                if emotion and last_user:
                    observed.append((emotion, classify_default(last_user)))
    if observed:
        correct = sum(1 for p, e in observed if p == e)
        value = correct / len(observed)
        numerator, denominator = correct, len(observed)
        note = "on-wire emotion labels vs lexicon prediction over user turns"
    else:
        ev = evaluate_emotion()
        value = ev["accuracy"]
        numerator, denominator = ev["correct"], ev["n"]
        note = f"lexicon classifier accuracy over {ev['n']} curated labelled utterances"
    return {
        "emotion_accuracy": MetricResult(
            value=value, numerator=numerator, denominator=denominator, note=note,
        )
    }