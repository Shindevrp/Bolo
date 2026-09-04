"""Word Error Rate (WER) computation.

Deterministic, dependency-free word-alignment edit distance between reference
text and a hypothesis (e.g. the agent's STT transcript vs ground truth).
"""
from __future__ import annotations

import re


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop numbers-only tokens that give WER
    spurious wins, and return the word list."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    words = text.split()
    return [w for w in words if w]


def _edit_dist(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein distance. Returns (dist, ref_len, hyp_len)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, m
    if m == 0:
        return n, n, 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        r = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m], n, m


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate in [0, inf]. 0.0 = perfect. Handles empty ref specially."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    dist, n, _ = _edit_dist(ref, hyp)
    exact = 1.0 if n == 0 else 0.0
    return 0.0 if n == 0 else min(dist / n, 10.0)


def wer_stats(references: list[str], hypotheses: list[str]) -> dict:
    """Aggregate WER + per-item values over paired lists."""
    if not references:
        return {"wer": None, "items": 0, "errors": 0, "words": 0}
    err = 0
    words = 0
    per = []
    for ref, hyp in zip(references, hypotheses):
        rw = normalize(ref)
        hw = normalize(hyp)
        dist, n, _ = _edit_dist(rw, hw)
        err += dist
        words += n
        per.append({"reference": ref, "hypothesis": hyp, "wer": wer(ref, hyp)})
    avg = (err / words) if words else 0.0
    return {
        "wer": round(avg, 4),
        "items": len(references),
        "errors": err,
        "words": words,
        "per_item": per,
    }
