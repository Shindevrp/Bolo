"""Word / Character Error Rate computation.

Deterministic, dependency-free edit distances between reference text and a
hypothesis (e.g. the agent's STT transcript vs ground truth).

  WER (Word Error Rate)         word-alignment Levenshtein distance
  CER (Character Error Rate)    character-level Levenshtein distance

CER is the traditional ASR companion to WER: it stays finite even for short /
heavily mixed-alphabet utterances where a single word error would otherwise
dominate the WER denominator. Both are computed from the same normalized text.
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


def _char_dist(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Character-level Levenshtein. Returns (dist, ref_chars, hyp_chars)."""
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return m, 0, m
    if m == 0:
        return n, n, 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        r = reference[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r == hypothesis[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m], n, m


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate in [0, inf]. 0.0 = perfect. Normalizes like WER."""
    r = "".join(normalize(reference))
    h = "".join(normalize(hypothesis))
    if not r:
        return 0.0 if not h else float(len(h))
    dist, n, _ = _char_dist(r, h)
    return 0.0 if n == 0 else min(dist / n, 10.0)


def cer_stats(references: list[str], hypotheses: list[str]) -> dict:
    """Aggregate CER + per-item values over paired lists."""
    if not references:
        return {"cer": None, "items": 0, "errors": 0, "chars": 0}
    err = 0
    chars = 0
    per = []
    for ref, hyp in zip(references, hypotheses):
        r = "".join(normalize(ref))
        h = "".join(normalize(hyp))
        dist, n, _ = _char_dist(r, h)
        err += dist
        chars += n
        per.append({"reference": ref, "hypothesis": hyp, "cer": cer(ref, hyp)})
    avg = (err / chars) if chars else 0.0
    return {
        "cer": round(avg, 4),
        "items": len(references),
        "errors": err,
        "chars": chars,
        "per_item": per,
    }
