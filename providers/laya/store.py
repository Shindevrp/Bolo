"""Disk-backed shadow log for the Laya System-1 decision layer.

Step 2 of the approved latency/quality plan: every turn's shadow rows
(Laya-vs-legacy decisions) are appended to a JSONL file so the collection
survives restarts, and `tasa-bench laya-export` can turn the agreed rows
into fine-tuning samples for Step 3 (retraining the probe on self-labels).

The write path is deliberately shallow: one append + flush per turn under a
lock. Any I/O failure is logged (the first few times only) and never
retried -- shadow logging must never block or slow the audio loop.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("laya.store")

DEFAULT_SHADOW_LOG = "~/.tasa/shadow/rows.jsonl"


def _expand(path: str) -> str:
    return os.path.expanduser(path)


class ShadowLogStore:
    """Append-only JSONL sink. Thread-safe and fail-open.

    ``append`` never raises: a broken path, a full disk, or bad JSON just
    logs and is skipped, exactly as the audio path demands.
    """

    def __init__(self, path: str | os.PathLike | None) -> None:
        self._path = _expand(str(path)) if path else None
        self._lock = threading.Lock()
        self._writes = 0
        self._failures = 0

    @property
    def path(self) -> str | None:
        return self._path

    @property
    def enabled(self) -> bool:
        return self._path is not None

    @property
    def writes(self) -> int:
        return self._writes

    def append(self, entry: dict | None) -> bool:
        """Write one JSONL line. True when written (or skipped as a no-op)."""
        if not self.enabled or not entry:
            return True
        with self._lock:
            try:
                path = Path(self._path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(entry, ensure_ascii=False, default=str) + "\n"
                    )
                    handle.flush()
                self._writes += 1
                return True
            except Exception as e:  # noqa: BLE001 - fail-open by contract
                self._failures += 1
                if self._failures <= 3:
                    logger.error(
                        f"shadow log write failed path={self._path} error={e!r}"
                    )
                return False


def read_shadow_log(path: str | os.PathLike | None) -> list[dict]:
    """Read every entry from a shadow JSONL. Malformed lines are skipped."""
    p = Path(_expand(str(path))) if path else None
    entries: list[dict] = []
    if p is None or not p.exists():
        return entries
    with p.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    f"shadow log: skipping malformed line {line_no} of {p}"
                )
    return entries


def training_samples(
    entries: list[dict],
    *,
    min_conf: float = 0.5,
    question: str | None = None,
) -> list[dict]:
    """Flatten shadow entries into Laya fine-tuning samples (Step 3).

    Only self-labelled rows survive: outcome labels (what actually happened)
    always, and legacy-agreement labels when Laya's confidence clears
    ``min_conf``. Each sample carries the exact ``state`` dict that was
    fed to the forward pass, so it can be re-asked verbatim at training time.
    """
    out: list[dict] = []
    for entry in entries:
        state = entry.get("state") or {}
        for row in entry.get("rows", []):
            q = row.get("question")
            if q is None or (question and q != question):
                continue
            label = row.get("self_label")
            if label is None:
                continue
            conf = row.get("laya_conf")
            # Outcome labels are ground truth: Laya's own confidence (or
            # absence -- the model may not even have run) doesn't gate them.
            outcome = row.get("label_source") == "outcome"
            if not outcome and (
                not isinstance(conf, (int, float)) or conf < min_conf
            ):
                continue
            out.append({
                "question": q,
                "answer": label,
                "source": row.get("label_source") or "agree",
                "confidence": float(conf) if isinstance(conf, (int, float)) else None,
                "state": state,
                "transcript": entry.get("transcript", ""),
                "session_id": entry.get("session_id", ""),
                "ts": entry.get("ts"),
            })
    return out