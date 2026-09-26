"""Export the disk-backed shadow log into Laya fine-tuning samples.

The Step 2 -> Step 3 bridge: `tasa-bench laya-export` reads the JSONL shadow
log written by the server (TASA_LAYA_SHADOW_LOG), keeps the self-labelled
rows -- where Laya agreed with the legacy classifier at sufficient
confidence -- and writes them as question/answer pairs ready for Step-3
retraining. The raw log is never modified or truncated.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from providers.laya.store import DEFAULT_SHADOW_LOG, read_shadow_log, training_samples


def run_laya_export(args: argparse.Namespace) -> int:
    source = args.input or DEFAULT_SHADOW_LOG
    entries = read_shadow_log(source)
    samples = training_samples(
        entries, min_conf=args.min_conf, question=args.question
    )

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            encoding="utf-8",
        )

    print(f"shadow log:   {source}")
    print(f"entries:      {len(entries)}")
    print(f"samples:      {len(samples)} (min-conf {args.min_conf}"
          f"{', question=' + args.question if args.question else ''})")
    if samples:
        by_q = Counter(s["question"] for s in samples)
        print("by question:  " + ", ".join(f"{q}={n}" for q, n in sorted(by_q.items())))
    if args.out and samples:
        print(f"wrote:        {args.out}")
    return 0 if samples else 1