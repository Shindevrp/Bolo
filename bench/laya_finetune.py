"""Step 3: fine-tuning scaffolding + quality gate for the Laya probe.

`tasa-bench laya-finetune` takes the self-labelled shadow samples that
`tasa-bench laya-export` (Step 2) produces and:

  1. builds a canonical train / held-out dataset (deduplicated by
     ``(question, state)``, keeping the highest-confidence label),
  2. fine-tunes the probe when the installed ``laya`` package exposes a
     supervised fine-tune API (best-effort invocation -- any signature
     mismatch just stages the data with a clear message),
  3. reruns the golden-label benchmark as a **quality gate** so a checkpoint
     can never silently regress below the baseline it replaces.

Deliberately staged so CI and environments without the optional ``laya``
dependency still work: dataset construction and the gate run with the
deterministic gold-ideal agent (accuracy 1.0), and "unsupported" never fails
the process -- only a real gate failure (``--real`` below ``--gate``) exits 1
to block a rollout.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

from providers.laya.store import read_shadow_log

# Macro accuracy of the uncalibrated base checkpoint on the golden corpus
# (measured via `tasa-bench laya --real`, report.laya.json). The default gate
# is "do not regress below the checkpoint this step replaces"; deployments
# can raise it once a fine-tuned checkpoint proves out.
DEFAULT_GATE = 0.4811
DEFAULT_DATA = "training.laya.jsonl"


def _capability() -> tuple[callable | None, str]:
    """Return (finetune-callable-or-None, capability note)."""
    try:
        import laya
    except Exception:  # noqa: BLE001 - optional dependency import safety
        return None, "laya package not installed"
    for name in ("finetune", "supervised_finetune"):
        fn = getattr(laya, name, None)
        if callable(fn):
            return fn, f"laya.{name}"
    return None, "laya exposes no supervised fine-tune API"


def load_samples(path: str | Path) -> list[dict]:
    """Read the self-labelled sample JSONL written by `tasa-bench laya-export`."""
    return read_shadow_log(str(path))


def _canonical_key(sample: dict) -> str:
    return json.dumps(
        {"question": sample.get("question"), "state": sample.get("state") or {}},
        sort_keys=True,
        default=str,
    )


def _bucket(key: str, buckets: int) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % buckets


def _rank(sample: dict) -> tuple[int, float]:
    """Outcome labels (ground truth) beat agreement labels; then confidence."""
    return (
        1 if sample.get("source") == "outcome" else 0,
        sample.get("confidence") or 0.0,
    )


def build_datasets(
    samples: list[dict], *, heldout_frac: float = 0.15
) -> dict:
    """Deduplicate by ``(question, state)`` (keep an outcome label over an
    agreement label, then the highest confidence) and
    partition deterministically into train / held-out buckets. With the
    default fraction the partition is 1-of-7 rows, so it is reproducible
    across machines and runs without a random seed."""
    kept: dict[str, dict] = {}
    dropped = 0
    for s in samples:
        key = _canonical_key(s)
        current = kept.get(key)
        if current is None:
            kept[key] = s
        elif _rank(s) > _rank(current):
            kept[key] = s
            dropped += 1
        else:
            dropped += 1
    ordered = sorted(kept.values(), key=lambda s: _canonical_key(s))
    buckets = max(2, round(1.0 / heldout_frac))
    train = [s for s in ordered if _bucket(_canonical_key(s), buckets) != 0]
    heldout = [s for s in ordered if _bucket(_canonical_key(s), buckets) == 0]
    return {
        "train": train,
        "heldout": heldout,
        "duplicates_dropped": dropped,
        "samples_kept": len(ordered),
        "buckets": buckets,
    }


def _write_dataset(ds: dict, out_dir: Path) -> None:
    def line(sample: dict) -> str:
        return json.dumps(
            {
                "question": sample["question"],
                "state": sample.get("state") or {},
                "answer": sample["answer"],
            },
            ensure_ascii=False,
        )

    (out_dir / "train.jsonl").write_text(
        "\n".join(line(s) for s in ds["train"]) + ("\n" if ds["train"] else ""),
        encoding="utf-8",
    )
    (out_dir / "heldout.jsonl").write_text(
        "\n".join(line(s) for s in ds["heldout"]) + ("\n" if ds["heldout"] else ""),
        encoding="utf-8",
    )
    by_q = Counter(s["question"] for s in ds["train"])
    (out_dir / "dataset.json").write_text(
        json.dumps(
            {
                "samples_kept": ds["samples_kept"],
                "train": len(ds["train"]),
                "heldout": len(ds["heldout"]),
                "duplicates_dropped": ds["duplicates_dropped"],
                "bucket_mod": ds["buckets"],
                "by_question": dict(sorted(by_q.items())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _finetune_attempt(
    fn: callable, api_name: str, train_path: Path, base: str, out_dir: Path
) -> str | None:
    """Best-effort supervised fine-tune. Returns a checkpoint path on success,
    else None with a clear message. The shim guesses a conventional signature
    and fails open on any mismatch -- the dataset is staged regardless."""
    checkpoint = out_dir / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    try:
        fn(
            model=base,
            train=str(train_path),
            output=str(checkpoint),
            epochs=3,
        )
        return str(checkpoint)
    except TypeError as e:
        print(
            f"  laya.{api_name} signature differs from the scaffolding shim "
            f"({e}); dataset staged for manual training -- no checkpoint "
            f"produced."
        )
        return None
    except Exception as e:  # noqa: BLE001 - fine-tune failure is non-fatal
        print(f"  laya fine-tune failed: {e!r}; dataset staged.")
        return None


def _gate(args, model: str | None) -> dict:
    """Run the golden-label quality gate over ``model`` (or the gold-ideal
    fake when ``--real`` is off)."""
    from bench.laya_eval import _BoundSystem1, _IdealAgent, _run_async, load_cases

    cases = load_cases(args.gold)
    if args.real:
        from providers.laya.client import LayaSystem1

        s1 = LayaSystem1(
            enabled=True,
            model=model or args.model,
            device=args.device,
            allow_cpu=True,
            shadow_timeout=120.0,
        )
        source = "real"
    else:
        by_text = {c["text"]: c["labels"] for c in cases}
        s1 = _BoundSystem1(_IdealAgent(by_text))
        source = "fake"
    report = asyncio.run(_run_async(s1, cases, source=source))
    report["gated_model"] = model or (args.model if args.real else "fake(gold-ideal)")
    return report


def _render_gate(report: dict, gate: float) -> tuple[str, bool]:
    macro = report.get("macro_accuracy")
    passed = macro is not None and macro >= gate
    lines = [
        f"gate model:  {report.get('gated_model')}",
        f"gate corpus: {report.get('caseload')} cases ({report.get('source')})",
        f"macro acc:   {macro if macro is None else round(macro, 4)}",
        f"gate floor:  {gate}",
        f"gate:        {'PASS' if passed else 'FAIL ' + ('' if macro is not None else '(no real measurement)')}",
    ]
    for qid, b in sorted(report.get("questions", {}).items()):
        lines.append(f"  {qid:<20}{b['n']:>4}{round(b['accuracy'], 4):>9}")
    return "\n".join(lines), passed


def run_laya_finetune(args: argparse.Namespace) -> int:
    data_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data)
    if not samples:
        print(f"no samples in {args.data} -- run `tasa-bench laya-export` first")
        return 1
    ds = build_datasets(samples, heldout_frac=args.heldout)
    _write_dataset(ds, data_dir)
    print(
        f"samples:      {len(samples)} -> kept {ds['samples_kept']}"
        f" (dropped {ds['duplicates_dropped']} dup/conflict)"
    )
    print(
        f"dataset:      train={len(ds['train'])} heldout={len(ds['heldout'])}"
        f" -> {data_dir}"
    )

    fn, capability = _capability()
    checkpoint: str | None = None
    if fn is not None:
        print(f"fine-tune:    {capability} (base={args.model}, epochs=3)")
        checkpoint = _finetune_attempt(
            fn, capability, data_dir / "train.jsonl", args.model, data_dir
        )
    else:
        print(f"fine-tune:    unsupported ({capability}); dataset staged.")

    report = _gate(args, model=checkpoint)
    text, passed = _render_gate(report, args.gate)
    print("\nquality gate")
    print(text)
    return 0 if passed else 1