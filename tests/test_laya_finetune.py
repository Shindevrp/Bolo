"""Step 3: fine-tuning scaffolding + gold-corpus quality gate."""
from __future__ import annotations

import argparse
import json

from bench.laya_finetune import (
    build_datasets,
    load_samples,
    run_laya_finetune,
)


def _sample(question="intent", answer="command", conf=0.9, text="turn it off") -> dict:
    return {
        "question": question,
        "answer": answer,
        "confidence": conf,
        "state": {"transcript": text, "turn_count": 1},
        "transcript": text,
        "session_id": "s",
        "ts": 1.0,
    }


class TestDatasetBuild:
    def test_dedup_keeps_highest_confidence(self) -> None:
        samples = [
            _sample(question="intent", answer="command", conf=0.9),
            _sample(question="intent", answer="command", conf=0.3),
            _sample(question="intent", answer="greeting", conf=0.95, text="hi"),
        ]
        ds = build_datasets(samples)
        assert ds["samples_kept"] == 2
        assert ds["duplicates_dropped"] == 1
        by_text = {s["state"]["transcript"]: s for s in ds["train"] + ds["heldout"]}
        assert by_text["turn it off"]["confidence"] == 0.9  # winning sample won

    def test_split_is_deterministic_and_totals(self) -> None:
        samples = [_sample(text=f"u{i}") for i in range(50)]
        a = build_datasets(samples, heldout_frac=0.15)
        b = build_datasets(samples, heldout_frac=0.15)
        assert [s["state"]["transcript"] for s in a["train"]] == [
            s["state"]["transcript"] for s in b["train"]
        ]
        assert len(a["train"]) + len(a["heldout"]) == 50
        assert a["buckets"] == 7
        assert 0 < len(a["heldout"]) < len(a["train"])

    def test_sample_order_does_not_change_split(self) -> None:
        samples = [_sample(text=f"u{i}") for i in range(30)]
        a = build_datasets(samples)
        b = build_datasets(list(reversed(samples)))
        assert {s["state"]["transcript"] for s in a["train"]} == {
            s["state"]["transcript"] for s in b["train"]
        }


class TestFinetuneCli:
    def _sample_file(self, tmp_path) -> str:
        p = tmp_path / "training.laya.jsonl"
        p.write_text(
            "\n".join(
                json.dumps(_sample(text=t)) for t in ["turn it off", "what time", "hello"]
            )
        )
        return str(p)

    def test_load_samples(self, tmp_path) -> None:
        path = self._sample_file(tmp_path)
        samples = load_samples(path)
        assert len(samples) == 3
        assert samples[0]["question"] == "intent"

    def _namespace(self, tmp_path) -> argparse.Namespace:
        return argparse.Namespace(
            data=self._sample_file(tmp_path),
            gold="bench/gold/laya_gold.json",
            model="convaiinnovations/laya",
            out_dir=str(tmp_path / "ft"),
            real=False,
            device=None,
            gate=0.4811,
            heldout=0.15,
        )

    def test_fake_run_stages_dataset_and_passes_gate(
        self, tmp_path, capsys
    ) -> None:
        args = self._namespace(tmp_path)
        rc = run_laya_finetune(args)
        out_dir = tmp_path / "ft"
        assert rc == 0
        train = out_dir.joinpath("train.jsonl").read_text().splitlines()
        assert train, "train set is not empty"
        assert json.loads(train[0])["answer"] == "command"
        assert json.loads(train[0])["state"]["transcript"]
        summary = json.loads(out_dir.joinpath("dataset.json").read_text())
        assert summary["train"] == len(train)
        assert summary["heldout"] >= 0
        captured = capsys.readouterr().out
        assert "quality gate" in captured
        assert "gate:        PASS" in captured
        # `laya` (or its absence) is optional: the dataset must stage either way.
        assert "unsupported (" in captured

    def test_empty_samples_fails_fast(self, tmp_path) -> None:
        data = tmp_path / "empty.jsonl"
        data.touch()
        args = argparse.Namespace(
            data=str(data), gold="x", model="m", out_dir=str(tmp_path / "o"),
            real=True, device=None, gate=0.0, heldout=0.15,
        )
        assert run_laya_finetune(args) == 1