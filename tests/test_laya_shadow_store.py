"""Step 2: disk-backed shadow logging with self-labels + export bridge.

The shadow deque stays bounded and in-memory for ``shadow_report()``; when a
``shadow_log`` path is configured every entry also goes to a JSONL file, and
``training_samples()`` / ``bolo-bench laya-export`` turn the agreed rows into
fine-tuning data for Step 3.
"""
from __future__ import annotations

import json
from collections import Counter

from core.config import CoreConfig
from providers.laya.client import LayaSystem1
from providers.laya.store import (
    ShadowLogStore,
    read_shadow_log,
    training_samples,
)
from tests.test_laya_adapters import FakeAgent, _answers


def _label_row(
    *, question: str = "intent",
    laya="command",
    conf: float = 0.9,
    legacy="command",
    match: bool = True,
) -> dict:
    return {
        "question": question,
        "laya": laya,
        "laya_conf": conf,
        "legacy": legacy,
        "match": match,
    }


class TestShadowLogStore:
    def test_disabled_without_path(self) -> None:
        store = ShadowLogStore(None)
        assert store.enabled is False
        assert store.path is None
        assert store.append({"rows": [1]}) is True  # no-op, never raises

    def test_append_writes_one_jsonl_line(self, tmp_path) -> None:
        store = ShadowLogStore(str(tmp_path / "rows.jsonl"))
        entry = {"session_id": "s", "ts": 1.0, "rows": [{"question": "intent"}]}
        assert store.append(entry) is True
        assert store.append(None) is True
        lines = (tmp_path / "rows.jsonl").read_text().splitlines()
        assert json.loads(lines[0]) == entry
        assert len(lines) == 1

    def test_append_only_multiple_writes(self, tmp_path) -> None:
        store = ShadowLogStore(str(tmp_path / "rows.jsonl"))
        expected: list[int] = []
        for i in range(5):
            store.append({"i": i})
            expected.append(i)
        got = [
            json.loads(line)["i"]
            for line in (tmp_path / "rows.jsonl").read_text().splitlines()
        ]
        assert got == expected

    def test_expands_tilde(self, monkeypatch, tmp_path) -> None:
        from providers.laya import store as store_mod

        home = str(tmp_path)
        monkeypatch.setattr(store_mod.os.path, "expanduser", lambda p: p.replace("~", home))
        s = ShadowLogStore("~/shadow/rows.jsonl")
        s.append({"ts": 1})
        assert home == str(tmp_path)
        assert (tmp_path / "shadow" / "rows.jsonl").exists()


class TestSelfLabels:
    def _client(self, path: str | None) -> LayaSystem1:
        return LayaSystem1(
            enabled=True, agent=FakeAgent(_answers()), shadow_log=path
        )

    def test_agree_high_conf_is_labeled_and_written(self, tmp_path) -> None:
        path = str(tmp_path / "rows.jsonl")
        c = self._client(path)
        c.log_shadow(
            session_id="s",
            transcript="turn it off",
            state={"transcript": "turn it off", "turn_count": 2},
            rows=[_label_row(question="intent", laya="command", conf=0.9)],
        )
        record = c._shadow[-1]
        assert record["rows"][0]["self_label"] == "command"
        assert record["labelable"] == 1
        assert record["state"]["turn_count"] == 2
        assert record["session_id"] == "s"

        # Disk entry carries the same annotated rows.
        disk = read_shadow_log(path)
        assert len(disk) == 1
        assert disk[0]["rows"][0]["self_label"] == "command"
        assert disk[0]["state"]["transcript"] == "turn it off"

    def test_low_conf_agree_not_labeled(self, tmp_path) -> None:
        c = self._client(str(tmp_path / "rows.jsonl"))
        c.log_shadow(
            session_id="s",
            transcript="x",
            rows=[_label_row(laya="command", conf=0.2)],
        )
        assert c._shadow[-1]["rows"][0]["self_label"] is None
        assert c._shadow[-1]["labelable"] == 0

    def test_disagree_not_labeled(self, tmp_path) -> None:
        c = self._client(str(tmp_path / "rows.jsonl"))
        c.log_shadow(
            session_id="s",
            transcript="x",
            rows=[_label_row(laya="command", legacy="greeting", match=False, conf=0.9)],
        )
        assert c._shadow[-1]["rows"][0]["self_label"] is None

    def test_false_boolean_is_a_valid_label(self, tmp_path) -> None:
        # A confident "NOT a question" agreement is a useful training label.
        c = self._client(str(tmp_path / "rows.jsonl"))
        c.log_shadow(
            session_id="s",
            transcript="ok",
            rows=[_label_row(question="is_question", laya=False, conf=0.8, legacy=False)],
        )
        assert c._shadow[-1]["rows"][0]["self_label"] is False

    def test_pipeline_rows_still_work_without_a_path(self) -> None:
        # No shadow_log -> no disk access at all; in-memory report still works.
        c = self._client(None)
        c.log_shadow(
            session_id="s",
            transcript="x",
            rows=[_label_row(question="intent")],
        )
        report = c.shadow_report()
        assert report["intent"]["n"] == 1
        assert report["intent"]["agreement"] == 1.0


class TestExport:
    def test_training_samples_filter_conf_and_question(self, tmp_path) -> None:
        path = str(tmp_path / "rows.jsonl")
        c = LayaSystem1(enabled=True, agent=FakeAgent(_answers()), shadow_log=path)
        c.log_shadow(
            session_id="a", transcript="t1",
            state={"transcript": "t1", "turn_count": 1},
            rows=[
                _label_row(question="intent", laya="command", conf=0.9),
                _label_row(question="intent", laya="greeting", conf=0.3),
                # No real legacy classifier: agreement is not a label ...
                _label_row(question="tool_needed", laya=False, conf=0.9, legacy=False),
                # ... but what the turn actually did is.
                {**_label_row(question="tool", laya="none", conf=0.3, legacy="none"),
                 "outcome": "get_weather"},
                _label_row(
                    question="urgency", laya=False, legacy=True, match=False, conf=0.9,
                ),
            ],
        )
        entries = read_shadow_log(path)
        samples = training_samples(entries, min_conf=0.5)
        qs = Counter(s["question"] for s in samples)
        assert qs["intent"] == 1
        assert "tool_needed" not in qs  # placeholder legacy -> no label
        assert "urgency" not in qs  # disagreement -> no label
        tool = next(s for s in samples if s["question"] == "tool")
        assert tool["answer"] == "get_weather"  # outcome beats Laya's guess
        assert tool["source"] == "outcome"

        intent = next(s for s in samples if s["question"] == "intent")
        assert intent["answer"] == "command"
        assert intent["state"]["turn_count"] == 1
        assert intent["transcript"] == "t1"

        # Question filter narrows further.
        only_tool = training_samples(entries, min_conf=0.5, question="tool")
        assert [s["question"] for s in only_tool] == ["tool"]

        # The confidence gate drops agreement labels, never outcome labels.
        strict = training_samples(entries, min_conf=0.95)
        assert [s["question"] for s in strict] == ["tool"]

    def test_read_skips_malformed_lines(self, tmp_path) -> None:
        p = tmp_path / "rows.jsonl"
        p.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
        entries = read_shadow_log(str(p))
        assert [e["ok"] for e in entries] == [1, 2]

    def test_read_missing_file_returns_empty(self, tmp_path) -> None:
        assert read_shadow_log(str(tmp_path / "nope.jsonl")) == []


class TestConfigWiring:
    def test_default_path_wired_for_server(self, monkeypatch) -> None:
        monkeypatch.delenv("BOLO_LAYA_SHADOW_LOG", raising=False)
        monkeypatch.delenv("BOLO_LAYA_SHADOW_LOG", raising=False)
        assert CoreConfig().laya_shadow_log == "~/.bolo/shadow/rows.jsonl"

    def test_empty_env_disables_disk_writes(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_LAYA_SHADOW_LOG", "")
        assert CoreConfig().laya_shadow_log == ""
        c = LayaSystem1(enabled=False, agent=None)
        assert c._store is None