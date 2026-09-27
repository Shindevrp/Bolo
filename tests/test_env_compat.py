"""Bolo reads config as ``BOLO_*`` and falls back to legacy ``TASA_*``."""

from __future__ import annotations

import pytest

from core.env import env, env_bool, env_float, env_int, env_list


class TestPrecedence:
    def test_bolo_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_LLM_MODEL", "bolo-model")
        monkeypatch.setenv("TASA_LLM_MODEL", "tasa-model")
        assert env("LLM_MODEL") == "bolo-model"

    def test_legacy_used_when_bolo_absent(self, monkeypatch) -> None:
        monkeypatch.delenv("BOLO_LLM_MODEL", raising=False)
        monkeypatch.setenv("TASA_LLM_MODEL", "tasa-model")
        assert env("LLM_MODEL") == "tasa-model"

    def test_default_when_neither_set(self, monkeypatch) -> None:
        monkeypatch.delenv("BOLO_LLM_MODEL", raising=False)
        monkeypatch.delenv("TASA_LLM_MODEL", raising=False)
        assert env("LLM_MODEL", "fallback") == "fallback"

    def test_bolo_empty_overrides_legacy(self, monkeypatch) -> None:
        # An explicitly empty BOLO_ value must win over a legacy value, or you
        # cannot un-set something the old .env still defines.
        monkeypatch.setenv("BOLO_LAYA_SHADOW_LOG", "")
        monkeypatch.setenv("TASA_LAYA_SHADOW_LOG", "/old/path.jsonl")
        assert env("LAYA_SHADOW_LOG") == ""

    def test_legacy_empty_overrides_default(self, monkeypatch) -> None:
        monkeypatch.delenv("BOLO_LAYA_SHADOW_LOG", raising=False)
        monkeypatch.setenv("TASA_LAYA_SHADOW_LOG", "")
        assert env("LAYA_SHADOW_LOG", "~/.bolo/shadow/rows.jsonl") == ""

    def test_bare_key_is_prefixed(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_VAD_THRESHOLD", "0.7")
        assert env("VAD_THRESHOLD") == "0.7"

    def test_whitespace_is_stripped(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_LLM_MODEL", "  spaced  ")
        assert env("LLM_MODEL") == "spaced"


class TestTypedHelpers:
    def test_float(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_VAD_THRESHOLD", "0.42")
        assert env_float("VAD_THRESHOLD", 0.5) == 0.42

    def test_float_falls_back_on_garbage(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_VAD_THRESHOLD", "not-a-number")
        assert env_float("VAD_THRESHOLD", 0.5) == 0.5

    def test_int(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_N", "7")
        assert env_int("N", 1) == 7

    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("0", False), ("true", True), ("false", False),
        ("no", False), ("off", False),
    ])
    def test_bool(self, monkeypatch, raw, expected) -> None:
        monkeypatch.setenv("BOLO_FLAG", raw)
        assert env_bool("FLAG", not expected) is expected

    def test_bool_empty_falls_back_to_default(self, monkeypatch) -> None:
        # Empty means "not specified". Every existing flag defaults to on and
        # is disabled with 0/false, so this preserves the old behaviour.
        monkeypatch.setenv("BOLO_FLAG", "")
        assert env_bool("FLAG", True) is True
        assert env_bool("FLAG", False) is False

    def test_list(self, monkeypatch) -> None:
        monkeypatch.setenv("BOLO_LIST", "a, b ,c")
        assert env_list("LIST") == ["a", "b", "c"]
