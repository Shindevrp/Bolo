from __future__ import annotations

import pytest

from bench.scoring.wer import cer, cer_stats, normalize, wer, wer_stats
from bench.scoring.entity_scan import (
    extract_capitalized_entities,
    user_provided_entities,
    has_uncertainty,
)


class TestWER:
    def test_identical(self):
        assert wer("the quick brown fox", "the quick brown fox") == 0.0

    def test_single_substitution(self):
        assert wer("the quick brown fox", "the quick red fox") == pytest.approx(0.25)

    def test_insertion(self):
        assert wer("hello world", "hello brave world") == pytest.approx(0.5)

    def test_deletion(self):
        assert wer("hello world", "hello") == pytest.approx(0.5)

    def test_none(self):
        assert wer("hello world", "") == 1.0

    def test_norm_ignores_case_and_punct(self):
        assert wer("Hello, World!", "hello world") == 0.0

    def test_stats(self):
        st = wer_stats(["a b c"], ["a b c"])
        assert st["wer"] == 0.0
        assert st["items"] == 1


class TestCER:
    def test_identical(self):
        assert cer("the quick brown fox", "the quick brown fox") == 0.0

    def test_single_char_substitution(self):
        assert cer("bat", "cat") == pytest.approx(1 / 3)

    def test_char_insertion(self):
        assert cer("abc", "abxc") == pytest.approx(1 / 3)

    def test_char_deletion(self):
        assert cer("abxc", "abc") == pytest.approx(0.25)

    def test_empty_hyp(self):
        assert cer("hello world", "") == 1.0

    def test_empty_ref_matches(self):
        assert cer("", "") == 0.0

    def test_caps_and_punct_normalized(self):
        assert cer("What time?", "what time") == 0.0

    def test_stats(self):
        st = cer_stats(["a b c"], ["a b c"])
        assert st["cer"] == 0.0
        assert st["items"] == 1
        assert st["chars"] == 3
        assert st["errors"] == 0
        assert len(st["per_item"]) == 1


class TestEntityScan:
    def test_caps_detected(self):
        ents = extract_capitalized_entities("The best place is Grand Canyon near the city")
        assert "Grand" in ents or "Grand Canyon" in ents

    def test_function_words_ignored(self):
        ents = extract_capitalized_entities("It is a simple fact that we should go")
        assert ents == []

    def test_user_provided(self):
        assert user_provided_entities("Near Hyderabad") == ["Hyderabad"]

    def test_uncertainty(self):
        assert has_uncertainty("I'm not sure that place exists")
        assert not has_uncertainty("There's a great place called Zanzibar")
