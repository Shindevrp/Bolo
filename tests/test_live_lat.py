"""Realtime latency recorder: turn segmentation + stage derivation + reports."""
from __future__ import annotations

from bench.live_lat import (
    TraceRecorder,
    load_trace,
    render_delta,
    render_summary,
    save_trace,
    summarize,
)


def _feed_simple_turn(r: TraceRecorder) -> dict:
    """A canonical turn; returns the closed turn dict."""
    r.feed("speech_start", 0.0)
    r.feed("speech_end", 0.5)
    r.feed("transcript", 0.6, text="what time is it")
    r.feed("llm_token", 1.2)
    r.feed("llm_done", 2.0, text="It is 10am.")
    r.feed("tts_chunk", 2.1, nbytes=1000)
    return r.feed("tts_done", 3.0)


class TestTraceRecorder:
    def test_stage_timings_are_exact(self) -> None:
        r = TraceRecorder()
        turn = _feed_simple_turn(r)
        assert len(r.turns) == 1
        assert turn["speech_ms"] == 500.0
        assert turn["commit_to_stt_ms"] == 100.0
        assert turn["stt_to_first_token"] == 600.0
        assert turn["first_token_to_done"] == 800.0
        assert turn["commit_to_reply_ms"] == 1500.0
        assert turn["reply_to_tts_first"] == 100.0
        assert turn["commit_to_tts_first"] == 1600.0
        assert turn["commit_to_tts_done"] == 2500.0
        assert turn["tts_audio_ms"] == 900.0
        assert turn["tts_bytes"] == 1000
        assert turn["user_text"] == "what time is it"
        assert turn["reply_text"] == "It is 10am."
        assert turn["aborted"] is False

    def test_two_turns_do_not_bleed(self) -> None:
        r = TraceRecorder()
        _feed_simple_turn(r)
        r.feed("speech_start", 10.0)
        r.feed("speech_end", 10.4)
        r.feed("transcript", 10.5, text="bye")
        r.feed("llm_done", 10.6, text="Goodbye!")
        r.feed("tts_done", 10.9)
        r.close()
        assert len(r.turns) == 2
        assert r.turns[1]["start_ts"] == 10.0
        assert r.turns[1]["user_text"] == "bye"
        assert r.turns[1]["commit_to_reply_ms"] == 200.0
        assert r.turns[1].get("stt_to_first_token") is None  # no llm_token event

    def test_interrupt_aborts_turn(self) -> None:
        r = TraceRecorder()
        r.feed("speech_start", 0.0)
        r.feed("speech_end", 0.4)
        closed = r.feed("interrupt", 0.5)
        assert closed["aborted"] is True
        assert closed["dur_ms"] == 400.0

    def test_partial_and_backchannel_are_ignored(self) -> None:
        r = TraceRecorder()
        r.feed("speech_start", 0.0)
        r.feed("partial_transcript", 0.2, text="what time")
        r.feed("backchannel", 0.3, text="mhm")
        r.feed("speech_end", 0.5)
        r.feed("transcript", 0.6, text="what time is it")
        r.feed("llm_done", 1.0, text="10am.")
        r.feed("tts_done", 1.4)
        turn = r.turns[0]
        assert turn["user_text"] == "what time is it"  # partial never overwrote
        assert turn["speech_ms"] == 500.0

    def test_new_speech_closes_incomplete_barge_turn(self) -> None:
        r = TraceRecorder()
        r.feed("speech_start", 0.0)
        r.feed("speech_end", 0.3)
        r.feed("transcript", 0.4, text="wait--")
        r.feed("speech_start", 2.0)  # user barged over their own turn
        assert len(r.turns) == 1
        assert r.turns[0]["aborted"] is True


class TestSummarizeAndRender:
    def test_summary_percentiles(self) -> None:
        r = TraceRecorder()
        _feed_simple_turn(r)
        t = r.turns[0]
        r2 = TraceRecorder()
        _feed_simple_turn(r2)
        r2.turns[0]["commit_to_tts_first"] = t["commit_to_tts_first"] + 100.0
        stats = summarize([t, r2.turns[0]])
        key = stats["commit_to_tts_first"]
        assert key["count"] == 2
        assert key["p50"] == 1650.0
        assert key["p99"] == 1700.0

    def test_render_summary_shape(self) -> None:
        r = TraceRecorder()
        _feed_simple_turn(r)
        stats = summarize(r.turns)
        out = render_summary(stats)
        assert "turns: 1" in out
        assert "perceived response" in out
        assert "1600.0ms" in out

    def test_save_load_roundtrip_and_delta(self, tmp_path) -> None:
        r = TraceRecorder()
        _feed_simple_turn(r)
        path = str(tmp_path / "trace.jsonl")
        save_trace(path, r.turns, {"label": "a"})
        loaded = load_trace(path)
        assert len(loaded) == 1
        assert loaded[0]["reply_text"] == "It is 10am."

        slower = [dict(t, commit_to_tts_first=t["commit_to_tts_first"] + 200.0)
                  for t in r.turns]
        delta = render_delta(r.turns, slower)
        assert "perceived response" in delta
        assert "+12.5%".replace("+", "") in delta or "+12.5%" in delta

    def test_empty_summary(self) -> None:
        stats = summarize([])
        assert stats["count"] == 0
        assert "commit_to_tts_first" not in stats