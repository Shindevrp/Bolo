"""Deterministic metric extraction from a ScenarioResult timeline.

All metrics here are computed from client-observed events, never from model
internals, so they remain comparable across agents. Qualitative categories are
handled separately by the optional judge LLM (scoring.judge).
"""
from __future__ import annotations

import time

from bench.agent.base import EventType
from bench.driver.runner import Timeline

EVENT_LABEL = {
    EventType.SPEECH_START: "speech_start",
    EventType.SPEECH_END: "speech_end",
    EventType.PARTIAL_TRANSCRIPT: "partial_transcript",
    EventType.TRANSCRIPT: "transcript",
    EventType.LLM_TOKEN: "llm_token",
    EventType.LLM_DONE: "llm_done",
    EventType.TTS_CHUNK: "tts_chunk",
    EventType.TTS_DONE: "tts_done",
    EventType.BACKCHANNEL: "backchannel",
    EventType.INTERRUPT: "interrupt",
    EventType.RESPONSE_DELAY: "response_delay",
}


def latency_buckets(tl: Timeline) -> dict:
    """Compute key user-end -> agent-response latencies (s).

    latencies are derived from the last TTS_CHUNK/LLM_TOKEN after SPEECH_END,
    using monotonic event timestamps. Where a required event is absent we
    return None (never invent a number).
    """
    speech_end = tl.first(EventType.SPEECH_END)
    if speech_end is None:
        return {
            "e2e_first_audio": None,
            "e2e_transcript": None,
            "speech_end_to_first_token": None,
            "speech_end_to_tts_done": None,
            "first_audio_bucket": "unobservable",
            "has_speech_end": False,
            "note": "no speech_end",
        }

    # First TTS audio after speech_end
    first_tts = tl.first(EventType.TTS_CHUNK)
    first_tts_after = None
    if first_tts is not None:
        # We use absolute monotonic; first speech_end as the anchor.
        first_tts_after = tl.first(EventType.TTS_CHUNK)

    def delta(a: "EventType", b: "EventType"):
        ea = tl.first(a)
        eb = tl.first(b)
        if ea is None or eb is None:
            return None
        return round(eb.mtime - ea.mtime, 3)

    # From user end (speech_end) to first agent audio:
    if first_tts_after is not None:
        ttfa = round(first_tts_after.mtime - speech_end.mtime, 3)
    else:
        ttfa = None
    # speech_end -> first LLM token
    tflt = delta(EventType.SPEECH_END, EventType.LLM_TOKEN)
    # speech_end -> full response done
    tfr = delta(EventType.SPEECH_END, EventType.TTS_DONE)

    return {
        "e2e_first_audio": ttfa,           # speech_end -> first TTS audio
        "speech_end_to_first_token": tflt,
        "speech_end_to_tts_done": tfr,
        # Categorical labels the report can cite without inventing precision
        "first_audio_bucket": _bucket(ttfa),
        "has_speech_end": speech_end is not None,
    }


def _bucket(seconds: float | None) -> str:
    if seconds is None:
        return "unobservable"
    s = seconds
    if s < 0.5:
        return "immediate"
    if s < 1.0:
        return "slight delay"
    if s < 2.0:
        return "noticeable delay"
    if s < 4.0:
        return "long delay"
    return "severe delay"


def turn_end_metrics(tl: Timeline) -> dict:
    """Detect if the agent ended (or responded to) a turn before the user
    finished, or split a turn. Derived purely from event ordering + a
    'pre-empt' signal (a response event occurring before an expected end)."""
    # Heuristic: if LLM_TOKEN/TTS starts before the user's speech_end, the
    # agent responded mid-utterance (early). Bolo only emits speech_end then
    # processes; so normally LLM follows speech_end. We surface the ordering.
    spe = tl.first(EventType.SPEECH_END)
    first_llm = tl.first(EventType.LLM_TOKEN)
    pre_empt = bool(
        spe is not None
        and first_llm is not None
        and first_llm.mtime < spe.mtime
    )
    # Did the agent wait an unusually long time? speech_end -> first audio
    lat = latency_buckets(tl)
    return {
        "pre_empt": pre_empt,
        "speech_end_seen": spe is not None,
        "first_llm_before_speech_end": pre_empt,
        "first_audio_after_speech_end": lat["e2e_first_audio"],
        "bucket": lat["first_audio_bucket"],
    }


def barge_in_metrics(tl: Timeline) -> dict:
    """Barge-in correctness. Requires an interrupt event + subsequent turn."""
    interrupt = tl.first(EventType.INTERRUPT)
    if interrupt is None:
        return {"interrupt_seen": False, "interrupted": False}

    # After the interrupt, a new turn should begin (new speech_start or new
    # transcript) indicating the agent recovered to listening.
    new_turn = any(
        e.type in (EventType.SPEECH_START, EventType.TRANSCRIPT)
        and e.mtime >= interrupt.mtime
        for e in tl.events
    )
    # How quickly the interrupt surfaced (from send to INTERRUPT event) is
    # computed by the caller using step timestamps. Here we mark detection.
    return {
        "interrupt_seen": True,
        "interrupted": True,
        "recovered_to_new_turn": new_turn,
    }


def is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return t.lower().startswith(
        ("what", "why", "how", "when", "where", "who", "which", "can", "could", "do", "does", "are", "is", "should")
    )


def count_questions(text: str) -> int:
    t = (text or "").strip()
    return t.count("?")


def context_facts(tl: Timeline) -> dict:
    """Whether the final answer references earlier provided values. Uses the
    accumulated user transcripts + final LLM output, checking key markers."""
    user = " ".join(tl.final_transcripts()).lower()
    out = tl.llm_output().lower()
    return {
        "mentioned_five": "five" in out or "5 " in out or "5 people" in out,
        "mentioned_budget": "6000" in out or "six thousand" in out,
        "mentioned_nature": ("nature" in out or "hiking" in out),
        "user_gave_five": "five" in user,
    }


def grounding_metrics(tl: Timeline) -> dict:
    """Detect fabricated capitalized proper nouns and uncertainty expression.

    Mirrors Bolo's EntityGate heuristic: extract capitalized runs, and flag
    entities not present in the user's own input. This is a deterministic
    proxy (never a ground truth about existence), which is the honest limit
    for a black-box harness.
    """
    from bench.scoring.entity_scan import (
        extract_capitalized_entities,
        user_provided_entities,
        UNCERTAINTY_PHRASES as uncertainty_phrases,
    )

    user_input = " ".join(tl.final_transcripts())
    response = tl.llm_output()
    supplied = user_provided_entities(user_input)
    entities = extract_capitalized_entities(response)
    unsupported = [
        e for e in entities if e.lower() not in {s.lower() for s in supplied}
    ]
    unc = [p for p in uncertainty_phrases if p in response.lower()]
    return {
        "named_entities": entities,
        "unsupported_entities": unsupported,
        "uncertainty_expressed": bool(unc),
        "uncertainty_phrases": unc,
        "fabrication_risk": bool(unsupported),
    }


def response_length_metrics(tl: Timeline) -> dict:
    out = tl.llm_output()
    words = len(out.split())
    return {
        "words": words,
        "sentences": max(1, out.count(". ") + out.count("!") + out.count("?")),
        "questions_asked": count_questions(out),
    }


def collect_scenario_metrics(tl: Timeline) -> dict:
    """Aggregate all deterministic metrics for one scenario timeline."""
    return {
        "latency": latency_buckets(tl),
        "turn_end": turn_end_metrics(tl),
        "barge_in": barge_in_metrics(tl),
        "context": context_facts(tl),
        "grounding": grounding_metrics(tl),
        "response": response_length_metrics(tl),
    }
