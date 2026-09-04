"""Optional judge-LLM rubric for qualitative categories.

The deterministic metrics (scoring.metrics) measure what can be measured.
Qualitative categories — naturalness, dialogue flow, question quality,
repetition, response length fit, role consistency, intent/uncertainty handling
— are scored here by a judge LLM over the transcript timeline. The judge is a
separate, configurable endpoint (e.g. Ollama / vLLM OpenAI-compatible).

If no judge URL is configured, these categories are recorded as 'not scored'
rather than guessed (evaluator rules 7-8).
"""
from __future__ import annotations

import json

JUDGE_SYSTEM = """You are a strict, unbiased evaluator of conversational voice agents.
Given a timed transcript of a user interacting with an agent, score these
categories from 1 (very poor) to 10 (excellent):

naturalness, dialogue_flow, question_quality, repetition_control,
response_length, role_consistency, intent_understanding, uncertainty_handling,
backchannel_handling, grounding, error_recovery, tts_quality

Rules:
- Be honest and conservative. Do not inflate.
- base every score on concrete evidence in the transcript.
- response_length: prefer short, direct, 1-3 sentence voice responses unless
  detail was requested. Penalize long monologues and over-questioning.
- question_quality: penalize asking 4-5 questions in one voice turn; reward
  asking only necessary questions (don't re-ask what was already answered).
- repetition_control: penalize repeated questions/explanations/filler phrases
  ("got it", "sure", "perfect") dominating.
- error_recovery: score how gracefully the agent acknowledged and recovered
  when the user corrected or re-stated a request ("no, that's not what I
  meant...").
- tts_quality: score from cues in the transcript (filler/interruption markers
  aside); if TTS was never heard or there is no evidence, give a neutral 5 and
  note it.
- Return ONLY a JSON object keyed by the categories above with integer values,
  plus a "notes" array of short concrete observations. No markdown."""


def _call_judge(url: str, model: str, transcript: str, timeout: float) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=url, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        temperature=0.0,
        max_tokens=700,
        timeout=timeout,
    )
    content = resp.choices[0].message.content or "{}"
    return _parse(content)


def _parse(content: str) -> dict:
    content = content.strip()
    # Strip code fences if present
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # try to extract the JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return data


def score_timeline(
    tl_transcript: str,
    *,
    judge_url: str | None = None,
    judge_model: str = "qwen2.5:3b",
    timeout: float = 45.0,
) -> dict:
    """Return per-category judge scores. Empty dict if no judge configured."""
    if not judge_url:
        return {"scored": False, "categories": {}, "notes": []}
    try:
        data = _call_judge(judge_url, judge_model, tl_transcript, timeout)
    except Exception as exc:
        return {"scored": False, "error": f"{type(exc).__name__}: {exc}", "categories": {}, "notes": []}
    cats = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    notes = data.get("notes", []) if isinstance(data.get("notes"), list) else []
    return {"scored": True, "categories": cats, "notes": notes}
