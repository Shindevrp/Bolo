"""Bolo Speech-to-Speech Benchmark Framework.

A reusable, agent-agnostic benchmark harness that scores any speech-to-speech
conversational agent (Bolo first, then other agents) on the evaluator's
categories, driven by scripted scenarios over real open-source audio corpora.

Design goals:
  - Reusable: adapters isolate the agent's transport; scenarios + scoring are
    agent-agnostic and directly comparable across agents.
  - Deterministic pipeline metrics (turn-end, barge-in, latency, WER, grounding)
    computed from the event timeline, not the judge LLM.
  - Optional judge-LLM rubric for qualitative categories (naturalness, flow,
    question quality, ...).
  - Honest about what cannot be measured (TTS spectral quality, fine ASR
    split-points) — never invented, per evaluator rules 7 & 8.
"""
