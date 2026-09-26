from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CoreConfig:
    # STT
    stt_model: str = field(
        default_factory=lambda: os.getenv("TASA_STT_MODEL", "base")
    )
    stt_device: str = field(
        default_factory=lambda: os.getenv("TASA_STT_DEVICE", "cpu")
    )
    stt_compute: str = field(
        default_factory=lambda: os.getenv("TASA_STT_COMPUTE", "int8")
    )

    # LLM
    llm_url: str = field(
        default_factory=lambda: os.getenv("TASA_LLM_URL", "http://localhost:8000/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "TASA_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("TASA_LLM_API_KEY", "EMPTY")
    )
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_top_p: float = 0.9

    # Escalation tier (Phase-2 "stronger LLM" fallback). Empty url disables
    # the fallback: escalate votes then fall back to the primary model.
    llm_fallback_url: str = field(
        default_factory=lambda: os.getenv("TASA_LLM_FALLBACK_URL", "")
    )
    llm_fallback_model: str = field(
        default_factory=lambda: os.getenv("TASA_LLM_FALLBACK_MODEL", "")
    )
    llm_fallback_api_key: str = field(
        default_factory=lambda: os.getenv("TASA_LLM_FALLBACK_API_KEY", "EMPTY")
    )

    # TTS
    tts_model: str = field(
        default_factory=lambda: os.getenv(
            "TASA_TTS_MODEL",
            "/usr/share/piper/voices/en_US-lessac-medium.onnx",
        )
    )

    # VAD
    vad_threshold: float = field(
        default_factory=lambda: float(os.getenv("TASA_VAD_THRESHOLD", "0.5"))
    )
    vad_sample_rate: int = 16000
    silence_ms: float = 400.0

    # Emotion classification
    emotion_enabled: bool = field(
        default_factory=lambda: os.getenv("TASA_EMOTION_ENABLED", "1") != "0"
    )
    emotion_model: str = field(
        default_factory=lambda: os.getenv(
            "TASA_EMOTION_MODEL",
            "j-hartmann/emotion-english-distilroberta-base",
        )
    )
    emotion_device: str | None = field(
        default_factory=lambda: os.getenv("TASA_EMOTION_DEVICE", "cuda")
    )

    # Pipeline
    audio_queue_size: int = 512
    output_queue_size: int = 512
    default_timeout: float = 30.0
    default_language: str = "en"
    session_timeout: float = 300.0

    # System-1 decision layer (Laya) -- shadow mode
    laya_model: str = field(
        default_factory=lambda: os.getenv(
            "TASA_LAYA_MODEL", "convaiinnovations/laya"
        )
    )
    laya_device: str | None = field(
        default_factory=lambda: (
            os.getenv("TASA_LAYA_DEVICE", "").strip() or None
        )
    )
    laya_enabled: bool = field(
        default_factory=lambda: os.getenv("TASA_LAYA_ENABLED", "1") != "0"
    )
    laya_conf_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("TASA_LAYA_CONF_THRESHOLD", "0.5")
        )
    )
    # Phase 2: let confident Laya answers drive intent/complexity/routing
    # (shadow rows keep recording either way). Off by default: base
    # checkpoints ship uncalibrated/over-confident temperatures, so Phase-2
    # stays opt-in until shadow agreement data justifies turning it on.
    laya_phase2: bool = field(
        default_factory=lambda: os.getenv("TASA_LAYA_PHASE2", "0") == "1"
    )
    # Phase 3: let confident cadence-1 Laya answers modulate endpoint lanes
    # (shorter turn-commit after a confident "utterance complete") and barge
    # decisions (backchannel suppression / immediate disagreement interrupt) --
    # always inside the deterministic safety floor (min speech duration,
    # endpoint cap, energy gates). Off by default, same over-confidence
    # rationale as Phase-2.
    laya_phase3: bool = field(
        default_factory=lambda: os.getenv("TASA_LAYA_PHASE3", "0") == "1"
    )
    # Phase 4: cadence-1 "utterance complete" endpoint authority on its own --
    # the shorter turn-commit during silence wait WITHOUT Phase-3's barge
    # enforcement. Lets the confident ``turn_complete`` verdict drive the
    # speculative early-start independently of the interrupt path. Same
    # safety floor (min speech, endpoint cap) and same default-off rationale.
    laya_phase4: bool = field(
        default_factory=lambda: os.getenv("TASA_LAYA_PHASE4", "0") == "1"
    )
    # Phase 5: complexity-routing gate -- a bare acknowledgment (multi-word
    # exact match, e.g. "sounds good") is answered deterministically instead
    # of the LLM when a confident Laya verdict marks it trivial (simple, not a
    # question, not urgent, no escalation). Runs a synchronous cadence-2 pass
    # ONLY for ack-shaped utterances. Off by default until shadow agreement
    # data justifies it.
    laya_phase5: bool = field(
        default_factory=lambda: os.getenv("TASA_LAYA_PHASE5", "0") == "1"
    )
    # Deterministic fast-path (canned greetings/farewells/backchannels +
    # templated time/date/calc replies). Independent of the Laya phases: it is
    # pure rule matching that costs no GPU, so it can serve replies even while
    # Laya runs fully shadowed. On by default.
    tasa_fast_path: bool = field(
        default_factory=lambda: os.getenv("TASA_FAST_PATH", "1") != "0"
    )
    # Single-miss tool prefetch (latency Step 1): while the user is still
    # talking, a live-listening Laya probe on the partial transcript can
    # predict a confident tool call and stash its result so the LLM's first
    # prompt already carries the data (saves the tool round-trip on a hit;
    # costs exactly one wasted lookup on a miss). On by default.
    tool_prefetch: bool = field(
        default_factory=lambda: os.getenv("TASA_TOOL_PREFETCH", "1") != "0"
    )
    # Cast Laya's probe weights to fp16 to halve memory bandwidth during the
    # live-listening / turn-level passes sharing the GPU with the LLM.
    # Best-effort: skipped when the loaded checkpoint exposes no convertible
    # torch modules. On by default.
    laya_fp16: bool = field(
        default_factory=lambda: os.getenv("TASA_LAYA_FP16", "1") != "0"
    )
    # Step 2: disk-backed shadow log. Each turn's Laya-vs-legacy rows (with
    # self-labels where Laya agreed with legacy at confidence) are appended
    # to this JSONL path so the collection survives restarts and can be
    # exported into training data (`tasa-bench laya-export`). Empty disables
    # disk writes (rows still stay in memory for shadow_report()).
    laya_shadow_log: str = field(
        default_factory=lambda: os.getenv(
            "TASA_LAYA_SHADOW_LOG", "~/.tasa/shadow/rows.jsonl"
        )
    )


