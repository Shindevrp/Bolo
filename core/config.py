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

    # Multi-speaker
    multi_speaker_enabled: bool = field(
        default_factory=lambda: os.getenv("TASA_MULTI_SPEAKER", "0") == "1"
    )
    speaker_1_name: str = field(
        default_factory=lambda: os.getenv("TASA_SPEAKER_1_NAME", "Sh")
    )
    speaker_1_voice: str = field(
        default_factory=lambda: os.getenv(
            "TASA_SPEAKER_1_VOICE",
            "/app/models/piper/en_GB-alan-low.onnx",
        )
    )
    speaker_2_name: str = field(
        default_factory=lambda: os.getenv("TASA_SPEAKER_2_NAME", "Ti")
    )
    speaker_2_voice: str = field(
        default_factory=lambda: os.getenv(
            "TASA_SPEAKER_2_VOICE",
            "/app/models/piper/en_US-kristin-medium.onnx",
        )
    )
    speaker_1_personality: str = field(
        default_factory=lambda: os.getenv(
            "TASA_SPEAKER_1_PERSONALITY",
            "thoughtful, analytical, warm British wit",
        )
    )
    speaker_1_style: str = field(
        default_factory=lambda: os.getenv(
            "TASA_SPEAKER_1_STYLE",
            "calm, measured, dry humor, uses precise language",
        )
    )
    speaker_2_personality: str = field(
        default_factory=lambda: os.getenv(
            "TASA_SPEAKER_2_PERSONALITY",
            "energetic, curious, optimistic American",
        )
    )
    speaker_2_style: str = field(
        default_factory=lambda: os.getenv(
            "TASA_SPEAKER_2_STYLE",
            "enthusiastic, expressive, asks great questions, warm",
        )
    )
    interrupt_mode: str = field(
        default_factory=lambda: os.getenv("TASA_INTERRUPT_MODE", "sentence")
    )
    interrupt_urgency_threshold: float = field(
        default_factory=lambda: float(os.getenv("TASA_INTERRUPT_URGENCY_THRESHOLD", "0.7"))
    )
    max_consecutive_turns: int = field(
        default_factory=lambda: int(os.getenv("TASA_MAX_CONSECUTIVE_TURNS", "2"))
    )
    overlap_ms: int = field(
        default_factory=lambda: int(os.getenv("TASA_OVERLAP_MS", "200"))
    )
    inter_speaker_pause_ms: int = field(
        default_factory=lambda: int(os.getenv("TASA_INTER_SPEAKER_PAUSE_MS", "400"))
    )
