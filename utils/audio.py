from __future__ import annotations


def normalize_audio_bytes(audio: bytes) -> bytes:
    """Placeholder audio normalization helper."""
    return audio


def audio_duration_seconds(samples: int, sample_rate: int = 16000) -> float:
    """Return audio duration in seconds for a sample count."""
    return samples / sample_rate
