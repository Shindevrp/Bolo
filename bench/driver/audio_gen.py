"""Synthetic user-speech generation for scenarios.

Produces 16kHz mono 16-bit PCM for scripted utterances using Piper TTS when a
model is available, otherwise a deterministic toned placeholder. Also provides
helpers to interleave natural pauses / filler silence so the harness can probe
an agent's turn-end thresholds.

Also generates pure-PCM silence for explicit steps.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_APP_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_CANDIDATE_MODELS = [
    _APP_MODELS_DIR / "en_US-lessac-medium.onnx",
    Path("/usr/share/piper/voices/en_US-lessac-medium.onnx"),
    Path("/usr/share/piper/voices/en_US-lessac-medium.onnx") if False else None,
]

_piper = None
_model_path = None


def _load_piper() -> object | None:
    global _piper, _model_path
    if _piper is not None:
        return _piper
    candidates = [c for c in _CANDIDATE_MODELS if c is not None]
    chosen = next((c for c in candidates if c.exists()), None)
    if chosen is None:
        return None
    try:
        from providers.tts.piper_tts import PiperTTS

        _piper = PiperTTS(model_path=str(chosen))
        _model_path = chosen
        return _piper
    except Exception as exc:
        print(f"[audio_gen] piper init failed: {exc}")
        return None


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x
    from scipy.signal import resample_poly

    if sr % 16000 == 0:
        return resample_poly(x, 1, sr // 16000)
    return resample_poly(x, 16000, sr)


def synthesize_pcm16(text: str, sample_rate: int = 16000) -> bytes:
    """Synthesize `text` synchronously and return raw 16-bit mono PCM at 16kHz."""
    tts = _load_piper()
    if tts is None:
        return _placeholder_pcm(text, sample_rate)

    import asyncio

    return asyncio.run(_synth_piper(tts, text, sample_rate))


async def synthesize_pcm16_async(text: str, sample_rate: int = 16000) -> bytes:
    """Async variant safe to call from within a running event loop."""
    tts = _load_piper()
    if tts is None:
        return _placeholder_pcm(text, sample_rate)
    return await _synth_piper(tts, text, sample_rate)


async def _synth_piper(tts, text: str, sample_rate: int) -> bytes:
    audio = await tts.synthesize(text)  # bytes at piper sample rate
    sr = tts.sample_rate
    n = len(audio) // 2
    if n == 0:
        return _placeholder_pcm(text, sample_rate)
    nums = np.frombuffer(audio[: n * 2], dtype=np.int16).astype(np.float32) / 32767.0
    resampled = _resample_to_16k(nums, sr)
    pcm = (np.clip(resampled, -1, 1) * 32767).astype(np.int16).tobytes()
    return pcm


def _placeholder_pcm(text: str, sample_rate: int = 16000) -> bytes:
    """Non-silent placeholder when Piper is unavailable. VAD-detectable as
    speech but NOT intelligible — only a CI fallback."""
    dur = max(0.6, 0.09 * len(text.split()))
    n = int(sample_rate * dur)
    t = np.arange(n) / sample_rate
    x = 0.4 * (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 300 * t))
    return (x * 32767).astype(np.int16).tobytes()


def silence_pcm(ms: float, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * ms / 1000.0)
    return b"\x00\x00" * (n // 2)


def interleave_silence(pcm: bytes, gap_ms: float, sample_rate: int = 16000) -> bytes:
    """Append a silence gap to the end of the PCM (used to separate clauses
    within a single turn for turn-end probing)."""
    return pcm + silence_pcm(gap_ms, sample_rate)
