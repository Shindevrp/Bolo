"""Agent interface abstraction.

Defines the normalized event protocol shared by every agent adapter so that the
scenario runner and scoring layers are completely agent-agnostic. Adding a new
agent = implementing one adapter class against this protocol; all scenarios,
metrics, and reports work unchanged.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class EventType(Enum):
    SPEECH_START = auto()
    SPEECH_END = auto()
    PARTIAL_TRANSCRIPT = auto()
    TRANSCRIPT = auto()          # final STT output for the turn
    LLM_TOKEN = auto()
    LLM_DONE = auto()
    TTS_CHUNK = auto()           # binary audio from the agent (bytes)
    TTS_DONE = auto()
    BACKCHANNEL = auto()
    INTERRUPT = auto()
    RESPONSE_DELAY = auto()
    STATUS = auto()
    ERROR = auto()
    PROSODY = auto()           # emitted TTS prosody profile (label + emotion)


@dataclass
class Event:
    type: EventType
    text: str | None = None
    audio: bytes | None = None
    mtime: float = field(default_factory=time.monotonic)

    @classmethod
    def make(cls, type: EventType, **kw) -> "Event":
        return cls(type=type, **kw)


def make_silence(ms: float, sr: int = 16000) -> bytes:
    """Raw 16-bit mono PCM silence of `ms` milliseconds."""
    n = int(sr * ms / 1000.0)
    return (b"\x00\x00" * (n // 2)) + (b"\x00" if n % 2 else b"")


def wav_to_pcm16(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> bytes:
    """Resample float32 audio in [-1,1] to 16-bit mono PCM at target_sr."""
    x = audio.astype(np.float32)
    if orig_sr != target_sr:
        from scipy.signal import resample_poly

        if orig_sr % target_sr == 0:
            x = resample_poly(x, 1, orig_sr // target_sr)
        else:
            x = resample_poly(x, target_sr, orig_sr)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()


class SpeechAgent:
    """Protocol every agent adapter must implement.

    The adapter exposes an async, pull-based event stream so the runner can
    consume events concurrently while executing scenario steps.
    """

    sample_rate: int = 16000

    async def connect(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    async def reset_session(self) -> None:
        raise NotImplementedError

    async def send_audio(self, pcm_bytes: bytes) -> None:
        raise NotImplementedError

    async def send_interrupt(self) -> None:
        raise NotImplementedError

    async def next_event(self, timeout: float = 60.0) -> Event | None:
        """Return the next event, or None on timeout / closed stream."""
        raise NotImplementedError
