"""Open-source audio corpus loader.

Streams real speech clips (with ground-truth text where available) from
HuggingFace `datasets`, resampled to 16 kHz mono 16-bit PCM — the format the
benchmark feeds to agent adapters. Used for the ASR/WER leg and for varied,
realistic user utterances in conversational scenarios.

Corpora supported:
  - librispeech test-clean / test-other  -> ASR/WER (gold-standard 16k audio)
  - mozilla common_voice (en)            -> robustness (accents, real variation)
  - ljspeech                             -> clean single-speaker clips
  - libritts_r (or libritts)             -> multi-speaker TTS-derived clips

Installing the dependency (required for HF corpora):
    pip install datasets
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Clip:
    """One audio utterance with optional ground-truth text."""

    id: str
    audio: np.ndarray  # float32 in [-1, 1]
    sr: int
    text: str = ""
    speaker: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sr if self.sr else 0.0

    def to_pcm16(self, target_sr: int = 16000) -> bytes:
        """Resample (if needed) and return raw 16-bit mono PCM at target_sr."""
        x = self.audio.astype(np.float32)
        if self.sr != target_sr:
            x = _resample(x, self.sr, target_sr)
        pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes()


def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    # Downsample/upsample by integer ratio via rational resampling.
    if orig_sr % target_sr == 0:
        down = orig_sr // target_sr
        return resample_poly(x, 1, down)
    up = target_sr
    down = orig_sr
    return resample_poly(x, up, down)


def load_parquet_hf(
    dataset: str,
    config: str | None = None,
    split: str = "test",
    limit: int | None = None,
) -> list[Clip]:
    """Load a HuggingFace audio dataset and return clips.

    dataset is the repo id, e.g. "librispeech_asr"; config is the config name
    (e.g. "clean") when the dataset has multiple configurations.
    Requires `datasets` package (with audio decoding) and network.
    """
    import datasets as hf

    if config is not None:
        ds = hf.load_dataset(dataset, config, split=split, streaming=False)
    else:
        ds = hf.load_dataset(dataset, split=split, streaming=False)

    out: list[Clip] = []
    for row in ds:
        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        text = row.get("text") or ""
        text = (text or "").strip()
        sid = row.get("id") or f"clip-{len(out)}"
        speaker = str(row.get("speaker_id") or "")
        out.append(Clip(id=sid, audio=arr, sr=sr, text=text, speaker=speaker))
        if limit and len(out) >= limit:
            break
    return out


def load_local_wav(directory: str, limit: int | None = None) -> list[Clip]:
    """Load .wav files from a directory (each file name = id; text optional
    via a sibling `<id>.txt`). Useful for offline/custom fixtures."""
    d = Path(directory)
    clips: list[Clip] = []
    for wav in sorted(d.glob("*.wav")):
        with wave.open(str(wav), "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            n = w.getnframes()
            raw = w.readframes(n)
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        txt = ""
        txt_path = wav.with_suffix(".txt")
        if txt_path.exists():
            txt = txt_path.read_text().strip()
        clips.append(Clip(id=wav.stem, audio=x, sr=sr, text=txt))
        if limit and len(clips) >= limit:
            break
    return clips


def load_corpus(
    name: str,
    limit: int | None = None,
    directory: str | None = None,
) -> list[Clip]:
    """Load clips by a shorthand corpus name.

    name in: librispeech-test-clean, librispeech-test-other,
             common-voice-en, ljspeech, libritts, local
    """
    key = name.lower().replace("_", "-").replace("/", "-").strip()
    if key == "local" or directory:
        if not directory:
            raise ValueError("directory required for 'local' corpus")
        return load_local_wav(directory, limit)

    if key in ("librispeech-test-clean", "librispeech", "librispeech-clean"):
        return _try_load(
            ("librispeech_asr:clean", "test"),
            ("hf-internal-testing/librispeech_asr_dummy:clean", "validation"),
            limit,
        )
    if key in ("librispeech-test-other", "librispeech-other"):
        return _try_load(
            ("librispeech_asr:other", "test"),
            ("hf-internal-testing/librispeech_asr_dummy:other", "validation"),
            limit,
        )
    if key in ("common-voice-en", "common-voice", "commonvoice"):
        return _try_load(
            ("mozilla-foundation/common_voice_17_0:en", "test"),
            ("hf-internal-testing/librispeech_asr_dummy:clean", "validation"),
            limit,
        )
    if key in ("ljspeech", "lj-speech"):
        return _try_load(
            ("keithito/lj_speech", "train"),
            ("hf-internal-testing/librispeech_asr_dummy:clean", "validation"),
            limit,
        )
    if key in ("libritts", "libritts-r"):
        return _try_load(
            ("libritts_r:clean", "test"),
            ("hf-internal-testing/librispeech_asr_dummy:clean", "validation"),
            limit,
        )

    raise ValueError(f"unknown corpus: {name}")


def _try_load(primary, fallback, limit):
    """Attempt the primary HF dataset, then a fallback; raise if both fail."""
    errs = []
    for spec, split in (primary, fallback):
        if ":" in spec:
            path, config = spec.split(":", 1)
        else:
            path, config = spec, None
        try:
            return load_parquet_hf(path, config=config, split=split, limit=limit)
        except Exception as e:  # noqa: BLE001 - try next source
            errs.append(f"{path}@{split}: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("no corpus source loadable: " + " | ".join(errs))
