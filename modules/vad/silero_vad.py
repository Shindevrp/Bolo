from __future__ import annotations

import numpy as np
import torch

from utils.logger import logger


class SileroVAD:
    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * frame_ms // 1000
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str | None) -> torch.nn.Module:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        model.eval()
        model.to(self.device)
        return model

    def reset(self) -> None:
        self._state = None
        reset_fn = getattr(self.model, "reset_states", None)
        if callable(reset_fn):
            reset_fn()

    def speech_probability(self, audio_chunk: bytes) -> float:
        """Return the max VAD speech probability over the chunk's sub-frames."""
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        num_samples = 512 if self.sample_rate == 16000 else 256

        if len(audio) < num_samples:
            return 0.0

        best = 0.0
        with torch.no_grad():
            for start in range(0, len(audio), num_samples):
                frame = audio[start:start + num_samples]
                if len(frame) < num_samples:
                    break
                audio_tensor = torch.from_numpy(frame).unsqueeze(0).to(self.device)
                prob = self.model(audio_tensor, self.sample_rate)
                if isinstance(prob, tuple):
                    prob = prob[0]
                best = max(best, float(prob.item()))
        return best

    def is_speech(self, audio_chunk: bytes) -> bool:
        return self.speech_probability(audio_chunk) >= self.threshold


class HysteresisVAD:
    """Stateful VAD wrapper around a base VAD that adds production-grade
    hysteresis/debounce to fix flickery speech/no-speech endpointing.

    Independent tunables:
      - onset_threshold      : probability to *start* treating as speech
      - offset_threshold     : probability (lower) to carry on being speech
      - onset_confirmation   : consecutive speech frames required to enter ON
      - offset_confirmation  : consecutive silence frames required to leave ON
      - hangover_frames      : speech frames kept alive after last true speech
      - min_speech_frames    : minimum speech run length before ON is 'armed'
      - min_silence_frames   : minimum silence run before reporting continued
                               silence (guard against micro-dips)
      - ema_alpha            : smoothing of per-frame probability
    """

    def __init__(
        self,
        base_vad: SileroVAD,
        sample_rate: int = 16000,
        onset_threshold: float = 0.60,
        offset_threshold: float = 0.35,
        onset_confirmation: int = 2,
        offset_confirmation: int = 2,
        hangover_frames: int = 2,
        min_speech_frames: int = 1,
        min_silence_frames: int = 2,
        ema_alpha: float = 0.5,
    ) -> None:
        self.base_vad = base_vad
        self.sample_rate = sample_rate
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.onset_confirmation = onset_confirmation
        self.offset_confirmation = offset_confirmation
        self.hangover_frames = hangover_frames
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.ema_alpha = ema_alpha
        self._smooth_prob = 0.0
        self._seen_frames = 0
        self._state = "OFF"  # OFF -> ARMED -> ON
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._hangover = 0
        self._speech_frames = 0

    def reset(self) -> None:
        self._smooth_prob = 0.0
        self._seen_frames = 0
        self._state = "OFF"
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._hangover = 0
        self._speech_frames = 0

    def speech_probability(self, audio_chunk: bytes) -> float:
        raw = self.base_vad.speech_probability(audio_chunk)
        self._seen_frames += 1
        if self._seen_frames == 1:
            self._smooth_prob = raw
        else:
            self._smooth_prob = (
                self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._smooth_prob
            )
        return self._smooth_prob

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Return the *debounced* speech decision for one audio frame."""
        prob = self.speech_probability(audio_chunk)
        return self._update_state(prob)

    def _update_state(self, prob: float) -> bool:
        if self._state == "OFF":
            if prob >= self.onset_threshold:
                self._consecutive_speech += 1
                if self._consecutive_speech >= self.onset_confirmation:
                    self._state = "ARMED"
                    self._consecutive_speech = 0
                    self._speech_frames = 0
            else:
                self._consecutive_speech = 0
            return False

        if self._state == "ARMED":
            # ARMED: track whether this run reaches min speech length.
            if prob >= self.offset_threshold:
                self._speech_frames += 1
                self._hangover = self.hangover_frames
                self._consecutive_silence = 0
                if self._speech_frames >= self.min_speech_frames:
                    self._state = "ON"
                return self._speech_frames >= self.min_speech_frames
            self._consecutive_silence += 1
            if self._consecutive_silence >= self.onset_confirmation:
                # run fizzled out before reaching min speech
                self._state = "OFF"
                self._consecutive_silence = 0
                self._speech_frames = 0
            return False

        # state == "ON"
        if prob >= self.offset_threshold:
            self._speech_frames += 1
            self._hangover = self.hangover_frames
            self._consecutive_silence = 0
            return True

        # below offset threshold: hangover + silence debounce
        if self._hangover > 0:
            self._hangover -= 1
            return True
        self._consecutive_silence += 1
        if self._consecutive_silence >= self.offset_confirmation:
            self._state = "OFF"
            self._consecutive_silence = 0
            self._speech_frames = 0
        return False

    def is_continuously_silent(self, silence_frames: int) -> bool:
        """True once the VAD has been free of speech for >= silence_frames."""
        return self._state == "OFF" and self._consecutive_silence >= silence_frames

    @property
    def state(self) -> str:
        return self._state