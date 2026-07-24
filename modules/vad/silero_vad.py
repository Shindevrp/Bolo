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
    ) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * frame_ms // 1000
        self.model = self._load_model(model_path)
        self._state = None

    def _load_model(self, model_path: str | None) -> torch.nn.Module:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        model.eval()
        return model

    def reset(self) -> None:
        self._state = None

    def is_speech(self, audio_chunk: bytes) -> bool:
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        with torch.no_grad():
            speech_prob, self._state = self.model(
                audio_tensor, self.sample_rate, state=self._state
            )
            speech_prob = speech_prob.item()
        return speech_prob >= self.threshold