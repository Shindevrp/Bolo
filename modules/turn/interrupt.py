from __future__ import annotations


class InterruptHandler:
    def __init__(
        self,
        speech_energy_threshold: float = 0.01,
        silence_confidence_threshold: float = 0.3,
        consecutive_speech_frames: int = 3,
    ) -> None:
        self.speech_energy_threshold = speech_energy_threshold
        self.silence_confidence_threshold = silence_confidence_threshold
        self.consecutive_speech_frames = consecutive_speech_frames
        self._speech_frame_count = 0

    def should_interrupt(
        self, energy: float, silence_duration: float, is_speaking: bool
    ) -> bool:
        if not is_speaking:
            return False

        if energy > self.speech_energy_threshold:
            self._speech_frame_count += 1
        else:
            self._speech_frame_count = 0

        if self._speech_frame_count >= self.consecutive_speech_frames:
            self._speech_frame_count = 0
            return True

        return False

    def reset(self) -> None:
        self._speech_frame_count = 0