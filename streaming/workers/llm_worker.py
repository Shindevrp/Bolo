from __future__ import annotations


class LLMWorker:
    """Worker responsible for language model inference."""

    def run(self, prompt: str) -> str:
        return f"LLM response for: {prompt}"
