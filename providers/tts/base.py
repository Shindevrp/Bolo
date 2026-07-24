from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class TTSProvider(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def synthesize_stream(
        self, text_chunks: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError