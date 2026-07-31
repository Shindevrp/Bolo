from __future__ import annotations

from modules.memory.vector_db import VectorDB
from modules.memory.session import SessionMemory


class RetrievalModule:
    def __init__(
        self,
        vector_db: VectorDB | None = None,
        max_context_tokens: int = 2048,
    ) -> None:
        self.vector_db = vector_db or VectorDB()
        self.max_context_tokens = max_context_tokens

    def retrieve_context(
        self,
        query: str,
        session_memory: SessionMemory,
        top_k: int = 3,
    ) -> list[str]:
        results = self.vector_db.search(query, top_k=top_k)
        recent = [e.content for e in session_memory.get_history(4)]
        combined = results + [f"[Recent] {r}" for r in recent]
        return combined

    def add_to_long_term(self, text: str) -> None:
        self.vector_db.add(text)

    def warm_up(self) -> None:
        self.vector_db.warm_up()