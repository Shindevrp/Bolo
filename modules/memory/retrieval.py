from __future__ import annotations

from modules.memory.vector_db import VectorDB
from modules.memory.session import SessionMemory


class RetrievalModule:
    def __init__(
        self,
        vector_db: VectorDB | None = None,
        max_context_tokens: int = 2048,
        min_score: float = 0.3,
    ) -> None:
        self.vector_db = vector_db or VectorDB()
        self.max_context_tokens = max_context_tokens
        self.min_score = min_score

    def retrieve_context(
        self,
        query: str,
        session_memory: SessionMemory,
        top_k: int = 3,
    ) -> list[str]:
        """Vector hits only, filtered by score and deduped against recent turns.

        Recent turns are already injected as full history messages, so echoing
        them here would only duplicate context and waste tokens.
        """
        results = self.vector_db.search_scored(query, top_k=top_k * 2)
        recent = {
            e.content.strip().lower() for e in session_memory.get_history(4)
        }
        hits = [
            doc
            for doc, score in results
            if score >= self.min_score and doc.strip().lower() not in recent
        ]
        return hits[:top_k]

    def add_to_long_term(self, text: str) -> None:
        self.vector_db.add(text)

    def warm_up(self) -> None:
        self.vector_db.warm_up()
