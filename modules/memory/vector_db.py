from __future__ import annotations

import numpy as np


class VectorDB:
    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self.documents: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self._encoder = None

    def _lazy_load_encoder(self) -> None:
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        try:
            self.embedding_dim = self._encoder.get_embedding_dimension()
        except AttributeError:
            self.embedding_dim = self._encoder.get_sentence_embedding_dimension()

    def warm_up(self) -> None:
        self._lazy_load_encoder()

    def add(self, text: str) -> None:
        self._lazy_load_encoder()
        emb = self._encoder.encode(text, normalize_embeddings=True)
        self.documents.append(text)
        self.embeddings.append(emb)

    def search_scored(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.documents:
            return []
        self._lazy_load_encoder()
        query_emb = self._encoder.encode(query, normalize_embeddings=True)
        scores = [float(np.dot(query_emb, doc_emb)) for doc_emb in self.embeddings]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_indices]

    def search(self, query: str, top_k: int = 3) -> list[str]:
        return [doc for doc, _ in self.search_scored(query, top_k)]