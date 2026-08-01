from __future__ import annotations

import threading

import numpy as np

_encoder = None
_encoder_lock = threading.Lock()


def _get_encoder():
    """Module-level singleton sentence encoder shared across VectorDBs.

    The encoder (~90MB) is expensive to load and holds the GIL during
    model/tokenizer init. Loading it once and warming it at server startup
    keeps it off the per-session audio hot path.
    """
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from sentence_transformers import SentenceTransformer

                _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


class VectorDB:
    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self.documents: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self._encoder = None

    def _lazy_load_encoder(self):
        if self._encoder is None:
            self._encoder = _get_encoder()
        return self._encoder

    def warm_up(self) -> None:
        self._lazy_load_encoder()

    def add(self, text: str) -> None:
        encoder = self._lazy_load_encoder()
        emb = encoder.encode(text, normalize_embeddings=True)
        self.documents.append(text)
        self.embeddings.append(emb)

    def search_scored(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.documents:
            return []
        encoder = self._lazy_load_encoder()
        query_emb = encoder.encode(query, normalize_embeddings=True)
        scores = [float(np.dot(query_emb, doc_emb)) for doc_emb in self.embeddings]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_indices]

    def search(self, query: str, top_k: int = 3) -> list[str]:
        return [doc for doc, _ in self.search_scored(query, top_k)]
