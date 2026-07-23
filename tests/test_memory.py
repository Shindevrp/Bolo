from __future__ import annotations

import pytest

from modules.memory.session import SessionMemory
from modules.memory.vector_db import VectorDB
from modules.memory.retrieval import RetrievalModule


class TestSessionMemory:
    def test_add_and_retrieve(self) -> None:
        m = SessionMemory(max_turns=3)
        m.add("user", "hello")
        m.add("assistant", "hi")
        hist = m.get_history()
        assert len(hist) == 2
        assert hist[0].role == "user"
        assert hist[1].role == "assistant"

    def test_max_turns(self) -> None:
        m = SessionMemory(max_turns=2)
        m.add("user", "a")
        m.add("assistant", "b")
        m.add("user", "c")
        hist = m.get_history()
        assert len(hist) == 2
        assert hist[0].content == "b"
        assert hist[1].content == "c"

    def test_context_messages(self) -> None:
        m = SessionMemory()
        m.add("user", "hello")
        msgs = m.context_messages(system_prompt="You are TASA.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestVectorDB:
    def test_search(self) -> None:
        db = VectorDB(embedding_dim=4)

        class FakeEncoder:
            def encode(self, text, **kw):
                import numpy as np
                if "weather" in text.lower():
                    return np.array([1, 0, 0, 0])
                return np.array([0, 1, 0, 0])

            def get_sentence_embedding_dimension(self):
                return 4

        db._encoder = FakeEncoder()
        db.add("the weather is nice")
        db.add("I like pizza")
        results = db.search("weather today", top_k=1)
        assert "weather" in results[0].lower()


class TestRetrievalModule:
    def test_retrieve_context(self) -> None:
        rm = RetrievalModule()
        sm = SessionMemory()
        sm.add("user", "hello")
        ctx = rm.retrieve_context("hello", sm, top_k=1)
        assert any("[Recent]" in c for c in ctx)