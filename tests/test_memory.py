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

    def test_search_scored_returns_scores(self) -> None:
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
        scored = db.search_scored("weather today", top_k=2)
        by_doc = {d: s for d, s in scored}
        assert by_doc["the weather is nice"] == pytest.approx(1.0)
        assert by_doc["I like pizza"] == pytest.approx(0.0)


class TestRetrievalModule:
    def test_retrieve_context_no_recent_echo(self) -> None:
        rm = RetrievalModule()
        sm = SessionMemory()
        sm.add("user", "hello")
        ctx = rm.retrieve_context("hello", sm, top_k=1)
        assert not any("[Recent]" in c for c in ctx)

    def test_retrieve_context_score_and_dedup(self) -> None:
        rm = RetrievalModule(min_score=0.5)
        sm = SessionMemory()

        class FakeEncoder:
            def encode(self, text, **kw):
                import numpy as np
                if "weather" in text.lower():
                    return np.array([1, 0, 0, 0])
                return np.array([0, 1, 0, 0])

            def get_sentence_embedding_dimension(self):
                return 4

        rm.vector_db._encoder = FakeEncoder()
        rm.vector_db.add("the weather is nice")
        rm.vector_db.add("I like pizza")
        sm.add("user", "I like pizza")  # also in recent -> deduped
        hits = rm.retrieve_context("weather today", sm, top_k=2)
        assert all("pizza" not in h for h in hits)
        assert len(hits) == 1
        assert "weather" in hits[0].lower()