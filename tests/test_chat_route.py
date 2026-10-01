"""Tests for the HTTP chat path in ``app/routes/chat.py``.

That route used to stream straight from the LLM, so it never executed a tool
and never kept history: it would invent places that SerpApi never returned,
and a follow-up like "anything in the morning?" had nothing to refer back to.
These tests pin the two behaviours that fix.

SerpApi is replayed from fixtures; nothing here touches the network.
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routes import chat as chat_mod  # noqa: E402
from modules.memory.session import SessionMemory  # noqa: E402
from providers.search.serpapi import (  # noqa: E402
    SerpApiClient,
    set_client,
)
from tests.test_serpapi import Replay  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Returns scripted replies in order; the last one repeats."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []
        self.calls = 0

    async def generate_stream(self, messages):
        self.seen.append(list(messages))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        for tok in reply.split(" "):
            yield tok + " "


def _pipeline(llm):
    return types.SimpleNamespace(llm=llm)


@pytest.fixture(autouse=True)
def _replay_serpapi():
    client = SerpApiClient(
        "test-key", transport=httpx.MockTransport(Replay())
    )
    set_client(client)
    chat_mod._sessions.clear()
    yield
    set_client(None)
    chat_mod._sessions.clear()


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def test_tool_call_is_executed_and_stripped_from_the_answer():
    """A {tool:...} reply must never reach the user verbatim."""
    llm = FakeLLM([
        "{tool:search_places(query=restaurants, location=Mumbai)}",
        "There's a great biryani place in Mumbai.",
    ])
    answer = asyncio.run(
        chat_mod._tool_rounds(
            _pipeline(llm), llm, SessionMemory(), "where should I eat?", _no_emit
        )
    )
    assert "{" not in answer
    assert "tool:" not in answer
    assert answer.strip().startswith("There's a great biryani place")
    assert llm.calls == 2


def test_place_answer_is_backed_by_a_search_result():
    """The second turn must be able to speak names a SerpApi result returned.

    Uses the recorded ``biryani in Hyderabad`` fixture, which is the only
    local_results fixture; anything else replays as no-results.
    """
    llm = FakeLLM([
        "{tool:search_places(query=biryani in Hyderabad)}",
        "There's a biryani place in Hyderabad.",
    ])
    memory = SessionMemory()
    asyncio.run(
        chat_mod._tool_rounds(
            _pipeline(llm), llm, memory, "biryani in Hyderabad", _no_emit
        )
    )
    tool_entries = [e for e in memory.entries if e.role == "tool"]
    assert tool_entries, "tool result must be recorded in memory"
    assert "No results" not in tool_entries[0].content
    assert tool_entries[0].content != ""


def test_no_tool_call_streams_the_answer_directly():
    llm = FakeLLM(["Hello, how can I help?"])
    answer = asyncio.run(
        chat_mod._tool_rounds(
            _pipeline(llm), llm, SessionMemory(), "hi", _no_emit
        )
    )
    assert llm.calls == 1
    assert "Hello" in answer


def test_tool_round_loop_is_bounded():
    """A model that keeps emitting tool calls must not loop forever."""
    llm = FakeLLM(["{tool:search_web(query=loop)}"])
    answer = asyncio.run(
        chat_mod._tool_rounds(
            _pipeline(llm), llm, SessionMemory(), "loop", _no_emit
        )
    )
    assert llm.calls == 3
    assert "{" not in answer


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_followup_turn_receives_prior_history():
    llm = FakeLLM(["Flights leave hourly from DEL."])
    memory = SessionMemory()
    asyncio.run(
        chat_mod._tool_rounds(
            _pipeline(llm), llm, memory, "flights DEL to BOM?", _no_emit
        )
    )
    llm2 = FakeLLM(["Yes, morning flights exist too."])
    asyncio.run(
        chat_mod._tool_rounds(
            _pipeline(llm2), llm2, memory, "anything in the morning?", _no_emit
        )
    )
    second_prompt = json.dumps(llm2.seen[0])
    assert "flights DEL to BOM?" in second_prompt
    assert "leave hourly" in second_prompt


def test_system_prompt_includes_the_tool_protocol():
    llm = FakeLLM(["ok"])
    memory = SessionMemory()
    asyncio.run(chat_mod._tool_rounds(_pipeline(llm), llm, memory, "hi", _no_emit))
    system = llm.seen[0][0]
    assert system["role"] == "system"
    assert "{tool:" in system["content"]
    assert "search_places" in system["content"]


def test_sessions_are_reused_per_id():
    a = chat_mod._memory_for("s1")
    a.add("user", "first")
    b = chat_mod._memory_for("s1")
    assert b is a
    c = chat_mod._memory_for("s2")
    assert c is not a


def test_idle_sessions_are_trimmed():
    chat_mod._sessions["old"] = (SessionMemory(), 0.0)
    chat_mod._sessions["new"] = (SessionMemory(), 1e12)
    chat_mod._trim("new")
    assert "old" not in chat_mod._sessions
    assert "new" in chat_mod._sessions


async def _no_emit(_token: str) -> None:
    return None