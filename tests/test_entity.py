from __future__ import annotations

import asyncio

from modules.turn.entity import EntityGate


class TestQueryNeedsVerification:
    def test_restaurant_query_is_flagged(self) -> None:
        assert EntityGate.query_needs_verification("any good restaurants near me?")

    def test_place_query_is_flagged(self) -> None:
        assert EntityGate.query_needs_verification("recommend a cafe in Pune")

    def test_general_chat_not_flagged(self) -> None:
        assert not EntityGate.query_needs_verification("how are you today?")
        assert not EntityGate.query_needs_verification("tell me a joke")


class TestExtractEntities:
    def test_captures_capitalized_name(self) -> None:
        ents = EntityGate.extract_entities("There's a great place called Happy Tiffins")
        assert any("Happy Tiffins" in e for e in ents)

    def test_extracts_multiple_names(self) -> None:
        ents = EntityGate.extract_entities("Try Spice Garden or Cafe Mocha")
        joined = " ".join(ents)
        assert "Spice Garden" in joined
        assert "Cafe Mocha" in joined


class TestUnsupportedEntities:
    def test_supported_entity_not_blocked(self) -> None:
        blocked = EntityGate.unsupported_entities(
            "Go to Happy Tiffins near you.", {"Happy Tiffins"}
        )
        assert not blocked

    def test_user_provided_entity_not_blocked(self) -> None:
        blocked = EntityGate.unsupported_entities(
            "Yes, I can help with Kullavi Kalyan.",
            set(),
            user_input_entities={"Kullavi Kalyan"},
        )
        assert not blocked

    def test_unverified_entity_blocked(self) -> None:
        blocked = EntityGate.unsupported_entities(
            "Try the new place called Bijli Bazaar.", set()
        )
        assert any("Bijli Bazaar" in b for b in blocked)


class TestSafetyResponse:
    def test_single_entity_wording(self) -> None:
        assert "not certain that place actually exists" in EntityGate.safety_response(
            ["Lau Bandrey"]
        )

    def test_multi_entity_wording(self) -> None:
        r = EntityGate.safety_response(["A", "B"])
        assert "not sure all of those places" in r


class TestPipelineEntityGuard:
    def _gate(self) -> EntityGate:
        return EntityGate()

    def test_verification_tracks_tool_results_as_supported(self) -> None:
        """Tool results for an entity query become supported so the verified
        answer is not flagged as fabrication."""
        from core.pipeline import StreamingPipeline
        from tests.test_streaming import (
            FakeSTT,
            FakeLLM,
            FakeTTS,
            FakeVAD,
            ConversationContext,
        )
        from modules.memory.session import SessionMemory

        async def run() -> list[str]:
            p = StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), FakeVAD())
            ctx = ConversationContext()
            sm = SessionMemory()
            results = [
                {"tool": "search_web", "result": "Happy Tiffins is a popular cafe."}
            ]
            supported: set[str] = set(EntityGate.extract_entities("any good restaurants near me"))
            for tr in results:
                supported.update(EntityGate.extract_entities(str(tr["result"])))
            fabrications = p._verify_entity_assertions(
                "any good restaurants near me",
                "Try Happy Tiffins, it's great.",
                supported,
            )
            return fabrications

        assert asyncio.run(run()) == []

    def test_unsupported_fabrication_flagged(self) -> None:
        from core.pipeline import StreamingPipeline
        from tests.test_streaming import (
            FakeSTT,
            FakeLLM,
            FakeTTS,
            FakeVAD,
            ConversationContext,
        )

        async def run() -> list[str]:
            p = StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), FakeVAD())
            ctx = ConversationContext()
            fabrications = p._verify_entity_assertions(
                "any good restaurants near me",
                "Try Bajji Anagar, it's the best.",
                set(),
            )
            return fabrications

        result = asyncio.run(run())
        assert any("Bajji Anagar" in b for b in result)

    def test_non_entity_query_not_gated(self) -> None:
        from core.pipeline import StreamingPipeline, ConversationContext
        from tests.test_streaming import FakeSTT, FakeLLM, FakeTTS, FakeVAD

        async def run() -> list[str]:
            p = StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), FakeVAD())
            fabrications = p._verify_entity_assertions(
                "how are you",
                "I'm doing great, thanks.",
                set(),
            )
            return fabrications

        assert asyncio.run(run()) == []


class TestEndToEndEntityGuard:
    def test_fabricated_place_name_blocked_from_memory(self) -> None:
        from core.pipeline import (
            StreamingPipeline,
            ConversationContext,
            PipelineEvent,
        )
        from modules.memory.session import SessionMemory
        from tests.test_streaming import (
            FakeSTTText,
            FakeTTSStream,
            FakeVADTrue,
        )

        class EntityQuerySTT:
            async def transcribe(self, audio_blob: bytes) -> str:
                return "is there a good restaurant near me?"

        class FabricatingLLM:
            async def generate_stream(self, messages):
                for token in [
                    "You should try ",
                    "Bajji Anagar",
                    ", it's the best place in town.",
                ]:
                    yield token

        async def run() -> tuple[str, list]:
            p = StreamingPipeline(EntityQuerySTT(), FabricatingLLM(), FakeTTSStream(), FakeVADTrue())
            sm = SessionMemory()
            p._memories["sess"] = sm
            q = p._session_output("sess")
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )

            assistant_entries = [e.content for e in sm.entries if e.role == "assistant"]
            events = []
            while not q.empty():
                try:
                    msg = q.get_nowait()
                    events.append(msg.event)
                except Exception:
                    break

            recorded = " ".join(assistant_entries)
            return recorded, events

        recorded, events = asyncio.run(run())
        assert "Bajji Anagar" not in recorded
        assert any(e == PipelineEvent.ENTITY_GUARD for e in events)
        assert "not certain that place actually exists" in recorded


class TestLiveFalsePositives:
    """Regressions from a live session: ordinary replies were replaced with
    "I'm not certain that place exists"."""

    def test_sentence_opening_words_are_not_names(self) -> None:
        for text in (
            "Based on the weather data, it's clear. Winds are light. Would you like more?",
            "For the current population, let me check. Could you repeat that?",
        ):
            assert EntityGate.extract_entities(text) == [], text

    def test_two_word_name_opening_a_sentence_still_counts(self) -> None:
        assert EntityGate.extract_entities("Paradise Biryani is great.") == ["Paradise Biryani"]

    def test_partial_name_backed_by_tool_text(self) -> None:
        result = "Paradise Biryani, 4.1 stars | Shah Ghouse, 4.0 stars"
        assert EntityGate.unsupported_entities("I'd go to Paradise first.", {result}) == []
        assert EntityGate.unsupported_entities("Try Bawarchi House.", {result}) == ["Bawarchi House"]

    def test_query_words_match_whole_words(self) -> None:
        assert not EntityGate.query_needs_verification("where is parking allowed")
        assert EntityGate.query_needs_verification("where's a good park")


class TestToolMarkerFilter:
    def _run(self, tokens, tool_names=frozenset()):
        from modules.tools.registry import ToolMarkerFilter

        f = ToolMarkerFilter(tool_names)
        out = "".join(f.feed(t) for t in tokens) + f.flush()
        return out, f.seen

    def test_split_marker_is_held_back(self) -> None:
        assert self._run(["Hi. ", "{", "to", "ol", ":x()}"]) == ("Hi. ", True)

    def test_brace_that_is_not_a_tool_is_released(self) -> None:
        assert self._run(["a {", "b} c"]) == ("a {b} c", False)
        assert self._run(["ends with {to"]) == ("ends with {to", False)

    def test_dropped_prefix_with_parens_is_held_back(self) -> None:
        # No "tool:" prefix at all -- open-paren syntax alone is enough,
        # even for a name the filter doesn't know about.
        assert self._run(["Hi. ", "{get_weather(Hyder", "abad)}"]) == ("Hi. ", True)

    def test_dropped_prefix_and_parens_needs_known_name(self) -> None:
        assert self._run(
            ["{", "get_weather", "}"], tool_names=frozenset({"get_weather"})
        ) == ("", True)
        # Same bare shape, unregistered name: not confident enough, release.
        assert self._run(["{", "x", "}"]) == ("{x}", False)
