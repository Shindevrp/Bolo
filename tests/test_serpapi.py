"""SerpApi client, tool-call parsing and SerpApi-backed tools.

Every request is answered by ``httpx.MockTransport`` from the recorded
fixtures in ``tests/fixtures/serpapi`` -- nothing touches the network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from modules.tools import local, news, web_search
from modules.tools.builtin import get_builtin_tools
from modules.tools.registry import ToolRegistry, ToolSpec, parse_args
from providers.search.serpapi import (
    SerpApiClient,
    SerpApiError,
    SerpApiUnavailable,
    fixture_name,
    normalize_query,
    set_client,
)

FIXTURES = Path(__file__).parent / "fixtures" / "serpapi"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _index() -> dict[tuple[str, str], dict]:
    """(engine, q) -> fixture body, keyed from each fixture's own params."""
    out = {}
    for path in FIXTURES.glob("*.json"):
        data = json.loads(path.read_text())
        sp = data["search_parameters"]
        out[(sp["engine"], normalize_query(sp.get("q", "")))] = data
    return out


class Replay:
    """MockTransport handler serving fixtures; records every request."""

    def __init__(self, override: dict | None = None, status: int = 200) -> None:
        self.fixtures = _index()
        self.override = override
        self.status = status
        self.requests: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.requests.append(params)
        if self.override is not None:
            return httpx.Response(self.status, json=self.override)
        body = self.fixtures.get((params["engine"], params.get("q", "")))
        if body is None:
            body = {"error": "Google hasn't returned any results for this query."}
        return httpx.Response(self.status, json=body)


def _client(handler=None, **kw) -> tuple[SerpApiClient, Replay]:
    replay = handler or Replay()
    client = SerpApiClient(
        "test-key", transport=httpx.MockTransport(replay), **kw
    )
    return client, replay


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TestClient:
    def test_search_sends_engine_key_and_params(self) -> None:
        client, replay = _client()
        data = run(client.search("google", q="best time to visit goa", gl="in", hl="en"))
        assert data["organic_results"][0]["title"] == "Best Time to Visit Goa"
        req = replay.requests[0]
        assert req["engine"] == "google"
        assert req["api_key"] == "test-key"
        assert req["gl"] == "in"

    def test_none_params_are_dropped(self) -> None:
        client, replay = _client(Replay({"news_results": []}))
        run(client.search("google_news", q=None, gl="in"))
        assert "q" not in replay.requests[0]

    def test_cache_hit_costs_no_credit(self) -> None:
        client, replay = _client()
        for _ in range(3):
            run(client.search("google", q="best time to visit goa"))
        assert len(replay.requests) == 1
        assert client.credits_used == 1
        assert client.cache_hits == 2

    def test_cache_expires(self) -> None:
        client, replay = _client(cache_ttl=0)
        run(client.search("google", q="best time to visit goa"))
        run(client.search("google", q="best time to visit goa"))
        assert len(replay.requests) == 2

    def test_credit_cap(self) -> None:
        client, replay = _client(credit_cap=1)
        run(client.search("google", q="best time to visit goa"))
        assert not client.available
        with pytest.raises(SerpApiUnavailable, match="credit cap"):
            run(client.search("google", q="who is aravind srinivas"))
        assert len(replay.requests) == 1

    def test_no_key_is_unavailable(self) -> None:
        client = SerpApiClient("")
        assert not client.enabled
        with pytest.raises(SerpApiUnavailable):
            run(client.search("google", q="x"))

    def test_api_error_message_is_bad_result(self) -> None:
        client, _ = _client(Replay({"error": "Invalid API key."}, status=401))
        with pytest.raises(SerpApiError) as e:
            run(client.search("google", q="x"))
        assert ToolRegistry._result_is_bad(str(e.value))
        assert "Invalid API key" in str(e.value)

    def test_empty_results_are_no_results(self) -> None:
        client, _ = _client()
        with pytest.raises(SerpApiError, match="No results found"):
            run(client.search("google", q="zxqv nonsense query"))

    def test_timeout(self) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        client, _ = _client(slow)
        with pytest.raises(SerpApiError) as e:
            run(client.search("google", q="x"))
        assert "timed out" in str(e.value)
        assert ToolRegistry._result_is_bad(str(e.value))

    def test_errors_are_not_cached(self) -> None:
        client, replay = _client(Replay({"error": "boom"}))
        for _ in range(2):
            with pytest.raises(SerpApiError):
                run(client.search("google", q="x"))
        assert len(replay.requests) == 2

    def test_record_dir_writes_fixture_without_metadata(self, tmp_path) -> None:
        body = {"search_metadata": {"json_endpoint": "secret"}, "organic_results": []}
        client, _ = _client(Replay(body), record_dir=tmp_path)
        run(client.search("google", q="Hello World"))
        files = list(tmp_path.glob("*.json"))
        assert [f.name for f in files] == [fixture_name("google", {"q": "hello world"})]
        saved = json.loads(files[0].read_text())
        assert "search_metadata" not in saved
        assert saved["search_parameters"] == {"engine": "google", "q": "hello world"}
        assert "test-key" not in files[0].read_text()

    def test_query_normalisation_shares_one_credit(self) -> None:
        client, replay = _client()
        run(client.search("google", q="Best time to visit Goa?"))
        run(client.search("google", q="  best  time to visit goa "))
        assert len(replay.requests) == 1
        assert replay.requests[0]["q"] == "best time to visit goa"

    def test_empty_result_is_negatively_cached(self) -> None:
        client, replay = _client()
        for _ in range(2):
            with pytest.raises(SerpApiError, match="No results found"):
                run(client.search("google", q="zxqv nonsense query"))
        assert len(replay.requests) == 1
        assert client.credits_used == 1

    def test_fatal_error_disables_client(self) -> None:
        client, replay = _client(Replay({"error": "Your account has run out of searches."}))
        with pytest.raises(SerpApiError):
            run(client.search("google", q="a"))
        assert not client.available
        with pytest.raises(SerpApiUnavailable):
            run(client.search("google", q="b"))
        assert len(replay.requests) == 1

    def test_rate_limit_is_not_fatal(self) -> None:
        client, _ = _client(Replay({"error": "Rate limit exceeded for account."}, status=429))
        with pytest.raises(SerpApiError):
            run(client.search("google", q="a"))
        assert client.available

    def test_concurrent_identical_requests_share_one_call(self) -> None:
        client, replay = _client()

        async def both():
            return await asyncio.gather(
                client.search("google", q="best time to visit goa"),
                client.search("google", q="Best time to visit Goa"),
            )

        a, b = run(both())
        assert a is b
        assert len(replay.requests) == 1
        assert client.dedup_hits == 1

    def test_cancelled_caller_does_not_cancel_shared_fetch(self) -> None:
        release = asyncio.Event()

        class Slow(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                await release.wait()
                return httpx.Response(200, json={"organic_results": [{"title": "t"}]})

        client = SerpApiClient("k", transport=Slow())

        async def scenario():
            prefetch = asyncio.create_task(client.search("google", q="x"))
            await asyncio.sleep(0)
            turn = asyncio.create_task(client.search("google", q="x"))
            await asyncio.sleep(0)
            prefetch.cancel()  # speech-end cancels the prefetch probe
            release.set()
            data = await turn
            return data, prefetch

        data, prefetch = run(scenario())
        assert data["organic_results"][0]["title"] == "t"
        assert prefetch.cancelled()
        assert client.credits_used == 1

    def test_connection_is_reused(self) -> None:
        client, _ = _client()

        async def two():
            await client.search("google", q="best time to visit goa")
            first = client._http
            await client.search("google", q="who is aravind srinivas")
            assert client._http is first
            await client.aclose()

        run(two())

    def test_cache_is_bounded(self, monkeypatch) -> None:
        from providers.search import serpapi

        monkeypatch.setattr(serpapi, "CACHE_MAX_ENTRIES", 2)
        client, _ = _client(Replay({"organic_results": []}))
        for q in ("a", "b", "c"):
            run(client.search("google", q=q))
        assert len(client._cache) == 2

    def test_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", " k ")
        monkeypatch.setenv("SERPAPI_CREDIT_CAP", "7")
        client = SerpApiClient.from_env()
        assert client.api_key == "k"
        assert client.credit_cap == 7


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------


class TestCallParsing:
    reg = ToolRegistry()

    def test_positional_unchanged(self) -> None:
        assert self.reg.find_calls("{tool:get_weather(Hyderabad)}") == [
            {"name": "get_weather", "args": ["Hyderabad"]}
        ]
        assert self.reg.find_calls("{tool:set_reminder(buy milk, 10)}") == [
            {"name": "set_reminder", "args": ["buy milk", "10"]}
        ]

    def test_kwargs(self) -> None:
        calls = self.reg.find_calls(
            "{tool:search_flights(from=DEL, to=BOM, date=2026-10-12)}"
        )
        assert calls == [{
            "name": "search_flights",
            "args": [],
            "kwargs": {"from": "DEL", "to": "BOM", "date": "2026-10-12"},
        }]

    def test_mixed_and_quoted(self) -> None:
        assert parse_args('biryani, location="Bandra, Mumbai"') == (
            ["biryani"], {"location": "Bandra, Mumbai"}
        )

    def test_json_object_arg(self) -> None:
        calls = self.reg.find_calls('{tool:search_flights({"from": "DEL", "to": "BOM"})}')
        assert calls[0]["kwargs"] == {"from": "DEL", "to": "BOM"}
        assert calls[0]["args"] == []

    def test_nested_parens_and_comparisons(self) -> None:
        assert self.reg.find_calls("{tool:calculate(max(1, 2) * 3)}")[0]["args"] == [
            "max(1, 2) * 3"
        ]
        assert self.reg.find_calls("{tool:calculate(2==2)}")[0]["args"] == ["2==2"]

    def test_apostrophes_and_unbalanced_quotes(self) -> None:
        assert self.reg.find_calls("{tool:search_web(what's the capital of France)}")[0][
            "args"
        ] == ["what's the capital of France"]
        assert self.reg.find_calls("{tool:search_web('90s music)}")[0]["args"] == [
            "90s music"
        ]

    def test_incomplete_call_not_found(self) -> None:
        assert self.reg.find_calls('{tool:search_flights({"from": "DEL"') == []

    def test_multiple_calls_and_strip(self) -> None:
        text = 'a {tool:x(1)} b {tool:y({"k": "}"})} c'
        assert [c["name"] for c in self.reg.find_calls(text)] == ["x", "y"]
        assert self.reg.strip_calls(text) == "a  b  c"

    def test_execute_passes_kwargs(self) -> None:
        reg = ToolRegistry()
        reg.register(ToolSpec(
            name="f", description="", parameters={},
            handler=lambda q, location="": f"{q}@{location}",
        ))
        call = reg.find_calls("{tool:f(tea, location=Pune)}")[0]
        assert run(reg.execute_call(call))["result"] == "tea@Pune"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@pytest.fixture
def serp():
    client, replay = _client()
    set_client(client)
    return replay


class TestSearchWeb:
    def test_answer_box_first_then_two_organic(self, serp) -> None:
        out = run(web_search.search_web("height of qutub minar"))
        parts = out.split(" | ")
        assert parts[0] == "Qutub Minar height: 72.5 m"
        assert len(parts) == 3
        assert "(en.wikipedia.org)" in parts[1]
        assert parts[2].startswith("Qutub Minar - ASI (asi.nic.in)")
        assert "Should not be read out" not in out

    def test_knowledge_graph(self, serp) -> None:
        out = run(web_search.search_web("who is aravind srinivas"))
        assert out.startswith("Aravind Srinivas (Computer scientist): Aravind Srinivas is")
        assert "Should not be read out" not in out

    def test_organic_only_takes_three(self, serp) -> None:
        out = run(web_search.search_web("best time to visit goa"))
        assert len(out.split(" | ")) == 3
        assert "goa-tourism.com" in out

    def test_ai_overview_leads_when_no_answer_box(self, serp) -> None:
        """Recorded live response: no answer box or knowledge graph, but an
        inlined AI overview with the direct answer."""
        out = run(web_search.search_web("population of pune"))
        parts = out.split(" | ")
        assert parts[0].startswith("The estimated population of Pune city in 2026 is approximately 4.7 million")
        assert parts[0].endswith("(census2011.co.in)")
        assert len(parts) == 3  # overview + two organic

    def test_ai_overview_page_token_only_is_skipped(self) -> None:
        assert web_search._ai_overview({"page_token": "x"}) == ""

    def test_sends_india_locale(self, serp) -> None:
        run(web_search.search_web("best time to visit goa"))
        assert serp.requests[0]["gl"] == "in"
        assert serp.requests[0]["hl"] == "en"

    def test_falls_back_without_key(self, monkeypatch) -> None:
        set_client(SerpApiClient(""))
        monkeypatch.setattr(web_search, "_ddg_instant", lambda q: ["ddg says hi"])
        assert run(web_search.search_web("anything")) == "ddg says hi"

    def test_falls_back_on_serpapi_error(self, serp, monkeypatch) -> None:
        monkeypatch.setattr(web_search, "_ddg_instant", lambda q: [])
        monkeypatch.setattr(web_search, "_wiki_search", lambda q: "wiki says hi")
        assert run(web_search.search_web("zxqv nonsense query")) == "wiki says hi"


class TestSearchPlaces:
    def test_top_three(self, serp) -> None:
        out = run(local.search_places("biryani", "Hyderabad"))
        parts = out.split(" | ")
        assert len(parts) == 3
        assert parts[0].startswith("Paradise Biryani, 4.1 stars (98,213 reviews)")
        assert "Fourth Place" not in out
        assert serp.requests[0]["engine"] == "google_maps"
        assert serp.requests[0]["q"] == "biryani in hyderabad"

    def test_single_place_result(self, serp) -> None:
        out = run(local.search_places("gateway of india"))
        assert out.startswith("Gateway Of India Mumbai, 4.6 stars")

    def test_no_fallback_without_key(self) -> None:
        set_client(SerpApiClient(""))
        out = run(local.search_places("biryani", "Hyderabad"))
        assert ToolRegistry._result_is_bad(out)

    def test_no_results_is_bad(self, serp) -> None:
        out = run(local.search_places("nothing here"))
        assert ToolRegistry._result_is_bad(out)

    def test_via_registry_kwargs(self, serp) -> None:
        reg = get_builtin_tools()
        call = reg.find_calls(
            "{tool:search_places(query=biryani, location=Hyderabad)}"
        )[0]
        out = run(reg.execute_call_with_retry(call))
        assert "failed" not in out
        assert out["result"].startswith("Paradise Biryani")


class TestNews:
    def test_topic_headlines_with_sources(self, serp) -> None:
        out = run(news.get_news("Goa"))
        assert out.split(" | ") == [
            "Goa beaches reopen after monsoon (The Hindu)",
            "Goa airport adds new flights to Delhi (Times of India)",
            "Tourist arrivals in Goa rise (Mint)",
        ]

    def test_category_maps_to_query(self, serp) -> None:
        run(news.get_news("tech"))
        assert serp.requests[0]["q"] == "technology"
        run(news.get_news("general"))
        assert "q" not in serp.requests[1]

    def test_rss_fallback_for_category(self, monkeypatch) -> None:
        set_client(SerpApiClient(""))
        monkeypatch.setattr(news, "_rss_headlines", lambda c: f"rss:{c}")
        assert run(news.get_news("business")) == "rss:business"

    def test_no_off_topic_rss_for_free_topic(self, monkeypatch) -> None:
        set_client(SerpApiClient(""))
        monkeypatch.setattr(news, "_rss_headlines", lambda c: "unrelated")
        out = run(news.get_news("Goa"))
        assert ToolRegistry._result_is_bad(out)


def test_tests_never_see_a_real_key() -> None:
    from providers.search.serpapi import get_client

    assert not get_client().enabled
