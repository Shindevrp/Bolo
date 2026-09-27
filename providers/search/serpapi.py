"""Async SerpApi client shared by every search-backed tool.

One client per process (see ``get_client``). It adds three things the raw
HTTP API doesn't give a voice agent:

- a hard 5 s timeout, so a slow search never stalls a spoken turn;
- an in-memory TTL cache, so a follow-up ("and tomorrow?") or a prefetch
  that already ran never spends a second credit on the same query;
- a credit counter with a cap (free tier = 250 searches/month), so a runaway
  prefetch loop can't burn the month's quota;
- in-flight de-duplication, a short negative cache for empty results (so the
  registry's retry doesn't spend a second credit), and a kill switch on
  fatal account errors (bad key, out of searches);
- one keep-alive HTTP connection, so only the first search pays for TLS.

Failures raise ``SerpApiError`` whose message starts with "Search failed:"
(or "Unable" when the client is disabled), so a tool that returns
``str(err)`` is recognised as bad by ``ToolRegistry._result_is_bad`` and
retried / reported as unavailable instead of being read out as data.

Set ``SERPAPI_RECORD_DIR`` to save every live response as a JSON fixture
(for ``tests/fixtures/serpapi`` and offline benchmarks).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger("serpapi")

SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_CACHE_TTL_S = 15 * 60
DEFAULT_CREDIT_CAP = 200
NEGATIVE_CACHE_TTL_S = 5 * 60
CACHE_MAX_ENTRIES = 256

# Account-level errors: any further search can only fail again. (A 429 rate
# limit is transient and deliberately not listed.)
_FATAL_ERRORS = ("invalid api key", "run out of searches", "searches for the month")


class SerpApiError(Exception):
    """A search that produced no usable data (message is tool-ready)."""


class SerpApiUnavailable(SerpApiError):
    """No key configured or the credit cap is reached -- use a fallback."""


def normalize_query(q: str) -> str:
    """Case/spacing/trailing-punctuation-insensitive query, so "Goa?" and
    "goa" share one cache entry (ours and SerpApi's free 1 h cache)."""
    return " ".join(q.split()).strip("?.!,").strip().lower()


def fixture_name(engine: str, params: dict[str, Any]) -> str:
    """Stable, readable fixture filename for an (engine, params) request."""
    q = str(params.get("q") or params.get("departure_id") or "")
    slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:40] or "noq"
    digest = hashlib.sha1(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{engine}__{slug}__{digest}.json"


class SerpApiClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        cache_ttl: float = DEFAULT_CACHE_TTL_S,
        credit_cap: int = DEFAULT_CREDIT_CAP,
        record_dir: str | Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.credit_cap = credit_cap
        self.record_dir = Path(record_dir).expanduser() if record_dir else None
        self._transport = transport
        # key -> (stored_at, data | SerpApiError); errors are negative entries.
        self._cache: OrderedDict[str, tuple[float, dict[str, Any] | SerpApiError]] = (
            OrderedDict()
        )
        self._inflight: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._http: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None
        self.disabled_reason = ""
        self.credits_used = 0
        self.cache_hits = 0
        self.dedup_hits = 0

    @classmethod
    def from_env(cls, **kw: Any) -> "SerpApiClient":
        cap = os.getenv("SERPAPI_CREDIT_CAP", "").strip()
        return cls(
            os.getenv("SERPAPI_API_KEY", ""),
            credit_cap=int(cap) if cap.isdigit() else DEFAULT_CREDIT_CAP,
            record_dir=os.getenv("SERPAPI_RECORD_DIR") or None,
            **kw,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def credits_left(self) -> int:
        return max(0, self.credit_cap - self.credits_used)

    @property
    def available(self) -> bool:
        return self.enabled and self.credits_left > 0 and not self.disabled_reason

    def stats(self) -> dict[str, Any]:
        """Counters for benchmarks and the UI."""
        return {
            "credits_used": self.credits_used,
            "credits_left": self.credits_left,
            "cache_hits": self.cache_hits,
            "dedup_hits": self.dedup_hits,
            "disabled_reason": self.disabled_reason,
        }

    def _http_client(self) -> httpx.AsyncClient:
        """One keep-alive client per event loop (a client can't cross loops)."""
        loop = asyncio.get_running_loop()
        if self._http is None or self._http_loop is not loop or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=self.timeout, transport=self._transport
            )
            self._http_loop = loop
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    @staticmethod
    def _cache_key(engine: str, params: dict[str, Any]) -> str:
        return json.dumps([engine, params], sort_keys=True, default=str)

    async def search(self, engine: str, **params: Any) -> dict[str, Any]:
        """Run one SerpApi search and return its JSON.

        ``None`` params are dropped, so callers can pass optional fields
        straight through. Cached results -- and a concurrent identical
        request -- cost no credit. Empty results are cached briefly too.
        """
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        if isinstance(params.get("q"), str):
            params["q"] = normalize_query(params["q"])
        key = self._cache_key(engine, params)

        hit = self._cache.get(key)
        if hit:
            stored_at, value = hit
            ttl = NEGATIVE_CACHE_TTL_S if isinstance(value, SerpApiError) else self.cache_ttl
            if time.monotonic() - stored_at < min(ttl, self.cache_ttl):
                self._cache.move_to_end(key)
                self.cache_hits += 1
                if isinstance(value, SerpApiError):
                    raise SerpApiError(str(value))
                return value
            del self._cache[key]

        task = self._inflight.get(key)
        if task is not None and not task.done():
            self.dedup_hits += 1
        else:
            if not self.enabled:
                raise SerpApiUnavailable("Unable to search: no SerpApi key configured")
            if self.disabled_reason:
                raise SerpApiUnavailable(f"Unable to search: {self.disabled_reason}")
            if self.credits_left <= 0:
                raise SerpApiUnavailable(
                    f"Unable to search: SerpApi credit cap ({self.credit_cap}) reached"
                )
            # The fetch is its own task: a cancelled caller (a prefetch at
            # speech-end) neither kills it for a concurrent waiter nor wastes
            # the credit -- the result still lands in the cache.
            task = asyncio.ensure_future(self._fetch(engine, params, key))
            self._inflight[key] = task
            task.add_done_callback(lambda t, k=key: self._settle(k, t))
        return await asyncio.shield(task)

    def _settle(self, key: str, task: asyncio.Future[dict[str, Any]]) -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()  # retrieved: every caller may have gone away

    async def _fetch(self, engine: str, params: dict[str, Any], key: str) -> dict[str, Any]:
        query = {"engine": engine, **params, "api_key": self.api_key}
        self.credits_used += 1
        start = time.perf_counter()
        try:
            resp = await self._http_client().get(SERPAPI_URL, params=query)
            data = resp.json()
        except httpx.TimeoutException:
            raise SerpApiError(
                f"Search failed: SerpApi timed out after {self.timeout:g}s"
            ) from None
        except (httpx.HTTPError, ValueError) as e:
            raise SerpApiError(f"Search failed: {type(e).__name__}") from None
        finally:
            logger.debug(
                f"serpapi engine={engine} ms={(time.perf_counter() - start) * 1000:.0f} "
                f"credits_used={self.credits_used}"
            )

        if not isinstance(data, dict):
            raise SerpApiError("Search failed: malformed SerpApi response")
        err = data.get("error")
        if err:
            # "Google hasn't returned any results for this query." is a real
            # empty result: cache it briefly so a retry doesn't re-spend.
            if "hasn't returned any results" in str(err):
                empty = SerpApiError(f"No results found for: {params.get('q', engine)}")
                self._store(key, empty)
                raise empty
            if resp.status_code in (401, 403) or any(
                f in str(err).lower() for f in _FATAL_ERRORS
            ):
                self.disabled_reason = str(err)
                logger.warning(f"serpapi disabled: {err}")
            raise SerpApiError(f"Search failed: {err}")
        if resp.status_code >= 400:
            raise SerpApiError(f"Search failed: HTTP {resp.status_code}")

        self._store(key, data)
        self._record(engine, params, data)
        return data

    def _store(self, key: str, value: dict[str, Any] | SerpApiError) -> None:
        self._cache[key] = (time.monotonic(), value)
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)

    def _record(self, engine: str, params: dict[str, Any], data: dict[str, Any]) -> None:
        if not self.record_dir:
            return
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(data)
            # search_metadata carries the account-scoped JSON URL; drop it.
            payload.pop("search_metadata", None)
            payload["search_parameters"] = {"engine": engine, **params}
            (self.record_dir / fixture_name(engine, params)).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False)
            )
        except OSError as e:
            logger.debug(f"serpapi fixture record failed: {e!r}")


def locale_params() -> dict[str, str]:
    """Google locale for every SerpApi call (India-first by default)."""
    return {
        "gl": os.getenv("SERPAPI_GL", "in"),
        "hl": os.getenv("SERPAPI_HL", "en"),
    }


_client: SerpApiClient | None = None


def get_client() -> SerpApiClient:
    """Process-wide client, built from the environment on first use."""
    global _client
    if _client is None:
        _client = SerpApiClient.from_env()
    return _client


def set_client(client: SerpApiClient | None) -> None:
    """Swap the process-wide client (tests, benchmarks). ``None`` resets."""
    global _client
    _client = client
