"""Async SerpApi client shared by every search-backed tool.

One client per process (see ``get_client``). It adds three things the raw
HTTP API doesn't give a voice agent:

- a hard 5 s timeout, so a slow search never stalls a spoken turn;
- an in-memory TTL cache, so a follow-up ("and tomorrow?") or a prefetch
  that already ran never spends a second credit on the same query;
- a credit counter with a cap (free tier = 250 searches/month), so a runaway
  prefetch loop can't burn the month's quota.

Failures raise ``SerpApiError`` whose message starts with "Search failed:"
(or "Unable" when the client is disabled), so a tool that returns
``str(err)`` is recognised as bad by ``ToolRegistry._result_is_bad`` and
retried / reported as unavailable instead of being read out as data.

Set ``SERPAPI_RECORD_DIR`` to save every live response as a JSON fixture
(for ``tests/fixtures/serpapi`` and offline benchmarks).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger("serpapi")

SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_CACHE_TTL_S = 15 * 60
DEFAULT_CREDIT_CAP = 200


class SerpApiError(Exception):
    """A search that produced no usable data (message is tool-ready)."""


class SerpApiUnavailable(SerpApiError):
    """No key configured or the credit cap is reached -- use a fallback."""


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
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.credits_used = 0
        self.cache_hits = 0

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
        return self.enabled and self.credits_left > 0

    @staticmethod
    def _cache_key(engine: str, params: dict[str, Any]) -> str:
        return json.dumps([engine, params], sort_keys=True, default=str)

    async def search(self, engine: str, **params: Any) -> dict[str, Any]:
        """Run one SerpApi search and return its JSON.

        ``None`` params are dropped, so callers can pass optional fields
        straight through. Cached results cost no credit.
        """
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        key = self._cache_key(engine, params)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl:
            self.cache_hits += 1
            return hit[1]

        if not self.enabled:
            raise SerpApiUnavailable("Unable to search: no SerpApi key configured")
        if self.credits_left <= 0:
            raise SerpApiUnavailable(
                f"Unable to search: SerpApi credit cap ({self.credit_cap}) reached"
            )

        query = {"engine": engine, **params, "api_key": self.api_key}
        self.credits_used += 1
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self._transport
            ) as http:
                resp = await http.get(SERPAPI_URL, params=query)
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
            # empty result, not a failure -- still report it as bad data.
            if "hasn't returned any results" in str(err):
                raise SerpApiError(f"No results found for: {params.get('q', engine)}")
            raise SerpApiError(f"Search failed: {err}")
        if resp.status_code >= 400:
            raise SerpApiError(f"Search failed: HTTP {resp.status_code}")

        self._cache[key] = (time.monotonic(), data)
        self._record(engine, params, data)
        return data

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
