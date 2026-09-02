from __future__ import annotations

import urllib.parse
import urllib.request
import urllib.error
import json
import html
import re
import time


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def _http_get_json(url: str, retries: int = 2) -> dict:
    """GET a JSON endpoint with retry + exponential backoff.

    Retries on HTTP 429/5xx (respecting Retry-After) and on transient network
    errors (timeouts, connection resets) so a rate-limited Wikipedia/DuckDuckGo
    doesn't immediately fail the tool call.
    """
    attempt = 0
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TASA/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else None
            if e.code not in (429, 500, 502, 503, 504) or attempt >= retries:
                raise
            attempt += 1
            time.sleep(wait if wait is not None else min(2 ** attempt, 4))
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt >= retries:
                raise
            attempt += 1
            time.sleep(min(2 ** attempt, 4))


def _ddg_instant(query: str) -> list[str]:
    encoded = urllib.parse.quote(query.strip())
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
    data = _http_get_json(url)

    results: list[str] = []
    abstract = data.get("AbstractText", "")
    if abstract:
        results.append(html.unescape(abstract))
    answer = data.get("Answer", "")
    if answer:
        results.append(f"Answer: {html.unescape(answer)}")
    for topic in data.get("RelatedTopics", [])[:3]:
        if isinstance(topic, dict):
            text = topic.get("Text", "")
            if text:
                results.append(html.unescape(text))
    return results[:3]


def _wiki_search(query: str) -> str:
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": 2,
        "format": "json",
        "utf8": "1",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    data = _http_get_json(url)
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return f"No results found for: {query}"
    lines = []
    for hit in hits:
        lines.append(f"{hit['title']}: {_strip_html(hit['snippet'])}")
    return " | ".join(lines)


async def search_web(query: str) -> str:
    """Search the web for a query. Uses DuckDuckGo Instant Answers first,
    then falls back to Wikipedia when no instant answer exists."""
    try:
        results = _ddg_instant(query)
    except Exception:
        results = []

    if results:
        return " | ".join(results)

    try:
        return _wiki_search(query)
    except Exception as e:
        return f"Search failed: {e}"
