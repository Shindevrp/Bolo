from __future__ import annotations

import asyncio
import urllib.parse
import urllib.request
import urllib.error
import json
import html
import re
import time
from typing import Any

from providers.search.serpapi import SerpApiError, get_client


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


def _domain(link: str) -> str:
    host = urllib.parse.urlparse(link or "").netloc
    return host[4:] if host.startswith("www.") else host


def _answer_box(box: dict[str, Any]) -> str:
    """The single most direct fact in an answer box, if any."""
    for key in ("answer", "result", "snippet"):
        val = box.get(key)
        if isinstance(val, str) and val.strip():
            title = box.get("title")
            if key != "snippet" and isinstance(title, str) and title.strip():
                return f"{title.strip()}: {val.strip()}"
            return val.strip()
    if box.get("temperature"):  # weather answer box
        where = box.get("location", "")
        unit = box.get("unit", "")
        cond = box.get("weather", "")
        return f"{where}: {cond}, {box['temperature']}°{unit[:1].upper()}".strip(": ")
    highlighted = box.get("snippet_highlighted_words")
    if isinstance(highlighted, list) and highlighted:
        return ", ".join(str(h) for h in highlighted)
    return ""


def _knowledge_graph(kg: dict[str, Any]) -> str:
    title = str(kg.get("title", "")).strip()
    desc = str(kg.get("description", "")).strip()
    kind = str(kg.get("type", "")).strip()
    if not title or not (desc or kind):
        return ""
    head = f"{title} ({kind})" if kind else title
    return f"{head}: {desc}" if desc else head


def condense_google(data: dict[str, Any], n_organic: int = 2) -> str:
    """Google results -> a short, speakable, sourced summary.

    Priority: answer box, then knowledge graph, then the top organic
    results; each organic line carries its source domain.
    """
    parts: list[str] = []
    box = data.get("answer_box")
    if isinstance(box, dict):
        if (a := _answer_box(box)):
            parts.append(a)
    kg = data.get("knowledge_graph")
    if isinstance(kg, dict):
        if (k := _knowledge_graph(kg)):
            parts.append(k)
    want = n_organic if parts else n_organic + 1
    for r in (data.get("organic_results") or [])[:want]:
        title = str(r.get("title", "")).strip()
        snippet = str(r.get("snippet", "")).strip()
        if not (title or snippet):
            continue
        src = _domain(r.get("link", ""))
        head = f"{title} ({src})" if src else title
        parts.append(f"{head}: {snippet}" if snippet else head)
    return " | ".join(parts)


async def _fallback_search(query: str) -> str:
    """Keyless DuckDuckGo -> Wikipedia search (blocking HTTP off-loop)."""
    try:
        results = await asyncio.to_thread(_ddg_instant, query)
    except Exception:
        results = []
    if results:
        return " | ".join(results)
    try:
        return await asyncio.to_thread(_wiki_search, query)
    except Exception as e:
        return f"Search failed: {e}"


async def search_web(query: str) -> str:
    """Search the web for a query.

    SerpApi Google (answer box -> knowledge graph -> top organic results)
    when a key and credits are available; DuckDuckGo Instant Answers then
    Wikipedia otherwise, or when SerpApi fails.
    """
    query = query.strip()
    if not query:
        return "No results found for: (empty query)"
    client = get_client()
    if client.available:
        try:
            data = await client.search("google", q=query, **_locale())
            summary = condense_google(data)
            if summary:
                return summary
        except SerpApiError:
            pass
    return await _fallback_search(query)


def _locale() -> dict[str, str]:
    """Google locale for every SerpApi call (India-first by default)."""
    import os

    return {
        "gl": os.getenv("SERPAPI_GL", "in"),
        "hl": os.getenv("SERPAPI_HL", "en"),
    }
