from __future__ import annotations

import urllib.parse
import urllib.request
import json
import html
import re


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def _ddg_instant(query: str) -> list[str]:
    encoded = urllib.parse.quote(query.strip())
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
    req = urllib.request.Request(url, headers={"User-Agent": "TASA/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

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
    req = urllib.request.Request(url, headers={"User-Agent": "TASA/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
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
