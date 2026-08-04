from __future__ import annotations

import urllib.parse
import urllib.request
import json
import html


async def search_web(query: str) -> str:
    """Search the web using DuckDuckGo Instant Answers API (free, no key)."""
    try:
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

        if not results:
            return f"No instant answer found for: {query}"

        return " | ".join(results[:3])
    except Exception as e:
        return f"Search failed: {e}"
