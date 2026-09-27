from __future__ import annotations

import asyncio
import urllib.request
import xml.etree.ElementTree as ET
import html
from typing import Any

from modules.tools.web_search import _clean
from providers.search.serpapi import SerpApiError, get_client, locale_params


_FEEDS = {
    "general": "https://feeds.bbci.co.uk/news/rss.xml",
    "tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
}

# Category words the old RSS tool accepted -> a Google News query.
# "general" means top stories (no query).
_CATEGORY_QUERIES = {
    "general": "",
    "tech": "technology",
    "science": "science",
    "business": "business",
}


def _story_line(item: dict[str, Any]) -> str:
    # Clustered stories nest the lead article under "highlight".
    lead = item.get("highlight") if not item.get("title") else item
    if not isinstance(lead, dict):
        return ""
    title = _clean(lead.get("title"))
    if not title:
        return ""
    source = lead.get("source")
    name = source.get("name") if isinstance(source, dict) else source
    return f"{title} ({_clean(name)})" if isinstance(name, str) and name else title


def condense_news(data: dict[str, Any], n: int = 3) -> str:
    """Google News results -> top *n* headlines with their outlets."""
    lines = [
        line for item in (data.get("news_results") or [])
        if isinstance(item, dict) and (line := _story_line(item))
    ]
    return " | ".join(lines[:n])


def _rss_headlines(category: str) -> str:
    feed_url = _FEEDS.get(category.lower().strip(), _FEEDS["general"])
    req = urllib.request.Request(feed_url, headers={"User-Agent": "Bolo/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        xml_data = resp.read().decode()

    root = ET.fromstring(xml_data)
    items = root.findall(".//item")[:3]

    if not items:
        return f"No news headlines available for category: {category}"

    headlines: list[str] = []
    for item in items:
        title = item.findtext("title", "")
        if title:
            headlines.append(html.unescape(title))

    if not headlines:
        return "No headlines found."

    return " | ".join(headlines)


async def get_news(topic: str = "general") -> str:
    """Top news headlines, optionally about a topic or place.

    SerpApi Google News when available; BBC RSS (by category) otherwise.
    """
    topic = topic.strip() or "general"
    client = get_client()
    if client.available:
        q = _CATEGORY_QUERIES.get(topic.lower(), topic)
        try:
            data = await client.search("google_news", q=q or None, **locale_params())
            summary = condense_news(data)
            if summary:
                return summary
        except SerpApiError:
            pass
    if topic.lower() not in _FEEDS:
        # RSS only has fixed categories; general headlines would be passed
        # off as news about the topic.
        return f"Unable to find news about {topic} right now"
    try:
        return await asyncio.to_thread(_rss_headlines, topic)
    except Exception as e:
        return f"News fetch failed: {e}"
