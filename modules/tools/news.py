from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
import html


_FEEDS = {
    "general": "https://feeds.bbci.co.uk/news/rss.xml",
    "tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
}


async def get_news(category: str = "general") -> str:
    """Fetch top news headlines from RSS feeds (free, no API key)."""
    try:
        feed_url = _FEEDS.get(category.lower().strip(), _FEEDS["general"])
        req = urllib.request.Request(feed_url, headers={"User-Agent": "TASA/1.0"})
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
    except Exception as e:
        return f"News fetch failed: {e}"
