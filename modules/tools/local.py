from __future__ import annotations

from typing import Any

from modules.tools.web_search import _clean
from providers.search.serpapi import SerpApiError, get_client, locale_params


def _place_line(p: dict[str, Any]) -> str:
    """One place -> "Name, 4.5 stars (1,234 reviews), cafe, ₹200–400, address"."""
    title = _clean(p.get("title"))
    if not title:
        return ""
    bits = [title]
    rating = p.get("rating")
    if rating:
        reviews = p.get("reviews")
        bits.append(
            f"{rating} stars ({reviews:,} reviews)" if isinstance(reviews, int)
            else f"{rating} stars"
        )
    for key in ("type", "price"):
        val = p.get(key)
        if isinstance(val, str) and val.strip():
            bits.append(_clean(val))
    state = p.get("open_state")
    if isinstance(state, str) and state.strip():
        bits.append(_clean(state))
    addr = p.get("address")
    if isinstance(addr, str) and addr.strip():
        bits.append(_clean(addr))
    return ", ".join(bits)


def condense_places(data: dict[str, Any], n: int = 3) -> str:
    """Google Maps results -> the top *n* places, one per line."""
    places = data.get("local_results")
    if not isinstance(places, list) or not places:
        single = data.get("place_results")
        places = [single] if isinstance(single, dict) else []
    lines = [line for p in places[:n] if (line := _place_line(p))]
    return " | ".join(lines)


async def search_places(query: str, location: str = "") -> str:
    """Find real places (restaurants, hotels, shops, sights) via SerpApi
    Google Maps. Returns the top 3 with rating, type, price and address.

    There is deliberately no keyless fallback: naming a place that didn't
    come from a real listing is exactly what the agent must never do.
    """
    query = query.strip()
    location = location.strip()
    if not query:
        return "No results found for: (empty query)"
    q = f"{query} in {location}" if location else query
    client = get_client()
    if not client.available:
        return "Unable to search places: SerpApi is not available"
    try:
        data = await client.search(
            "google_maps", q=q, type="search", hl=locale_params()["hl"]
        )
    except SerpApiError as e:
        return str(e)
    return condense_places(data) or f"No results found for: {q}"
