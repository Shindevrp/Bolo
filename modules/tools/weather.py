from __future__ import annotations

import asyncio
import urllib.parse
import urllib.request
import json
from typing import Any

from providers.search.serpapi import SerpApiError, get_client, locale_params


def _serpapi_weather(box: dict[str, Any]) -> str:
    """SerpApi's weather answer box -> the same shape as the wttr.in line."""
    where = str(box.get("location", "")).strip()
    temp = str(box.get("temperature", "")).strip()
    if not (where and temp):
        return ""
    unit = "F" if str(box.get("unit", "")).lower().startswith("f") else "C"
    cond = str(box.get("weather", "")).strip()
    humidity = str(box.get("humidity", "")).strip()
    wind = str(box.get("wind", "")).strip()
    line = f"{where}: {cond}, {temp}°{unit}" if cond else f"{where}: {temp}°{unit}"
    if humidity:
        line += f", humidity {humidity}"
    if wind:
        line += f", wind {wind}"
    return line


def _wttr_weather(city: str) -> str:
    """Current weather for a city from wttr.in (free, no API key)."""
    try:
        encoded = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "Bolo/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        current = data.get("current_condition", [{}])[0]
        if not current:
            return f"Weather data not available for {city}"

        temp_c = current.get("temp_C", "?")
        feels_like = current.get("FeelsLikeC", "?")
        desc = current.get("weatherDesc", [{}])[0].get("value", "unknown")
        humidity = current.get("humidity", "?")
        wind_kmph = current.get("windspeedKmph", "?")

        return (
            f"{city}: {desc}, {temp_c}°C (feels like {feels_like}°C), "
            f"humidity {humidity}%, wind {wind_kmph} km/h"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


async def get_weather(city: str) -> str:
    """Current weather for a city.

    SerpApi Google's weather answer box when a key and credits are
    available; wttr.in otherwise, or when SerpApi fails.
    """
    city = city.strip()
    client = get_client()
    if client.available:
        try:
            data = await client.search("google", q=f"weather in {city}", **locale_params())
            box = data.get("answer_box")
            if isinstance(box, dict):
                summary = _serpapi_weather(box)
                if summary:
                    return summary
        except SerpApiError:
            pass
    return await asyncio.to_thread(_wttr_weather, city)
