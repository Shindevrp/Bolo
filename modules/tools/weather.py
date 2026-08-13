from __future__ import annotations

import urllib.parse
import urllib.request
import json


async def get_weather(city: str) -> str:
    """Get current weather for a city using wttr.in (free, no API key)."""
    try:
        encoded = urllib.parse.quote(city.strip())
        url = f"https://wttr.in/{encoded}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "TASA/1.0"})
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
