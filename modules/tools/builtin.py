from __future__ import annotations

import datetime
import math
import random

from modules.tools.registry import ToolRegistry, ToolSpec
from modules.tools.weather import get_weather
from modules.tools.web_search import search_web
from modules.tools.news import get_news
from modules.tools.reminder import set_reminder


def _get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p").lstrip("0")


def _get_date() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y")


def _calculate(expression: str) -> str:
    allowed = {"abs", "acos", "asin", "atan", "ceil", "cos", "degrees",
               "exp", "factorial", "floor", "fmod", "log", "log10",
               "pow", "radians", "sin", "sqrt", "tan", "pi", "e"}
    safe_dict = {k: getattr(math, k, None) for k in allowed}
    safe_dict.update({"abs": abs, "int": int, "float": float, "round": round,
                      "min": min, "max": max, "sum": sum, "str": str})
    safe_dict["__builtins__"] = None
    result = eval(expression, safe_dict)
    return str(result)


def _roll_dice(sides: str = "6") -> str:
    n = max(1, int(sides))
    return str(random.randint(1, n))


def _echo(text: str) -> str:
    return text


def get_builtin_tools() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(ToolSpec(
        name="get_time",
        description="Get the current time",
        parameters={"properties": {}},
        handler=_get_time,
        proactive_hint="If the user asks about the time or what time it is, use this tool.",
    ))

    registry.register(ToolSpec(
        name="get_date",
        description="Get today's date",
        parameters={"properties": {}},
        handler=_get_date,
        proactive_hint="If the user asks about the date or what day it is, use this tool.",
    ))

    registry.register(ToolSpec(
        name="calculate",
        description="Evaluate a mathematical expression",
        parameters={
            "properties": {
                "expression": {"description": "math expression to evaluate"},
            },
        },
        handler=_calculate,
        proactive_hint="If the user asks you to calculate or compute something mathematical, use this tool.",
    ))

    registry.register(ToolSpec(
        name="roll_dice",
        description="Roll a dice with the given number of sides",
        parameters={
            "properties": {
                "sides": {"description": "number of sides (default 6)"},
            },
        },
        handler=_roll_dice,
    ))

    registry.register(ToolSpec(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "properties": {
                "city": {"description": "city name to get weather for"},
            },
        },
        handler=get_weather,
        proactive_hint="If the user mentions weather, temperature, rain, sun, outdoor plans, or how hot/cold it is somewhere, proactively offer the weather.",
    ))

    registry.register(ToolSpec(
        name="search_web",
        description="Search the web for information",
        parameters={
            "properties": {
                "query": {"description": "search query"},
            },
        },
        handler=search_web,
        proactive_hint="If the user asks about current events, facts you're unsure about, or needs information you don't have, search the web.",
    ))

    registry.register(ToolSpec(
        name="get_news",
        description="Get top news headlines",
        parameters={
            "properties": {
                "category": {"description": "news category: general, tech, science, or business"},
            },
        },
        handler=get_news,
        proactive_hint="If the user asks about current events, what's happening in the world, or wants news updates, use this tool.",
    ))

    registry.register(ToolSpec(
        name="set_reminder",
        description="Set a reminder for later",
        parameters={
            "properties": {
                "text": {"description": "what to be reminded about"},
                "minutes": {"description": "minutes from now (default 5)"},
            },
        },
        handler=set_reminder,
        proactive_hint="If the user asks to be reminded about something later, use this tool.",
    ))

    return registry
