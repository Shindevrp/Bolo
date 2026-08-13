from __future__ import annotations

import ast
import datetime
import math
import operator
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
    """Safely evaluate a mathematical expression using an AST whitelist.

    Only numeric literals, arithmetic/comparison/logical operators, and the
    whitelisted math functions are allowed. Attribute access, calls to
    non-whitelisted names, and any other node types are rejected, so there is
    no way to reach arbitrary Python objects (no ``eval`` / ``exec`` / dunder
    access).
    """
    _MATH_FUNCS: dict[str, object] = {
        "abs": abs, "acos": math.acos, "asin": math.asin, "atan": math.atan,
        "atan2": math.atan2, "ceil": math.ceil, "cos": math.cos, "degrees": math.degrees,
        "exp": math.exp, "factorial": math.factorial, "floor": math.floor,
        "fmod": math.fmod, "hypot": math.hypot, "log": math.log, "log10": math.log10,
        "log2": math.log2, "pow": math.pow, "radians": math.radians, "sin": math.sin,
        "sqrt": math.sqrt, "tan": math.tan, "trunc": math.trunc,
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "int": int, "float": float, "round": round, "min": min, "max": max,
        "sum": sum,
    }
    _OPS: dict[type[ast.operator], object] = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.UAdd: operator.pos, ast.USub: operator.neg,
        ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
    }

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        return f"Error: invalid expression ({e.msg})"

    def _eval(node: ast.AST):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)) or node.value is None:
                return node.value
            raise ValueError("unsupported literal")
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported operator")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported operator")
            return op(_eval(node.operand))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("unsupported comparison")
            op = _OPS.get(type(node.ops[0]))
            if op is None:
                raise ValueError("unsupported operator")
            return op(_eval(node.left), _eval(node.comparators[0]))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = _eval(node.values[0])
                for v in node.values[1:]:
                    if not bool(result):
                        return result
                    result = _eval(v)
                return result
            if isinstance(node.op, ast.Or):
                result = _eval(node.values[0])
                for v in node.values[1:]:
                    if bool(result):
                        return result
                    result = _eval(v)
                return result
            raise ValueError("unsupported boolean operator")
        if isinstance(node, ast.Name):
            if node.id not in _MATH_FUNCS:
                raise ValueError(f"unknown name: {node.id}")
            return _MATH_FUNCS[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCS:
                raise ValueError("unsupported call")
            func = _MATH_FUNCS[node.func.id]
            args = [_eval(a) for a in node.args]
            if node.keywords:
                raise ValueError("keyword arguments not supported")
            if isinstance(func, (int, float)):
                raise ValueError("cannot call a constant")
            return func(*args)
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    try:
        result = _eval(tree)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, TypeError, OverflowError) as e:
        return f"Error: {e}"
    if isinstance(result, bool) or result is None:
        return str(result)
    if isinstance(result, float):
        if math.isnan(result) or math.isinf(result):
            return str(result)
        if result.is_integer():
            return str(int(result))
        return f"{result:.10g}"
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
        proactive_hint="the user asks about weather, temperature, rain, sun, or how hot/cold it is somewhere — ALWAYS use get_weather, never search_web for weather.",
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
        proactive_hint="the user asks for a fact, a general question you're unsure about, or specific information like a person, event, or topic — use for general web search.",
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
        proactive_hint="the user asks about news, headlines, or what's happening in the world — use this instead of search_web.",
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
