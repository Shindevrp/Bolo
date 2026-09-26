from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str | Coroutine[Any, Any, str]]
    proactive_hint: str = ""


_CALL_START_RE = re.compile(r"\{tool:(\w+)\(")
_KWARG_RE = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$", re.S)
_CLOSE = {"(": ")", "[": "]", "{": "}"}


def _unquote(s: str) -> str:
    return s.strip().strip('"').strip("'")


def _scan_call(text: str, start: int) -> tuple[str, int] | None:
    """Scan the argument body of a call whose ``(`` ends at *start*.

    Balances ()/[]/{} and skips quoted strings, so JSON arguments and
    parenthesised values survive. Returns (body, end) where ``text[:end]``
    includes the closing ``)}``, or None when the call isn't closed yet
    (e.g. still streaming).
    """
    stack: list[str] = []
    quote = ""
    i = start
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = ""
        elif c in "\"'" and (i == start or not text[i - 1].isalnum()):
            # An apostrophe inside a word ("what's") is not a quote.
            quote = c
        elif c in _CLOSE:
            stack.append(_CLOSE[c])
        elif stack and c == stack[-1]:
            stack.pop()
        elif not stack and c == ")":
            if text.startswith(")}", i):
                return text[start:i], i + 2
            return None
        i += 1
    # Unbalanced quote or bracket (e.g. "'90s music"): fall back to the
    # first ")}" like the old flat regex did.
    end = text.find(")}", start)
    if end < 0:
        return None
    return text[start:end], end + 2


def _split_args(body: str) -> list[str]:
    """Split on top-level commas (not inside quotes or brackets)."""
    parts: list[str] = []
    depth = 0
    quote = ""
    cur: list[str] = []
    for i, c in enumerate(body):
        if quote:
            if c == quote:
                quote = ""
        elif c in "\"'" and (i == 0 or not body[i - 1].isalnum()):
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    parts.append("".join(cur))
    return [p for p in parts if p.strip()]


def parse_args(body: str) -> tuple[list[str], dict[str, str]]:
    """Parse a call body into positional args and keyword args.

    Accepts ``a, b``, ``from=DEL, to=BOM`` (mixable) and a single JSON
    object ``{"from": "DEL"}``. Values stay strings.
    """
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return [], {str(k): str(v) for k, v in obj.items()}
    args: list[str] = []
    kwargs: dict[str, str] = {}
    for part in _split_args(body):
        m = _KWARG_RE.match(part.strip())
        if m:
            kwargs[m.group(1)] = _unquote(m.group(2))
        else:
            args.append(_unquote(part))
    return args, kwargs


def iter_calls(text: str):
    """Yield (name, body, start, end) for every complete tool call."""
    pos = 0
    while True:
        m = _CALL_START_RE.search(text, pos)
        if not m:
            return
        scanned = _scan_call(text, m.end())
        if scanned is None:
            pos = m.end()
            continue
        body, end = scanned
        yield m.group(1), body, m.start(), end
        pos = end


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def system_prompt_block(self) -> str:
        if not self._tools:
            return ""
        names = ", ".join(sorted(self._tools.keys()))
        lines = [
            f"\nYou have access to tools: {{{names}}}.",
            "",
            "TOOL PROTOCOL:",
            "When you need external data, your ENTIRE response must be exactly one "
            "tool call in this exact format: {tool:name(args)} — no other text, "
            "no speaker tags, no commentary before or after.",
            "Examples: {tool:get_weather(Hyderabad)}  {tool:get_news(general)}  "
            "{tool:search_web(Who is Nikola Tesla)}  "
            "{tool:search_places(query=biryani, location=Hyderabad)}",
            "Arguments may be positional or name=value pairs.",
            "",
            "Choose the correct tool:",
        ]
        for t in sorted(self._tools.values(), key=lambda s: s.name):
            if not t.proactive_hint:
                continue
            hint = t.proactive_hint
            if not hint.lower().startswith(("use", "the user", "if the user")):
                hint = f"use if {hint}"
            lines.append(f"- {t.name}: {hint}")
        lines += [
            "",
            "RULES:",
            "- NEVER name, recommend, or describe a specific place, restaurant, "
            "shop, or business that did not come from a search_web or search_places result. "
            "Fabricating a plausible-sounding name is prohibited.",
            "- NEVER say 'let me check', 'let me look that up', or 'I'll search' — "
            "just output the tool call and wait for the result.",
            "- NEVER invent or improvise data when a tool result is missing.",
            "- NEVER call a second tool as a fallback after a failed one — the "
            "system retries automatically. Wait for the result.",
            "- After a tool result is provided, present the data naturally and "
            "accurately. If the result says something is unavailable, say so briefly.",
        ]
        return "\n".join(lines)

    def find_calls(self, text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for name, body, _, _ in iter_calls(text):
            args, kwargs = parse_args(body)
            call: dict[str, Any] = {"name": name, "args": args}
            if kwargs:
                call["kwargs"] = kwargs
            calls.append(call)
        return calls

    def strip_calls(self, text: str) -> str:
        out: list[str] = []
        pos = 0
        for _, _, start, end in iter_calls(text):
            out.append(text[pos:start])
            pos = end
        out.append(text[pos:])
        return "".join(out).strip()

    async def execute_call(self, call: dict[str, Any]) -> dict[str, str]:
        spec = self._tools.get(call["name"])
        if not spec:
            return {"tool": call["name"], "result": f"Unknown tool: {call['name']}"}
        try:
            args = call.get("args", [])
            kwargs = call.get("kwargs") or {}
            if asyncio.iscoroutinefunction(spec.handler):
                result = await spec.handler(*args, **kwargs)
            else:
                result = spec.handler(*args, **kwargs)
        except Exception as e:
            result = f"Error: {e}"
        return {"tool": call["name"], "result": result}

    @staticmethod
    def _result_is_bad(result: str) -> bool:
        if not result or not result.strip():
            return True
        lowered = result.strip().lower()
        bad_prefixes = (
            "error", "search failed", "no instant answer", "no results found",
            "no data", "unknown tool", "failed", "exception", "unable", "timed out",
        )
        return lowered.startswith(bad_prefixes)

    async def execute_call_with_retry(
        self, call: dict[str, Any], retries: int = 1
    ) -> dict[str, str]:
        last: dict[str, str] | None = None
        for _ in range(retries + 1):
            last = await self.execute_call(call)
            if not self._result_is_bad(str(last["result"])):
                return last
            await asyncio.sleep(0.5)
        assert last is not None
        last["failed"] = True
        return last

    async def execute_all(self, text: str) -> tuple[str, list[dict[str, str]]]:
        calls = self.find_calls(text)
        if not calls:
            return text, []
        results = await asyncio.gather(*[self.execute_call(c) for c in calls])
        cleaned = self.strip_calls(text)
        return cleaned, results
