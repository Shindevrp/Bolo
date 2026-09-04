from __future__ import annotations

import asyncio
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


TOOL_CALL_RE = r"\{tool:(\w+)\(([^}]*)\)\}"


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
            "{tool:search_web(Who is Nikola Tesla)}",
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
            "shop, or business that did not come from a search_web result. "
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

    def find_calls(self, text: str) -> list[dict[str, str]]:
        calls = []
        for match in re.finditer(TOOL_CALL_RE, text):
            calls.append({
                "name": match.group(1),
                "args": [
                    a.strip().strip('"').strip("'")
                    for a in match.group(2).split(",")
                    if a.strip()
                ],
            })
        return calls

    def strip_calls(self, text: str) -> str:
        return re.sub(TOOL_CALL_RE, "", text).strip()

    async def execute_call(self, call: dict[str, Any]) -> dict[str, str]:
        spec = self._tools.get(call["name"])
        if not spec:
            return {"tool": call["name"], "result": f"Unknown tool: {call['name']}"}
        try:
            if asyncio.iscoroutinefunction(spec.handler):
                result = await spec.handler(*call["args"])
            else:
                result = spec.handler(*call["args"])
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
