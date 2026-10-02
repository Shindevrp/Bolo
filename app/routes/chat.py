from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from modules.memory.session import SessionMemory
from modules.tools.builtin import get_builtin_tools
from modules.tools.registry import ToolMarkerFilter

router = APIRouter(prefix="/chat", tags=["chat"])

# Bound per session so a long demo can't grow history without limit.
MAX_TURNS = 12

BASE_PROMPT = (
    "You are Bolo, a real-time conversational voice assistant. "
    "Respond concisely and naturally: short, spoken, human. You are speaking, "
    "not writing, so never use lists, markdown, or long paragraphs."
)

# Tools are stateless but the key client must not be, so it is built once per
# process and reused. Building it per request re-reads env and drops caches.
_registry = get_builtin_tools()
_NO_ANSWER = "Sorry, I couldn't find an answer to that right now."

# session_id -> (SessionMemory, created_at). A demo client sends the same
# conversation repeatedly, so history has to outlive the request.
_sessions: dict[str, tuple[SessionMemory, float]] = {}


def _memory_for(session_id: str) -> SessionMemory:
    entry = _sessions.get(session_id)
    if entry is None:
        entry = (SessionMemory(max_turns=MAX_TURNS), time.time())
        _sessions[session_id] = entry
    return entry[0]


def _trim(touch: str | None = None) -> None:
    """Refresh ``touch`` and drop sessions idle for over an hour.

    The map lives for the life of the process, so without this a long-running
    server accumulates one entry per session id it has ever been sent.
    """
    now = time.time()
    if touch is not None and touch in _sessions:
        _sessions[touch] = (_sessions[touch][0], now)
    for sid in [s for s, (_, seen) in _sessions.items() if now - seen > 3600]:
        _sessions.pop(sid, None)


def _system_prompt() -> str:
    block = _registry.system_prompt_block()
    return f"{BASE_PROMPT}\n{block}" if block else BASE_PROMPT


async def _tool_rounds(
    pipeline, llm, memory: SessionMemory, message: str, emit
) -> str:
    """Answer one turn, executing any {tool:...} calls the model emits.

    Mirrors the voice pipeline's loop: the model is asked for exactly one tool
    call when it needs external data, the call is executed, and the result is
    fed back as a follow-up turn. Up to 3 rounds so a chained request
    (flights, then hotels) can resolve inside a single turn.
    """
    memory.add("user", message)

    messages = memory.context_messages(_system_prompt())
    full = ""
    for _round in range(3):
        full = ""
        # Stream only what is safe to show: a "{tool:...}" call (which may
        # arrive split across tokens) is cut out, never sent to the client.
        marker = ToolMarkerFilter()
        async for token in llm.generate_stream(messages):
            full += token
            if safe := marker.feed(token):
                await emit(safe)
        if held := marker.flush():
            await emit(held)

        calls = _registry.find_calls(full)
        if not calls:
            break

        # Each call executes exactly once. The tool result goes into memory for
        # later turns AND into the follow-up messages so this turn can use it.
        messages = messages + [{"role": "assistant", "content": full}]
        for call in calls:
            result = await _registry.execute_call_with_retry(call)
            payload = f"{call['name']}: {result['result']}"
            memory.add("tool", payload)
            messages.append({"role": "tool", "content": payload})
        messages.append(
            {"role": "user", "content": "Continue naturally with the tool results."}
        )

    answer = _registry.strip_calls(full).strip()
    if not answer:
        # The model was still asking for tools when the round cap hit (or
        # produced nothing): say so rather than return a blank reply.
        answer = _NO_ANSWER
        await emit(answer)
    memory.add("assistant", answer)
    return answer


@router.post("/stream")
async def chat_stream(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if pipeline.llm is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    session_id = str(body.get("session_id") or uuid.uuid4())[:8]
    _trim(session_id)
    memory = _memory_for(session_id)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

        # _tool_rounds pushes tokens through a queue; this generator relays
        # them as SSE events while the turn (and its tool calls) runs.
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def emit(token: str) -> None:
            await queue.put(token)

        async def run() -> str:
            try:
                return await _tool_rounds(
                    pipeline, pipeline.llm, memory, message, emit
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while (token := await queue.get()) is not None:
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            answer = await task
        finally:
            if not task.done():
                task.cancel()
        yield f"data: {json.dumps({'type': 'done', 'text': answer})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/")
async def chat_once(request: Request):
    t0 = time.perf_counter()
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if pipeline.llm is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    session_id = str(body.get("session_id") or uuid.uuid4())[:8]
    _trim(session_id)
    memory = _memory_for(session_id)

    async def collect(token: str) -> None:
        return None

    text = await _tool_rounds(pipeline, pipeline.llm, memory, message, collect)
    elapsed = time.perf_counter() - t0
    return {
        "response": text,
        "elapsed": round(elapsed, 2),
        "session_id": session_id,
    }