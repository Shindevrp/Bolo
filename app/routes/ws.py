from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.pipeline import StreamingPipeline, PipelineEvent
from core.dialogue_manager import DialogueManager
from core.state import SessionState, DialogueState
from modules.memory.session import SessionMemory
from modules.memory.retrieval import RetrievalModule
from modules.backchannel.generator import BackchannelGenerator
from modules.backchannel.timing import BackchannelTiming
from modules.turn.detector import TurnDetector
from modules.turn.interrupt import InterruptHandler
from utils.logger import logger

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Session {session_id} connected")

    pipeline: StreamingPipeline | None = websocket.app.state.pipeline
    if pipeline is None:
        await websocket.send_json({"error": "Pipeline not initialized"})
        await websocket.close()
        return

    session = SessionState(session_id=session_id)
    memory = SessionMemory()
    retrieval = RetrievalModule()
    dialogue = DialogueManager()
    turn_detector = TurnDetector()
    interrupt_handler = InterruptHandler()
    backchannel_gen = BackchannelGenerator()
    backchannel_timing = BackchannelTiming()
    last_backchannel_time = 0.0

    async def pump_output():
        async for msg in pipeline.output_stream():
            try:
                if msg.event == PipelineEvent.SPEECH_START:
                    dialogue.on_speech_start(session)
                    await websocket.send_json({"type": "speech_start"})

                elif msg.event == PipelineEvent.SPEECH_END:
                    dialogue.on_speech_end(session)
                    await websocket.send_json({"type": "speech_end"})

                elif msg.event == PipelineEvent.FINAL_TRANSCRIPT:
                    dialogue.on_transcript(session, str(msg.data))
                    memory.add("user", str(msg.data))
                    retrieval.add_to_long_term(str(msg.data))
                    await websocket.send_json({
                        "type": "transcript",
                        "text": str(msg.data),
                    })

                elif msg.event == PipelineEvent.LLM_TOKEN:
                    dialogue.on_response_token(session, str(msg.data))
                    if session.state == DialogueState.INTERRUPTIBLE:
                        await websocket.send_json({
                            "type": "llm_token",
                            "token": str(msg.data),
                        })

                elif msg.event == PipelineEvent.LLM_DONE:
                    dialogue.on_response_done(session, str(msg.data))
                    memory.add("assistant", str(msg.data))
                    await websocket.send_json({
                        "type": "llm_done",
                        "text": str(msg.data),
                    })

                elif msg.event == PipelineEvent.TTS_CHUNK:
                    audio_bytes = msg.data
                    if isinstance(audio_bytes, bytes):
                        await websocket.send_bytes(audio_bytes)

                elif msg.event == PipelineEvent.TTS_DONE:
                    session.set_state(DialogueState.IDLE)
                    await websocket.send_json({"type": "tts_done"})

                elif msg.event == PipelineEvent.BACKCHANNEL:
                    await websocket.send_json({
                        "type": "backchannel",
                        "text": str(msg.data),
                    })

                elif msg.event == PipelineEvent.INTERRUPT:
                    dialogue.on_interrupt(session)
                    await websocket.send_json({"type": "interrupt"})

            except Exception:
                break

    pump_task = None
    try:
        pump_task = asyncio.create_task(pump_output())

        while True:
            raw = await websocket.receive()
            logger.info(f"Session {session_id} raw keys: {list(raw.keys())} type={raw.get('type','?')}")

            msg_type = raw.get("type", "")
            if msg_type == "websocket.disconnect":
                logger.info(f"Session {session_id} disconnect msg")
                break

            if "bytes" in raw and raw["bytes"] is not None:
                chunk = raw["bytes"]
                logger.info(f"Session {session_id} audio chunk {len(chunk)} bytes")
                await pipeline.push_audio(chunk, session_id)

            elif "text" in raw and raw["text"] is not None:
                text = raw["text"]
                logger.info(f"Session {session_id} text msg: {text[:100]}")
                data = json.loads(text)
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            else:
                logger.info(f"Session {session_id} unhandled msg: {raw}")

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Session {session_id} error: {e}")
    finally:
        if pump_task:
            pump_task.cancel()
        logger.info(f"Session {session_id} cleaned up")