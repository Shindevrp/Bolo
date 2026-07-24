from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.routes.health import router as health_router
from app.routes.ws import router as ws_router
from app.routes.chat import router as chat_router
from app.routes.metrics import router as metrics_router
from core.pipeline import StreamingPipeline
from core.config import CoreConfig
from utils.logger import get_logger

logger = get_logger("server")

config = CoreConfig()
pipeline: StreamingPipeline | None = None


def _build_providers():
    from providers.stt.faster_whisper_stt import FasterWhisperSTT
    from providers.llm.vllm_llm import VLLMProvider
    from providers.tts.piper_tts import PiperTTS
    from modules.vad.silero_vad import SileroVAD
    from modules.turn.detector import TurnDetector
    from modules.turn.interrupt import InterruptHandler
    from modules.turn.timing import TurnTiming
    from modules.turn.backchannel import TurnBackchannel
    from modules.backchannel.generator import BackchannelGenerator
    from modules.backchannel.timing import BackchannelTiming

    stt = FasterWhisperSTT(
        model_size=os.getenv("TASA_STT_MODEL", "base"),
        device=os.getenv("TASA_STT_DEVICE", "cpu"),
        compute_type=os.getenv("TASA_STT_COMPUTE", "int8"),
    )

    llm = VLLMProvider(
        base_url=os.getenv("TASA_LLM_URL", "http://localhost:8000/v1"),
        model=os.getenv("TASA_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    )

    tts = PiperTTS(
        model_path=os.getenv(
            "TASA_TTS_MODEL",
            "/usr/share/piper/voices/en_US-lessac-medium.onnx",
        ),
    )

    vad = SileroVAD(threshold=float(os.getenv("TASA_VAD_THRESHOLD", "0.5")))

    turn_detector = TurnDetector()
    interrupt_handler = InterruptHandler()
    turn_timing = TurnTiming()
    backchannel_gen = BackchannelGenerator()
    backchannel_timing = BackchannelTiming()
    turn_backchannel = TurnBackchannel(
        generator=backchannel_gen, timing=backchannel_timing
    )

    return stt, llm, tts, vad, turn_detector, interrupt_handler, turn_timing, turn_backchannel, backchannel_gen, backchannel_timing


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    try:
        stt, llm, tts, vad, td, ih, tt, tbc, bcg, bct = _build_providers()
        pipeline = StreamingPipeline(
            stt=stt, llm=llm, tts=tts, vad=vad,
            turn_detector=td, interrupt_handler=ih,
            turn_timing=tt, turn_backchannel=tbc,
            backchannel_generator=bcg, backchannel_timing=bct,
        )
        app.state.pipeline = pipeline
        app.state.start_time = time.time()
        await pipeline.start()
        logger.info("pipeline initialized")
    except Exception as e:
        logger.error(f"pipeline init failed: {e}")
        pipeline = None
        app.state.pipeline = None
        app.state.start_time = time.time()

    yield

    if pipeline:
        await pipeline.stop()
        logger.info("pipeline stopped")


app = FastAPI(
    title="TASA",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)

app.include_router(health_router)
app.include_router(ws_router)
app.include_router(chat_router)
app.include_router(metrics_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error"},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "TASA",
        "version": "0.2.0",
        "status": "running" if app.state.pipeline else "degraded",
    }


@app.get("/mic", response_class=HTMLResponse)
def mic_ui():
    return (Path(__file__).parent / "mic.html").read_text()


@app.get("/ui", response_class=HTMLResponse)
def full_ui():
    return (Path(__file__).parent / "ui.html").read_text()
