from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.ws import router as ws_router
from core.pipeline import StreamingPipeline
from core.config import CoreConfig
from utils.logger import logger

config = CoreConfig()

pipeline: StreamingPipeline | None = None


def _build_providers():
    from providers.stt.faster_whisper_stt import FasterWhisperSTT
    from providers.llm.vllm_llm import VLLMProvider
    from providers.tts.piper_tts import PiperTTS
    from modules.vad.silero_vad import SileroVAD

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

    return stt, llm, tts, vad


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    try:
        stt, llm, tts, vad = _build_providers()
        pipeline = StreamingPipeline(stt=stt, llm=llm, tts=tts, vad=vad)
        app.state.pipeline = pipeline
        await pipeline.start()
        logger.info("TASA pipeline initialized and running")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.warning("Server running without pipeline — connect providers later via API")
        pipeline = None
        app.state.pipeline = None

    yield

    if pipeline:
        await pipeline.stop()
        logger.info("TASA pipeline stopped")


app = FastAPI(title="TASA", lifespan=lifespan)

app.include_router(health_router)
app.include_router(ws_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "TASA real-time voice agent is running"}


@app.get("/config")
def get_config() -> dict[str, str]:
    return {
        "stt_model": os.getenv("TASA_STT_MODEL", "base"),
        "stt_device": os.getenv("TASA_STT_DEVICE", "cpu"),
        "llm_url": os.getenv("TASA_LLM_URL", "http://localhost:8000/v1"),
        "llm_model": os.getenv("TASA_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
        "tts_model": os.getenv("TASA_TTS_MODEL", "/usr/share/piper/voices/en_US-lessac-medium.onnx"),
        "vad_threshold": os.getenv("TASA_VAD_THRESHOLD", "0.5"),
    }