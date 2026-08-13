from __future__ import annotations

import asyncio
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
from app.routes.webrtc import router as webrtc_router
from app.routes.sessions import router as sessions_router
from core.pipeline import StreamingPipeline
from core.config import CoreConfig
from utils.logger import get_logger

logger = get_logger("server")

config = CoreConfig()
pipeline: StreamingPipeline | None = None


def _resolve_device(env_key: str) -> str:
    """Resolve a TASA_*_DEVICE env var.

    ``auto`` (the default when unset) picks cuda when available, otherwise cpu,
    so the same image runs on GPU and CPU machines without manual config.
    """
    value = os.getenv(env_key, "").strip().lower()
    if value in ("", "auto"):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return value


def _build_providers():
    from providers.stt.faster_whisper_stt import FasterWhisperSTT
    from providers.llm.vllm_llm import VLLMProvider
    from providers.tts.piper_tts import PiperTTS, PiperMultiVoice
    from modules.vad.silero_vad import SileroVAD
    from modules.turn.detector import TurnDetector
    from modules.turn.interrupt import InterruptHandler
    from modules.turn.timing import TurnTiming
    from modules.turn.backchannel import TurnBackchannel
    from modules.backchannel.generator import BackchannelGenerator
    from modules.backchannel.timing import BackchannelTiming
    from modules.emotion.classifier import EmotionClassifier
    from modules.speaker.profile import SpeakerProfile
    from modules.speaker.coordinator import SpeakerCoordinator

    stt = FasterWhisperSTT(
        model_size=os.getenv("TASA_STT_MODEL", "tiny"),
        device=_resolve_device("TASA_STT_DEVICE"),
        compute_type=os.getenv("TASA_STT_COMPUTE", "int8"),
    )

    llm = VLLMProvider(
        base_url=os.getenv("TASA_LLM_URL", "http://localhost:8000/v1"),
        model=os.getenv("TASA_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    )

    # Build TTS: multi-speaker or single voice
    speaker_coordinator = None
    if config.multi_speaker_enabled:
        from modules.speaker.interrupt_policy import InterruptPolicy

        s1 = SpeakerProfile(
            name=config.speaker_1_name,
            voice_model_path=config.speaker_1_voice,
            personality=config.speaker_1_personality,
            speaking_style=config.speaker_1_style,
            color="#3B82F6",
        )
        s2 = SpeakerProfile(
            name=config.speaker_2_name,
            voice_model_path=config.speaker_2_voice,
            personality=config.speaker_2_personality,
            speaking_style=config.speaker_2_style,
            color="#10B981",
        )
        tts_1 = PiperTTS(model_path=s1.voice_model_path)
        tts_2 = PiperTTS(model_path=s2.voice_model_path)
        tts = PiperMultiVoice(voices={s1.name: tts_1, s2.name: tts_2})

        interrupt_policy = InterruptPolicy(
            urgency_threshold=config.interrupt_urgency_threshold,
            max_consecutive_turns=config.max_consecutive_turns,
            interrupt_mode=config.interrupt_mode,
        )
        speaker_coordinator = SpeakerCoordinator(
            [s1, s2],
            interrupt_policy=interrupt_policy,
            overlap_ms=config.overlap_ms,
        )
        logger.info(
            f"multi-speaker enabled: {s1.name} + {s2.name} "
            f"interrupt_mode={config.interrupt_mode} "
            f"overlap={config.overlap_ms}ms"
        )
    else:
        tts = PiperTTS(
            model_path=os.getenv(
                "TASA_TTS_MODEL",
                str(Path(__file__).parent.parent / "models" / "en_US-lessac-medium.onnx"),
            ),
        )

    vad = SileroVAD(
        threshold=float(os.getenv("TASA_VAD_THRESHOLD", "0.5")),
        device=_resolve_device("TASA_VAD_DEVICE"),
    )

    turn_detector = TurnDetector()
    interrupt_handler = InterruptHandler()
    turn_timing = TurnTiming()
    backchannel_gen = BackchannelGenerator()
    backchannel_timing = BackchannelTiming()
    turn_backchannel = TurnBackchannel(
        generator=backchannel_gen, timing=backchannel_timing
    )

    emotion = EmotionClassifier(
        enabled=config.emotion_enabled,
        model_name=config.emotion_model,
        device=_resolve_device("TASA_EMOTION_DEVICE"),
    )

    return (stt, llm, tts, vad, turn_detector, interrupt_handler, turn_timing,
            turn_backchannel, backchannel_gen, backchannel_timing, emotion,
            speaker_coordinator)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    try:
        (stt, llm, tts, vad, td, ih, tt, tbc, bcg, bct, emotion,
         speaker_coordinator) = (
            _build_providers()
        )
        pipeline = StreamingPipeline(
            stt=stt, llm=llm, tts=tts, vad=vad,
            turn_detector=td, interrupt_handler=ih,
            turn_timing=tt, turn_backchannel=tbc,
            backchannel_generator=bcg, backchannel_timing=bct,
            emotion_classifier=emotion,
            speaker_coordinator=speaker_coordinator,
        )
        app.state.pipeline = pipeline
        app.state.start_time = time.time()
        await pipeline.start()
        # Warm up LLM so first user request doesn't pay 30s model load
        try:
            logger.info("warming up LLM...")
            warmup_msgs = [{"role": "user", "content": "hi"}]
            async for _ in llm.generate_stream(warmup_msgs):
                pass
            logger.info("LLM warmed up")
        except Exception as e:
            logger.warning(f"LLM warmup failed (non-critical): {e}")
        # Warm up emotion classifier so first turn doesn't pay model load
        try:
            await asyncio.to_thread(emotion.load)
            if emotion.loaded:
                logger.info(
                    f"emotion classifier warmed up ({emotion.model_name})"
                )
        except Exception as e:
            logger.warning(f"emotion warmup failed (non-critical): {e}")
        # Warm up the shared retrieval encoder so it is never loaded on the
        # audio hot path (loading it mid-session stalls the VAD loop).
        try:
            from modules.memory.vector_db import _get_encoder

            await asyncio.to_thread(_get_encoder)
            logger.info("retrieval encoder warmed up")
        except Exception as e:
            logger.warning(f"retrieval encoder warmup failed (non-critical): {e}")
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
app.include_router(webrtc_router)
app.include_router(sessions_router)


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


@app.get("/webrtc", response_class=HTMLResponse)
def webrtc_ui():
    return (Path(__file__).parent / "webrtc.html").read_text()
