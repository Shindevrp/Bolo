FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio && \
    pip install --no-cache-dir ".[all,dashboard]"


FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY app/ app/
COPY core/ core/
COPY modules/ modules/
COPY providers/ providers/
COPY utils/ utils/
COPY bench/ bench/
COPY streamlit_app.py ./

ENV BOLO_STT_MODEL=base
ENV BOLO_STT_DEVICE=cpu
ENV BOLO_STT_COMPUTE=int8
ENV BOLO_LLM_URL=http://vllm:8000/v1
ENV BOLO_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
ENV BOLO_TTS_MODEL=/app/models/piper/en_US-lessac-medium.onnx
ENV BOLO_VAD_THRESHOLD=0.5

RUN mkdir -p /app/models/piper && python3 <<EOF
import urllib.request, pathlib
voices = {
    'en_US-lessac-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx',
    'en_GB-alan-low': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/low/en_GB-alan-low.onnx',
    'en_US-kristin-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium/en_US-kristin-medium.onnx',
}
dest_dir = pathlib.Path('/app/models/piper')
for name, url in voices.items():
    onnx = dest_dir / f'{name}.onnx'
    cfg = dest_dir / f'{name}.onnx.json'
    if not onnx.exists():
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(url, onnx)
        urllib.request.urlretrieve(url + '.json', cfg)
        print(f'{name} downloaded.')
    else:
        print(f'{name} already exists.')
EOF

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
