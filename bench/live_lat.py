"""Realtime latency recorder for TASA's WebSocket voice endpoint.

Streams the user's microphone (or a WAV file) to ``/ws/audio`` and, on the
background, records every pipeline event with a client-side timestamp to
segment each turn and compute a per-stage latency breakdown:

    speech_ms            speech_start -> speech_end      (utterance + endpoint wait)
    commit_to_stt_ms     speech_end -> transcript        (STT tail after commit)
    stt_to_first_token   transcript -> first llm_token
    first_token_to_done  first llm_token -> llm_done
    commit_to_reply_ms   speech_end -> llm_done          (brain: reply text ready)
    reply_to_tts_first   llm_done -> first TTS chunk
    commit_to_tts_first  speech_end -> first TTS chunk   (user-perceived response)
    commit_to_tts_done   speech_end -> tts_done          (full turn incl. audio)

Turns are printed live as they finish and summarized (p50/p95/p99/mean) at the
end; ``--baseline`` diffs two traces to compare deployments (e.g. Laya shadow
off vs on) turn for turn against the same input.

The pure turn-tracking logic (``TraceRecorder`` / ``summarize``) has no
dependency on network, mic, or speaker and is unit-tested in
``tests/test_live_lat.py``; capture pieces import sounddevice/websockets lazily
so the summarizer works headlessly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

# Stage derivation: (key, label, later-ts-key, earlier-ts-key).
_STAGES: list[tuple[str, str, str, str]] = [
    ("speech_ms", "utterance+endpoint", "end_ts", "start_ts"),
    ("commit_to_stt_ms", "stt tail", "transcript_ts", "end_ts"),
    ("stt_to_first_token", "stt->first token", "first_token_ts", "transcript_ts"),
    ("first_token_to_done", "llm rest", "done_ts", "first_token_ts"),
    ("commit_to_reply_ms", "reply ready", "done_ts", "end_ts"),
    ("reply_to_tts_first", "tts first byte", "tts_first_ts", "done_ts"),
    ("commit_to_tts_first", "perceived response", "tts_first_ts", "end_ts"),
    ("commit_to_tts_done", "full turn (rtt)", "tts_done_ts", "end_ts"),
    ("tts_audio_ms", "tts audio", "tts_done_ts", "tts_first_ts"),
]

STAGES: list[tuple[str, str]] = [(k, label) for k, label, _, _ in _STAGES]

_RECOGNIZED = {
    "speech_start", "speech_end", "partial_transcript", "transcript",
    "llm_token", "llm_done", "tts_done", "backchannel", "interrupt",
    "status", "error", "prosody",
}


class TraceRecorder:
    """Segments a WS event stream into turns and derives stage timings.

    Feed events via :meth:`feed(kind, ts, text=None, nbytes=0)`` where *kind*
    is a ``/ws/audio`` event name (or ``"tts_chunk"`` for binary audio); *ts*
    is a monotonic wall-clock timestamp in seconds.
    """

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []
        self._cur: dict[str, Any] | None = None

    def feed(
        self,
        kind: str,
        ts: float,
        text: str | None = None,
        nbytes: int = 0,
    ) -> dict[str, Any] | None:
        """Push one event. Returns the freshly-closed turn, if any."""
        if kind == "tts_chunk":
            if self._cur is not None:
                if self._cur.get("tts_first_ts") is None:
                    self._cur["tts_first_ts"] = ts
                self._cur["tts_bytes"] = self._cur.get("tts_bytes", 0) + nbytes
            return None

        if kind == "speech_start":
            if self._cur is not None:
                self._close(aborted=True)
            self._cur = {"start_ts": ts, "aborted": False, "tts_bytes": 0}
            return None

        if self._cur is None:
            return None

        if kind == "speech_end":
            self._cur["end_ts"] = ts
        elif kind == "transcript":
            self._cur["transcript_ts"] = ts
            if text:
                self._cur["user_text"] = text
        elif kind == "llm_token":
            if self._cur.get("first_token_ts") is None:
                self._cur["first_token_ts"] = ts
        elif kind == "llm_done":
            self._cur["done_ts"] = ts
            if text:
                self._cur["reply_text"] = text
        elif kind == "tts_done":
            self._cur["tts_done_ts"] = ts
            return self._close()
        elif kind == "interrupt":
            return self._close(aborted=True)
        return None

    def _close(self, *, aborted: bool = False) -> dict[str, Any]:
        assert self._cur is not None
        cur = dict(self._cur)
        cur["aborted"] = cur.get("aborted") or aborted
        anchor = cur.get(
            "done_ts", cur.get("tts_done_ts", cur.get("end_ts", cur["start_ts"]))
        )
        cur["dur_ms"] = round((anchor - cur["start_ts"]) * 1000.0, 1)
        self._derive(cur)
        self.turns.append(cur)
        self._cur = None
        return cur

    @staticmethod
    def _derive(cur: dict[str, Any]) -> None:
        for key, _label, later, earlier in _STAGES:
            a = cur.get(later)
            b = cur.get(earlier)
            if a is not None and b is not None:
                cur[key] = round((a - b) * 1000.0, 1)

    def close(self) -> None:
        if self._cur is not None:
            self._close(aborted=True)


def load_trace(path: str) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "dur_ms" in obj:
                turns.append(obj)
    return turns


def summarize(turns: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"count": len(turns)}
    for key, _label in STAGES:
        # Negative deltas happen when TASA streams TTS before llm_done; those
        # buckets carry no latency meaning and are excluded.
        vals = sorted(
            t[key] for t in turns if key in t and t[key] is not None and t[key] >= 0
        )
        if not vals:
            continue
        stats[key] = {
            "count": len(vals),
            "p50": round(statistics.median(vals), 1),
            "p95": round(vals[min(len(vals) - 1, int(len(vals) * 0.95))], 1),
            "p99": round(vals[min(len(vals) - 1, int(len(vals) * 0.99))], 1),
            "mean": round(statistics.mean(vals), 1),
        }
    return stats


def render_summary(stats: dict[str, Any]) -> str:
    lines = ["", f"turns: {stats.get('count', 'n/a')}", ""]
    lines.append(f"{'stage':<22}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}")
    lines.append("-" * 58)
    for key, label in STAGES:
        s = stats.get(key)
        if s is None:
            continue
        lines.append(
            f"{label:<22}{s['p50']:>7.1f}ms {s['p95']:>7.1f}ms "
            f"{s['p99']:>7.1f}ms {s['mean']:>7.1f}ms"
        )
    lines.append("-" * 58)
    return "\n".join(lines)


def render_delta(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> str:
    labels = dict(STAGES)
    lines = [
        "",
        f"delta (baseline={len(a)} turns vs new={len(b)} turns), ms",
        f"{'stage':<22}{'baseline':>10}{'new':>10}{'Δms':>9}{'Δ%':>9}",
        "-" * 60,
    ]
    for key, _label in STAGES:
        a_ = [t[key] for t in a if key in t and t[key] is not None and t[key] >= 0]
        b_ = [t[key] for t in b if key in t and t[key] is not None and t[key] >= 0]
        if not a_ or not b_:
            continue
        ma, mb = statistics.mean(a_), statistics.mean(b_)
        d = mb - ma
        pct = (d / ma) * 100.0 if ma else 0.0
        lines.append(
            f"{labels[key]:<22}{ma:>9.1f}ms{mb:>9.1f}ms{d:>+8.1f}{pct:>+8.1f}%"
        )
    lines.append("-" * 60)
    return "\n".join(lines)


def save_trace(path: str, turns: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        f.writelines(json.dumps(turn) + "\n" for turn in turns)


def print_turn(turn: dict[str, Any]) -> None:
    user = (turn.get("user_text") or "").strip()[:40]
    rtt = turn.get("commit_to_tts_first")
    rtt_s = f"{rtt:.1f}ms" if rtt is not None else "-"
    aborted = " [aborted]" if turn.get("aborted") else ""
    print(f"  {user:<42}response {rtt_s}{aborted}")


# --------------------------------------------------------------------------
# Live capture (lazy imports: sounddevice, websockets, soundfile)
# --------------------------------------------------------------------------

_MIC_RATE = 16000
_DEFAULT_WS_URL = "ws://localhost:8000/ws/audio"


async def _recv_loop(ws: Any, recorder: TraceRecorder, playback: Any) -> None:
    async for raw in ws:
        ts = time.monotonic()
        if isinstance(raw, bytes):
            recorder.feed("tts_chunk", ts, nbytes=len(raw))
            if playback is not None:
                try:
                    playback.write(raw)
                except Exception:  # noqa: BLE001, S110 - playback best-effort
                    pass
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = data.get("type")
        if kind not in _RECOGNIZED:
            continue
        if kind == "transcript":
            closed = recorder.feed("transcript", ts, text=data.get("text"))
        elif kind == "llm_done":
            closed = recorder.feed("llm_done", ts, text=data.get("text"))
        else:
            closed = recorder.feed(kind, ts)
        if closed is not None and not closed.get("aborted"):
            print_turn(closed)


async def _mic_loop(ws: Any, queue: asyncio.Queue[bytes], stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.2)
        except TimeoutError:
            continue
        await ws.send(chunk)


async def _file_loop(ws: Any, frames: list[bytes], stop: asyncio.Event) -> None:
    for frame in frames:
        if stop.is_set():
            return
        await ws.send(frame)
        await asyncio.sleep(0.02)


def get_playback(rate: int | None):
    if rate is None:
        return None
    import sounddevice as sd

    return sd.OutputStream(samplerate=rate, channels=1, dtype="int16")


def run_capture(
    *,
    url: str = _DEFAULT_WS_URL,
    source: str | None = None,
    mic_device: int | None = None,
    playback_rate: int = 22050,
    mute: bool = False,
    seconds: float = 0.0,
    label: str = "live",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import sounddevice as sd
    from websockets.asyncio.client import connect

    recorder = TraceRecorder()
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    stop = asyncio.Event()
    playback = get_playback(None if mute else playback_rate)

    async def _main() -> None:
        async with connect(url, max_size=64 * 1024 * 1024) as ws:
            tasks: list[asyncio.Task] = []

            if source is not None:
                import soundfile as sf

                data, sr = sf.read(source, dtype="int16", always_2d=True)
                if sr != _MIC_RATE:
                    raise SystemExit(
                        f"--file must be {_MIC_RATE}Hz mono PCM, got sr={sr}"
                    )
                frames = [data[i:i + 320].tobytes() for i in range(0, len(data), 320)]
                tasks.append(asyncio.create_task(_file_loop(ws, frames, stop)))
            else:
                def _cb(indata, _frames, _t, _status) -> None:
                    if len(indata):
                        mic_queue.put_nowait(indata.tobytes())

                stream = sd.InputStream(
                    samplerate=_MIC_RATE, channels=1, dtype="int16",
                    blocksize=320, callback=_cb, device=mic_device,
                )
                stream.start()
                tasks.append(asyncio.create_task(_mic_loop(ws, mic_queue, stop)))

            recv_task = asyncio.create_task(_recv_loop(ws, recorder, playback))
            tasks.append(recv_task)

            if seconds > 0:
                await asyncio.sleep(seconds)
                stop.set()
                await asyncio.sleep(0.5)
                recv_task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    stop.set()
                    recv_task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            if source is None:
                stream.stop()
                stream.close()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    finally:
        if playback is not None:
            try:
                playback.close()
            except Exception:  # noqa: BLE001, S110 - teardown best-effort
                pass

    recorder.close()
    return recorder.turns, {"label": label, "source": source or "mic", "url": url}


def cmd_live(args) -> int:
    turns, meta = run_capture(
        url=args.url,
        source=args.file,
        mic_device=args.device,
        playback_rate=args.playback_rate,
        mute=args.mute,
        seconds=args.seconds,
        label=args.label,
    )
    if not turns:
        print("no turns recorded")
        return 1
    stats = summarize(turns)
    print(render_summary(stats))
    print(f"\ntrace: {args.out} ({len(turns)} turns)")
    save_trace(args.out, turns, meta)
    if args.baseline:
        base = load_trace(args.baseline)
        print(render_delta(base, turns))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tasa-live", description=__doc__.splitlines()[0])
    p.add_argument("--url", default=_DEFAULT_WS_URL, help="TASA /ws/audio endpoint")
    p.add_argument("--file", default=None, help="replay a 16000Hz mono WAV instead of mic")
    p.add_argument("--device", type=int, default=None, help="input device index (mic)")
    p.add_argument("--playback-rate", type=int, default=22050, help="TTS playback rate")
    p.add_argument("--mute", action="store_true", help="do not play replies")
    p.add_argument("--seconds", type=float, default=0.0, help="capture window (0 = until Ctrl-C)")
    p.add_argument("--out", default="live_trace.jsonl", help="trace output path")
    p.add_argument("--baseline", default=None, help="a previous trace to diff against")
    p.add_argument("--label", default="live", help="session label stored in the trace")
    args = p.parse_args(argv)
    return cmd_live(args)


if __name__ == "__main__":
    raise SystemExit(main())