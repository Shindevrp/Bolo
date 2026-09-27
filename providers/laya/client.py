"""System-1 decision layer backed by Laya.

Laya is a 421M non-autoregressive decision model: a single forward pass
evaluates typed ``choice`` / ``score`` / ``noul`` questions over a text or JSON
state and returns typed answers with probabilities. It never generates text,
so there is nothing to parse and nothing to hallucinate.

Phase 1 = **shadow mode**: the pipeline runs Laya alongside the existing
deterministic classifiers, records agreement/skew, and NEVER overrides them.
Phase 2 = **enforcement** (opt-in, ``BOLO_LAYA_PHASE2=1``): the pipeline writes
confident answers onto the conversation context; this client stays policy-free
-- it only extracts typed answers and applies confidence thresholds
(``choice`` for classification, ``action_noul``'s stricter 0.9 for routing
flags). Enforcement lives in ``core.pipeline._shadow_cadence2``. Every call
fails open: any load error, timeout, or low-confidence answer degrades to
today's behavior. Nothing in this module can block the audio orchestration
loop for more than ``shadow_timeout`` seconds.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from statistics import median

from providers.laya.store import ShadowLogStore
from utils.logger import get_logger

logger = get_logger("laya")

try:  # optional dependency: the pipeline must run without `laya` installed
    import laya as _laya
except Exception:  # pragma: no cover - import safety
    _laya = None  # type: ignore[assignment]

SHADOW_TIMEOUT = 2.0

#: Questions with a real legacy classifier, so Laya-agrees-with-legacy is a
#: usable (if imitative) label. Everything else is labelled from outcomes only.
AGREE_LABEL_QUESTIONS = frozenset({
    "intent", "is_question", "query_complexity", "topic_changed",
    "needs_verify", "sentiment",
})
_ACT_PROB_REQ = 0.9


def _resolve_device(device: str | None) -> str:
    """Map a BOLO_LAYA_DEVICE value ('', 'auto') to a concrete device."""
    value = (device or "").strip().lower()
    if value in ("", "auto"):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return value


class LayaSystem1:
    """Async wrapper around the Laya agent.

    Loads lazily on first use (so startup is never blocked), serializes
    inference behind a single semaphore, bounds shadow-forward latency, and
    keeps a bounded log of Laya-vs-legacy decisions for Phase-2 rollout.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        model: str = "convaiinnovations/laya",
        device: str | None = None,
        conf_threshold: float = 0.5,
        shadow_timeout: float = SHADOW_TIMEOUT,
        max_shadow: int = 2000,
        use_fp16: bool = True,
        shadow_log: str | None = None,
        agent=None,
        allow_cpu: bool = False,
    ) -> None:
        self._enabled = enabled and _laya is not None
        self._model = model
        self._device = _resolve_device(device)
        # fp16 probe weights: halves memory bandwidth / speeds the forward
        # pass when sharing the GPU with the LLM (see ``_convert_fp16``).
        self._use_fp16 = use_fp16
        # A turn-level pass takes seconds on CPU, far past ``shadow_timeout``:
        # every call would time out and only add latency. Without a GPU the
        # layer stays off unless the device is set explicitly (or allow_cpu).
        explicit = (device or "").strip().lower() not in ("", "auto")
        if (
            self._enabled
            and agent is None
            and self._device == "cpu"
            and not (explicit or allow_cpu)
        ):
            logger.warning(
                "laya disabled: no GPU found (set BOLO_LAYA_DEVICE=cpu to force)"
            )
            self._enabled = False
        self._conf_threshold = conf_threshold
        self._shadow_timeout = shadow_timeout
        # Injection point for tests / bench fakes (an object with a sync
        # `.predict(state, questions)` returning {"answers": {...}}).
        self._agent = agent
        self._load_error: str | None = None
        self._semaphore = asyncio.Semaphore(1)
        self._priority_waiters = 0
        self._latency_ms: deque[float] = deque(maxlen=200)
        self._shadow: deque[dict] = deque(maxlen=max_shadow)
        # Step 2: optional disk-backed shadow log (see providers.laya.store).
        # Off unless the caller passes a path -- keeps tests and one-off
        # evaluations side-effect free.
        self._store = ShadowLogStore(shadow_log) if shadow_log else None

    # ------------------------------------------------------------------
    # availability / load
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def device(self) -> str:
        return self._device

    @property
    def available(self) -> bool:
        return self._enabled and (self._agent is not None or self._load_error is None)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def warmup(self) -> None:
        """Preload the checkpoint so the first live turn never pays model load."""
        if not self._enabled:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self):  # plain lazy loader
        if self._agent is not None or self._load_error is not None:
            return self._agent
        if _laya is None:
            self._load_error = "laya package not installed"
            return None
        try:
            t0 = time.perf_counter()
            self._agent = _laya.load(self._model, device=self._device)
            if self._use_fp16:
                self._convert_fp16()
            logger.info(
                f"laya loaded model={self._model} device={self._device} "
                f"in {(time.perf_counter() - t0) * 1000:.0f}ms"
            )
        except Exception as e:
            self._load_error = str(e)
            logger.warning(
                f"laya unavailable, shadow mode disabled: "
                f"model={self._model} error={e!r}"
            )
            self._agent = None
        return self._agent

    def _convert_fp16(self) -> None:
        """Best-effort fp16 cast of the probe weights.

        Halves memory bandwidth for the shared-GPU turn-level and
        live-listening passes. Introspects the loaded agent object graph for
        ``torch.nn.Module`` instances and casts them to float16. Anything the
        introspection cannot find is left untouched -- this is fail-open: if
        the checkpoint exposes no obvious torch modules, Laya simply stays
        fp32 and nothing here ever raises.
        """
        try:
            import torch
        except Exception:
            return
        modules: list[torch.nn.Module] = []
        seen: set[int] = set()

        def _collect(obj) -> None:
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                modules.append(obj)
            if isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)
            elif isinstance(obj, (list, tuple, set)):
                for v in obj:
                    _collect(v)
            elif not isinstance(obj, (str, bytes, int, float, bool, type(None))):
                try:
                    for v in vars(obj).values():
                        _collect(v)
                except TypeError:
                    pass

        _collect(self._agent)
        converted = 0
        skipped = 0
        for m in modules:
            try:
                m.half()
                converted += 1
            except Exception as e:  # noqa: BLE001 - fp16 is best-effort
                skipped += 1
                if skipped <= 3:
                    logger.debug(f"laya: could not cast module to fp16: {e!r}")
        if converted:
            logger.info(f"laya: cast {converted} module(s) to fp16")

    async def _load_async(self):
        if not self._enabled:
            return None
        if self._agent is not None or self._load_error is not None:
            return self._agent
        return await asyncio.to_thread(self._load_sync)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    async def predict(
        self, state: dict, questions: dict, *, priority: bool = True
    ) -> dict:
        """Run one batched forward pass; returns ``answers`` dict or {}.

        One inference runs at a time. The caller waits at most
        ``_shadow_timeout`` in total (lock wait + inference); on timeout it
        gets {} and the legacy path decides. The lock is released only when
        the inference thread really finishes, so calls never pile up.

        ``priority=False`` marks an opportunistic call (the live-listening
        probe): it never waits, and is skipped when the model is busy or a
        priority (turn-level) call is waiting for it.
        """
        if not self._enabled:
            return {}
        agent = await self._load_async()
        if agent is None:
            return {}
        t0 = time.perf_counter()
        if not priority:
            if self._semaphore.locked() or self._priority_waiters:
                return {}
            await self._semaphore.acquire()  # free: returns without waiting
        else:
            self._priority_waiters += 1
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(), timeout=self._shadow_timeout
                )
            except TimeoutError:
                logger.debug("laya predict: model busy, skipping")
                return {}
            finally:
                self._priority_waiters -= 1

        remaining = self._shadow_timeout - (time.perf_counter() - t0)
        if remaining <= 0.0:
            # The lock wait used the whole budget: don't start an inference
            # whose answer nobody will read.
            self._semaphore.release()
            return {}

        holder: dict = {}

        async def _run() -> None:
            try:
                result = await asyncio.to_thread(agent.predict, state, questions)
                answers = result.get("answers") if isinstance(result, dict) else None
                if isinstance(answers, dict):
                    holder.update(answers)
            except Exception as e:
                logger.debug(f"laya predict failed: {e!r}")
            finally:
                self._semaphore.release()

        bg = asyncio.create_task(_run())
        try:
            await asyncio.wait_for(asyncio.shield(bg), timeout=remaining)
        except TimeoutError:
            logger.debug(
                f"laya predict: over {self._shadow_timeout}s budget, skipping "
                f"(lock held until the thread finishes)"
            )
            return {}
        if not holder:
            return {}
        self._latency_ms.append((time.perf_counter() - t0) * 1000.0)
        return holder

    # ------------------------------------------------------------------
    # typed-answer extraction (returns None when absent / low confidence)
    # ------------------------------------------------------------------

    @staticmethod
    def _entry(answers: dict, key: str) -> dict | None:
        if not isinstance(answers, dict):
            return None
        e = answers.get(key)
        return e if isinstance(e, dict) else None

    def choice(self, answers: dict, key: str, default=None):
        """Return the selected label, or ``default`` when confidence is weak."""
        e = self._entry(answers, key)
        if e is None or "choice" not in e:
            return default
        if e.get("confidence", 0.0) < self._conf_threshold:
            return default
        return e["choice"]

    def choice_of(self, answers: dict, key: str):
        """Return (label, confidence, probabilities) without a threshold."""
        e = self._entry(answers, key)
        if e is None or "choice" not in e:
            return None, 0.0, {}
        return e["choice"], float(e.get("confidence", 0.0)), e.get("probabilities")

    def noul_prob(self, answers: dict, key: str) -> float | None:
        """Return P(true) for a noul question (no threshold applied)."""
        e = self._entry(answers, key)
        if e is None or "noul" not in e:
            return None
        return float(e["noul"])

    def noul(self, answers: dict, key: str, default: bool = False) -> bool:
        prob = self.noul_prob(answers, key)
        return (prob is not None and prob >= 0.5) if prob is not None else default

    def action_noul(self, answers: dict, key: str, default: bool = False) -> bool:
        """Strict-threshold noul for routing flags (Phase-2 enforcement).

        Base checkpoints ship uncalibrated, over-confident temperatures, so
        routing flags (``invoke_llm`` / ``escalate`` / ``urgent`` /
        ``tool_needed``) require P >= ``_ACT_PROB_REQ``. Anything weaker or
        missing returns ``default`` -- fail-open to legacy behavior.
        """
        prob = self.noul_prob(answers, key)
        if prob is None:
            return default
        return prob >= _ACT_PROB_REQ

    def score(self, answers: dict, key: str, default=None):
        """Return the expected ordinal level, or ``default`` when absent."""
        e = self._entry(answers, key)
        if e is None or "score" not in e:
            return default
        return e["score"]

    # ------------------------------------------------------------------
    # shadow bookkeeping
    # ------------------------------------------------------------------

    def _self_label(self, row: dict):
        """Training label for one row, or None.

        An ``outcome`` (what actually happened: the tool the LLM really
        called, whether the user kept talking after a partial) always wins --
        it is ground truth. Otherwise a row is labelled only when Laya agrees,
        confidently, with a REAL legacy classifier (``AGREE_LABEL_QUESTIONS``);
        questions whose "legacy" value is a hard-coded placeholder would only
        teach the placeholder. ``laya_conf`` is the confidence of the answer
        given, so a confident ``False`` counts too.
        """
        outcome = row.get("outcome")
        if outcome is not None:
            return outcome
        if row.get("question") not in AGREE_LABEL_QUESTIONS:
            return None
        if not row.get("match"):
            return None
        conf = row.get("laya_conf")
        if not isinstance(conf, (int, float)) or conf < self._conf_threshold:
            return None
        value = row.get("laya")
        if value is None or value == "":
            return None
        return value

    def log_shadow(
        self,
        *,
        session_id: str,
        transcript: str,
        rows: list[dict],
        state: dict | None = None,
    ) -> None:
        """Record Laya-vs-legacy rows for one turn. ``rows`` entries:
        {"question", "laya", "laya_conf", "legacy", "match"}. Each row gets a
        computed ``self_label`` (see ``_self_label``); the entry carries the
        ``state`` dict fed to the forward pass so Step-3 training can re-ask
        the exact question. When a ``shadow_log`` path was configured the
        entry is also appended to disk (fail-open, never raises)."""
        if not rows:
            return
        annotated = []
        for row in rows:
            label = self._self_label(row)
            source = None
            if label is not None:
                source = "outcome" if row.get("outcome") is not None else "agree"
            annotated.append({**row, "self_label": label, "label_source": source})
        entry = {
            "session_id": session_id,
            "transcript": transcript,
            "ts": time.time(),
            "state": state or {},
            "rows": annotated,
            "labelable": sum(
                1 for r in annotated if r.get("self_label") is not None
            ),
        }
        self._shadow.append(entry)
        if self._store:
            self._store.append(entry)

    def shadow_report(self) -> dict:
        """Per-question agreement between Laya and the legacy classifiers."""
        agg: dict[str, dict] = {}
        for entry in self._shadow:
            for row in entry.get("rows", []):
                q = row.get("question", "?")
                bucket = agg.setdefault(q, {"n": 0, "matched": 0, "skipped": 0})
                bucket["n"] += 1
                if row.get("laya") is None:
                    bucket["skipped"] += 1
                elif row.get("match"):
                    bucket["matched"] += 1
        out: dict[str, dict] = {}
        for q, b in agg.items():
            n = max(1, b["n"])
            out[q] = {
                "n": b["n"],
                "agreement": round(b["matched"] / n, 4),
                "skipped": b["skipped"],
            }
        return out

    def latency_report_ms(self) -> dict:
        samples = list(self._latency_ms)
        if not samples:
            return {}
        s = sorted(samples)
        return {
            "p50": round(median(samples), 2),
            "p95": round(s[int(len(s) * 0.95)], 2),
            "count": len(samples),
        }