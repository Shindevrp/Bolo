"""Deterministic endpointing / barge-in diagnostics from captured event timelines.

These are the "conversational mechanics" metrics that answer, per run, how well
the agent manages turn-taking boundaries and interruption behaviour:

  False Endpoint Rate       How often TASA cuts the user off
  Missed Endpoint Rate      How often TASA waits after the user finished
  Endpoint P50/P95          Typical/worst response-start delay
  Utterance Fragment Rate   How often one thought becomes multiple turns
  Completion Capture Rate   % of utterances captured completely
  Barge-in Detection Rate   % of real interruptions detected
  False Barge-in Rate       % of harmless backchannels/noise treated as interruptions
  Barge-in Stop P50/P95     How quickly TTS stops
  Recovery Success Rate     % of interruptions that recover to a new turn

Everything is computed from client-observed events (never model internals), and
only from events that were actually captured -- a missing event yields None, we
never invent a number. Rates are 0.0-1.0 with an explicit denominator so the
reader can judge the sample size. All latency deltas are in seconds.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from bench.agent.base import EventType
from bench.driver.runner import ScenarioResult, Timeline

# Thresholds (seconds) for the diagnostics; documented + tunable so results are
# reproducible.
ENDPOINT_LATE_S = 3.0    # speech_end -> response beyond this = missed endpoint
BARGE_STOP_TAIL_S = 2.0  # how long after a barge-in start we look for the last TTS audio


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UserTurn:
    """One user utterance (speech window) plus the agent's response timing."""
    speech_start: float
    speech_end: float | None
    transcript: str | None = None
    transcript_time: float | None = None
    partials: list[tuple[float, str]] = field(default_factory=list)
    # True if the agent began generating/responding before the user finished
    # (i.e. between speech_start and speech_end) -> a "cut the user off" event.
    pre_end_response: bool = False
    # First agent output (LLM or TTS) strictly AFTER speech_end; None if the
    # agent never responded after the user finished.
    response_after_end: float | None = None
    interrupt_time: float | None = None


@dataclass
class BargeAttempt:
    """A scripted real-interruption injection and how the agent reacted."""
    start: float                  # when the interruption audio began going in
    text: str
    saw_interrupt: bool = False
    interrupt_event_t: float | None = None
    tts_stop: float | None = None   # last TTS audio still playing when interrupted
    already_idle: bool = False      # playback had already ended before the cut-in
    recovered: bool | None = None


@dataclass
class BackchannelAttempt:
    """A scripted harmless backchannel/noise injection."""
    start: float
    text: str
    saw_interrupt: bool = False
    interrupt_event_t: float | None = None


@dataclass
class EndpointSample:
    turns: list[UserTurn] = field(default_factory=list)
    spoken_texts: list[tuple[str, float]] = field(default_factory=list)  # (ground-truth text, t)
    barge_attempts: list[BargeAttempt] = field(default_factory=list)
    backchannels: list[BackchannelAttempt] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Turn reconstruction from a single timeline.
# ---------------------------------------------------------------------------

def reconstruct_user_turns(tl: Timeline) -> list[UserTurn]:
    ev = [(e.type, e.mtime, e.text) for e in tl.events]
    starts = [(t, x) for (et, t, x) in ev if et == EventType.SPEECH_START]
    ends = [t for (et, t, _) in ev if et == EventType.SPEECH_END]
    transcripts = [(t, x) for (et, t, x) in ev if et == EventType.TRANSCRIPT]
    partials = [(t, x) for (et, t, x) in ev if et == EventType.PARTIAL_TRANSCRIPT]
    llms = [t for (et, t, _) in ev if et == EventType.LLM_TOKEN]
    ttss = [t for (et, t, _) in ev if et == EventType.TTS_CHUNK]
    tts_dones = [t for (et, t, _) in ev if et == EventType.TTS_DONE]
    interrupts = [t for (et, t, _) in ev if et == EventType.INTERRUPT]

    def first_after(times, ref):
        a = [x for x in times if x > ref + 1e-6]
        return min(a) if a else None

    turns: list[UserTurn] = []
    for (st, _) in starts:
        end_t = next((e for e in ends if e > st + 1e-6), None)
        turns.append(UserTurn(speech_start=st, speech_end=end_t))

    # Attach transcripts/partials to whichever turn's window contains them.
    for (t, x) in transcripts:
        for turn in turns:
            if turn.speech_end is None:
                if t >= turn.speech_start:
                    turn.transcript = turn.transcript or x
                    turn.transcript_time = turn.transcript_time if turn.transcript_time is not None else t
                    break
            elif turn.speech_start - 1e-3 <= t < turn.speech_end + 9999:
                turn.transcript = turn.transcript or x
                if turn.transcript_time is None:
                    turn.transcript_time = t
                break
    for (t, x) in partials:
        for turn in turns:
            if (turn.speech_end is None and t >= turn.speech_start) or (
                turn.speech_end is not None and turn.speech_start - 1e-3 <= t < turn.speech_end + 9999
            ):
                turn.partials.append((t, x))
                break

    response_times = sorted(
        [t for t in llms] + [t for t in ttss]
    )
    for idx, turn in enumerate(turns):
        if turn.speech_end is None:
            continue
        se = turn.speech_end
        # Upper bound for this turn's response window = the start of the next
        # user turn (so we never attribute the NEXT turn's response to this one).
        ub = turns[idx + 1].speech_start if idx + 1 < len(turns) else float("inf")
        # A response begun while the user was still talking (before speech_end
        # but after speech_start) is a "cut off" / false endpoint.
        pre = [t for t in response_times if turn.speech_start - 1e-3 <= t < se - 1e-3]
        turn.pre_end_response = bool(pre)
        # The legit response to this turn = earliest agent output after its
        # speech_end but before the next user turn begins.
        after = [t for t in response_times if se + 1e-6 < t < ub]
        turn.response_after_end = after[0] if after else None
        turn.interrupt_time = first_after(interrupts, se)

    return turns


# ---------------------------------------------------------------------------
# Harvest samples from scenario results.
# ---------------------------------------------------------------------------

def collect_endpoint_samples(
    results: list[ScenarioResult],
    spoken_by_scenario: dict[str, list[tuple[str, float]]] | None = None,
) -> EndpointSample:
    samples = EndpointSample()
    for r in results:
        tl = r.timeline
        turns = reconstruct_user_turns(tl)
        samples.turns.extend(turns)

        # spoken ground truth from the scenario script if supplied
        spoken = (spoken_by_scenario or {}).get(r.scenario.id, [])
        base = min((t.speech_start for t in turns), default=0.0)
        for (text, t_off) in spoken:
            samples.spoken_texts.append((text, base + t_off))

        # classify injections: real barge-ins vs backchannels
        steps = r.scenario.steps
        all_irq = sorted(e.mtime for e in tl.events if e.type == EventType.INTERRUPT)
        tts_times = sorted(e.mtime for e in tl.events if e.type == EventType.TTS_CHUNK)
        speech_starts = sorted(e.mtime for e in tl.events if e.type == EventType.SPEECH_START)
        playback_t = tts_times[0] if tts_times else None
        # Injection onsets are the user speech_starts that occur once playback
        # has begun (the original question is before playback and is skipped).
        # Each injection step consumes the NEXT such speech_start in order, so
        # distinct injections get distinct onsets instead of all mapping to the
        # first post-playback utterance.
        cursor = next((i for i, st in enumerate(speech_starts) if playback_t is not None and st > playback_t + 1e-6), 0)
        injection_steps = [s for s in steps if s.kind in ("interrupt_during_playback", "interrupt_during_llm")]
        for step in injection_steps:
            text = (step.text or "").strip()
            is_bc = any(k in text.lower() for k in ("uh-huh", "uh huh", "yeah", "right", "okay", "ok"))
            if cursor < len(speech_starts):
                inject = speech_starts[cursor]
                cursor += 1
            elif playback_t is not None:
                inject = playback_t + 0.1
            else:
                inject = 0.0
            # The INTERRUPT event the agent emits once it detects the cut-in.
            irq_t = next(
                (x for x in all_irq if x > inject + 1e-6 and (playback_t is None or x > playback_t)),
                None,
            )
            # TTS actually running when interrupted = last audio emitted strictly
            # between the interruption onset and detection. If none, playback had
            # already finished (nothing to stop) -> we do not clock a stop latency.
            playing = [t for t in tts_times if inject + 1e-6 < t < (irq_t if irq_t else float("inf"))]
            tts_stop = playing[-1] if playing else None
            already_idle = tts_stop is None
            if is_bc:
                samples.backchannels.append(
                    BackchannelAttempt(start=inject, text=text, saw_interrupt=irq_t is not None, interrupt_event_t=irq_t)
                )
            else:
                samples.barge_attempts.append(
                    BargeAttempt(
                        start=inject, text=text,
                        saw_interrupt=irq_t is not None, interrupt_event_t=irq_t,
                        tts_stop=tts_stop, already_idle=already_idle,
                    )
                )
    return samples


# ---------------------------------------------------------------------------
# The 10 metrics.
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    value: float | None
    numerator: int
    denominator: int
    note: str = ""


def _p(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    frac = k - f
    return sorted_vals[f] + frac * (sorted_vals[c] - sorted_vals[f])


def compute_endpoint_metrics(samples: EndpointSample) -> dict[str, MetricResult]:
    out: dict[str, MetricResult] = {}
    turns = samples.turns

    # --- 1. False Endpoint Rate ---
    false_end, den = 0, 0
    for t in turns:
        if t.speech_end is None:
            continue
        den += 1
        if t.pre_end_response:
            false_end += 1
    out["false_endpoint_rate"] = MetricResult(
        value=(false_end / den) if den else None, numerator=false_end, denominator=den,
        note="agent response began before user speech_end (cut the user off)",
    )

    # --- 2. Missed Endpoint Rate ---
    # TASA waits too long after the user has finished: the response arrives
    # only beyond ENDPOINT_LATE_S after speech_end, or never. Turns where the
    # agent actually cut the user off (false endpoint) do NOT count as missed —
    # the agent did not wait, it pre-empted (but they stay in the denominator).
    missed, den = 0, 0
    for t in turns:
        if t.speech_end is None:
            continue
        den += 1
        if t.pre_end_response:
            continue  # cut the user off; not a "waited too long"
        resp = t.response_after_end
        if resp is None or (resp - t.speech_end) > ENDPOINT_LATE_S:
            missed += 1
    out["missed_endpoint_rate"] = MetricResult(
        value=(missed / den) if den else None, numerator=missed, denominator=den,
        note=f"speech_end -> response > {ENDPOINT_LATE_S}s or no response",
    )

    # --- 3. Endpoint P50/P95 ---
    # Response-start delay (speech_end -> first agent audio), restricted to
    # turns where the agent responded after the user finished (a real wait).
    delays = []
    for t in turns:
        if t.speech_end is None or t.response_after_end is None:
            continue
        delay = t.response_after_end - t.speech_end
        if delay >= 0:
            delays.append(delay)
    delays = sorted(delays)
    n_endpoint = sum(1 for t in turns if t.speech_end is not None)
    out["endpoint_p50"] = MetricResult(
        value=_p(delays, 50), numerator=len(delays), denominator=n_endpoint,
        note="speech_end -> first agent audio (P50)",
    )
    out["endpoint_p95"] = MetricResult(
        value=_p(delays, 95), numerator=len(delays), denominator=n_endpoint,
        note="speech_end -> first agent audio (P95)",
    )

    # --- 4. Utterance Fragment Rate ---
    # One thought delivered as multiple committed transcript turns. We compare
    # repeated transcript text across turns within a scenario: an identical
    # transcript appearing again after a prior one is evidence of a split.
    fragments, frag_den = 0, 0
    seen: dict[str, int] = {}
    per_scene = {}
    for t in turns:
        if t.transcript is None:
            continue
        frag_den += 1
        key = t.transcript.strip().lower()
        seen[key] = seen.get(key, 0) + 1
    fragments = sum(1 for k, c in seen.items() if c > 1)
    out["utterance_fragment_rate"] = MetricResult(
        value=(fragments / frag_den) if frag_den else None, numerator=fragments, denominator=frag_den,
        note="same transcript committed more than once (split turn)",
    )

    # --- 5. Completion Capture Rate ---
    if samples.spoken_texts:
        captured, cap_den = 0, len(samples.spoken_texts)
        all_texts = [(t.transcript or "").strip().lower() for t in turns]
        for (text, _t) in samples.spoken_texts:
            low = (text or "").strip().lower()
            hit = any(low in x or x in low for x in all_texts if x)
            if hit:
                captured += 1
        note = "utterance fully captured (reference contained/matched by transcript)"
    else:
        captured = sum(1 for t in turns if t.transcript)
        cap_den = max(1, len(turns))
        note = "turn produced a committed final transcript"
    out["completion_capture_rate"] = MetricResult(
        value=(captured / cap_den) if cap_den else None, numerator=captured, denominator=cap_den, note=note,
    )

    # --- 6. Barge-in Detection Rate ---
    bi = samples.barge_attempts
    det = sum(1 for a in bi if a.saw_interrupt)
    out["barge_in_detection_rate"] = MetricResult(
        value=(det / len(bi)) if bi else None, numerator=det, denominator=len(bi),
        note="real interruption injection surfaced an INTERRUPT event",
    )

    # --- 7. False Barge-in Rate ---
    fb = samples.backchannels
    false_bc = sum(1 for b in fb if b.saw_interrupt)
    out["false_barge_in_rate"] = MetricResult(
        value=(false_bc / len(fb)) if fb else None, numerator=false_bc, denominator=len(fb),
        note="backchannel/noise incorrectly fired an INTERRUPT",
    )

    # --- 8. Barge-in Stop P50/P95: how quickly TTS stops after a barge-in. ---
    # Only measured when TTS was actively playing at the cut-in (otherwise there
    # is nothing to stop). stop latency = last TTS audio still playing minus the
    # interruption onset.
    stops = sorted(
        (a.tts_stop - a.start)
        for a in samples.barge_attempts
        if a.tts_stop is not None and a.start is not None
    )
    n_active = sum(1 for a in samples.barge_attempts if not a.already_idle)
    out["barge_in_stop_p50"] = MetricResult(
        value=_p(stops, 50), numerator=len(stops), denominator=n_active,
        note="interruption onset -> TTS audio stopped (P50; active playback only)",
    )
    out["barge_in_stop_p95"] = MetricResult(
        value=_p(stops, 95), numerator=len(stops), denominator=n_active,
        note="interruption onset -> TTS audio stopped (P95; active playback only)",
    )

    # --- 9. Recovery Success Rate ---
    # Of the interruptions the agent actually detected, how many returned to a
    # new listening turn (a fresh user speech_start or transcript) afterward.
    rec_ok, rec_den = 0, 0
    all_speech_starts = sorted(t.speech_start for t in turns)
    for a in samples.barge_attempts:
        if not a.saw_interrupt or a.interrupt_event_t is None:
            continue
        rec_den += 1
        recovered = any(s > a.interrupt_event_t + 1e-6 for s in all_speech_starts)
        if recovered:
            rec_ok += 1
    out["recovery_success_rate"] = MetricResult(
        value=(rec_ok / rec_den) if rec_den else None, numerator=rec_ok, denominator=rec_den,
        note="after a detected interrupt, agent returned to listening (new turn)",
    )

    return out


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def compute_eot_f1(samples: EndpointSample) -> dict[str, MetricResult]:
    """End-of-Turn detection F1.

    Each user turn with a committed speech_end is a single end-of-turn
    detection. The agent's turn-taking decision is classified against the
    document window [speech_end, speech_end + ENDPOINT_LATE_S]:

      True Positive  response started in the window         (correct EOT)
      False Positive response started before speech_end     (false EOT,
                                                             cut the user off)
      False Negative no response by the window end          (missed EOT)

    Precision/recall/F1 follow directly; every turn with a speech_end is in the
    denominator, so the reader can always judge the sample size.
    """
    tp = fp = fn = 0
    turns = [t for t in samples.turns if t.speech_end is not None]
    for t in turns:
        if t.pre_end_response:
            fp += 1
        elif t.response_after_end is None or (t.response_after_end - t.speech_end) > ENDPOINT_LATE_S:
            fn += 1
        else:
            tp += 1
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    return {
        "eot_true_positives": MetricResult(value=float(tp), numerator=tp, denominator=len(turns),
                                           note="responses started within the EOT window"),
        "eot_false_positives": MetricResult(value=float(fp), numerator=fp, denominator=len(turns),
                                            note="responses started before speech_end (cut-offs)"),
        "eot_false_negatives": MetricResult(value=float(fn), numerator=fn, denominator=len(turns),
                                            note="no response within the EOT window (missed)"),
        "eot_precision": MetricResult(value=precision, numerator=tp, denominator=tp + fp,
                                      note="TP / (TP + FP)"),
        "eot_recall": MetricResult(value=recall, numerator=tp, denominator=tp + fn,
                                   note="TP / (TP + FN)"),
        "eot_f1": MetricResult(value=_f1(precision, recall), numerator=tp,
                               denominator=tp + fp + fn,
                               note="harmonic mean of precision and recall"),
    }


def analyze_results(results: list[ScenarioResult], spoken_by_scenario: dict[str, list[tuple[str, float]]] | None = None) -> dict:
    samples = collect_endpoint_samples(results, spoken_by_scenario)
    metrics = compute_endpoint_metrics(samples)
    metrics.update(compute_eot_f1(samples))
    return metrics


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_ROWS = [
    ("False Endpoint Rate",     "false_endpoint_rate",     "how often TASA cuts the user off"),
    ("Missed Endpoint Rate",    "missed_endpoint_rate",    "how often TASA waits after the user finished"),
    ("Endpoint P50",            "endpoint_p50",            "typical response-start delay"),
    ("Endpoint P95",            "endpoint_p95",            "worst response-start delay"),
    ("Utterance Fragment Rate", "utterance_fragment_rate", "how often one thought becomes multiple turns"),
    ("Completion Capture Rate", "completion_capture_rate", "% of utterances captured completely"),
    ("Barge-in Detection Rate", "barge_in_detection_rate", "% of real interruptions detected"),
    ("False Barge-in Rate",     "false_barge_in_rate",     "% of harmless backchannels/noise treated as interruptions"),
    ("Barge-in Stop P50",       "barge_in_stop_p50",       "how quickly TTS stops"),
    ("Barge-in Stop P95",       "barge_in_stop_p95",       "how quickly TTS stops (worst)"),
    ("Recovery Success Rate",   "recovery_success_rate",   "% of interruptions that recover"),
    ("EOT Precision",           "eot_precision",           "correct EOT response / all responses begun"),
    ("EOT Recall",              "eot_recall",              "correct EOT response / all turns ended"),
    ("EOT F1",                  "eot_f1",                  "harmonic mean of EOT precision and recall"),
]


def render_metrics(metrics: dict[str, MetricResult]) -> str:
    lines = ["Endpointing / Barge-in / EOT metrics", "-" * 78]
    lines.append(f"{'Metric':<26} {'value':>12} {'n/d':>10}  note")
    lines.append("-" * 78)
    for label, key, note in _ROWS:
        m = metrics.get(key)
        if m is None or m.value is None:
            lines.append(f"{label:<26} {'n/a':>12} {'-':>10}  {note}")
            continue
        v = m.value
        if "EOT" in label or "Precision" in label or "Recall" in label:
            vs = f"{v:.3f}"
        elif "Rate" in label or "Capture" in label:
            vs = f"{v*100:.1f}%"
        else:
            vs = f"{v:.3f}s"
        nd = f"{m.numerator}/{m.denominator}"
        lines.append(f"{label:<26} {vs:>12} {nd:>10}  {m.note or note}")
    return "\n".join(lines)
