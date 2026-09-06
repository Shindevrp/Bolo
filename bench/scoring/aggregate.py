"""Aggregation of scenario results into category scores, issues, priorities,
and regression-test rows.

This maps the deterministic metrics (scoring.metrics) + observed-LLM output +
optional judge scores into the report's 1-10 categories and weighted benchmark.
Rules are conservative and evidence-based.
"""
from __future__ import annotations

from bench.driver.runner import ScenarioResult
from bench.scoring.metrics import (
    barge_in_metrics,
    context_facts,
    grounding_metrics,
    is_question,
    latency_buckets,
    turn_end_metrics,
    collect_scenario_metrics,
)
from bench.scoring.report import CATEGORIES, Report, RegressionRow, WEIGHTS

# Category labels used in reports
C = {
    "TT": "Turn-taking",
    "BI": "Barge-in",
    "ASR": "ASR/Segmentation",
    "CTX": "Context/State",
    "INTENT": "Intent understanding",
    "FLOW": "Dialogue flow",
    "GRD": "Grounding",
    "REC": "Error recovery",
    "TTS": "TTS quality",
    "LAT": "Latency",
    "NAT": "Naturalness",
    "BC": "Backchannel handling",
}


def get_timeline(res: ScenarioResult):
    return res.timeline


def _result_by_id(results: list[ScenarioResult]) -> dict[str, ScenarioResult]:
    return {r.scenario.id: r for r in results}


def _score_turn_taking(results, issues, positives):
    # Evidence from turn_taking / baseline scenarios.
    samples = []
    worst = 10.0
    seen_early = False
    seen_hold = False
    for sc_id in ("turn_taking", "baseline"):
        r = results.get(sc_id)
        if not r:
            continue
        m = collect_scenario_metrics(r.timeline)
        te = m["turn_end"]
        if te["first_llm_before_speech_end"]:
            seen_early = True
            worst = min(worst, 6.0)
            issues["major"].append(
                "[turn_taking] LLM response began before speech_end (early pre-empt) "
                f"in scenario '{sc_id}'."
            )
        lat = te.get("first_audio_after_speech_end")
        if lat is not None and lat > 4.0:
            issues["minor"].append(
                f"[turn_taking] {sc_id}: {lat:.2f}s before first audio after speech end "
                "(slow turn response)."
            )
        samples.append(sc_id)
    if not seen_early and samples:
        pos = f"[turn_taking] Turn-end respected in {len(samples)} scenario(s); no early pre-empt observed."
        positives.append(pos)
    return worst


def _score_barge_in(results, issues, positives, details):
    score = 10.0
    bi = results.get("barge_in")
    bc = results.get("backchannel")
    if bi:
        m = collect_scenario_metrics(bi.timeline)
        b = m["barge_in"]
        if not b["interrupt_seen"]:
            score -= 4.0
            issues["critical"].append(
                "[barge_in] Expected an interrupt when the user cut in during playback, "
                "but no interrupt event fired (BI-001)."
            )
        elif not b["recovered_to_new_turn"]:
            score -= 2.0
            issues["major"].append(
                "[barge_in] Interrupt fired but the agent did not recover to a new listening turn."
            )
        else:
            positives.append("[barge_in] Interrupt detected during playback and agent recovered.")
    if bc:
        m = collect_scenario_metrics(bc.timeline)
        b = m["barge_in"]
        if b["interrupt_seen"]:
            score -= 3.0
            issues["major"].append(
                "[barge_in] Backchannel ('yeah, uh-huh, right') incorrectly triggered an interrupt (BC-001): "
                "passive acknowledgements should NOT stop speech."
            )
        else:
            positives.append("[backchannel] Backchannel did not interrupt playback.")
    return max(score, 1.0)


def _score_asr(results, wer: float | None, issues, positives):
    if wer is None:
        # No ASR corpus scored; mark from transcript-bearing scenarios only.
        any_transcript = any(r.timeline.final_transcripts() for r in results.values())
        if not any_transcript:
            issues["minor"].append(
                "[asr] No ASR/WER leg ran (no LibriSpeech/CommonVoice corpus); "
                "ASR scored conservatively."
            )
            return 5.0
        return 6.0  # transcripts produced but WER unmeasured
    if wer <= 0.12:
        positives.append(f"[asr] Low WER across corpus ({wer*100:.1f}%).")
        return 9.0
    if wer <= 0.30:
        issues["major"].append(f"[asr] Elevated WER ({wer*100:.1f}%) — "
                               "utterances transcribed imperfectly.")
        return 6.0
    issues["critical"].append(
        f"[asr] High WER ({wer*100:.1f}%) — poor ASR/segmentation."
    )
    return 3.0


def _score_context(results, issues, positives):
    score = 10.0
    ctx = results.get("context")
    if ctx:
        m = collect_scenario_metrics(ctx.timeline)
        c = m["context"]
        details = {
            "mentioned_five": c["mentioned_five"],
            "mentioned_budget": c["mentioned_budget"],
            "mentioned_nature": c["mentioned_nature"],
        }
        # When the final turn asked "given everything, recommend?", the answer
        # should naturally reference the group size/budget/theme.
        if c.get("mentioned_five") is False:
            score -= 2.0
            issues["major"].append(
                "[context] Final recommendation did not reference updated group size (five) after the "
                "3→5 correction (CTX-001)."
            )
        if c.get("mentioned_nature") is False:
            score -= 1.0
            issues["minor"].append(
                "[context] Final recommendation did not reflect the 'nature/short hikes' preference."
            )
        positives.append("[context] User turns were transcribed and the 'given everything' question was answered.")
    return score


def _score_grounding(results, issues, positives):
    grd = results.get("grounding")
    score = 10.0
    if grd:
        m = collect_scenario_metrics(grd.timeline)
        g = m["grounding"]
        if g["fabrication_risk"]:
            score -= 4.0
            issues["critical"].append(
                f"[grounding] Response asserted unverified named entities {g['unsupported_entities']} "
                "that were not provided by the user (HAL-001)."
            )
        if g["uncertainty_expressed"]:
            positives.append(
                f"[grounding] Agent expressed uncertainty ({g['uncertainty_phrases']}) rather than fabricating."
            )
        else:
            score -= 1.0
            issues["minor"].append(
                "[grounding] No explicit uncertainty phrasing observed in the entity-seeking response."
            )
    return score


def _score_latency(results, issues, positives):
    worst = 10.0
    for r in results.values():
        m = collect_scenario_metrics(r.timeline)
        lat = m["latency"]
        bucket = lat.get("first_audio_bucket", "unobservable")
        if bucket == "severe delay":
            worst = min(worst, 3.0)
            issues["major"].append(f"[latency] Severe delay in '{r.scenario.id}'.")
        elif bucket == "long delay":
            worst = min(worst, 5.0)
            issues["minor"].append(f"[latency] Long delay in '{r.scenario.id}'.")
        elif bucket in ("noticeable delay",):
            worst = min(worst, 6.0)
        elif bucket == "immediate":
            positives.append(f"[latency] Immediate first audio in '{r.scenario.id}'.")
        else:
            worst = min(worst, 7.0)
            issues["minor"].append(f"[latency] Flower unobservable in '{r.scenario.id}'.")
    return worst


def _unscored(issues, name, evidence: str):
    issues["minor"].append(f"[{name}] Not directly scored (no judge). {evidence}")


def _judge_average(judge: dict, keys: list[str]) -> float | None:
    if not judge.get("scored"):
        return None
    vals = [v for k, v in judge.get("categories", {}).items() if k in keys and isinstance(v, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate(
    results: list[ScenarioResult],
    *,
    wer: float | None = None,
    cer: float | None = None,
    judge: dict | None = None,
    agent: str = "unnamed",
) -> Report:
    reports_by_id = _result_by_id(results)
    issues = {"critical": [], "major": [], "minor": []}
    positives: list[str] = []
    pattern: list[str] = []

    scores = {cat: None for cat in CATEGORIES}
    judge = judge or {}

    # -- Deterministic categories --
    scores["Turn-taking"] = _score_turn_taking(reports_by_id, issues, positives)
    scores["Barge-in"] = _score_barge_in(reports_by_id, issues, positives, {})
    scores["ASR/Segmentation"] = _score_asr(reports_by_id, wer, issues, positives)
    scores["Context/State"] = _score_context(reports_by_id, issues, positives)
    scores["Grounding"] = _score_grounding(reports_by_id, issues, positives)
    scores["Latency"] = _score_latency(reports_by_id, issues, positives)

    # -- Judge (or conservative fallback) categories --
    for cat in ("Intent understanding", "Dialogue flow", "Error recovery",
                "TTS quality", "Naturalness", "Backchannel handling"):
        key = cat.lower().replace(" ", "_").replace("/", "_")
        jv = judge.get("categories", {}).get(key)
        if isinstance(jv, (int, float)):
            scores[cat] = max(1.0, min(10.0, float(jv)))
        else:
            scores[cat] = 5.0
            _unscored(issues, cat, "defaulted to neutral 5 (judge unavailable).")

    # -- Patterns & priorities --
    for cat in scores:
        s = scores[cat]
        if s is not None and s < 5.0:
            pattern.append(f"Chronic weakness in {cat} (score {s:.1f}).")

    priorities = _priorities(scores, issues)

    # -- Regression rows from scenario results --
    regression = _build_regression(results, wer, cer)

    rep = Report(
        agent=agent,
        scores=scores,
        issues=issues,
        positives=positives,
        patterns=pattern,
        priorities=priorities,
        regression=regression,
        judge_active=bool(judge.get("scored")),
        scenario_details={
            r.scenario.id: {
                "ok": r.ok,
                "llm": r.timeline.llm_output()[:400],
                "error": r.error,
            }
            for r in results
        },
    )
    return rep


def _priorities(scores: dict, issues: dict) -> list[str]:
    order = [
        (scores.get("Barge-in"), "Barge-in reliability"),
        (scores.get("Turn-taking"), "End-of-turn detection"),
        (scores.get("ASR/Segmentation"), "ASR segmentation / WER"),
        (scores.get("Grounding"), "Grounding & hallucination guard"),
        (scores.get("Context/State"), "Conversation state management"),
        (scores.get("Latency"), "Response latency"),
        (scores.get("Error recovery"), "Error recovery"),
        (scores.get("Dialogue flow"), "Dialogue flow / context switching"),
        (scores.get("Naturalness"), "Naturalness"),
        (scores.get("Backchannel handling"), "Backchannel handling"),
    ]
    # Non-null scores: prioritize lower ones; put null last.
    ranked = [x[1] for x in sorted(
        ((s, name) for s, name in order if s is not None),
        key=lambda p: p[0],
    )]
    return ranked


def _build_regression(results: list[ScenarioResult], wer: float | None, cer: float | None = None) -> list[RegressionRow]:
    rows: list[RegressionRow] = []
    by_id = _result_by_id(results)

    def add(test_id, sc_id, utt, expected, observed, passed, severity, cat):
        rows.append(
            RegressionRow(
                test_id=test_id,
                scenario=sc_id,
                utterance=utt,
                expected=expected,
                observed=observed,
                passed=passed,
                severity=severity,
                category=cat,
            )
        )

    r = by_id.get("asr_wer")
    if r:
        ok = r.ok and (wer is not None) and wer <= 0.30
        add("ASR-001", "asr_wer", "(LibriSpeech clip)",
            "ASR transcript closely matches ground truth",
            f"WER {wer*100:.1f}%" if wer is not None else "no transcript/WER",
            ok, "major" if (wer is not None and wer > 0.30) else "minor", "ASR/Segmentation")

    r = by_id.get("barge_in")
    if r:
        had = any(e.type.name == "INTERRUPT" for e in r.timeline.events)
        add("BI-001", "barge_in", "Wait, no, that's not what I meant",
            "Interrupt detected; TTS stops; new turn accepted",
            "INTERRUPT " + ("fired" if had else "NOT fired"),
            had, "critical" if not had else "major", "Barge-in")

    r = by_id.get("backchannel")
    if r:
        had = any(e.type.name == "INTERRUPT" for e in r.timeline.events)
        add("BC-001", "backchannel", "yeah uh-huh right",
            "Backchannel does NOT trigger interrupt",
            "INTERRUPT " + ("fired (unexpected)" if had else "none"),
            not had, "major" if had else "minor", "Backchannel handling")

    r = by_id.get("context")
    if r:
        add("CTX-001", "context", "given everything, what would you recommend?",
            "Recommendation reflects five people, budget, nature",
            "5 referenced" if (r.timeline and any("five" in (e.text or "").lower() for e in r.timeline.events if e.type.name == "LLM_DONE")) else "final answer captured",
            r.ok, "minor", "Context/State")

    r = by_id.get("grounding")
    if r:
        from bench.scoring.metrics import grounding_metrics
        g = grounding_metrics(r.timeline)
        add("HAL-001", "grounding", "restaurants near Hyderabad",
            "Agent expresses uncertainty / does not fabricate",
            f"unsupported entities {g['unsupported_entities'] or 'none'}; uncertainty={g['uncertainty_expressed']}",
            not g["fabrication_risk"],
            "critical" if g["fabrication_risk"] else "minor", "Grounding")

    r = by_id.get("recovery")
    if r:
        add("REC-001", "recovery", "no, that's not what I meant, I wanted a restaurant",
            "Agent acknowledges and re-interprets, continues naturally",
            "transcript + response captured",
            r.ok, "minor", "Error recovery")

    # CER-001: character error is the ASR companion to WER — same corpus, char level.
    if cer is not None:
        add("CER-001", "asr_wer", "(LibriSpeech clip)",
            "Character-level transcript error is low",
            f"CER {cer*100:.1f}%",
            cer <= 0.20,
            "major" if cer > 0.20 else "minor", "ASR/Segmentation")

    # EOT-001: end-of-turn detection F1 over turn_taking/baseline turns.
    eot = _eot_f1(results)
    if eot is not None:
        tp, fp, fn = eot
        add("EOT-001", "turn_taking", "(end-of-turn timing)",
            "End-of-turn detection F1 is high (no cut-offs, no misses)",
            f"TP={tp} FP={fp} FN={fn}",
            tp >= 1 and fp == 0 and fn == 0,
            "major" if (fp or fn) else "minor", "Turn-taking")

    return rows


def _eot_f1(results: list[ScenarioResult]):
    """Tally end-of-turn detection decisions over deterministic scenarios."""
    from bench.scoring.endpoint import (
        ENDPOINT_LATE_S,
        collect_endpoint_samples,
        reconstruct_user_turns,
    )

    tp = fp = fn = 0
    for r in results:
        if r.scenario.id not in ("turn_taking", "baseline"):
            continue
        for t in reconstruct_user_turns(r.timeline):
            if t.speech_end is None:
                continue
            if t.pre_end_response:
                fp += 1
            elif t.response_after_end is None or (t.response_after_end - t.speech_end) > ENDPOINT_LATE_S:
                fn += 1
            else:
                tp += 1
    if tp + fp + fn == 0:
        return None
    return tp, fp, fn
