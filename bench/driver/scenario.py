"""Scenario schema and registry.

A scenario is an ordered list of deterministic steps that the runner executes
against an agent adapter, plus expected-outcome metadata used by the scorer.

Step types:
  speak(text, pause_after_ms, silence_after_ms)
      Synthetically voice `text`, push PCM, and optionally append intra-
      utterance silence to probe turn-end thresholds.
  corpus(scope, kind)
      Pull one clip from a loaded corpus as a user utterance (real audio).
  silence(ms)
      Push explicit silence frames (turn-end probing / waiting).
  wait_for(event, timeout_ms)
      Block until the named event arrives (or timeout) — used to measure
      timing and to know when playback is active.
  interrupt_during_playback(text, after_ms)
      Wait until a TTS_CHUNK is observed, then inject `text` (barge-in).
  interrupt_during_llm(text)
      Inject `text` after LLM_TOKEN starts but before TTS output.
  expect(event, within_ms)
      Assert the named event must occur within a window.
  verify_text(contains=[...], not_contains=[...], field="llm_done")
      Assert on accumulated response text.
  assert_metric(key, op, value)
      Free-form deterministic assertion on the captured timeline.

Scenarios are target-independent: they only use buildin steps + corpus scopes.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


@dataclass
class Step:
    kind: str
    text: str | None = None
    pause_after_ms: float = 0.0
    silence_after_ms: float = 0.0
    ms: float = 0.0           # generic duration/silence param
    timeout_ms: float = 0.0   # wait/expect window
    event: str | None = None  # event name for expect/wait_for
    contains: list[str] = field(default_factory=list)
    not_contains: list[str] = field(default_factory=list)
    text_field: str = "llm_done"
    key: str | None = None
    op: str | None = None
    value: float | None = None
    scope: str | None = None   # corpus scope name
    corpus: str | None = None
    after_ms: float = 0.0
    absent: bool = False   # expect: assert event does NOT occur
    contain: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    name: str
    category: str
    steps: list[Step] = field(default_factory=list)
    corpus: str = ""          # default corpus name
    meta: dict = field(default_factory=dict)

    @property
    def regression_id(self) -> str:
        return self.meta.get("regression_id", self.id.upper())


def step_from_dict(d: dict) -> Step:
    known = {f.name for f in dataclasses.fields(Step)}
    return Step(**{k: v for k, v in d.items() if k in known})


def load_scenario(path: Path) -> Scenario:
    data = json.loads(path.read_text())
    steps = [step_from_dict(s) for s in data.pop("steps", [])]
    meta = data.pop("meta", {})
    return Scenario(steps=steps, meta=meta, **data)


def load_scenarios(subset: list[str] | str | None = None) -> dict[str, Scenario]:
    """Load scenarios, optionally filtered by id glob / comma-list."""
    out: dict[str, Scenario] = {}
    for p in sorted(SCENARIOS_DIR.glob("*.json")):
        sc = load_scenario(p)
        out[sc.id] = sc

    if subset is None or subset == "all":
        return out
    if isinstance(subset, str):
        subset = [s.strip() for s in subset.split(",") if s.strip()]
    wanted = set()
    for s in subset:
        for sc_id in out:
            if sc_id == s or s in sc_id:
                wanted.add(sc_id)
    return {k: v for k, v in out.items() if k in wanted}
