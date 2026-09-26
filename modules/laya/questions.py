"""Typed-question schemas for the Laya System-1 decision layer.

Cadence 2 (turn-level): evaluated once per final transcript, batched into ONE
forward pass. Grouped by policy concern for the Phase-2 policy engine:

  UNDERSTANDING: what the user said / what they want
  ACTION:        what (if anything) the pipeline should do differently
  SIGNAL:        scalar modifiers that tune pacing / delivery, never decisions

Phase 1 runs these in shadow: the pipeline logs Laya's answers and compares
them to the legacy classifiers but never overrides behavior.

Schema budget: the English checkpoint has 512-token context with ~192 tokens
reserved for the option head (`head_max_len`), so the state payload must stay
small (raw transcript, not dialogue history) and every `choice` must stay well
under 20 options. All questions here comply.
"""
from __future__ import annotations

UNDERSTANDING_QUESTIONS: dict = {
    "intent": {
        "type": "choice",
        "instructions": "What is the user doing with this spoken utterance?",
        "criteria": {
            "statement": "sharing info, making a claim, casual filler",
            "question": "asking anything, direct or implied",
            "greeting": "opening remarks, hello",
            "farewell": "closing, goodbye",
            "backchannel": "short acknowledgment: yeah, ok, uh-huh, got it",
            "correction": "correcting the assistant or changing their mind",
            "command": "imperative or a request to do something",
            "continuation": "continuing their prior thought with and/but/then/so",
        },
    },
    "is_question": {
        "type": "noul",
        "instructions": "Is this utterance a question?",
    },
    "query_complexity": {
        "type": "choice",
        "instructions": "How complex should the answer be?",
        "criteria": {
            "simple": "quick greeting or yes/no",
            "standard": "normal short conversational answer",
            "complex": "needs explanation, comparison, or detail",
        },
    },
    "topic_changed": {
        "type": "noul",
        "instructions": "Did the user just switch to a new topic?",
    },
    "topic": {
        "type": "choice",
        "instructions": "What is the conversation topic?",
        "criteria": {
            "smalltalk": "greetings, personal, casual chat",
            "weather": "weather, temperature, climate",
            "news": "news, headlines, current events",
            "tech": "code, computers, programming",
            "food": "restaurants, food, dining",
            "travel_places": "places, directions, travel, landmarks",
            "services": "shops, businesses, booking, services",
            "math_logic": "calculations, reasoning, dates, times",
            "other": "anything else",
        },
    },
    "entity_type": {
        "type": "choice",
        "instructions": "Is the user asking about a specific verifiable named thing?",
        "criteria": {
            "none": "general chat, opinion, ordinary question",
            "place": "a specific restaurant, cafe, hotel, address, directions, location",
            "business": "a specific shop, company, brand, bank",
            "person": "a specific named person",
            "other": "any other specific named entity",
        },
    },
    "needs_verify": {
        "type": "noul",
        "instructions": (
            "Can the truthful answer require an external web or tool lookup?"
        ),
    },
}

ACTION_QUESTIONS: dict = {
    "tool_needed": {
        "type": "noul",
        "instructions": (
            "Does answering require external data "
            "(time, date, weather, web, news, math, reminder)?"
        ),
    },
    "tool": {
        "type": "choice",
        "instructions": "Which tool, if any, must be called?",
        "criteria": {
            "none": "answer from conversation alone",
            "get_time": "current time",
            "get_date": "current date",
            "calculate": "arithmetic or math expression",
            "get_weather": "weather for a place",
            "get_news": "news headlines",
            "search_web": "a fact or general question needing lookup",
            "set_reminder": "set a reminder",
            "roll_dice": "roll a dice",
        },
    },
    "invoke_llm": {
        "type": "noul",
        "instructions": (
            "Does this turn need a generative answer? Answer true unless this is "
            "pure greeting/farewell/backchannel or a templated tool reply."
        ),
    },
    "escalate": {
        "type": "noul",
        "instructions": (
            "Should this turn be routed to a stronger/fallback model "
            "for safety or confidence reasons?"
        ),
    },
}

SIGNAL_QUESTIONS: dict = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the user's sentiment?",
        "criteria": {
            "positive": "happy, appreciative, excited",
            "neutral": "flat, factual",
            "negative": "sad, angry, frustrated, disappointed",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": (
            "Is the user time-sensitive, frustrated, or in need of an immediate answer?"
        ),
    },
    "high_stakes": {
        "type": "noul",
        "instructions": (
            "Does this concern health, safety, money, legal, or another "
            "high-stakes topic?"
        ),
    },
}

# One batched schema for the turn-level (cadence 2) shadow pass.
TURN_QUESTIONS: dict = {
    **UNDERSTANDING_QUESTIONS,
    **ACTION_QUESTIONS,
    **SIGNAL_QUESTIONS,
}

# Cadence-1 (live-listening) schema: evaluated repeatedly during the user's
# turn at the partial-transcript cadence (~1s) on the partial text only. It
# feeds two Phase-3 decisions that the deterministic safety floor always still
# gates: endpoint shortening when the utterance is confidently complete, and
# barge-in suppression/ack for backchannel vs disagreement speech.
CADENCE1_QUESTIONS: dict = {
    "turn_complete": {
        "type": "noul",
        "instructions": "Has the user just finished their utterance?",
    },
    "completion_conf": {
        "type": "score",
        "instructions": (
            "How sure are you that the utterance is complete right now? "
            "(0 very unsure, 10 certain)"
        ),
        "criteria": [
            "0: the user is mid-word / clearly still talking",
            "1: barely any signal the turn is ending",
            "2: mostly still talking",
            "3: weak hint of a pause",
            "4: pause but likely continuing",
            "5: neutral, could go either way",
            "6: pause with a falling edge",
            "7: wording sounds final",
            "8: sentence ends cleanly",
            "9: clearly finished the thought",
            "10: certain the turn is over",
        ],
    },
    "barge_type": {
        "type": "choice",
        "instructions": (
            "During assistant playback, what is this user speech?"
        ),
        "criteria": {
            "none": "not interrupting; ordinary speech",
            "backchannel": "acknowledgment: yeah, uh-huh, okay, right",
            "disagreement": "firm interrupt: no, stop, wait, that's wrong",
        },
    },
}

# Latency Step-1 prefetch schema: evaluated on the live partial transcript
# while the user is still talking. Far lighter than the full cadence-2 pass --
# it only asks the tool decision, so a confident "tool_needed -> tool" answer
# can be executed once and its result seeds the LLM's first prompt.
PREFETCH_QUESTIONS: dict = {
    "tool": ACTION_QUESTIONS["tool"],
}

PREFETCH_STATE_KEYS = ("transcript", "turn_count")

CADENCE1_STATE_KEYS = (
    "transcript",
    "turn_count",
)

TURN_STATE_KEYS = (
    "transcript",
    "prev_intent",
    "turn_count",
    "topic",
    "last_transcript",
)


def guard_questions() -> dict | None:
    """Laya's guardrail preset (jailbreak/injection/leak nouls), if installed."""
    try:
        from laya import guard_questions as _guard

        return _guard()
    except Exception:
        return None