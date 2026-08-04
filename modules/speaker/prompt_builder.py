from __future__ import annotations

from modules.speaker.profile import SpeakerProfile

MULTI_SPEAKER_RULES = """
You are having a conversation with the user alongside your partner {partner_names}.
Each speaker is identified by [Name]. Format your response as:
[YourName] what you want to say.
[PartnerName] what your partner would say.

Rules:
- Alternate speakers naturally; don't let one dominate.
- Build on each other's ideas, add new perspectives.
- Disagree respectfully when you have a different view.
- Reference your partner by name occasionally.
- Keep each speaker's turn to 1-3 sentences.
- Never output two consecutive turns from the same speaker.
- When the user asks a question, one speaker answers, the other adds commentary.
- Don't use speaker tags for the user's words — only for your two speakers.
- Your two speakers are discussing WITH each other and the user, not monologuing.
- One speaker might ask the other a question mid-response for natural flow.
"""


def build_multi_speaker_prompt(
    speakers: list[SpeakerProfile],
    lead_speaker: str | None = None,
) -> str:
    """Build the system prompt for multi-speaker conversations."""
    if len(speakers) < 2:
        return ""

    partner_names = " and ".join(s.name for s in speakers if s.name != lead_speaker)
    if not partner_names:
        partner_names = " and ".join(s.name for s in speakers)

    parts: list[str] = []

    # Speaker personas
    for s in speakers:
        parts.append(s.system_prompt_block())

    parts.append("")

    # Conversation rules
    parts.append(MULTI_SPEAKER_RULES.format(partner_names=partner_names))

    return "\n".join(parts)
