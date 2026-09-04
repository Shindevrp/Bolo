from modules.dialogue.prompts import build_system_prompt


class TestSystemPrompt:
    def test_no_repetitive_question_guidance(self) -> None:
        p = build_system_prompt()
        assert "Check if the user is following along" not in p

    def test_anti_repetition_block_included(self) -> None:
        p = build_system_prompt()
        assert "NEVER end every response with a question" in p

    def test_no_premature_goodbye(self) -> None:
        p = build_system_prompt()
        assert "Do not greet again or say goodbye" in p

    def test_disengaged_prompt_not_pushing_questions(self) -> None:
        p = build_system_prompt(engagement=0.2)
        assert "Ask simple follow-ups" not in p

    def test_ongoing_context_after_many_turns(self) -> None:
        p = build_system_prompt(turn_count=5)
        assert "ongoing conversation" in p

    def test_grounding_block_always_included(self) -> None:
        p = build_system_prompt()
        assert "Never invent facts" in p
        assert "aren't sure rather than guessing" in p

    def test_grounding_block_never_fabricates(self) -> None:
        p = build_system_prompt(engagement=0.9, complexity="complex")
        assert "do not contradict it" in p

    def test_role_enforcement_always_included(self) -> None:
        p = build_system_prompt()
        assert "Never switch roles" in p
        assert "ALWAYS the assistant/agent" in p
        assert "never ask the user to respond to themselves" in p

    def test_base_prompt_establishes_assistant_and_user(self) -> None:
        p = build_system_prompt()
        assert "You are the ASSISTANT" in p
        assert "the USER" in p
