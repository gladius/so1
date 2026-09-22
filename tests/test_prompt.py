"""Prompt construction: every state form, label handling, criteria rendering."""

from __future__ import annotations

import pytest

from so1 import prompt
from so1.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion


def noul(**kw) -> NoulQuestion:
    return NoulQuestion.model_validate({"type": "noul", **kw})


def choice(**kw) -> ChoiceQuestion:
    return ChoiceQuestion.model_validate({"type": "choice", **kw})


def score(**kw) -> ScoreQuestion:
    return ScoreQuestion.model_validate({"type": "score", **kw})


def test_state_comes_first_and_question_last():
    messages = prompt.build("the state", noul(instructions="Urgent?"), ["Yes", "No"])
    user = messages[1]["content"]
    assert messages[0]["content"] == prompt.SYSTEM
    assert user.startswith("STATE:\nthe state")
    assert user.endswith("Reply with only Yes or No.")


def test_questions_share_a_byte_identical_prefix():
    """This is what makes vLLM's prefix cache work across the branches of one request."""
    state = "a long support thread"
    a = prompt.build(state, noul(instructions="Urgent?"), ["Yes", "No"])[1]["content"]
    b = prompt.build(state, choice(instructions="Route?", criteria={"x": None, "y": None}), ["A", "B"])[1]["content"]
    shared = f"STATE:\n{state}\n\n"
    assert a.startswith(shared) and b.startswith(shared)
    assert prompt.prefix_messages(state)[1]["content"] == f"STATE:\n{state}"


def test_object_state_is_serialised_json():
    messages = prompt.build({"order": {"id": "A-104"}}, noul(instructions="Paid?"), ["Yes", "No"])
    assert '"order"' in messages[1]["content"]
    assert '"id": "A-104"' in messages[1]["content"]


def test_array_state_is_serialised_json():
    messages = prompt.build(["first", "second"], noul(instructions="Paid?"), ["Yes", "No"])
    assert '"first"' in messages[1]["content"]
    assert len(messages) == 2


@pytest.mark.parametrize(
    "state",
    [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
    ],
)
def test_chat_state_keeps_turns_and_appends_the_question(state):
    messages = prompt.build(state, noul(instructions="Greeting?"), ["Yes", "No"])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "hi"
    assert messages[-1]["content"].startswith("QUESTION: Greeting?")
    assert prompt.prefix_messages(state)[-1] == {"role": "assistant", "content": "hello"}


def test_chat_state_folds_unsupported_roles_into_user_turns():
    state = [{"role": "tool", "content": "result=7"}, {"role": "user", "content": "ok?"}]
    messages = prompt.build(state, noul(instructions="Done?"), ["Yes", "No"])
    assert messages[1] == {"role": "user", "content": "TOOL: result=7"}


def test_a_list_that_is_not_chat_messages_stays_json():
    assert prompt.as_chat_messages(["a", "b"]) is None
    assert prompt.as_chat_messages([{"role": "user"}]) is None
    assert prompt.as_chat_messages({"messages": [], "other": 1}) is None


def test_choice_options_follow_request_order_and_hide_keys():
    question = choice(instructions="Route?", criteria={"billing": "money things", "tech": None})
    body = prompt.build("s", question, ["A", "B"])[1]["content"]
    assert "A. money things" in body
    assert "B. tech" in body  # null description falls back to the key
    assert "billing" not in body  # keys are never shown to the model
    assert body.endswith("Reply with only the letter of the best option (A, B).")


def test_score_levels_keep_their_order():
    question = score(instructions="How bad?", criteria=["calm", "cross", "furious"])
    body = prompt.build("s", question, ["A", "B", "C"])[1]["content"]
    assert "A. calm\nB. cross\nC. furious" in body


def test_noul_criteria_are_spelled_out():
    question = noul(instructions="Spam?", criteria={"true": "unsolicited advertising", "false": "a real conversation"})
    body = prompt.build("s", question, ["Yes", "No"])[1]["content"]
    assert "Yes means: unsolicited advertising" in body
    assert "No means: a real conversation" in body


def test_structured_instructions_are_serialised():
    question = noul(instructions={"question": "Same person as `dup`?", "dup": {"name": "John"}})
    body = prompt.build("s", question, ["Yes", "No"])[1]["content"]
    assert "Same person as `dup`?" in body
    assert '"name": "John"' in body


def test_missing_instructions_fall_back_to_a_default():
    body = prompt.build("s", noul(), ["Yes", "No"])[1]["content"]
    assert prompt.DEFAULT_INSTRUCTIONS["noul"] in body


def test_option_count_must_match_label_count():
    with pytest.raises(ValueError):
        prompt.build("s", choice(instructions="q", criteria={"a": None, "b": None}), ["A"])
