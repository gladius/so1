"""Prompt construction.

One user message per question, state FIRST and question LAST, so every question in a
request shares a byte-identical prefix and hits vLLM's prefix cache.
"""

from __future__ import annotations

import json
from typing import Any

from so1.schemas import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

SYSTEM = (
    "You are a precise decision classifier. Read the state carefully, then answer the "
    "question about it. Reply with only the requested label and nothing else."
)

NOUL_LABELS = ("Yes", "No")

DEFAULT_INSTRUCTIONS = {
    "noul": "Is the statement above true of the state?",
    "choice": "Which option best describes the state?",
    "score": "Which level best describes the state?",
}

_CHAT_ROLES = {"user", "assistant"}


def render(value: Any) -> str:
    """Instructions, criteria and state fragments may be str, object or array."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def as_chat_messages(state: Any) -> list[dict[str, str]] | None:
    """Recognise chat-message state: [{role, content}, ...] or {"messages": [...]}."""
    messages = state.get("messages") if isinstance(state, dict) and len(state) == 1 else state
    if not isinstance(messages, list) or not messages:
        return None
    if not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
        return None
    out: list[dict[str, str]] = []
    for message in messages:
        role, content = str(message["role"]), render(message["content"])
        if role not in _CHAT_ROLES:
            # Keep the model's chat template on a path it definitely supports.
            role, content = "user", f"{role.upper()}: {content}"
        out.append({"role": role, "content": content})
    return out


def option_descriptions(question: Question) -> list[str]:
    """Option/level text in request order. A null choice description falls back to its key."""
    if isinstance(question, ChoiceQuestion):
        return [render(desc) if desc is not None else key for key, desc in question.criteria.items()]
    if isinstance(question, ScoreQuestion):
        return [render(level) for level in question.criteria]
    return []


def question_block(question: Question, labels: list[str]) -> str:
    instructions = render(question.instructions) or DEFAULT_INSTRUCTIONS[question.type]
    lines = [f"QUESTION: {instructions}"]
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        if criteria is not None and criteria.true_ is not None:
            lines.append(f"Yes means: {render(criteria.true_)}")
        if criteria is not None and criteria.false_ is not None:
            lines.append(f"No means: {render(criteria.false_)}")
        lines += ["", "Reply with only Yes or No."]
    else:
        lines.append("OPTIONS:")
        lines += [f"{label}. {desc}" for label, desc in zip(labels, option_descriptions(question), strict=True)]
        lines += ["", f"Reply with only the letter of the best option ({', '.join(labels)})."]
    return "\n".join(lines)


def prefix_messages(state: Any) -> list[dict[str, str]]:
    """The shared part of every branch: what the warm-up call sends."""
    chat = as_chat_messages(state)
    if chat is not None:
        return [{"role": "system", "content": SYSTEM}, *chat]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"STATE:\n{render(state)}"}]


def build(state: Any, question: Question, labels: list[str]) -> list[dict[str, str]]:
    """Full branch prompt. Chat-message state gets the question as an extra user turn."""
    block = question_block(question, labels)
    chat = as_chat_messages(state)
    if chat is not None:
        return [{"role": "system", "content": SYSTEM}, *chat, {"role": "user", "content": block}]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"STATE:\n{render(state)}\n\n{block}"},
    ]
