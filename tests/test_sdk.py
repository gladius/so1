"""The official typesafe-sdk, pointed at us with TYPESAFE_BASE_URL, must work unchanged.

Runs against the fake vLLM by default so it needs no GPU. Point SO1_URL at a live service
to run it against the real thing instead.
"""

from __future__ import annotations

import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from tests.conftest import FakeVLLM

from so1.app import create_app

pytestmark = pytest.mark.sdk

typesafe_sdk = pytest.importorskip("typesafe_sdk")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    """A real socket: the SDK brings its own HTTP client, so TestClient will not do."""
    if os.environ.get("SO1_URL"):
        yield os.environ["SO1_URL"]
        return

    from so1.config import Config, ModelConfig

    fake = FakeVLLM()
    config = Config(
        default_model="gemma4-e4b",
        models={
            "gemma4-e4b": ModelConfig(
                name="gemma4-e4b",
                url="http://vllm.test",
                upstream_model="gemma4-e4b",
                description="fake",
                release_date="2026-09-21",
            )
        },
        aliases={"jev-latest": "gemma4-e4b", "jev-preview": "gemma4-e4b"},
    )
    app = create_app(config, transport=httpx.MockTransport(fake.handler))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "service did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def sdk_client(base_url, monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", base_url)
    monkeypatch.setenv("TYPESAFE_API_KEY", os.environ.get("SO1_API_KEY", "not-checked"))
    from typesafe_sdk import TypeSafeClient

    with TypeSafeClient() as client:  # picks the base url up from the environment, as a team would
        yield client


def test_the_sdk_runs_one_of_each_question_type(sdk_client):
    from typesafe_sdk import Choice, Noul, Score

    response = sdk_client.system_one(
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "billing": Noul(instructions="Is this ticket about billing?"),
            "tone": Choice(
                instructions="What is the customer's tone?", criteria={"calm": None, "frustrated": None, "angry": None}
            ),
            "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
        },
    )
    assert 0.0 <= response.nouls["billing"].noul <= 1.0
    assert response.choices["tone"].choice in {"calm", "frustrated", "angry"}
    assert 0.0 <= response.choices["tone"].confidence <= 1.0
    assert 0.0 <= response.scores["urgency"].score <= 2.0
    assert set(response.scores["urgency"].legend) == {0, 1, 2}  # the SDK coerces the keys to ints
    assert response.usage.input_tokens and response.usage.output_tokens
    assert response.model


def test_the_sdk_lists_models(sdk_client):
    names = [model.name for model in sdk_client.models.list().models]
    assert "jev-latest" in names


def test_the_sdk_raises_its_own_typed_error_for_a_rejected_request(sdk_client):
    """Over the score-level ceiling: the SDK sends it, we reject it, the SDK types the failure."""
    from typesafe_sdk import Score, TypeSafeAPIError

    with pytest.raises(TypeSafeAPIError) as caught:
        sdk_client.system_one(
            state="x",
            questions={"q": Score(instructions="rate it", criteria=[f"level {i}" for i in range(11)])},
            model="jev-latest",
        )
    assert caught.value.status == 400
    assert "Too many score levels" in str(caught.value)


def test_the_sdk_reports_an_unknown_model(sdk_client):
    from typesafe_sdk import (
        Noul,
        TypeSafeAPIError,
    )

    with pytest.raises(TypeSafeAPIError) as caught:
        sdk_client.system_one(state="x", questions={"q": Noul(instructions="ok?")}, model="definitely-not-a-model")
    assert caught.value.status == 400
    assert "Unknown model" in str(caught.value)
