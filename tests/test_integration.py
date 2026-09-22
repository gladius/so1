"""Against the real RunPod vLLM endpoint. Skipped unless RUNPOD_API_KEY is set.

uv run pytest -m upstream -s
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from so1.app import create_app
from so1.config import Config, Limits, ModelConfig

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="needs RUNPOD_API_KEY"),
]

URL = os.environ.get("E4B_URL", "https://fcy67kixeo37lf.api.runpod.ai")


@pytest.fixture(scope="module")
def live():
    config = Config(
        default_model="gemma4-e4b",
        models={
            "gemma4-e4b": ModelConfig(
                name="gemma4-e4b",
                url=URL,
                upstream_model="gemma4-e4b",
                description="Gemma 4 E4B on RunPod",
                release_date="2026-09-21",
            )
        },
        aliases={"jev-latest": "gemma4-e4b", "jev-preview": "gemma4-e4b"},
        upstream_api_key=os.environ["RUNPOD_API_KEY"],
        limits=Limits(upstream_timeout_s=330.0, request_timeout_s=600.0),
    )
    with TestClient(create_app(config)) as client:
        yield client


def test_startup_probes_pick_a_readout_mode(live):
    health = live.get("/health").json()
    upstream = health["upstreams"]["gemma4-e4b"]
    print(f"\n  probe: {upstream['detail']}")
    assert upstream["ready"], upstream["detail"]
    assert upstream["mode"] in ("exact", "fallback")
    assert upstream["strategy"] in ("chat", "prefix")


def test_limits_are_clamped_to_the_real_context_window(live):
    body = live.get("/v1/limits").json()
    assert body["models"]["gemma4-e4b"]["max_prompt_tokens"] <= 8192


def test_one_request_with_all_three_types(live):
    response = live.post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "Customer: our CSV export has failed every time since yesterday with Error 500, "
            "and month-end close is Friday. We are on the annual Business plan.",
            "questions": {
                "is_bug": {"type": "noul", "instructions": "Is the customer reporting a software defect?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments and invoices",
                        "technical": "Bugs and outages",
                        "sales": "Pricing and upgrades",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today"],
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    print(f"\n  {body['answers']}\n  usage={body['usage']} timing={response.headers.get('Server-Timing')}")

    assert body["model"] == "gemma4-e4b"
    noul, choice, score = body["answers"]["is_bug"], body["answers"]["team"], body["answers"]["urgency"]

    assert 0.0 <= noul["noul"] <= 1.0
    assert noul["noul"] > 0.5  # it plainly is a bug report
    assert set(choice["probabilities"]) == {"billing", "technical", "sales"}
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert choice["choice"] == "technical"
    assert sum(score["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= score["score"] <= 2.0
    assert set(score["legend"]) == {"0", "1", "2"}
    assert body["usage"]["input_tokens"] > 0
    assert response.headers["Server-Timing"]


def test_the_state_is_billed_once_not_once_per_question(live):
    payload = {"model": "jev-latest", "state": "A customer is upset about a duplicate charge. " * 40}
    one = live.post(
        "/v1/systemone",
        json={**payload, "questions": {"a": {"type": "noul", "instructions": "Is the customer upset?"}}},
    ).json()
    four = live.post(
        "/v1/systemone",
        json={
            **payload,
            "questions": {
                "a": {"type": "noul", "instructions": "Is the customer upset?"},
                "b": {"type": "noul", "instructions": "Is this about billing?"},
                "c": {"type": "noul", "instructions": "Does the customer want a refund?"},
                "d": {"type": "noul", "instructions": "Is this urgent?"},
            },
        },
    ).json()
    print(f"\n  1 question: {one['usage']}   4 questions: {four['usage']}")
    assert four["usage"]["input_tokens"] < 4 * one["usage"]["input_tokens"]
    assert four["usage"]["output_tokens"] == 4


def test_a_long_state_over_the_context_window_is_rejected_cleanly(live):
    response = live.post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "filler sentence for padding. " * 6000,
            "questions": {"a": {"type": "noul", "instructions": "Is this long?"}},
        },
    )
    assert response.status_code == 400, response.text
    assert "too long" in response.json()["detail"].lower()
