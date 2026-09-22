"""End-to-end service behaviour against the fake vLLM: answers, limits, errors, execution."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.conftest import FakeVLLM, FaultTransport, is_branch, running

from so1.app import create_app
from so1.config import Config, ModelConfig
from so1.readout import choice_confidence

NOUL = {"type": "noul", "instructions": "Is this urgent?"}
CHOICE = {
    "type": "choice",
    "instructions": "Route this.",
    "criteria": {"billing": "money", "tech": None, "sales": None},
}
SCORE = {"type": "score", "instructions": "How bad?", "criteria": ["calm", "cross", "furious"]}


def ask(client: TestClient, questions: dict, state="a customer is angry", model="jev-latest"):
    return client.post("/v1/systemone", json={"model": model, "state": state, "questions": questions})


def completions(fake: FakeVLLM) -> list[dict]:
    return [body for path, body in fake.requests if path.endswith("/completions")]


def branches(fake: FakeVLLM) -> list[dict]:
    return [b for b in completions(fake) if (b.get("top_logprobs") or b.get("logprobs")) != 1]


def test_fallback_readout_refuses_more_options_than_it_can_see(client, fake):
    """Top-20 logprobs cannot resolve 30 labels; answering anyway would be silent garbage."""
    criteria = {f"opt{i}": None for i in range(25)}
    response = ask(client, {"a": {"type": "choice", "instructions": "x", "criteria": criteria}})
    assert response.status_code == 400
    assert "fallback readout" in response.json()["detail"]


# ------------------------------------------------------------------ answers


def test_noul_returns_probability_of_yes_and_no_confidence(client, fake):
    fake.weights = {"Yes": 0.9, "No": 0.1}
    answer = ask(client, {"urgent": NOUL}).json()["answers"]["urgent"]
    assert answer == {"type": "noul", "noul": pytest.approx(0.9)}


def test_choice_returns_the_winning_key_and_a_distribution_over_keys(client, fake):
    fake.weights = {"A": 0.2, "B": 0.7, "C": 0.1}
    answer = ask(client, {"route": CHOICE}).json()["answers"]["route"]
    assert answer["type"] == "choice"
    assert answer["choice"] == "tech"  # second option, keyed back from label B
    assert answer["probabilities"] == {
        "billing": pytest.approx(0.2),
        "tech": pytest.approx(0.7),
        "sales": pytest.approx(0.1),
    }
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
    assert answer["confidence"] == pytest.approx(choice_confidence([0.2, 0.7, 0.1]))


def test_score_returns_the_expected_value_with_a_legend(client, fake):
    fake.weights = {"A": 0.0, "B": 0.95, "C": 0.05}
    answer = ask(client, {"anger": SCORE}).json()["answers"]["anger"]
    assert answer["type"] == "score"
    assert answer["score"] == pytest.approx(1.05)
    assert answer["legend"] == {"0": "calm", "1": "cross", "2": "furious"}
    assert answer["probabilities"] == {"0": pytest.approx(0.0), "1": pytest.approx(0.95), "2": pytest.approx(0.05)}


def test_answers_come_back_under_the_question_ids(client):
    body = ask(client, {"a": NOUL, "b": CHOICE, "c": SCORE}).json()
    assert set(body["answers"]) == {"a", "b", "c"}
    assert [body["answers"][k]["type"] for k in ("a", "b", "c")] == ["noul", "choice", "score"]
    assert body["model"] == "gemma4-e4b"  # the resolved model, not the alias


def test_usage_counts_the_shared_state_once(client):
    body = ask(client, {"a": NOUL, "b": NOUL, "c": NOUL}).json()
    single = ask(client, {"a": NOUL}).json()
    assert body["usage"]["output_tokens"] == 3
    # Three branches each resend the state; usage must not bill it three times.
    assert body["usage"]["input_tokens"] < 3 * single["usage"]["input_tokens"]


def test_temperature_calibration_is_applied(config, fake):
    config = Config(**{**vars(config), "temperature": {"noul": 4.0, "choice": 1.0, "score": 1.0}})
    fake.weights = {"Yes": 0.9, "No": 0.1}
    with running(config, fake) as client:
        flattened = ask(client, {"q": NOUL}).json()["answers"]["q"]["noul"]
    assert 0.5 < flattened < 0.9  # temperature > 1 pulls it towards even odds


# ------------------------------------------------------------------ execution


def test_a_multi_question_request_warms_the_prefix_once(client, fake):
    ask(client, {"a": NOUL, "b": CHOICE, "c": SCORE})
    warmups = [b for b in completions(fake) if (b.get("top_logprobs") or b.get("logprobs")) == 1]
    assert len(warmups) == 1
    assert len(branches(fake)) == 3
    assert "QUESTION:" not in warmups[0]["messages"][-1]["content"]


def test_a_single_question_request_skips_the_warm_up(client, fake):
    ask(client, {"a": NOUL})
    assert len(completions(fake)) == 1


def test_every_branch_asks_for_one_token_only(client, fake):
    ask(client, {"a": NOUL, "b": CHOICE})
    assert all(b["max_tokens"] == 1 for b in completions(fake))


def test_server_timing_header_reports_the_phases(client):
    header = ask(client, {"a": NOUL, "b": NOUL}).headers["Server-Timing"]
    assert {part.split(";")[0] for part in header.split(", ")} == {"prepare", "warmup", "branches"}


def test_a_failing_branch_cancels_its_siblings(config, fake):
    """One upstream failure must not leave the others burning GPU time."""
    state = {"seen": 0, "completed": 0}

    async def fault(request: httpx.Request) -> httpx.Response | None:
        if not is_branch(request):
            return None
        state["seen"] += 1
        if state["seen"] == 1:
            return httpx.Response(400, json={"error": {"message": "bad branch"}})
        await asyncio.sleep(5)
        state["completed"] += 1
        return None

    with running(config, fake, FaultTransport(fake, fault)) as client:
        response = ask(client, {"a": NOUL, "b": NOUL, "c": NOUL, "d": NOUL})
    assert response.status_code == 502
    assert state["completed"] == 0  # siblings were cancelled, not awaited


def test_retryable_upstream_status_is_retried_then_succeeds(config, fake):
    calls = {"n": 0}

    async def fault(request: httpx.Request) -> httpx.Response | None:
        if not is_branch(request):
            return None
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "no workers available"}}) if calls["n"] == 1 else None

    with running(config, fake, FaultTransport(fake, fault)) as client:
        assert ask(client, {"a": NOUL}).status_code == 200
    assert calls["n"] >= 2


def test_upstream_timeout_surfaces_as_504(config, fake):
    async def fault(request: httpx.Request) -> httpx.Response | None:
        if is_branch(request):
            raise httpx.ReadTimeout("too slow", request=request)
        return None

    with running(config, fake, FaultTransport(fake, fault)) as client:
        response = ask(client, {"a": NOUL})
    assert response.status_code == 504
    assert response.json()["detail"]["error_type"] == "timeout_error"


# ------------------------------------------------------------------ readout mode


def test_raw_logprob_servers_fall_back(client, fake):
    assert fake.logprobs_mode == "raw"
    assert client.get("/health").json()["upstreams"]["gemma4-e4b"]["mode"] == "fallback"


def test_processed_logprob_servers_get_the_exact_readout(config):
    fake = FakeVLLM()
    fake.logprobs_mode = "processed"
    with running(config, fake) as client:
        assert client.get("/health").json()["upstreams"]["gemma4-e4b"]["mode"] == "exact"
        ask(client, {"a": CHOICE})
        assert branches(fake)[-1]["allowed_token_ids"]  # sampling restricted to the label ids


def test_exact_mode_does_not_truncate_the_distribution(config):
    fake = FakeVLLM()
    fake.logprobs_mode = "processed"
    with running(config, fake) as client:
        ask(client, {"a": CHOICE})
    body = branches(fake)[-1]
    assert (body["temperature"], body["top_p"], body["top_k"], body["min_p"]) == (1.0, 1.0, -1, 0.0)


# ------------------------------------------------------------------ errors


def test_unknown_model_is_a_400_usage_error(client):
    response = ask(client, {"a": NOUL}, model="nope-9")
    assert response.status_code == 400
    assert response.json() == {"detail": {"error_type": "api_usage_error", "message": "Unknown model: nope-9"}}


def test_disabled_model_is_unknown(config, fake):
    config.models["off"] = ModelConfig(name="off", url="http://x", upstream_model="off", enabled=False)
    with running(config, fake) as client:
        assert ask(client, {"a": NOUL}, model="off").status_code == 400


def test_missing_state_is_422_with_the_offending_field(client):
    response = client.post("/v1/systemone", json={"model": "jev-latest", "questions": {"a": NOUL}})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "state"] and detail[0]["type"] == "missing"


def test_unknown_question_type_is_masked_as_400(client):
    response = ask(client, {"a": {"type": "bogus", "instructions": "x"}})
    assert response.status_code == 400
    assert response.json() == {"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}


def test_missing_question_type_stays_422(client):
    response = ask(client, {"a": {"instructions": "x"}})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "union_tag_not_found"


def test_empty_questions_is_422(client):
    assert ask(client, {}).status_code == 422


def test_too_many_choices_is_400_with_a_plain_detail(client, config):
    criteria = {f"opt{i}": None for i in range(config.limits.max_choice_options + 1)}
    response = ask(client, {"a": {"type": "choice", "instructions": "x", "criteria": criteria}})
    assert response.status_code == 400
    assert response.json() == {"detail": "Too many choices. Must have at most 64 choices."}


def test_too_many_score_levels_is_400(client):
    response = ask(client, {"a": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}})
    assert response.json() == {"detail": "Too many score levels. Must have at most 10 levels."}


def test_too_many_questions_is_400(client):
    response = ask(client, {f"q{i}": NOUL for i in range(65)})
    assert response.json() == {"detail": "Too many questions. Must have at most 64 questions."}


def test_more_options_than_the_tokenizer_has_labels(config, fake):
    with running(config, fake) as client:
        criteria = {f"opt{i}": None for i in range(32)}  # fake tokenizer has 26 + AA/AB/AC
        response = ask(client, {"a": {"type": "choice", "instructions": "x", "criteria": criteria}})
    assert response.status_code == 400
    assert "single-token labels" in response.json()["detail"]


def test_oversized_body_is_413(client):
    response = client.post(
        "/v1/systemone",
        content=json.dumps({"model": "jev-latest", "state": "x" * (2 * 1024 * 1024 + 10), "questions": {"a": NOUL}}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_overload_returns_529_with_retry_after(config, fake):
    config = Config(**{**vars(config), "limits": type(config.limits)(max_concurrent_requests=0)})
    with running(config, fake) as client:
        response = ask(client, {"a": NOUL})
    assert response.status_code == 529
    assert response.headers["Retry-After"] == "1"
    assert response.json()["detail"]["error_type"] == "overloaded_error"


# ------------------------------------------------------------------ auth and metadata


def test_bearer_auth_when_configured(config, fake):
    config = Config(**{**vars(config), "api_key": "secret"})
    with running(config, fake) as client:
        assert (
            client.post(
                "/v1/systemone", json={"model": "jev-latest", "state": "x", "questions": {"a": NOUL}}
            ).status_code
            == 403
        )
        bad = client.post(
            "/v1/systemone",
            json={"model": "jev-latest", "state": "x", "questions": {"a": NOUL}},
            headers={"Authorization": "Bearer wrong"},
        )
        assert bad.status_code == 401
        assert bad.json()["detail"]["error_type"] == "authentication_error"
        good = client.post(
            "/v1/systemone",
            json={"model": "jev-latest", "state": "x", "questions": {"a": NOUL}},
            headers={"Authorization": "Bearer secret"},
        )
        assert good.status_code == 200
        assert client.get("/health").status_code == 200  # health is never behind the key


def test_models_lists_served_ids_and_aliases(client):
    models = client.get("/v1/models").json()["models"]
    names = [m["name"] for m in models]
    assert names == ["gemma4-e4b", "jev-latest", "jev-preview"]
    assert all(set(m) == {"name", "description", "release_date"} for m in models)


def test_limits_reports_the_effective_ceilings(client):
    body = client.get("/v1/limits").json()
    assert body["limits"]["max_questions"] == 64
    assert body["models"]["gemma4-e4b"]["max_prompt_tokens"] == 8191  # clamped to max_model_len - 1


def test_health_endpoints(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["upstreams"]["gemma4-e4b"]["ready"] is True


def test_health_is_503_when_no_upstream_probes(config):
    def dead(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "no workers available"}})

    with TestClient(create_app(config, transport=httpx.MockTransport(dead))) as client:
        assert client.get("/health").status_code == 503
        assert ask(client, {"a": NOUL}).status_code == 503


def test_openapi_docs_are_served(client):
    assert client.get("/docs").status_code == 200
    assert "/v1/systemone" in client.get("/openapi.json").json()["paths"]
