"""Our responses must be the same shape as the real Jev's, field by field.

Goldens are recorded from live api.typesafe.ai by scripts/record_golden.py. The check is
structural (keys and types), not numeric: a different model gives different numbers.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from so1.schemas import SystemOneRequest

GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "golden"
SPEC = json.loads((pathlib.Path(__file__).parent / "fixtures" / "typesafe_openapi.json").read_text())

CASES = sorted(p.stem for p in GOLDEN.glob("*.json"))
OK_CASES = [
    name
    for name in CASES
    if json.loads((GOLDEN / f"{name}.json").read_text())["status"] == 200 and name != "models_list"
]


def load(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text())


def shape(value, path="") -> dict[str, str]:
    """Flatten a JSON value into {path: type}. Numbers collapse so 0 and 0.0 agree."""
    if isinstance(value, dict):
        out: dict[str, str] = {path: "object"}
        for key, item in value.items():
            out |= shape(item, f"{path}.{key}" if path else key)
        return out
    if isinstance(value, list):
        out = {path: "array"}
        for index, item in enumerate(value):
            out |= shape(item, f"{path}[{index}]")
        return out
    if isinstance(value, bool):
        return {path: "boolean"}
    if isinstance(value, (int, float)):
        return {path: "number"}
    if value is None:
        return {path: "null"}
    return {path: "string"}


@pytest.fixture
def answer_for(client):
    def run(request_body: dict):
        response = client.post("/v1/systemone", json=request_body)
        assert response.status_code == 200, response.text
        return response.json()

    return run


@pytest.mark.parametrize("name", OK_CASES)
def test_our_response_matches_the_golden_shape(name, answer_for):
    golden = load(name)
    ours = answer_for(golden["request"])
    assert shape(ours) == shape(golden["response"])


@pytest.mark.parametrize("name", OK_CASES)
def test_the_official_sdk_parses_our_response(name, answer_for):
    """The strict, frozen pydantic models the real SDK decodes responses with."""
    from typesafe_sdk._core.response_types import SystemOneResponse as SdkResponse

    golden = load(name)
    ours = answer_for(golden["request"])
    # The SDK decodes from JSON bytes, which is what lets its int-keyed score maps work.
    parsed = SdkResponse.model_validate_json(json.dumps(ours))
    assert set(parsed.answers) == set(golden["response"]["answers"])
    for qid, answer in parsed.answers.items():
        assert answer.type == golden["response"]["answers"][qid]["type"]


@pytest.mark.parametrize("name", OK_CASES)
def test_golden_requests_validate_against_our_request_schema(name):
    SystemOneRequest.model_validate(load(name)["request"])


def test_score_legend_and_probabilities_share_integer_keys(answer_for):
    golden = load("score_ten_levels")
    answer = answer_for(golden["request"])["answers"]["anger"]
    assert set(answer["legend"]) == set(answer["probabilities"])
    assert sorted(int(k) for k in answer["legend"]) == list(range(10))
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)


def test_noul_answers_carry_no_confidence(answer_for):
    answer = answer_for(load("noul_basic")["request"])["answers"]["is_urgent"]
    assert set(answer) == {"type", "noul"}


def test_choice_probabilities_are_keyed_by_the_requested_options(answer_for):
    golden = load("choice_basic")
    ours = answer_for(golden["request"])["answers"]["department"]
    assert set(ours["probabilities"]) == set(golden["request"]["questions"]["department"]["criteria"])
    assert ours["choice"] in ours["probabilities"]


def test_models_list_matches_the_golden_shape(client):
    golden = load("models_list")["response"]
    ours = client.get("/v1/models").json()
    assert set(ours) == set(golden)
    assert {k for m in ours["models"] for k in m} == {k for m in golden["models"] for k in m}


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    ("name", "request_body"),
    [
        (
            "err_missing_state",
            {"model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "Is this urgent?"}}},
        ),
        ("err_empty_questions", {"model": "jev-latest", "state": "x", "questions": {}}),
        (
            "err_unknown_type",
            {"model": "jev-latest", "state": "x", "questions": {"q": {"type": "bogus", "instructions": "y"}}},
        ),
    ],
)
def test_error_bodies_match_the_golden_shape(name, request_body, client):
    golden = load(name)
    response = client.post("/v1/systemone", json=request_body)
    assert response.status_code == golden["status"]
    ours, theirs = response.json()["detail"], golden["response"]["detail"]
    assert type(ours) is type(theirs)
    if isinstance(theirs, list):
        # A validation entry: same keys carrying the same meaning.
        assert {"type", "loc", "msg"} <= set(ours[0])
        assert ours[0]["loc"] == theirs[0]["loc"]
        assert ours[0]["type"] == theirs[0]["type"]
    else:
        assert set(ours) == set(theirs)


def test_missing_api_key_matches_the_golden_403(config, fake):
    from tests.conftest import running

    from so1.config import Config

    golden = load("err_no_auth")
    with running(Config(**{**vars(config), "api_key": "secret"}), fake) as client:
        response = client.post("/v1/systemone", json={"model": "jev-latest", "state": "x", "questions": {}})
    assert response.status_code == golden["status"] == 403
    assert response.json() == golden["response"]


# ------------------------------------------------------------------ against the published spec


def _schema(name: str) -> dict:
    return SPEC["components"]["schemas"][name]


@pytest.mark.parametrize(
    "name", ["SystemOneRequest", "SystemOneResponse", "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "Usage"]
)
def test_our_openapi_keeps_the_published_fields_and_requirements(name, client):
    theirs = _schema(name)
    ours = client.get("/openapi.json").json()["components"]["schemas"][name]
    assert set(ours["properties"]) == set(theirs["properties"]), name
    assert set(ours.get("required", [])) == set(theirs.get("required", [])), name


def test_the_published_spec_still_only_has_the_two_routes():
    """If TypeSafe adds a route, re-record the goldens before trusting this suite."""
    assert set(SPEC["paths"]) == {"/v1/systemone", "/v1/models"}
