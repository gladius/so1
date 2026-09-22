"""A fake vLLM endpoint, so unit tests drive the real code path end to end."""

from __future__ import annotations

import contextlib
import json
import math
import re
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from so1 import config as config_module
from so1.app import create_app

# Anything not in here tokenizes to more than one token, like a real tokenizer's rare pairs.
SINGLE_TOKENS = [
    "Yes",
    "No",
    "yes",
    "no",
    "a",
    "b",
    *(chr(c) for c in range(ord("A"), ord("Z") + 1)),
    "AA",
    "AB",
    "AC",
    "Kumquat",
    "Xylophone",
    "Zamboni",  # the tokens the exact-mode probe restricts sampling to
]
TOKEN_IDS = {text: 1000 + i for i, text in enumerate(SINGLE_TOKENS)}
IDS_TOKEN = {i: t for t, i in TOKEN_IDS.items()}

_OPTION_LINE = re.compile(r"^([A-Z]{1,2})\. ", re.MULTILINE)


class FakeVLLM:
    """Parses the prompt for its labels, then answers with a distribution the test picks."""

    def __init__(self) -> None:
        self.model = "gemma4-e4b"
        self.max_model_len = 8192
        self.max_logprobs = 20
        self.logprobs_mode = "raw"  # "processed" once the server masks logits before reporting
        self.weights: dict[str, float] | None = None
        self.noise = 0.0  # probability mass parked on non-label tokens
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.fail_with: tuple[int, str] | None = None
        self.fail_after: int = 0

    # -------------------------------------------------------------- helpers

    def labels_in(self, text: str) -> list[str]:
        options = _OPTION_LINE.findall(text)
        return options if options else ["Yes", "No"]

    def distribution(self, labels: list[str]) -> list[tuple[str, float]]:
        if self.weights:
            raw = [max(self.weights.get(label, 0.0), 1e-12) for label in labels]
        else:
            raw = [2.0**-i for i in range(len(labels))]  # first label favoured
        total = math.fsum(raw)
        scale = 1.0 - self.noise
        top = [(label, math.log(value / total * scale)) for label, value in zip(labels, raw, strict=True)]
        if self.noise:
            top.append(("The", math.log(self.noise)))
        return sorted(top, key=lambda pair: -pair[1])

    @staticmethod
    def _text_of(body: dict[str, Any]) -> str:
        return "\n".join(m["content"] for m in body.get("messages", []))

    # -------------------------------------------------------------- transport

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.url.path, body))
        if self.fail_with and len(self.requests) > self.fail_after:
            status, message = self.fail_with
            return httpx.Response(status, json={"error": {"message": message}})

        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [{"id": self.model, "root": "google/gemma-4-E4B-it", "max_model_len": self.max_model_len}]
                },
            )
        if path == "/tokenize":
            return self._tokenize(body)
        if path in ("/v1/chat/completions", "/v1/completions"):
            return self._completion(path, body)
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})

    def _tokenize(self, body: dict[str, Any]) -> httpx.Response:
        if "prompt" in body:
            prompt = body["prompt"]
            if isinstance(prompt, str):
                tokens = [TOKEN_IDS[prompt]] if prompt in TOKEN_IDS else [7, 8, 9]
            else:
                tokens = list(prompt)
        else:
            tokens = list(range(max(1, len(self._text_of(body)) // 4)))
        return httpx.Response(200, json={"count": len(tokens), "tokens": tokens, "max_model_len": self.max_model_len})

    def _completion(self, path: str, body: dict[str, Any]) -> httpx.Response:
        requested = body.get("top_logprobs") if path.endswith("chat/completions") else body.get("logprobs")
        if requested and requested > self.max_logprobs:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": f"Requested sample logprobs of {requested}, which is greater than "
                        f"max allowed: {self.max_logprobs} (parameter=logprobs, value={requested})"
                    }
                },
            )
        allowed = body.get("allowed_token_ids")
        if allowed is not None and self.logprobs_mode == "processed":
            top = [(IDS_TOKEN.get(i, "?"), math.log(1.0 / len(allowed))) for i in allowed]
        else:
            top = self.distribution(self.labels_in(self._text_of(body) or ""))
            if allowed is not None:
                # Raw mode: sampling is masked but the reported logprobs are the unmasked ones.
                top = [*top, (IDS_TOKEN.get(allowed[0], "?"), math.log(1e-9))]
        top = top[: requested or 1]
        prompt_tokens = max(1, len(json.dumps(body)) // 4)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 1}
        if path.endswith("chat/completions"):
            content = [
                {
                    "token": top[0][0],
                    "logprob": top[0][1],
                    "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top],
                }
            ]
            return httpx.Response(200, json={"choices": [{"logprobs": {"content": content}}], "usage": usage})
        return httpx.Response(
            200,
            json={"choices": [{"logprobs": {"tokens": [top[0][0]], "top_logprobs": [dict(top)]}}], "usage": usage},
        )


@pytest.fixture
def fake() -> FakeVLLM:
    return FakeVLLM()


@pytest.fixture
def config() -> config_module.Config:
    return config_module.Config(
        default_model="gemma4-e4b",
        models={
            "gemma4-e4b": config_module.ModelConfig(
                name="gemma4-e4b",
                url="http://vllm.test",
                upstream_model="gemma4-e4b",
                description="test",
                release_date="2026-09-21",
            )
        },
        aliases={"jev-latest": "gemma4-e4b", "jev-preview": "gemma4-e4b"},
        upstream_api_key="upstream-key",
    )


class FaultTransport(httpx.AsyncBaseTransport):
    """Delegates to the fake, but applies a fault once armed — i.e. after the startup probes."""

    def __init__(self, fake: FakeVLLM, fault) -> None:
        self.fake = fake
        self.fault = fault
        self.armed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.armed:
            injected = await self.fault(request)
            if injected is not None:
                return injected
        return self.fake.handler(request)


def is_branch(request: httpx.Request) -> bool:
    """A scoring call, as opposed to a probe, a tokenize or the single-token warm-up."""
    if not request.url.path.endswith("/completions"):
        return False
    body = json.loads(request.content)
    return (body.get("top_logprobs") or body.get("logprobs")) != 1


@contextlib.contextmanager
def running(config: config_module.Config, fake: FakeVLLM, transport: httpx.AsyncBaseTransport | None = None):
    """A started service with probe traffic already cleared away and any fault armed."""
    transport = transport if transport is not None else httpx.MockTransport(fake.handler)
    with TestClient(create_app(config, transport=transport)) as test_client:
        if isinstance(transport, FaultTransport):
            transport.armed = True
        fake.requests.clear()
        yield test_client


@pytest.fixture
def client(config: config_module.Config, fake: FakeVLLM):
    with running(config, fake) as test_client:
        yield test_client
