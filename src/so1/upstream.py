"""vLLM client: label verification, startup probes and the one-token scoring call."""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from so1.config import Limits, ModelConfig
from so1.prompt import NOUL_LABELS, SYSTEM
from so1.readout import label_distribution

log = logging.getLogger("so1.upstream")

# Gemma 4's empty thought block. Larger variants emit it before the answer even with
# thinking disabled, which pushes the answer off the first generated token.
THOUGHT_PREFIX = "<|channel>thought\n<channel|>"

RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_TEXT = ("no workers available", "worker is not ready")
_MAX_LOGPROBS_RE = re.compile(r"greater than max allowed:\s*(\d+)")

# Tokens the model will not pick for the probe question. If the server masks logits before
# reporting logprobs, only these come back; if it reports raw logprobs, the real answer leaks in.
_IMPROBABLE = ("Kumquat", "Xylophone", "Zamboni")

_PROBE_STATE = "Weather note: it is a clear, sunny day with no clouds."
_PROBE_QUESTION = "QUESTION: Is the sky blue?\n\nReply with only Yes or No."


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retryable = retryable


@dataclass
class Branch:
    """One question's forward pass."""

    probs: list[float]
    coverage: float
    prompt_tokens: int
    top_token: str
    latency_ms: float


@dataclass
class Upstream:
    config: ModelConfig
    api_key: str | None
    client: httpx.AsyncClient
    limits: Limits

    served_model: str = ""
    max_model_len: int = 0
    max_logprobs: int = 20
    mode: str = "fallback"  # "exact" once the server is known to report processed logprobs
    strategy: str = "chat"  # "prefix" when the model emits a thought block first
    labels: list[str] = field(default_factory=list)
    label_ids: dict[str, int] = field(default_factory=dict)
    thought_ids: list[int] = field(default_factory=list)
    ready: bool = False
    tokenizer_ok: bool = True
    detail: str = "not probed"

    # ---------------------------------------------------------------- HTTP

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None, attempts: int = 3, base: str | None = None
    ) -> Any:
        last: UpstreamError | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.request(
                    method,
                    (base or self.config.url) + path,
                    json=body,
                    headers=self._headers,
                    timeout=self.limits.upstream_timeout_s,
                )
            except httpx.TimeoutException as exc:
                last = UpstreamError(504, f"upstream timeout: {exc}", retryable=True)
            except httpx.HTTPError as exc:
                last = UpstreamError(502, f"upstream unreachable: {exc}", retryable=True)
            else:
                if response.status_code < 400:
                    return response.json()
                text = response.text[:400]
                retryable = response.status_code in RETRY_STATUSES or any(t in text.lower() for t in RETRY_TEXT)
                last = UpstreamError(response.status_code, text, retryable=retryable)
            if not last.retryable or attempt == attempts - 1:
                raise last
            await asyncio.sleep(0.5 * 2**attempt)
        raise last  # pragma: no cover - loop always raises or returns

    async def tokenize(self, **body: Any) -> list[int]:
        payload = await self._request(
            "POST", "/tokenize", {"model": self.served_model, **body}, base=self.config.tokenize_url
        )
        return payload["tokens"]

    # ---------------------------------------------------------------- probes

    async def probe(self) -> None:
        """Discover the served model, verify labels and pick a readout mode. Idempotent."""
        try:
            await self._probe_model()
            await self._probe_labels()
            await self._probe_strategy()
            await self._probe_exact_mode()
        except UpstreamError as exc:
            self.ready = False
            self.detail = f"probe failed: {exc.message[:200]}"
            log.warning("%s: %s", self.config.name, self.detail)
            return
        self.ready = True
        self.detail = (
            f"{self.mode} readout, {self.strategy} strategy, {len(self.labels)} labels"
            f"{'' if self.tokenizer_ok else ' (assumed, no /tokenize)'}, "
            f"max_model_len={self.max_model_len}, max_logprobs={self.max_logprobs}"
        )
        log.info("%s: %s", self.config.name, self.detail)

    async def _probe_model(self) -> None:
        payload = await self._request("GET", "/v1/models")
        entry = next((m for m in payload["data"] if m["id"] == self.config.upstream_model), payload["data"][0])
        self.served_model = entry["id"]
        # A gateway normalises /v1/models to the OpenAI schema and drops max_model_len.
        self.max_model_len = int(entry.get("max_model_len") or self.config.max_model_len or 0)

    async def _probe_labels(self) -> None:
        """Keep only labels that are a single token: A..Z, then two-letter combinations.

        Falls back to assuming A..Z / Yes / No when POST /tokenize is unreachable, which is the
        case behind an OpenAI-only gateway. Single letters are one token in every current
        tokenizer, so this is safe; two-letter labels are not assumed, and the per-question
        `coverage` diagnostic still catches a tokenizer that disagrees.
        """
        try:
            await self.tokenize(prompt="A", add_special_tokens=False)
        except UpstreamError as exc:
            self.tokenizer_ok = False
            self.labels = list(string.ascii_uppercase)
            self.label_ids = {}
            log.warning(
                "%s: POST /tokenize unreachable at %s (%s). Assuming single-token labels A-Z; "
                "options per question are capped at 26 and the exact readout and prefix mode are "
                "unavailable. Set models.%s.tokenizer_url to a direct vLLM route to remove this.",
                self.config.name, self.config.tokenize_url, exc.message[:80], self.config.name,
            )
            return
        self.tokenizer_ok = True
        wanted = max(self.limits.max_choice_options, len(NOUL_LABELS))
        candidates = [*NOUL_LABELS, *string.ascii_uppercase]
        pairs = ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))
        candidates += list(itertools.islice(pairs, 200))

        gate = asyncio.Semaphore(16)

        async def one(text: str) -> tuple[str, list[int] | None]:
            async with gate:
                try:
                    return text, await self.tokenize(prompt=text, add_special_tokens=False)
                except UpstreamError:
                    return text, None

        results = await asyncio.gather(*(one(c) for c in candidates))
        letters: list[str] = []
        for text, tokens in results:
            if tokens is None or len(tokens) != 1:
                continue
            self.label_ids[text] = tokens[0]
            if text not in NOUL_LABELS:
                letters.append(text)
        # Keep A..Z in order first, then the two-letter labels, up to the configured ceiling.
        order = {c: i for i, c in enumerate(candidates)}
        self.labels = sorted(letters, key=lambda t: order[t])[:wanted]
        missing = [label for label in NOUL_LABELS if label not in self.label_ids]
        if missing or len(self.labels) < 2:
            raise UpstreamError(500, f"tokenizer has no single-token labels for {missing or 'A/B'}")
        # Recorded for diagnostics only: which variants the model might emit instead.
        for variant in ("yes", "no", "a", "b"):
            try:
                tokens = await self.tokenize(prompt=variant, add_special_tokens=False)
            except UpstreamError:
                continue
            if len(tokens) == 1:
                self.label_ids.setdefault(variant, tokens[0])

    def _probe_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"STATE:\n{_PROBE_STATE}\n\n{_PROBE_QUESTION}"},
        ]

    async def _probe_strategy(self) -> None:
        """Plain chat works for E2B/E4B. Bigger Gemma 4 models need the thought block prepended."""
        messages = self._probe_messages()
        strategies = ("chat", "prefix") if self.tokenizer_ok else ("chat",)
        for strategy in strategies:
            self.strategy = strategy
            try:
                if strategy == "prefix" and not self.thought_ids:
                    self.thought_ids = await self.tokenize(prompt=THOUGHT_PREFIX, add_special_tokens=False)
                top, _, _ = await self._raw_logprobs(messages, list(NOUL_LABELS), self.max_logprobs)
            except UpstreamError as exc:
                log.warning("%s: strategy %r failed: %s", self.config.name, strategy, exc.message[:150])
                continue
            _, coverage = label_distribution(top, list(NOUL_LABELS))
            log.info("%s: strategy %r label coverage %.3f", self.config.name, strategy, coverage)
            if coverage >= 0.5:
                return
        self.strategy = "chat"
        raise UpstreamError(500, "no readout strategy put an answer label on the first token")

    async def _probe_exact_mode(self) -> None:
        """Exact mode needs the server started with --logprobs-mode processed_logprobs.

        Restricting sampling to tokens the model would never choose separates the two: with
        processed logprobs only those come back, with raw logprobs the real answer leaks in.
        """
        self.max_logprobs = await self._discover_max_logprobs()
        if not self.tokenizer_ok:
            return  # cannot map labels to ids without the tokenizer
        ids: list[int] = []
        for text in _IMPROBABLE:
            try:
                tokens = await self.tokenize(prompt=text, add_special_tokens=False)
            except UpstreamError:
                continue
            ids.append(tokens[0])
        if len(ids) < 2:
            log.warning("%s: could not build an exact-mode probe; staying on fallback", self.config.name)
            return
        try:
            top, _, _ = await self._raw_logprobs(self._probe_messages(), None, len(ids), allowed_token_ids=ids)
        except UpstreamError as exc:
            log.info("%s: exact readout rejected (%s); using fallback", self.config.name, exc.message[:120])
            return
        allowed = {t.strip().lower() for t in _IMPROBABLE} | {t.strip().lower()[:1] for t in _IMPROBABLE}
        leaked = [token for token, _ in top if token.strip().lower() not in allowed]
        total = math.fsum(math.exp(lp) for _, lp in top)
        if leaked:
            log.warning(
                "%s: server returns raw logprobs (%r leaked past allowed_token_ids); using fallback readout. "
                "Restart vLLM with --logprobs-mode processed_logprobs --max-logprobs %d for exact mode.",
                self.config.name,
                leaked[:3],
                self.limits.max_choice_options,
            )
            return
        if total < 0.99:
            log.warning("%s: allowed-token mass %.3f < 1; using fallback readout", self.config.name, total)
            return
        self.mode = "exact"

    async def _discover_max_logprobs(self) -> int:
        wanted = self.limits.max_choice_options
        try:
            await self._raw_logprobs(self._probe_messages(), None, wanted)
        except UpstreamError as exc:
            found = _MAX_LOGPROBS_RE.search(exc.message)
            return int(found.group(1)) if found else 20
        return wanted

    # ---------------------------------------------------------------- scoring

    async def _render_prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        ids = await self.tokenize(
            messages=messages,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        tail = self.thought_ids
        return ids if tail and ids[-len(tail) :] == tail else ids + tail

    async def _raw_logprobs(
        self,
        messages: list[dict[str, str]],
        labels: list[str] | None,
        top_k: int,
        allowed_token_ids: list[int] | None = None,
    ) -> tuple[list[tuple[str, float]], str, int]:
        """One forward pass. Returns (top [(token, logprob)], sampled token, prompt tokens)."""
        sampling: dict[str, Any] = {"max_tokens": 1}
        if allowed_token_ids is not None:
            # The sampled token is discarded; these keep the distribution untruncated.
            sampling |= {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "allowed_token_ids": allowed_token_ids,
            }
        else:
            sampling |= {"temperature": 0.0}

        if self.strategy == "prefix":
            body = {
                "model": self.served_model,
                "prompt": await self._render_prompt_ids(messages),
                "logprobs": top_k,
                **sampling,
            }
            payload = await self._request("POST", "/v1/completions", body)
            logprobs = payload["choices"][0]["logprobs"]
            top = list(logprobs["top_logprobs"][0].items())
            sampled = logprobs["tokens"][0]
        else:
            body = {
                "model": self.served_model,
                "messages": messages,
                "logprobs": True,
                "top_logprobs": top_k,
                "chat_template_kwargs": {"enable_thinking": False},
                **sampling,
            }
            payload = await self._request("POST", "/v1/chat/completions", body)
            entry = payload["choices"][0]["logprobs"]["content"][0]
            top = [(t["token"], t["logprob"]) for t in entry["top_logprobs"]]
            sampled = entry["token"]
        del labels  # kept in the signature for call-site readability
        return top, sampled, int(payload.get("usage", {}).get("prompt_tokens") or 0)

    async def score(self, messages: list[dict[str, str]], labels: list[str]) -> Branch:
        """Score one question: a single forward pass read as a distribution over labels."""
        started = time.perf_counter()
        use_exact = self.mode == "exact" and len(labels) <= self.max_logprobs
        allowed = [self.label_ids[label] for label in labels] if use_exact else None
        top_k = len(labels) if use_exact else self.max_logprobs
        top, sampled, prompt_tokens = await self._raw_logprobs(messages, labels, top_k, allowed_token_ids=allowed)
        probs, coverage = label_distribution(top, labels)
        return Branch(
            probs=probs,
            coverage=coverage,
            prompt_tokens=prompt_tokens,
            top_token=sampled,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def warm(self, messages: list[dict[str, str]]) -> int:
        """Send the shared prefix once so parallel branches hit the prefix cache."""
        _, _, prompt_tokens = await self._raw_logprobs(messages, None, 1)
        return prompt_tokens
