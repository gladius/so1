"""Request orchestration: validate, warm the prefix cache, fan out, assemble answers."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from so1 import prompt as prompts
from so1.config import Config
from so1.readout import calibrate, choice_confidence, expected_score, score_confidence
from so1.schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from so1.upstream import Branch, Upstream, UpstreamError

log = logging.getLogger("so1.service")

# Rough bytes-per-token; only used to decide whether an exact tokenizer check is worth a round trip.
_CHARS_PER_TOKEN = 3.5
_EXACT_CHECK_AT = 0.8

_UPSTREAM_ERROR_TYPES = {502: "upstream_error", 504: "timeout_error", 429: "rate_limit_error"}


class ServiceError(Exception):
    """An error whose `detail` is already in the exact shape the real Jev returns."""

    def __init__(self, status: int, detail: Any, retry_after: int | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.retry_after = retry_after

    @property
    def message(self) -> str:
        return self.detail if isinstance(self.detail, str) else str(self.detail.get("message", self.detail))


def usage_error(message: str) -> ServiceError:
    """400, the shape Jev uses for an unusable-but-well-formed request."""
    return ServiceError(400, {"error_type": "api_usage_error", "message": message})


def limit_error(message: str) -> ServiceError:
    """400 with a bare string detail, the shape Jev uses for its size ceilings."""
    return ServiceError(400, message)


@dataclass
class Result:
    response: SystemOneResponse
    timings: dict[str, float] = field(default_factory=dict)


class Service:
    def __init__(self, config: Config, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client
        self.upstreams = {
            name: Upstream(config=model, api_key=config.upstream_api_key, client=client, limits=config.limits)
            for name, model in config.models.items()
            if model.enabled
        }
        self._branch_gate = asyncio.Semaphore(config.limits.max_upstream_concurrency)
        self._inflight = 0

    async def startup(self) -> None:
        await asyncio.gather(*(u.probe() for u in self.upstreams.values()))

    def upstream_for(self, name: str) -> Upstream:
        model = self.config.resolve(name)
        if model is None:
            raise usage_error(f"Unknown model: {name}")
        upstream = self.upstreams[model.name]
        if not upstream.ready:
            raise ServiceError(
                503, {"error_type": "service_unavailable", "message": f"Model {name!r} is not ready: {upstream.detail}"}
            )
        return upstream

    # ---------------------------------------------------------------- validation

    def _labels_for(self, question: Question, upstream: Upstream, name: str) -> list[str]:
        limits = self.config.limits
        if isinstance(question, NoulQuestion):
            return list(prompts.NOUL_LABELS)
        options = question.criteria
        count = len(options)
        kind, ceiling = (
            ("choice", limits.max_choice_options)
            if isinstance(question, ChoiceQuestion)
            else (
                "score",
                limits.max_score_levels,
            )
        )
        if count > ceiling:
            raise limit_error(
                f"Too many choices. Must have at most {ceiling} choices."
                if kind == "choice"
                else f"Too many score levels. Must have at most {ceiling} levels."
            )
        if count > len(upstream.labels):
            raise limit_error(
                f"Too many choices. Model {upstream.config.name!r} has only {len(upstream.labels)} single-token labels."
            )
        # The fallback readout only sees the server's top-k, so a label outside it reads as zero.
        # Never answer a question we cannot actually resolve.
        if upstream.mode != "exact" and count > upstream.max_logprobs:
            raise limit_error(
                f"Too many choices. Model {upstream.config.name!r} is on the fallback readout and resolves at "
                f"most {upstream.max_logprobs} {'options' if kind == 'choice' else 'levels'} per question. "
                f"Restart vLLM with --logprobs-mode processed_logprobs to raise this to {ceiling}."
            )
        return upstream.labels[:count]

    def _budget(self, upstream: Upstream) -> int:
        configured = self.config.limits.max_prompt_tokens
        # Leave room for the single sampled token.
        return min(configured, upstream.max_model_len - 1) if upstream.max_model_len else configured

    async def _check_budget(self, upstream: Upstream, branches: list[tuple[str, list[dict[str, str]]]]) -> None:
        budget = self._budget(upstream)
        name, messages = max(branches, key=lambda b: sum(len(m["content"]) for m in b[1]))
        characters = sum(len(m["content"]) for m in messages)
        if characters / _CHARS_PER_TOKEN < budget * _EXACT_CHECK_AT:
            return
        try:
            tokens = len(await upstream.tokenize(messages=messages, add_generation_prompt=True))
        except UpstreamError:
            return  # the upstream will reject it on the real call if it truly is too long
        if tokens > budget:
            raise limit_error(
                f"Request is too long. Question {name!r} renders to {tokens} tokens; the limit for "
                f"{upstream.config.name!r} is {budget}."
            )

    # ---------------------------------------------------------------- execution

    async def answer(self, request: SystemOneRequest) -> Result:
        limits = self.config.limits
        if self._inflight >= limits.max_concurrent_requests:
            raise ServiceError(
                529,
                {"error_type": "overloaded_error", "message": "Service is temporarily overloaded. Retry shortly."},
                retry_after=1,
            )
        if len(request.questions) > limits.max_questions:
            raise limit_error(f"Too many questions. Must have at most {limits.max_questions} questions.")
        self._inflight += 1
        try:
            async with asyncio.timeout(limits.request_timeout_s):
                return await self._answer(request)
        except TimeoutError as exc:
            raise ServiceError(504, {"error_type": "timeout_error", "message": "Request timed out."}) from exc
        finally:
            self._inflight -= 1

    async def _answer(self, request: SystemOneRequest) -> Result:
        started = time.perf_counter()
        upstream = self.upstream_for(request.model)

        labels = {name: self._labels_for(q, upstream, name) for name, q in request.questions.items()}
        branches = [(name, prompts.build(request.state, q, labels[name])) for name, q in request.questions.items()]
        await self._check_budget(upstream, branches)
        prepared = time.perf_counter()

        # One batched call when the upstream can render prompts locally, otherwise one call per
        # question. Batching removes the warm-up and K-1 round trips, which dominate when the
        # GPU is remote; see results/EXPERIMENTS.md experiment 14.
        contents = {
            name: prompts.user_content(request.state, question, labels[name])
            for name, question in request.questions.items()
        }
        batchable = (
            self.config.limits.batch_questions
            and upstream.can_batch
            and all(value is not None for value in contents.values())
        )

        prefix_tokens = 0
        results: dict[str, Branch] = {}
        if batchable:
            warmed = time.perf_counter()
            names = list(request.questions)
            try:
                scored = await self._guarded(
                    upstream.score_batch([contents[n] for n in names], [labels[n] for n in names])
                )
            except UpstreamError as exc:
                raise self._upstream_error(exc) from exc
            results = dict(zip(names, scored, strict=True))
        else:
            if len(branches) > 1:
                try:
                    prefix_tokens = await self._guarded(upstream.warm(prompts.prefix_messages(request.state)))
                except UpstreamError as exc:
                    log.warning("prefix warm-up failed (%s); continuing", exc.message[:150])
            warmed = time.perf_counter()
            try:
                async with asyncio.TaskGroup() as group:
                    tasks = {
                        name: group.create_task(self._guarded(upstream.score(messages, labels[name])))
                        for name, messages in branches
                    }
            except* UpstreamError as group_error:
                first = group_error.exceptions[0]
                assert isinstance(first, UpstreamError)
                raise self._upstream_error(first) from first
            results = {name: task.result() for name, task in tasks.items()}
        finished = time.perf_counter()

        answers: dict[str, Answer] = {}
        for name, question in request.questions.items():
            branch = results[name]
            probs = calibrate(branch.probs, self.config.temperature.get(question.type, 1.0))
            answers[name] = self._assemble(question, labels[name], probs)
            log.info(
                "q=%s type=%s model=%s mode=%s ms=%.0f coverage=%.3f top=%.4f first_token=%r",
                name,
                question.type,
                upstream.config.name,
                upstream.mode,
                branch.latency_ms,
                branch.coverage,
                max(probs),
                branch.top_token,
            )

        response = SystemOneResponse(
            model=upstream.config.name,
            answers=answers,
            usage=self._usage(results.values(), prefix_tokens),
        )
        return Result(
            response=response,
            timings={
                "prepare": (prepared - started) * 1000,
                "warmup": (warmed - prepared) * 1000,
                "branches": (finished - warmed) * 1000,
            },
        )

    async def _guarded(self, coro: Any) -> Any:
        async with self._branch_gate:
            return await coro

    def _upstream_error(self, error: UpstreamError) -> ServiceError:
        status = self._map_status(error.status)
        return ServiceError(
            status,
            {"error_type": _UPSTREAM_ERROR_TYPES[status], "message": f"Upstream error: {error.message[:200]}"},
        )

    @staticmethod
    def _map_status(status: int) -> int:
        if status in (408, 504):
            return 504
        if status == 429:
            return 429
        return 502

    @staticmethod
    def _usage(branches: Any, prefix_tokens: int) -> Usage:
        """State is sent once per branch but is logically ingested once, so count it once."""
        rows = list(branches)
        if not rows:
            return Usage(input_tokens=0, output_tokens=0)
        if prefix_tokens <= 0:
            prefix_tokens = min(b.prompt_tokens for b in rows)
        suffixes = sum(max(0, b.prompt_tokens - prefix_tokens) for b in rows)
        return Usage(input_tokens=prefix_tokens + suffixes, output_tokens=len(rows))

    @staticmethod
    def _assemble(question: Question, labels: list[str], probs: list[float]) -> Answer:
        if isinstance(question, NoulQuestion):
            return NoulAnswer(type="noul", noul=probs[0])
        if isinstance(question, ChoiceQuestion):
            keys = list(question.criteria)
            probabilities = dict(zip(keys, probs, strict=True))
            return ChoiceAnswer(
                type="choice",
                choice=max(zip(keys, probs, strict=True), key=lambda kv: kv[1])[0],
                probabilities=probabilities,
                confidence=choice_confidence(probs),
            )
        assert isinstance(question, ScoreQuestion)
        del labels
        return ScoreAnswer(
            type="score",
            score=expected_score(probs),
            legend={str(i): level for i, level in enumerate(question.criteria)},
            probabilities={str(i): p for i, p in enumerate(probs)},
            confidence=score_confidence(probs),
        )
