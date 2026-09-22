# so1 — a Jev-compatible System One service on our own vLLM

`POST /v1/systemone`, wire-compatible with [TypeSafe's Jev API](https://docs.typesafe.ai/api), answered by
our own Gemma 4 endpoints. Agent teams point the official SDK at it and change nothing else:

```bash
export TYPESAFE_BASE_URL=http://so1.internal:8080
export TYPESAFE_API_KEY=...        # only if SO1_API_KEY is configured
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "billing": Noul(instructions="Is this ticket about billing?"),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "frustrated": None}),
            "urgency": Score(instructions="How urgent?", criteria=["can wait", "this week", "today"]),
        },
    )
```

## How it works

Every question is answered by **one forward pass, one token**. Nothing is generated and nothing is parsed.

1. **One prompt per question**, with the state FIRST and the question LAST, so all questions in a request
   share a byte-identical prefix. With two or more questions the prefix is sent once as a warm-up
   (`max_tokens: 1`, result discarded) so the parallel branches hit vLLM's prefix cache instead of all
   recomputing the state at once.
2. **Answers are single tokens.** `Yes`/`No` for noul, `A`, `B`, `C`… for choice and score. At startup every
   label is verified to be exactly one token via `POST /tokenize`; labels that are not are dropped. Option
   keys are never shown to the model — only the descriptions, in request order.
3. **The distribution is read from the logprobs** of that one position, folded over label variants
   (`"A"`, `" a."`, `"A)"`), renormalised over the labels, then temperature-scaled.
4. **The answer is computed from the distribution**: `noul` = P(Yes), `choice` = argmax plus the full
   distribution, `score` = `sum(i * p_i)` over zero-based levels.

All branches run concurrently under a global semaphore. If any branch fails the siblings are cancelled
immediately rather than left burning GPU time.

### Two readout modes

| mode | how | when |
|---|---|---|
| **exact** | `allowed_token_ids` restricted to the label ids, `logprobs: true`, `top_logprobs: K`, sampling left untruncated (`temperature 1.0, top_p 1.0, top_k -1, min_p 0`). The returned distribution *is* the distribution over the labels. | only when the server reports **processed** logprobs |
| **fallback** | plain `top_logprobs: 20`, sum the label variants, renormalise, record `coverage` | otherwise |

The mode is chosen per upstream at startup, by restricting sampling to tokens the model would never pick
and checking whether anything else leaks into the reported logprobs. **This matters:** the obvious probe —
"do the returned probabilities sum to ~1?" — passes under both modes and silently selects exact mode on a
server that is really reporting raw logprobs, which would produce wrong numbers. We check set membership.

**Our RunPod E4B endpoint currently runs raw logprobs, so it uses the fallback readout.** It already has
`--max-logprobs 64`; the missing flag is:

```
--logprobs-mode processed_logprobs
```

The service picks it up on its next start, with no config change. Meanwhile options per question are
capped at whatever the server's `max_logprobs` actually is (64 today, discovered at startup), because the
fallback readout cannot see past the server's top-k and a label outside it would silently read as
probability zero. We return `400` rather than answer a question we cannot resolve.

Exact mode is still worth turning on: it removes the top-k assumption entirely, so the reported
distribution is exactly the distribution over the labels rather than whatever share of the top-k they
happened to occupy (`coverage`, which we log per question).

### Thought blocks

Gemma 4 models larger than E2B/E4B can emit an empty thought block before the answer even with thinking
disabled, which pushes the answer off the first generated token. At startup each upstream is probed; if the
first token is not a label, that upstream switches to `prefix` mode: render the chat template via
`/tokenize`, append the thought-block token ids, and score via `/v1/completions` on the token-id prompt.
E4B does not need this; the probe runs anyway.

## Configuration

Copy `config.example.yaml` to `config.yaml`. `${VAR}` is expanded from the environment.

```yaml
default_model: gemma4-e4b
aliases: {jev-latest: gemma4-e4b, jev-preview: gemma4-e4b}
models:
  gemma4-e4b:
    url: https://fcy67kixeo37lf.api.runpod.ai
    upstream_model: gemma4-e4b
  gemma4-26b-a4b:
    enabled: false          # endpoint is down
upstream_api_key: ${RUNPOD_API_KEY}
api_key: ${SO1_API_KEY}     # optional bearer auth on OUR api; unset = open
temperature: {noul: 1.0, choice: 1.0, score: 1.0}
```

| env | meaning |
|---|---|
| `SO1_CONFIG` | config path (default `config.yaml`) |
| `SO1_HOST`, `SO1_PORT` | bind address (default `0.0.0.0:8080`) |
| `SO1_API_KEY` | bearer token for our API, if the config references it |
| `RUNPOD_API_KEY` | credential for the upstream endpoints |

```bash
uv sync --extra dev
uv run python -m so1                     # http://127.0.0.1:8080, docs at /docs
docker build -t so1 . && docker run -p 8080:8080 -v $PWD/config.yaml:/app/config.yaml \
    -e RUNPOD_API_KEY so1
```

## Routes

| route | notes |
|---|---|
| `POST /v1/systemone` | the Jev contract; adds a `Server-Timing` header (`prepare`, `warmup`, `branches`) |
| `GET /v1/models` | served ids and aliases, in the real API's `{"models": [...]}` shape |
| `GET /v1/limits` | effective ceilings, including what each upstream can actually do |
| `GET /health` | probes upstreams; `503` when none is ready |
| `GET /health/live` | process liveness only |
| `/docs` | OpenAPI |

## Limits

Defaults, all configurable under `limits:`.

| limit | default |
|---|---|
| questions per request | 64 |
| choice options | 64, further capped to the upstream's `max_logprobs` on the fallback readout (64 today) |
| score levels | 10 (what Jev documents) |
| body size | 2 MiB → `413` |
| prompt tokens per branch | 32768, **clamped at startup to the upstream's `max_model_len` − 1** (8191 on E4B today) |
| concurrent requests | 16 → `529` with `Retry-After` |
| concurrent upstream calls | 64 |
| request timeout | 120 s → `504` |

Upstream `429/5xx` and "no workers available" are retried up to 3 times with backoff.

## Contract fidelity

The schema is mirrored from the live `https://api.typesafe.ai/openapi.json`, and `tests/fixtures/golden/`
holds responses recorded from the real API (`scripts/record_golden.py`). Contract tests assert our
responses are the same shape field by field, and that the official SDK's own strict pydantic models parse
them. Error mapping matches the real API exactly, including some things the docs do not mention:

| case | status | body |
|---|---|---|
| missing API key | `403` | `{"detail": {"error_type": "authentication_error", ...}}` |
| invalid API key | `401` | same shape |
| schema violation | `422` | `{"detail": [{"type", "loc", "msg", "input", ...}]}` |
| unknown question `type` | `400` | `{"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}` |
| *missing* question `type` | `422` | `union_tag_not_found` — note this differs from the above |
| unknown model | `400` | `{"detail": {"error_type": "api_usage_error", "message": "Unknown model: x"}}` |
| over a size ceiling | `400` | `{"detail": "Too many choices. Must have at most N choices."}` — a bare string |

### Confidence

The brief specified `1 - H(p)/ln K`. **That is not what Jev does.** Fitted against 66 live responses
(`tests/fixtures/jev_confidence_samples.json`, mean absolute error 0.0085 — the residual is Jev's own
2-decimal rounding):

- **choice** — `clamp((K·p_max − 1) / (K − 1), 0, 1)`. Scaled peak probability: 1 when all the mass is on
  one option, 0 when uniform.
- **score** — `clamp(1 − Σ pᵢ·|i − mode| / D_K, 0, 1)` where `D_K` is the mean `|level − centre|` of a
  uniform distribution. This is ordinal: a distribution split across *opposite ends* of the rubric scores
  near zero even when its peak is tall, which the nominal formula would call confident.
- **noul** — carries no `confidence` field at all.

The two agree exactly when `K == 2`.

### Deliberate differences from the real Jev

- **Full float precision.** Jev rounds probabilities and scores to 2 decimals; we don't. Same types, more
  information. If you need bit-identical output, round at the call site.
- **`GET /v1/models` lists our model ids as well as the aliases.** Jev lists only aliases.
- **`usage`** counts the shared state once, plus each question's own suffix — we re-send the state per
  branch, but billing it per branch would be misleading. `output_tokens` is the honest count: one sampled
  token per question. Jev reports its own internal numbers (~20 per question).
- **`release_date`** comes from config. Jev returns an ISO-8601 timestamp; the docs claim `YYYY-MM-DD`.
- Score accepts 1 level and choice accepts 1 option, matching the published OpenAPI (`min_length: 1`)
  rather than the brief's "2 to 10".

## Measured results

### On public benchmarks with real gold labels

The only evaluation where neither the inputs nor the labels are ours
(`scripts/bench_public.py`, data pulled live from the HuggingFace dataset server):

| | so1 (gemma4-e4b) | jev-1.13.0 |
|---|---|---|
| BoolQ, n=300 (noul) | 82.7% [78.0, 86.5] | **94.3%** [91.1, 96.4] |
| BoolQ ECE / Brier | 0.151 / 0.159 | **0.039 / 0.051** |
| SST-5, n=250 (score) | 56.0% [49.8, 62.0] | 60.4% [54.2, 66.3] |
| SST-5 within 1 level | 85.6% | 92.4% |
| SST-5 ECE / Brier | 0.364 / 0.363 | **0.187 / 0.265** |

The BoolQ gap is **11.6 points and highly significant** (McNemar p < 0.0001, 35 discordant pairs, none
in our favour). The SST-5 gap is **not** significant (p = 0.19). Calibration is worse on both, by ~4x on
BoolQ and ~2x on SST-5.

**Weight these above the synthetic numbers below.** On our own battery the gap looked negligible
(42/45 vs 43/45, p = 1.00); on third-party data it is large and one-sided. See `results/DATASETS.md`.

### On our synthetic battery

45 questions (the 15-question baseline pack plus the 30-question battery), same cases sent to both,
`gemma4-e4b` on RunPod vs `jev-1.13.0`, after a warm-up request:

| | so1 (gemma4-e4b) | jev-1.13.0 |
|---|---|---|
| accuracy | 42/45 (93%) | 43/45 (96%) |
| &nbsp;&nbsp;noul | 16/18 | 17/18 |
| &nbsp;&nbsp;choice | 16/17 | 16/17 |
| &nbsp;&nbsp;score | 10/10 | 10/10 |
| latency p50 / request | 371 ms | 341 ms |
| latency p90 / request | 784 ms | 384 ms |
| latency / question | 416 ms | 299 ms |
| ECE | 0.109 | 0.073 |
| Brier | 0.111 | 0.089 |

The two systems gave identical predictions on 36/45 questions. Reproduce with
`uv run python scripts/eval_service.py --suite all --compare-jev`.

**Read this honestly:**

- **One real accuracy gap.** `rules/annual-3-exports`: a policy saying "within 30 days" against a charge
  dated 30 August and a request dated 19 September. Jev does the subtraction, E4B does not — it answers
  "no" at p=1.00. This is the blind spot from the original run and it is unchanged. Do the date maths in
  code.
- **Two of the three so1 misses are shared with Jev.** They are deliberately ambiguous inputs where the
  right answer is a flat distribution. so1 returns confidence 1.00 and Jev returns 0.97 — both are
  over-confident, so this measures a property of the approach, not a defect unique to us. They are counted
  against both in the table.
- **On the baseline pack alone so1 scores 14/15**, matching the original direct-endpoint run exactly, with
  the same single miss.
- **Latency degrades with state size and question count; Jev is nearly flat on both.** A 12-question
  fan-out over a 1.6 KB state takes ~1.02 s against Jev's ~0.37 s. This is architectural: Jev ingests the
  state once and evaluates every question in a single pass, while we issue one upstream call per question.
  A single question over a 6500-token state costs 2374 ms against Jev's 490 ms, so prefill dominates.
  Lowering the requested top-logprobs from 64 to 20 changed nothing, so it is GPU work, not payload.
  Expect 2-3x latency swing on this endpoint depending on how many RunPod workers are warm: with
  `workersMin: 0` a quiet period drops to one worker, and the concurrent branches then serialise on it.
  Cross-request prefix caching is worth only ~8%, so it is worker count, not cache state, that moves the
  number (`results/EXPERIMENTS.md` experiment 8). Run the service in the GPU's region.
- **The prefix warm-up costs a full round trip (~450 ms of that ~1.02 s).** It is a throwaway call that
  exists only to populate vLLM's prefix cache before the branches fan out. Sending the *first real branch*
  instead would warm the same prefix and return an answer, removing one serial round trip. Not yet
  implemented; it is the single largest easy win available.
- **Jev is better calibrated (ECE 0.073 vs 0.109), which is expected** — our `temperature` is still 1.0.

### A complex request, head to head

One request, 1.6 KB structured state (account records, a 4-rule refund policy, a 6-message thread), 12
questions across all three types:

| | so1 (gemma4-e4b) | jev-1.13.0 |
|---|---|---|
| accuracy | 10/12 | 12/12 |
| latency (median) | 1020 ms | 370 ms |
| per question | 85 ms | 31 ms |
| input_tokens | 1302 | 1552 |

so1 got right: open-vs-resolved issue tracking, a conditional cancellation threat, explicit negation,
absence of evidence, 4-way and 6-way routing, best-next-action out of 5, and both score questions. Both
misses were the same two refund-eligibility questions, for the reason described under Known limitations —
and both become correct when the conversation is removed from the state.

### About that calibration

`scripts/fit_temperature.py` on the 33 labelled rows from the run above wants `noul: 3.53`, which is a
real signal — E4B is over-confident. But the script's own two-fold held-out check rejects it (held-out NLL
1.11 fitted vs 0.43 at T=1), so it reports 1.0 and the config stays uncalibrated. 33 rows is far too few,
and fitting on the set you then evaluate is circular. Collect a few hundred labelled outcomes from real
traffic, on cases you are not also reporting accuracy against, before trusting a temperature.

## Known limitations

- **Probabilities are not calibrated.** `temperature` is `1.0` everywhere until it is fitted. The numbers
  are directionally useful but are not the calibrated probabilities Jev is trained to produce. Do not port
  Jev-tuned confidence thresholds across without re-checking them.
- **Scope the state to the question.** The sharpest failure we found is not arithmetic. Asked *"would this
  charge be refundable?"* against a state whose conversation contains *"I am not asking for any money
  back"*, E4B answers no (p=0.005) — it collapses the hypothetical into the factual. Remove the thread from
  the state and the same question is answered correctly (p=0.82). Precomputing the day count did **not**
  help; removing the conversation did. Send policy-evaluation questions in their own request, against a
  state holding only the policy and the records. Jev is not fooled by this (p=0.91), so it is a
  small-model limitation, not a property of the API - though Jev documents the same failure mode
  ("large state full of irrelevant detail") and simply tolerates far more of it. Worse, the effect is
  unstable rather than a clean threshold: on one fixed question we measured P(yes) swinging between 0.014
  and 0.958 purely on which irrelevant filler was present, so a test that passes at one state size tells
  you little about the next.
  **TODO: re-test this on Gemma 4 26B-A4B** when that endpoint is back up (`models.gemma4-26b-a4b`,
  `enabled: false` in config). The open question is whether hypothetical-vs-factual conflation is a
  capacity limit that a bigger model clears, or a prompt-shape problem that persists. Reproduce with
  `tests/fixtures/eval_suite.py` case `noul/7..9` plus the refund questions in the complex case.
- **Date maths is still better done in code**, but it was not the cause here: asked in isolation, E4B gets
  "30 Aug to 19 Sep is within 30 days" right at p=0.98.
- **Small models have blind spots.** Decompose multi-rule policies into simple yes/no checks and combine
  them in code (see `tests/fixtures/eval_suite.py` for the shape of this). A single "is this eligible?"
  over a four-rule policy is the least reliable thing you can ask.
- One upstream is live; the 26B endpoint is disabled in config until it is back.
- No rate limiting or per-tenant quotas. A single static bearer token is the whole auth model.
- Observability is logs plus `Server-Timing`; there is no metrics endpoint.

## Guidance for agent teams

- Ask **many small questions in one request**. They share the state's forward pass via the prefix cache, so
  the marginal cost of an extra question is small. This is the single biggest speed lever.
- **Put facts in the state, judgements in the questions.** Anything you can compute — dates, sums,
  counts, lookups — compute it and put the answer in the state.
- **Use `confidence` as a second axis.** Route low-confidence answers to a human or a reasoning model
  rather than treating every answer as equally good.
- **A flat distribution is a real answer.** "I don't know" is information; don't threshold it away.

## Tests

```bash
uv run pytest                    # unit + contract, no network
uv run pytest -m upstream -s     # against the real RunPod endpoint (needs RUNPOD_API_KEY)
uv run pytest -m sdk             # the official SDK against a live local service
```

| suite | what it covers |
|---|---|
| `test_prompt.py` | prompt construction, every state form, criteria and label handling |
| `test_readout.py` | variant folding, calibration, both confidence formulas vs 66 real samples |
| `test_service.py` | answers, warm-up, cancellation, retries, limits, auth, error mapping |
| `test_contract.py` | our responses vs recorded Jev goldens, and against the published OpenAPI |
| `test_integration.py` | the real endpoint: probes, all three types, usage, context clamping |
| `test_sdk.py` | the official `typesafe-sdk`, unchanged, via `TYPESAFE_BASE_URL` |

## Scripts

```bash
# Record goldens from the real API (prefers TYPESAFE_API_KEY, falls back to OPENROUTER_API_KEY).
uv run python scripts/record_golden.py

# Evaluate, head to head with the real Jev on identical cases.
uv run python scripts/eval_service.py --suite all --compare-jev --dump runs.jsonl

# Fit the calibration temperature from a labelled run, then paste it into config.yaml.
uv run python scripts/fit_temperature.py runs.jsonl
```

All measured results, the raw CSVs and the run logs are committed under `results/`, with
`results/EXPERIMENTS.md` recording every experiment, its numbers and how to reproduce it. The scaling
sweep is `scripts/bench_scaling.py` (`--pool easy` for latency, `--pool hard` for accuracy) and the
12-question complex fan-out is `scripts/bench_complex.py`; both write tidy CSV/JSON ready to plot.

`eval_service.py` reports accuracy overall and per type, per-request latency (p50/p90, after a warm-up
request so a cold worker does not pollute the numbers), ECE and Brier, every miss, and exactly which
questions the two systems disagree on. Case packs live in `tests/fixtures/`: `eval_cases.py` is the
original 15-question run (vendored from `longest.py`, the one that scored 14/15) and `eval_suite.py` is a
30-question battery covering all three types from trivial to known-hard.
