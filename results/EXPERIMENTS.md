# Recorded experiments

Everything measured against `gemma4-e4b` on RunPod (A40, endpoint `fcy67kixeo37lf`) versus the real
`jev-1.13.0` at `api.typesafe.ai`, on identical requests. Raw data sits beside this file.

| file | what |
|---|---|
| `bench_scaling_easy.csv` | state size x question count sweep, saturated question pool (latency signal) |
| `bench_scaling_hard.csv` | same sweep, discriminating question pool (accuracy signal) |
| `eval_all.json` / `eval.log` | 45-question head-to-head (baseline pack + battery), full answers |
| `eval_all_labelled.jsonl` | labelled probabilities from that run, input to `fit_temperature.py` |
| `complex_case.json` | the 12-question complex fan-out, both systems, 3 runs |
| `bench.log`, `bench_hard.log`, `integ.log` | raw run logs |

Reproduce: `scripts/eval_service.py`, `scripts/bench_scaling.py`, `scripts/bench_complex.py`.

---

## 1. Head-to-head accuracy, 45 questions

| | so1 | jev-1.13.0 |
|---|---|---|
| accuracy | 42/45 (93%) | 43/45 (96%) |
| noul / choice / score | 16/18, 16/17, 10/10 | 17/18, 16/17, 10/10 |
| latency p50 / p90 | 371 / 784 ms | 341 / 384 ms |
| ECE / Brier | 0.109 / 0.111 | 0.073 / 0.089 |

Identical predictions on 36/45. On the baseline pack alone so1 scores 14/15, matching the original
direct-endpoint run exactly, same single miss.

## 2. Complex fan-out: 12 questions, 1.6 KB structured state

| | so1 | jev-1.13.0 |
|---|---|---|
| accuracy | 10/12 | 12/12 |
| latency (median) | 1020 ms | 370 ms |
| per question | 85 ms | 31 ms |
| input_tokens | 1302 | 1552 |

Both misses were refund-eligibility questions. See experiment 3.

**Latency here moved between sessions and the earlier figure did not reproduce.** An initial measurement
gave 2358 ms (branches 1891 ms); three later invocations, each with its own warm-up and three timed runs,
agreed on 1003/1017/1033 ms (branches ~560 ms). The later number is the reproducible one and is what the
table reports. Experiment 8 identifies the cause: RunPod had scaled from 1 worker to 3.

Of the ~1020 ms, roughly 450 ms is the throwaway prefix warm-up that precedes the branches. Replacing it
with the first real branch would cut a whole round trip - see README, Known limitations.

## 3. The refund miss is state scope, not arithmetic

Same two questions, three framings:

| variant | refund_raw | refund_derived |
|---|---|---|
| full state + all 12 questions | NO (0.294) | NO (0.005) |
| full state + only the 2 refund questions | NO (0.269) | NO (0.005) |
| state **without the conversation thread** | yes (0.947) | yes (0.818) |

Correct answer is yes in every variant. It is not the fan-out (row 2 rules that out) and not the date
arithmetic — asked in isolation, E4B gets "30 Aug to 19 Sep is within 30 days" right at p=0.98, and
precomputing the day count did not help. The thread contains *"I am not asking for any money back"*, and
the model collapses the hypothetical ("would this be refundable?") into the factual ("is the customer
asking?"). Jev is not fooled: 0.80 / 0.91.

Decomposing the rule works: `is_annual` 1.000, `within_30_days` 1.000, `exports <= 5` 1.000 — AND them in
code and you get the right answer.

**TODO: re-test on Gemma 4 26B-A4B** when that endpoint returns, to see whether this is a capacity limit
or a prompt-shape problem.

## 4. Size scaling

Saturated pool — both systems 100% at every size, so this isolates latency (median ms):

| state tokens | q=1 | q=2 | q=4 | q=8 | q=16 | jev q=1 | jev q=16 |
|---|---|---|---|---|---|---|---|
| 500 | 476 | 1268 | 1463 | 1312 | 2051 | 339 | 383 |
| 1500 | 541 | 1014 | 1084 | 1164 | 2709 | 356 | 573 |
| 3000 | 756 | 1192 | 1315 | 1386 | 2694 | 345 | 762 |
| 6500 | 2374 | 2262 | 2478 | 2576 | 3501 | 490 | 789 |

Two separable costs: state size drives prefill (~5x from 500 to 6500 tokens on a single question) and
question count adds decodes. Jev is nearly flat on both because it ingests the state once.

Discriminating pool — accuracy (%) as state grows:

| state tokens | so1 q=1 | q=2 | q=4 | q=8 | jev q=1 | q=8 |
|---|---|---|---|---|---|---|
| 500 | 100 | 100 | 100 | 100 | 100 | 88 |
| 1500 | 0 | 50 | 50 | 50 | 100 | 75 |
| 3000 | 0 | 50 | 75 | 50 | 100 | 75 |
| 6500 | 0 | 50 | 50 | 38 | 100 | 75 |

so1's hard-question accuracy falls away as the padded thread grows, while Jev degrades far more gently.
Experiment 9 shows this is **not** a clean threshold: it is instability around the decision boundary,
driven by which irrelevant content happens to be present, not by token count alone.

## 5. p90 was complexity, not cold start

Within the 45-question run: the slowest request was `buried_fact/6500` at 2145 ms with a *single*
question, and the first half of the run (395 ms median) was no slower than the second (368 ms). A warm-up
request precedes all timing. Latency tracks state length, not position in the run.

## 6. Requested top-logprobs does not affect latency

The endpoint advertises `max_logprobs=64` and we request that per branch on the fallback readout.
Re-running the 12-question complex case with the ceiling lowered to 20: 2358 ms vs 2372 ms, branches
1891 ms vs 1899 ms. Both readings are from the slower session described in experiment 2, but they were
taken back to back under identical conditions, so the comparison holds: the cost is GPU decode, not
logprob payload.

## 7. Cost

Jev publishes $0.042/Mtok input, output free. Ours is time-billed: A40 at $0.49/hr secure
($0.35 community) per the RunPod API.

Break-even sustained input throughput — what we must push to match Jev's price:

| GPU rate | required |
|---|---|
| $0.35/hr | 2,315 input tok/s |
| $0.49/hr | 3,241 input tok/s |
| $1.22/hr | 8,069 input tok/s |

Measured sequentially (one request at a time, no batching), we reach 220-790 tok/s, so cost per request
is 4-15x Jev's. At low volume the idle timeout dominates: `workersMin=0, idleTimeout=300s` means a single
request holds a worker for 5 minutes, so 1 req/hr costs ~$0.041/hr against Jev's $0.00007/hr.

These are sequential numbers and therefore a floor, not a verdict — vLLM batches concurrent requests, so
the honest figure needs a throughput test under load. **Not yet measured.**

---

## 8. The rerun speed-up was worker autoscaling, not prefix caching

Suspicion: the timed runs reuse an identical state, so vLLM's prefix cache could be flattering them.
Measured directly, sending the same 12-question request with an identical state versus with a unique
`request_id` first in the state (so every block hash differs and the cache misses):

| system | identical state | unique state | cross-request cache benefit |
|---|---|---|---|
| so1 | 1039 ms | 1127 ms | 8% |
| jev-1.13.0 | 396 ms | 372 ms | -6% (noise) |

So cross-request prefix caching is worth ~8%, nowhere near the 2.3x shift in experiment 2. The real cause
is RunPod autoscaling. `Server-Timing` separates the serial and parallel halves and settles it:

| | earlier session | later session |
|---|---|---|
| warm-up, 1 serial call | 455 ms | 460 ms (unchanged) |
| 12 branches, concurrent | 1891 ms | 560 ms (3.4x faster) |

The serial part did not move; only the concurrent part sped up, by roughly the worker count. The endpoint
reported `workers.running = 3` (`workersMax: 3`) during the later runs. With `workersMin: 0`, a quiet
period drops back to a cold single worker, so **expect roughly 2-3x latency variation on this endpoint
depending on how warm it is**. Note this measures *cross-request* caching only; the prefix cache still
does useful work *within* a request, where the warm-up and all branches share one state.

## 9. Distractor sensitivity is instability, not a threshold

One policy question whose correct answer is always yes, with the policy and records fixed and only the
volume of irrelevant conversation varying:

| padding tokens | so1 P(yes) | jev P(yes) |
|---|---|---|
| 0 | 0.958 | 0.960 |
| 250 | 0.679 | 0.940 |
| 500 | 0.706 | 0.950 |
| 750 | **0.133** | 0.930 |
| 1000 | 0.731 | 0.930 |
| 1250 | 0.562 | 0.930 |
| 1500 | **0.014** | 0.930 |
| 2000 | 0.706 | 0.920 |
| 3000 | **0.245** | 0.920 |

so1 does not degrade monotonically; it oscillates across the 0.5 boundary depending on which filler text
is present, ranging from 0.014 to 0.958 on a question whose answer never changes. Jev holds 0.92-0.96
throughout. The practical consequence is that a passing test at one state size does not predict the next
one: for policy-style questions, filter the state down to the policy and the records.

## 10. Why Jev holds up better on large states

It is not immune - TypeSafe documents "Large state full of irrelevant detail" as a known `jev-1.13`
failure mode: *"Accuracy falls as the state grows with content unrelated to the decision. Unrelated
detail acts as a distractor."* We measured Jev dropping to 75% on the 8-question hard pool. Its tolerance
is simply far higher, plausibly because it is purpose-trained for exactly this task rather than a general
instruct model repurposed, it is a larger model, and it has a 64k budget against E4B's 8192. The
mitigation their docs recommend is the one experiment 3 arrived at independently: filter first, and send
only what the question needs.

## 11. Chain-of-thought: fixes the hard cases, destroys the calibration

`scripts/probe_cot.py`, five deliberately hard questions, E4B, 5 sampled chains at temperature 0.8.
Not part of the service - a probe to price the option.

| case | want | single pass | cot-1 | cot-5 mean | cot-5 vote |
|---|---|---|---|---|---|
| refund, no thread | Yes | yes 0.958 | yes 1.000 | yes 1.000 | 5/5 |
| refund, 1500t thread | Yes | **NO 0.018** | yes 0.998 | yes 0.999 | 5/5 |
| refund, 3000t thread | Yes | **NO 0.269** | yes 1.000 | yes 1.000 | 5/5 |
| date math in isolation | Yes | **NO 0.000** | yes 1.000 | yes 1.000 | 5/5 |
| billing already resolved | No | No 0.182 | No 0.000 | No 0.000 | 0/5 |
| **accuracy** | | **2/5** | **5/5** | **5/5** | **5/5** |
| median latency per answer | | 941 ms | 3083 ms | 15417 ms | |
| generated tokens | | 0 | ~500 | 2499 | |

CoT fixed every hard case, including the date arithmetic that the single pass gets exactly wrong
(0.000). It costs 3.3x latency for one chain and 16x for five, and turns 0 generated tokens into 2499.

**It also collapses the probability.** Every CoT answer is 0.000 or ~1.000, and all five chains agreed
on every case. Conditioning on a committed line of reasoning makes the final token near-deterministic:
the uncertainty does not disappear, it moves out of the answer distribution and into the distribution
over chains. So a CoT probability is *not* a usable confidence signal unless you sample many chains and
marginalise - and then your resolution is limited by how many chains you can afford.

Estimator shape, which matters if anyone plans to threshold on it:

- support is **discrete** either way: K labels, unchanged by CoT.
- `P(label | chain)` from a single chain is continuous in value but, measured here, saturated at 0/1.
- majority **vote** over N chains is **quantised to N+1 values** (k/N). With N=5 you get 0, 0.2, 0.4,
  0.6, 0.8, 1.0 and nothing between.
- the **mean** of the per-chain distributions is continuous and is the better estimator of
  `P(y|x) = sum_c P(y|c,x) P(c|x)`. Use the mean, never the vote.

## 12. Latency versus state size is linear, not exponential

One question per request, so there is no fan-out and worker count cannot matter, and a unique id first
in the state so the prefix cache always misses. Three runs per point.

| state tokens | input_tokens | so1 median | ms per 1k tokens | jev median |
|---|---|---|---|---|
| 250 | 478 | 592 ms | 1238 | 383 ms |
| 500 | 735 | 527 ms | 717 | 329 ms |
| 1000 | 1282 | 567 ms | 442 | 359 ms |
| 2000 | 2329 | 701 ms | 301 | 367 ms |
| 3000 | 3488 | 934 ms | 268 | 370 ms |
| 4000 | 4503 | 1130 ms | 251 | 637 ms |
| 5000 | 5619 | 1229 ms | 219 | 626 ms |
| 6500 | 7352 | 2283 ms | 311 | 634 ms |

Cost per 1k tokens *falls* from 1238 to ~220 and then rises slightly, which is a fixed overhead plus a
linear term, not exponential growth. Fitting the middle of the range gives roughly
**370 ms fixed + 0.15 ms per input token**. The last point overshoots that fit (2283 ms against a
predicted ~1500 ms) at 7352 tokens, which is 90% of E4B's 8192 context - consistent with KV-cache
pressure near the limit rather than with any exponential term. Jev is flat to 3000 tokens then steps to
~630 ms.

So the two latency effects are fully separated: **worker count** explains the session-to-session 2.3x
(experiment 8) and **linear prefill** explains growth with state size. Neither is exponential.

## 13. The temperature knob cannot change an answer

20,000 random distributions (K in 2,3,5,10) x random temperature in 0.5-10:

| | changed |
|---|---|
| noul decision crossed 0.5 | **0** |
| choice argmax changed | **0** |
| score expected value moved > 0.25 level | 7297 |

Softmax temperature is a monotone transform of the logits, so it cannot reorder them. Calibration is
therefore **not** an accuracy lever for noul or choice - it only reshapes confidence. It does move a
score, because a score is a weighted mean over the whole distribution rather than an argmax.
