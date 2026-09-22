#!/usr/bin/env python3
"""Bisect where per-request latency is actually going.

Point it at a LiteLLM proxy, a raw vLLM, or anything OpenAI-compatible. It separates
fixed overhead from compute from queueing from a broken prefix cache.

    uv run python scripts/diagnose_latency.py --base-url http://litellm.prod/v1 --model gemma-4-26b-a4b
    # and, if you can reach vLLM directly, run it again against that to isolate the proxy:
    uv run python scripts/diagnose_latency.py --base-url http://vllm.internal:8000/v1 --model ...

Every call uses max_tokens=1, so generation is never the variable.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
import uuid

import httpx

SYSTEM = "You are a precise decision classifier. Reply with only the requested label and nothing else."
QUESTION = "\n\nQUESTION: Is this about billing?\n\nReply with only Yes or No."
FILLER = ("The customer mentioned their quarterly planning meeting moved to Thursday. "
          "The onboarding recording was helpful. The template gallery saved setup time. ")


def body(model: str, state: str, top_logprobs: int = 20) -> dict:
    return {
        "model": model, "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": top_logprobs,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"STATE:\n{state}{QUESTION}"}],
    }


def timed(client: httpx.Client, url: str, payload: dict) -> tuple[float, int, int]:
    start = time.perf_counter()
    response = client.post(url, json=payload)
    elapsed = (time.perf_counter() - start) * 1000
    tokens = 0
    if response.status_code == 200:
        tokens = response.json().get("usage", {}).get("prompt_tokens", 0)
    return elapsed, response.status_code, tokens


def stats(values: list[float]) -> str:
    values = sorted(values)
    return (f"p50 {statistics.median(values):7.0f} ms   min {values[0]:7.0f}   "
            f"max {values[-1]:7.0f}   spread {values[-1] - values[0]:6.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="e.g. http://litellm:4000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    chat = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # keep-alive ON, matching how so1 calls upstreams
    with httpx.Client(headers=headers, timeout=600, limits=httpx.Limits(max_keepalive_connections=8)) as client:
        print("warming (result discarded)...")
        timed(client, chat, body(args.model, "warm up"))

        print("\n1. FLOOR - tiny prompt, repeated. This is fixed overhead: network + proxy + scheduler.")
        tiny = [timed(client, chat, body(args.model, "x"))[0] for _ in range(args.runs)]
        print(f"   {stats(tiny)}")
        floor = statistics.median(tiny)

        print("\n2. COMPUTE - does latency track prompt size? If flat, it is NOT the model.")
        slope = []
        for reps in (0, 10, 40, 120):
            state = FILLER * reps if reps else "x"
            ms, _status, tokens = timed(client, chat, body(args.model, state))
            ms = statistics.median([ms] + [timed(client, chat, body(args.model, state))[0]
                                           for _ in range(max(0, args.runs // 3 - 1))])
            slope.append((tokens, ms))
            print(f"   {tokens:6} prompt tokens -> {ms:7.0f} ms   (+{ms - floor:6.0f} over floor)")
        if len(slope) >= 2 and slope[-1][0] > slope[0][0]:
            per_1k = (slope[-1][1] - slope[0][1]) / max(1, (slope[-1][0] - slope[0][0])) * 1000
            print(f"   => ~{per_1k:.0f} ms per 1k prompt tokens; fixed floor ~{floor:.0f} ms")
            share = 100 * floor / slope[-1][1]
            print(f"   => at the largest size, {share:.0f}% of the time is still fixed overhead")

        print("\n3. PREFIX CACHE - identical prompt vs a unique one. No gain => caching off,")
        print("   or a load balancer is spraying requests across replicas with cold caches.")
        state = FILLER * 40
        same = [timed(client, chat, body(args.model, state))[0] for _ in range(args.runs)]
        uniq = [timed(client, chat, body(args.model, f"{uuid.uuid4().hex}\n{state}"))[0] for _ in range(args.runs)]
        gain = (statistics.median(uniq) - statistics.median(same)) / statistics.median(uniq) * 100
        print(f"   identical: {stats(same)}")
        print(f"   unique   : {stats(uniq)}")
        print(f"   => cache benefit {gain:+.0f}%   ({'working' if gain > 10 else 'NOT WORKING - investigate'})")

        print("\n4. VARIANCE - wide spread means queueing or cold/scaling workers, not compute.")
        print(f"   tiny-prompt spread was {max(tiny) - min(tiny):.0f} ms over {args.runs} runs")
        if max(tiny) > 2 * statistics.median(tiny):
            print("   => outliers present: suspect autoscaling, cold workers, or proxy retries")

        print("\n5. TOP_LOGPROBS COST - is the proxy serialising a big payload?")
        for k in (1, 20):
            v = statistics.median([timed(client, chat, body(args.model, "x", k))[0] for _ in range(3)])
            print(f"   top_logprobs={k:<3} -> {v:7.0f} ms")


if __name__ == "__main__":
    main()
