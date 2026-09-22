#!/usr/bin/env python3
"""Research probe: what happens to the probability if we add chain-of-thought?

NOT part of the service. so1 answers in one forward pass and never generates. This script
talks to the vLLM endpoint directly to measure what CoT would buy and what it would cost.

Three readouts of the same question:
  single    one forward pass, read P(label) at the answer position        (what so1 does)
  cot-1     sample ONE reasoning chain, then read P(label | that chain)
  cot-N     sample N chains, read P(label | chain_i) for each, then
              mean    = (1/N) sum_i P(label | chain_i)   -> continuous marginal estimate
              vote    = fraction of chains whose argmax is the label -> quantised to k/N

    uv run python scripts/probe_cot.py --chains 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_cases  # noqa: E402

from bench_scaling import KEY_MESSAGES, POLICY_STATE  # noqa: E402

SYSTEM = ("You are a precise decision classifier. Read the state carefully, then answer the "
          "question about it. Reply with only the requested label and nothing else.")
COT_SYSTEM = ("You are a precise decision analyst. Read the state carefully and reason step by step "
              "about the question. Be brief: at most four short sentences.")

REFUND_Q = ("Using `refund_policy`, `derived.days_since_renewal_charge` and "
            "`account.report_exports_this_period`, would the annual renewal charge be refundable "
            "if it were requested today?")


def state_with(padding: int):
    state = dict(POLICY_STATE)
    if padding:
        state["thread"] = eval_cases.build_thread(padding, KEY_MESSAGES, seed=padding)
    return state


CASES = [
    ("refund, no thread", state_with(0), REFUND_Q, "Yes"),
    ("refund, 1500t thread", state_with(1500), REFUND_Q, "Yes"),
    ("refund, 3000t thread", state_with(3000), REFUND_Q, "Yes"),
    ("date math in isolation",
     {"charge_date": "2026-08-30", "today": "2026-09-19"},
     "Counting from `charge_date` to `today`, have 30 or fewer days passed?", "Yes"),
    ("billing already resolved", state_with(1500),
     "Is there still an unresolved billing problem in the thread?", "No"),
]


def user_block(state, question: str) -> str:
    body = state if isinstance(state, str) else json.dumps(state, indent=2)
    return f"STATE:\n{body}\n\nQUESTION: {question}\n\nReply with only Yes or No."


class VLLM:
    def __init__(self, url: str, key: str, model: str) -> None:
        self.url, self.model = url.rstrip("/"), model
        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=600)
        self.calls, self.gen_tokens = 0, 0

    def _post(self, body: dict) -> dict:
        self.calls += 1
        response = self.client.post(f"{self.url}/v1/chat/completions", json=body)
        response.raise_for_status()
        return response.json()

    def p_yes(self, messages: list[dict]) -> float:
        payload = self._post({
            "model": self.model, "messages": messages, "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False},
        })
        top = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        mass = {"yes": 0.0, "no": 0.0}
        for entry in top:
            token = entry["token"].strip().lower().rstrip(".:)")
            if token in mass:
                mass[token] += math.exp(entry["logprob"])
        total = mass["yes"] + mass["no"]
        return mass["yes"] / total if total > 0 else 0.5

    def reason(self, state, question: str, temperature: float) -> str:
        payload = self._post({
            "model": self.model, "max_tokens": 220, "temperature": temperature, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": user_block(state, question).replace(
                    "Reply with only Yes or No.", "Reason step by step. Do not state a final answer yet.")},
            ],
        })
        self.gen_tokens += payload.get("usage", {}).get("completion_tokens", 0)
        return (payload["choices"][0]["message"].get("content") or "").strip()

    def p_yes_after(self, state, question: str, reasoning: str) -> float:
        return self.p_yes([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_block(state, question)},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": "Given that reasoning, reply with only Yes or No."},
        ])

    def single(self, state, question: str) -> float:
        return self.p_yes([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_block(state, question)},
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--url", default=os.environ.get("E4B_URL", "https://fcy67kixeo37lf.api.runpod.ai"))
    parser.add_argument("--model", default="gemma4-e4b")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "cot_probe.json")
    args = parser.parse_args()

    vllm = VLLM(args.url, os.environ["RUNPOD_API_KEY"], args.model)
    n = args.chains
    rows = []
    print(f"{n} chains at temperature {args.temperature}, model {args.model}\n")
    header = f"{'case':26}{'want':6}{'single':>18}{'cot-1':>18}{f'cot-{n} mean':>18}{f'cot-{n} vote':>14}"
    print(header)
    print("-" * len(header))

    for label, state, question, want in CASES:
        t0 = time.perf_counter()
        single = vllm.single(state, question)
        single_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        chain_ps = []
        for _ in range(n):
            reasoning = vllm.reason(state, question, args.temperature)
            chain_ps.append(vllm.p_yes_after(state, question, reasoning))
        cot_ms = (time.perf_counter() - t0) * 1000 / n

        mean_p = statistics.mean(chain_ps)
        vote = sum(p >= 0.5 for p in chain_ps) / n
        target = 1.0 if want == "Yes" else 0.0

        def mark(p):
            return ("yes" if p >= 0.5 else "NO ") + f" {p:.3f}"

        print(f"{label:26}{want:6}{mark(single):>18}{mark(chain_ps[0]):>18}"
              f"{mark(mean_p):>18}{f'{vote:.2f}':>14}")
        rows.append({"case": label, "want": want, "single": single, "chains": chain_ps,
                     "cot_mean": mean_p, "cot_vote": vote, "target": target,
                     "single_ms": single_ms, "cot_ms_per_chain": cot_ms})

    def acc(key, getter):
        hits = sum((getter(r) >= 0.5) == (r["target"] == 1.0) for r in rows)
        return f"{hits}/{len(rows)}"

    print("-" * len(header))
    print(f"{'ACCURACY':32}{acc('single', lambda r: r['single']):>18}"
          f"{acc('cot1', lambda r: r['chains'][0]):>18}{acc('mean', lambda r: r['cot_mean']):>18}"
          f"{acc('vote', lambda r: r['cot_vote']):>14}")
    print(f"{'median ms per answer':32}{statistics.median(r['single_ms'] for r in rows):>15.0f} ms"
          f"{statistics.median(r['cot_ms_per_chain'] for r in rows):>15.0f} ms"
          f"{statistics.median(r['cot_ms_per_chain'] * n for r in rows):>15.0f} ms")
    print(f"\nupstream calls: {vllm.calls}   generated tokens: {vllm.gen_tokens} "
          f"(single-pass generates 0)")
    print(f"distinct values a {n}-chain vote can take: {n + 1} "
          f"({', '.join(f'{i}/{n}' for i in range(n + 1))})  -- the mean is continuous")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"chains": n, "temperature": args.temperature, "rows": rows}, indent=1) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
