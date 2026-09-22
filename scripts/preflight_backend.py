#!/usr/bin/env python3
"""Can so1's readout method work against a given OpenAI-compatible backend?

Point it at vLLM directly, at a LiteLLM proxy, or at anything else that speaks the OpenAI
API, and it reports which readout modes are available and what will silently break.

    uv run python scripts/preflight_backend.py --base-url http://litellm.prod/v1 --model gemma4-e4b
    uv run python scripts/preflight_backend.py --base-url $E4B_URL/v1 --api-key $RUNPOD_API_KEY

Exit code 0 = usable (at least the fallback readout), 1 = not usable.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import httpx

OK, WARN, BAD = "PASS", "WARN", "FAIL"

MESSAGES = [
    {"role": "system", "content": "You are a precise decision classifier."},
    {"role": "user", "content": "STATE:\nIt is a clear, sunny day.\n\nQUESTION: Is the sky blue?\n\n"
                                "Reply with only Yes or No."},
]
IMPROBABLE = ["Kumquat", "Xylophone", "Zamboni"]


class Backend:
    def __init__(self, base: str, key: str | None, model: str, timeout: float) -> None:
        self.base = base.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.Client(headers=headers, timeout=timeout)

    def chat(self, **body):
        return self.client.post(f"{self.base}/chat/completions", json={"model": self.model, **body})

    def post(self, path: str, body: dict):
        return self.client.post(f"{self.base.removesuffix('/v1')}{path}", json=body)


def report(status: str, title: str, detail: str = "") -> None:
    print(f"  [{status:4}] {title}" + (f"\n         {detail}" if detail else ""))


def top_of(payload: dict):
    entry = payload["choices"][0].get("logprobs")
    if not entry or not entry.get("content"):
        return None
    return entry["content"][0].get("top_logprobs") or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=(os.environ.get("E4B_URL", "") + "/v1"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--model", default="gemma4-e4b")
    parser.add_argument("--want-options", type=int, default=64, help="largest choice question you intend to ask")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    backend = Backend(args.base_url, args.api_key, args.model, args.timeout)
    print(f"preflight: {args.base_url}  model={args.model}\n")
    fallback_ok = exact_ok = False
    max_top = 0

    # 1. the shape trap: logprobs must be a BOOL on chat completions, with the count in top_logprobs
    response = backend.chat(messages=MESSAGES, max_tokens=1, temperature=0, logprobs=5)
    if response.status_code >= 400:
        report(OK, "int `logprobs` is rejected on chat completions (correct)",
               "use logprobs=true + top_logprobs=<n>; an int here is the LEGACY /completions shape")
    else:
        report(WARN, "int `logprobs` was accepted on chat completions",
               "it was probably coerced to true; check top_logprobs is populated below")

    # 2. logprobs without top_logprobs -> the empty-array symptom
    response = backend.chat(messages=MESSAGES, max_tokens=1, temperature=0, logprobs=True)
    top = top_of(response.json()) if response.status_code < 400 else None
    if not top:
        report(OK, "logprobs=true alone returns no distribution (expected)",
               "this is the 'logprob present, top_logprobs empty' symptom - you MUST set top_logprobs")
    else:
        report(WARN, f"logprobs=true alone returned {len(top)} entries", "unusual but harmless")

    # 3. the fallback readout
    response = backend.chat(messages=MESSAGES, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    if response.status_code >= 400:
        report(BAD, "top_logprobs=20 rejected", response.text[:160])
    else:
        top = top_of(response.json())
        if not top:
            report(BAD, "top_logprobs=20 accepted but returned nothing",
                   "the backend is stripping logprobs - the method cannot work here")
        else:
            labels = {t["token"].strip().lower() for t in top}
            hit = labels & {"yes", "no"}
            mass = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() in {"yes", "no"})
            fallback_ok = bool(hit)
            report(OK if hit else BAD, f"FALLBACK readout: {len(top)} entries returned",
                   f"answer labels present: {sorted(hit) or 'NONE'}; label mass {mass:.3f}")
            max_top = len(top)

    # 4. how wide can the top-k go
    for want in (args.want_options, 20):
        response = backend.chat(messages=MESSAGES, max_tokens=1, temperature=0, logprobs=True, top_logprobs=want)
        if response.status_code < 400:
            max_top = max(max_top, len(top_of(response.json()) or []))
            report(OK if want >= args.want_options else WARN, f"top_logprobs={want} accepted",
                   f"ceiling on options per choice/score question: {max_top}")
            break
        if want == 20:
            report(BAD, "even top_logprobs=20 was rejected", response.text[:160])

    # 5. exact readout: allowed_token_ids + processed logprobs
    ids = []
    for text in IMPROBABLE:
        response = backend.post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False})
        if response.status_code < 400:
            ids.append(response.json()["tokens"][0])
    tokenize_ok = len(ids) >= 2
    if not tokenize_ok:
        report(WARN, "POST /tokenize unavailable",
               "cannot verify labels are single tokens, and prefix mode is unavailable. "
               "so1 needs a direct vLLM route for this even when inference goes via a proxy.")
    else:
        report(OK, "POST /tokenize available", "labels can be verified against the real tokenizer")
        response = backend.chat(messages=MESSAGES, max_tokens=1, temperature=1.0, top_p=1.0,
                                logprobs=True, top_logprobs=len(ids),
                                extra_body=None, allowed_token_ids=ids)
        if response.status_code >= 400:
            report(WARN, "allowed_token_ids rejected", "fallback readout only")
        else:
            top = top_of(response.json()) or []
            allowed = {t.strip().lower() for t in IMPROBABLE} | {t.strip().lower()[:1] for t in IMPROBABLE}
            leaked = [t["token"] for t in top if t["token"].strip().lower() not in allowed]
            if leaked:
                report(WARN, "allowed_token_ids accepted but logprobs are RAW",
                       f"{leaked[:3]} leaked past the mask; start vLLM with --logprobs-mode processed_logprobs")
            else:
                exact_ok = True
                report(OK, "EXACT readout available", "server reports processed logprobs")

    # 6. chat template control
    response = backend.chat(messages=MESSAGES, max_tokens=1, temperature=0, logprobs=True, top_logprobs=5,
                            chat_template_kwargs={"enable_thinking": False})
    report(OK if response.status_code < 400 else WARN, "chat_template_kwargs passthrough",
           "" if response.status_code < 400 else "cannot disable thinking; larger Gemma models may need prefix mode")

    print("\nVERDICT")
    if not fallback_ok:
        print("  UNUSABLE - no answer-label distribution is retrievable from this backend.")
        sys.exit(1)
    print(f"  usable via the {'EXACT' if exact_ok else 'FALLBACK'} readout, "
          f"up to {max_top} options per question.")
    if not tokenize_ok:
        print("  so1 additionally needs a direct vLLM route for POST /tokenize at startup.")
    sys.exit(0)


if __name__ == "__main__":
    main()
