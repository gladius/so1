#!/usr/bin/env python3
"""One complex fan-out request - 12 questions over a structured state - graded and timed.

    uv run python scripts/bench_complex.py --runs 3
    uv run python scripts/bench_complex.py --skip-jev

Sends the identical request to our service and to the real Jev, grades every answer against a
known-correct value, and reports accuracy, per-request latency and token usage side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
JEV_URL = "https://api.typesafe.ai"

STATE = {
    "today": "2026-09-19",
    "account": {
        "company": "Northwind Studio",
        "plan": "Annual Business",
        "seats": 40,
        "nonprofit_discount": False,
        "renewal_charge": {"invoice": "INV-88120", "amount_usd": 1188.00, "date": "2026-08-30"},
        "report_exports_this_period": 3,
    },
    "derived": {"days_since_renewal_charge": 20},
    "refund_policy": [
        "1. Monthly plans: refundable in full if requested within 14 days of the charge.",
        "2. Annual plans: refundable in full if requested within 30 days of the charge, provided the account "
        "has made 5 or fewer report exports in the current billing period.",
        "3. Duplicate charges are always refundable, on any plan, with no time limit.",
        "4. Accounts on the nonprofit discount are not eligible for goodwill refunds, but rule 3 still applies.",
    ],
    "thread": [
        {
            "date": "2026-09-02",
            "from": "customer",
            "text": "We were double charged on invoice INV-88119 - $99 went out twice on 1 September.",
        },
        {
            "date": "2026-09-03",
            "from": "agent",
            "text": "Confirmed, I have refunded the duplicate $99 charge. It should land in 3-5 business days.",
        },
        {"date": "2026-09-08", "from": "customer", "text": "The $99 refund arrived, thanks for sorting that."},
        {
            "date": "2026-09-18",
            "from": "customer",
            "text": "Since yesterday every CSV report export fails with 'Error 500: export worker timeout'. "
            "Month-end close is Friday and the whole finance team is blocked.",
        },
        {"date": "2026-09-19", "from": "agent", "text": "Escalating this to engineering now."},
        {
            "date": "2026-09-19",
            "from": "customer",
            "text": "To be clear I am not asking for any money back. But if this is not fixed well before our "
            "renewal comes round again, we will not be renewing.",
        },
    ],
}

ROUTING = {
    "billing": "Payments, charges, invoices, refunds or subscriptions",
    "technical": "Bugs, errors, outages or integration problems",
    "sales": "Pricing, quotes, plan upgrades or new seats",
    "general": "Onboarding, how-to questions or feedback with no problem to fix",
}
AREA = {
    "reporting_exports": "Report generation and CSV/data exports",
    "billing_system": "Invoicing and payment processing",
    "authentication": "Login, SSO and session handling",
    "mobile_app": "The iOS and Android applications",
    "public_api": "REST endpoints, webhooks and SDKs",
    "notifications": "Email and in-app alerts",
}
ACTION = {
    "wait_for_customer": "Nothing to do until the customer replies",
    "escalate_to_engineering": "Hand the open defect to the engineering team",
    "issue_refund": "Refund a charge back to the customer",
    "close_ticket": "Everything raised has been resolved",
    "transfer_to_sales": "Route to sales for a pricing or renewal conversation",
}
URGENCY = [
    "Very low: no action needed",
    "Low: can wait a week or more",
    "Medium: should be handled within a few days",
    "High: blocks important work, handle today",
    "Critical: outage, data loss or security risk, handle immediately",
]
FRUSTRATION = ["Calm and satisfied", "Mildly annoyed", "Angry"]

QUESTIONS = {
    "open_technical": {"type": "noul", "instructions": "Is there an unresolved technical problem in this thread?"},
    "open_billing": {"type": "noul", "instructions": "Is there still an unresolved billing problem in this thread?"},
    "cancel_threat": {"type": "noul", "instructions": "Does the customer indicate they may not renew or may cancel?"},
    "wants_refund_now": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund in their most recent message?",
    },
    "mentions_security": {"type": "noul", "instructions": "Does this case mention a security vulnerability or breach?"},
    "refund_raw": {
        "type": "noul",
        "instructions": "Using `refund_policy`, `account.renewal_charge.date` and `today`, would the annual "
        "renewal charge be refundable if the customer asked for it today?",
    },
    "refund_derived": {
        "type": "noul",
        "instructions": "Using `refund_policy`, `derived.days_since_renewal_charge` and "
        "`account.report_exports_this_period`, would the annual renewal charge be refundable today?",
    },
    "routing": {
        "type": "choice",
        "instructions": "Which team should handle the issue that is currently open?",
        "criteria": ROUTING,
    },
    "area": {"type": "choice", "instructions": "Which product area is the open defect in?", "criteria": AREA},
    "next_action": {
        "type": "choice",
        "instructions": "What is the single best next action on this ticket?",
        "criteria": ACTION,
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the issue that is currently open?",
        "criteria": URGENCY,
    },
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": FRUSTRATION},
}

# (kind, expected) - 'range' is inclusive, on the zero-based level index
EXPECT = {
    "open_technical": ("noul", "yes"),
    "open_billing": ("noul", "no"),
    "cancel_threat": ("noul", "yes"),
    "wants_refund_now": ("noul", "no"),
    "mentions_security": ("noul", "no"),
    "refund_raw": ("noul", "yes"),
    "refund_derived": ("noul", "yes"),
    "routing": ("choice", "technical"),
    "area": ("choice", "reporting_exports"),
    "next_action": ("choice", "escalate_to_engineering"),
    "urgency": ("range", (2.5, 4.0)),
    "frustration": ("range", (0.7, 2.0)),
}
NOTES = {
    "open_billing": "the duplicate charge was refunded and confirmed - must notice it is closed",
    "wants_refund_now": "explicit negation: 'I am not asking for any money back'",
    "mentions_security": "absence of evidence",
    "refund_raw": "HARD: needs 30 Aug -> 19 Sep = 20 days, i.e. within 30",
    "refund_derived": "same rule, but the day count is precomputed in the state",
    "cancel_threat": "'we will not be renewing' phrased conditionally",
}


def grade(qid: str, answer: dict) -> tuple[bool, str]:
    kind, want = EXPECT[qid]
    if kind == "noul":
        got = "yes" if answer["noul"] >= 0.5 else "no"
        return got == want, f"{got} ({answer['noul']:.3f})"
    if kind == "choice":
        return answer["choice"] == want, answer["choice"]
    value = sum(i * p for i, p in enumerate(answer["probabilities"].values()))
    low, high = want
    return low <= value <= high, f"{value:.2f}"


def run(client: httpx.Client, url: str, headers: dict, model: str, runs: int) -> dict:
    body = {"model": model, "state": STATE, "questions": QUESTIONS}
    client.post(f"{url}/v1/systemone", json=body, headers=headers, timeout=600)  # warm
    times, payload, timing = [], None, None
    for _ in range(runs):
        started = time.perf_counter()
        response = client.post(f"{url}/v1/systemone", json=body, headers=headers, timeout=600)
        times.append((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        payload, timing = response.json(), response.headers.get("Server-Timing")
    return {"payload": payload, "times": times, "server_timing": timing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("SO1_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-jev", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "complex_case.json")
    args = parser.parse_args()

    systems = [("so1", args.url, {}, args.model)]
    if not args.skip_jev:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("set TYPESAFE_API_KEY or pass --skip-jev")
        systems.append(("jev-1.13.0", JEV_URL, {"Authorization": f"Bearer {key}"}, "jev-latest"))

    print(f"state: {len(json.dumps(STATE))} chars   questions: {len(QUESTIONS)}   runs: {args.runs}\n")
    out = {}
    with httpx.Client() as client:
        for name, url, headers, model in systems:
            out[name] = run(client, url, headers, model, args.runs)

    names = list(out)
    width = 26
    print(f"{'question':20}{'expected':26}" + "".join(f"{n:>{width}}" for n in names))
    print("-" * (46 + width * len(names)))
    score = dict.fromkeys(names, 0)
    for qid in QUESTIONS:
        _, want = EXPECT[qid]
        cells = ""
        for name in names:
            ok, shown = grade(qid, out[name]["payload"]["answers"][qid])
            score[name] += ok
            cells += f"{('OK  ' if ok else 'MISS') + ' ' + shown:>{width}}"
        print(f"{qid:20}{want!s:26}{cells}")
        if qid in NOTES:
            print(f"{'':20}-> {NOTES[qid]}")
    print("-" * (46 + width * len(names)))

    total = len(QUESTIONS)
    print(f"{'ACCURACY':46}" + "".join(f"{f'{score[n]}/{total}':>{width}}" for n in names))
    for label, fn in (("latency median", statistics.median), ("latency min", min), ("latency max", max)):
        print(f"{label:46}" + "".join(f"{fn(out[n]['times']):>{width - 3}.0f} ms" for n in names))
    print(
        f"{'ms per question (median)':46}"
        + "".join(f"{statistics.median(out[n]['times']) / total:>{width - 3}.0f} ms" for n in names)
    )
    print(f"{'input_tokens':46}" + "".join(f"{out[n]['payload']['usage']['input_tokens']:>{width}}" for n in names))
    print(f"\nso1 Server-Timing: {out['so1']['server_timing']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                n: {
                    "answers": out[n]["payload"]["answers"],
                    "usage": out[n]["payload"]["usage"],
                    "ms": out[n]["times"],
                    "server_timing": out[n]["server_timing"],
                    "correct": score[n],
                    "total": total,
                }
                for n in names
            },
            indent=1,
        )
        + "\n"
    )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
