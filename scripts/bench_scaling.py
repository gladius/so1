#!/usr/bin/env python3
"""Sweep request size against accuracy and latency, for us and for the real Jev.

Two axes, identical requests to both systems:
  state size      padded support thread, 500 -> 6500 tokens (eval_cases.build_thread)
  question count  1 -> 16, drawn in order from a fixed pool of thread-answerable questions

    uv run python scripts/bench_scaling.py --csv bench.csv
    uv run python scripts/bench_scaling.py --sizes 500,2000 --counts 1,4 --runs 1   # quick

Writes a tidy CSV (one row per system x size x count x run) ready to plot.

The pool deliberately excludes policy-eligibility questions: those have a known small-model
failure (see README, "Scope the state to the question") that would swamp a size-scaling signal.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import statistics
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
import eval_cases  # noqa: E402

JEV_URL = "https://api.typesafe.ai"

# Decisive facts, placed at fixed points in an otherwise neutral thread.
KEY_MESSAGES = [
    (0.20, "Customer (Alex Chen)", "We were double charged on invoice INV-88119 - $99 went out twice on 1 September."),
    (
        0.30,
        "Agent (Priya, Support)",
        "Confirmed, I have refunded the duplicate $99 charge. It should land in 3-5 business days.",
    ),
    (0.45, "Customer (Alex Chen)", "The $99 refund arrived, thanks for sorting that out."),
    (
        0.80,
        "Customer (Alex Chen)",
        "Since yesterday every CSV report export fails with 'Error 500: export worker "
        "timeout'. Month-end close is this Friday and the whole finance team is blocked.",
    ),
    (0.90, "Agent (Priya, Support)", "Escalating this to engineering now."),
    (
        1.00,
        "Customer (Alex Chen)",
        "To be clear I am not asking for any money back. But if this is not fixed well "
        "before our renewal comes round again, we will not be renewing.",
    ),
]

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


def noul(text, expected):
    return {"type": "noul", "instructions": text}, ("noul", expected)


def choice(text, criteria, expected):
    return {"type": "choice", "instructions": text, "criteria": criteria}, ("choice", expected)


def score(text, criteria, low, high):
    return {"type": "score", "instructions": text, "criteria": criteria}, ("range", (low, high))


# Ordered: taking the first K gives a sensible mix at every K.
POOL: list[tuple[str, tuple]] = [
    ("open_technical", noul("Is there an unresolved technical problem in this thread?", "yes")),
    ("routing", choice("Which team should handle the issue that is currently open?", ROUTING, "technical")),
    ("urgency", score("How urgent is the issue that is currently open?", URGENCY, 2.4, 4.0)),
    ("open_billing", noul("Is there still an unresolved billing problem in this thread?", "no")),
    ("cancel_threat", noul("Does the customer indicate they may not renew or may cancel?", "yes")),
    ("area", choice("Which product area is the open defect in?", AREA, "reporting_exports")),
    ("wants_refund_now", noul("Is the customer asking for a refund in their most recent message?", "no")),
    ("frustration", score("How frustrated is the customer?", FRUSTRATION, 0.6, 2.0)),
    ("next_action", choice("What is the single best next action on this ticket?", ACTION, "escalate_to_engineering")),
    ("mentions_security", noul("Does this thread mention a security vulnerability or breach?", "no")),
    ("is_defect", noul("Is the customer reporting a software defect?", "yes")),
    ("has_deadline", noul("Does the customer mention a deadline?", "yes")),
    ("refund_was_paid", noul("Has the duplicate charge already been refunded?", "yes")),
    ("needs_engineering", noul("Does resolving the open issue require engineering involvement?", "yes")),
    ("is_angry_rant", noul("Is this message abusive or insulting towards the agent?", "no")),
    ("export_broken", noul("Is the CSV export feature currently broken?", "yes")),
]


# A second pool, chosen because these discriminate. Same thread, plus the records and policy the
# eligibility questions need. Both systems score 100% on the easy pool at every size, so accuracy
# only becomes a signal here.
POLICY_STATE = {
    "today": "2026-09-19",
    "account": {
        "company": "Northwind Studio",
        "plan": "Annual Business",
        "renewal_charge": {"invoice": "INV-88120", "amount_usd": 1188.00, "date": "2026-08-30"},
        "report_exports_this_period": 3,
    },
    "derived": {"days_since_renewal_charge": 20},
    "refund_policy": [
        "1. Monthly plans: refundable in full if requested within 14 days of the charge.",
        "2. Annual plans: refundable in full if requested within 30 days of the charge, provided the "
        "account has made 5 or fewer report exports in the current billing period.",
        "3. Duplicate charges are always refundable, on any plan, with no time limit.",
        "4. Accounts on the nonprofit discount are not eligible for goodwill refunds, but rule 3 still applies.",
    ],
}

FIRST_ISSUE = {"billing": "A duplicate or incorrect charge", "technical": "A software defect or outage",
               "sales": "Pricing or plan changes", "general": "A how-to or onboarding question"}

HARD_POOL: list[tuple[str, tuple]] = [
    ("refund_eligible", noul(
        "Using `refund_policy`, `derived.days_since_renewal_charge` and `account.report_exports_this_period`, "
        "would the annual renewal charge be refundable if it were requested today?", "yes")),
    ("latest_is_billing", noul(
        "Is the customer's most recent complaint about something they were charged for?", "no")),
    ("agent_resolved_latest", noul(
        "Has the agent already resolved the issue the customer raised most recently?", "no")),
    ("billing_before_export", noul(
        "Was the billing problem raised before the export problem?", "yes")),
    ("first_issue", choice("What kind of issue did the customer raise first in this thread?",
                           FIRST_ISSUE, "billing")),
    ("issue_count", score("How many distinct problems has the customer raised in this thread?",
                          ["Zero", "One", "Two", "Three or more"], 1.5, 2.5)),
    ("refund_eligible_raw", noul(
        "Using `refund_policy`, `account.renewal_charge.date` and `today`, would the annual renewal charge "
        "be refundable if it were requested today?", "yes")),
    ("all_resolved", noul("Have all the problems raised in this thread been resolved?", "no")),
]


def grade(kind_expected, answer) -> bool:
    kind, want = kind_expected
    if kind == "noul":
        return ("yes" if answer["noul"] >= 0.5 else "no") == want
    if kind == "choice":
        return answer["choice"] == want
    value = sum(i * p for i, p in enumerate(answer["probabilities"].values()))
    return want[0] <= value <= want[1]


def pool_for(name: str) -> list[tuple[str, tuple]]:
    return HARD_POOL if name == "hard" else POOL


def build(state_tokens: int, count: int, pool: str) -> tuple[object, dict]:
    thread = eval_cases.build_thread(state_tokens, KEY_MESSAGES, seed=state_tokens)
    state: object = thread
    if pool == "hard":
        # The eligibility questions need the records; the thread is what makes them hard.
        state = {**POLICY_STATE, "thread": thread}
    chosen = pool_for(pool)[:count]
    return state, {qid: spec for qid, (spec, _) in chosen}


def expectations(count: int, pool: str) -> dict:
    return {qid: exp for qid, (_, exp) in pool_for(pool)[:count]}


def call(client, url, headers, model, state, questions, timeout=600):
    started = time.perf_counter()
    response = client.post(
        f"{url}/v1/systemone",
        headers=headers,
        timeout=timeout,
        json={"model": model, "state": state, "questions": questions},
    )
    elapsed = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("SO1_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--sizes", default="500,1500,3000,6500")
    parser.add_argument("--counts", default="1,2,4,8,16")
    parser.add_argument("--pool", choices=("easy", "hard"), default="easy",
                        help="easy = saturated, measures latency; hard = discriminates accuracy")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--csv", type=pathlib.Path, default=ROOT / "results" / "bench_scaling.csv")
    parser.add_argument("--skip-jev", action="store_true")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    counts = [int(c) for c in args.counts.split(",") if int(c) <= len(pool_for(args.pool))]
    systems = [("so1", args.url, {}, "jev-latest")]
    if not args.skip_jev:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            sys.exit("set TYPESAFE_API_KEY or pass --skip-jev")
        systems.append(("jev-1.13.0", JEV_URL, {"Authorization": f"Bearer {key}"}, "jev-latest"))

    rows = []
    with httpx.Client() as client:
        for _name, url, headers, model in systems:
            state, questions = build(sizes[0], 1, args.pool)
            call(client, url, headers, model, state, questions)  # warm
        for state_tokens in sizes:
            for count in counts:
                state, questions = build(state_tokens, count, args.pool)
                expected = expectations(count, args.pool)
                for name, url, headers, model in systems:
                    for run in range(args.runs):
                        payload, elapsed = call(client, url, headers, model, state, questions)
                        correct = sum(grade(expected[q], payload["answers"][q]) for q in questions)
                        rows.append(
                            {
                                "system": name,
                                "state_tokens": state_tokens,
                                "questions": count,
                                "run": run,
                                "latency_ms": round(elapsed, 1),
                                "correct": correct,
                                "total": count,
                                "accuracy": round(correct / count, 4),
                                "input_tokens": payload["usage"]["input_tokens"],
                                "ms_per_question": round(elapsed / count, 1),
                            }
                        )
                        print(
                            f"  {name:12} state={state_tokens:<5} q={count:<3} run={run} "
                            f"{elapsed:7.0f} ms  {correct}/{count}",
                            flush=True,
                        )

    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'state':>7}{'q':>4}" + "".join(f"{n[0]:>26}" for n in systems))
    print("-" * (11 + 26 * len(systems)))
    for state_tokens in sizes:
        for count in counts:
            cells = ""
            for name, *_ in systems:
                sel = [
                    r
                    for r in rows
                    if r["system"] == name and r["state_tokens"] == state_tokens and r["questions"] == count
                ]
                ms = statistics.median(r["latency_ms"] for r in sel)
                acc = statistics.mean(r["accuracy"] for r in sel)
                cells += f"{f'{ms:6.0f} ms  {acc * 100:3.0f}%':>26}"
            print(f"{state_tokens:>7}{count:>4}{cells}")
    print(f"\n{len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
