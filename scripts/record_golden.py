#!/usr/bin/env python3
"""Record golden responses from the real Jev API into tests/fixtures/golden/.

Prefers a direct TypeSafe key (faithful error bodies and model ids); falls back to
OpenRouter, which wraps the API and rewrites errors and the response envelope.

    uv run python scripts/record_golden.py [--only NAME ...]

Synthetic states only. Nothing here is used at inference time.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "golden"

DIRECT = ("https://api.typesafe.ai", "TYPESAFE_API_KEY")
VIA_OPENROUTER = ("https://openrouter.ai/api", "OPENROUTER_API_KEY")

STATE = "Help! My payouts have been failing for 3 days and nobody has replied."

CASES: dict[str, dict] = {
    "noul_basic": {
        "state": STATE,
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    },
    "noul_criteria": {
        "state": "Buy cheap watches now!!! Limited offer, click here.",
        "questions": {
            "is_spam": {
                "type": "noul",
                "instructions": "Is this message spam?",
                "criteria": {"true": "Unsolicited advertising", "false": "A legitimate conversation"},
            }
        },
    },
    "choice_basic": {
        "state": STATE,
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            }
        },
    },
    "choice_null_descriptions": {
        "state": STATE,
        "questions": {
            "tone": {
                "type": "choice",
                "instructions": "What is the customer's tone?",
                "criteria": {"calm": None, "frustrated": None, "angry": None},
            }
        },
    },
    "score_basic": {
        "state": STATE,
        "questions": {
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            }
        },
    },
    "score_ten_levels": {
        "state": "This is mildly annoying but I can live with it.",
        "questions": {
            "anger": {
                "type": "score",
                "instructions": "Rate the anger from lowest to highest.",
                "criteria": [f"level {i}" for i in range(10)],
            }
        },
    },
    "mixed_all_types": {
        "state": STATE,
        "questions": {
            "billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
            "tone": {
                "type": "choice",
                "instructions": "What is the customer's tone?",
                "criteria": {"calm": None, "frustrated": None, "angry": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["can wait", "this week", "today"],
            },
        },
    },
    "state_object": {
        "state": {
            "ticket": {"subject": "Duplicate charge", "body": "I was charged twice for order A-104."},
            "order": {"id": "A-104", "charges": [49, 49]},
        },
        "questions": {"duplicate": {"type": "noul", "instructions": "Do the records show a duplicate charge?"}},
    },
    "state_chat_messages": {
        "state": [{"role": "user", "content": "where is my refund"}, {"role": "assistant", "content": "checking now"}],
        "questions": {"refund": {"type": "noul", "instructions": "Is the user asking about a refund?"}},
    },
    "structured_instructions": {
        "state": {"resume": {"name": "John Smith", "location": "Oakland, California", "employer": "Google"}},
        "questions": {
            "same_person": {
                "type": "noul",
                "instructions": {
                    "potential_duplicate": {
                        "name": "John Smith",
                        "location": "Oakland, California",
                        "last_employer": "Google",
                    },
                    "question": "Is the resume for the same person as `potential_duplicate`?",
                },
            }
        },
    },
    # Error shapes. Only faithful through the direct API.
    "err_missing_state": {"questions": {"q": {"type": "noul", "instructions": "Is this urgent?"}}},
    "err_unknown_type": {"state": "x", "questions": {"q": {"type": "bogus", "instructions": "y"}}},
    "err_empty_questions": {"state": "x", "questions": {}},
}


def pick_endpoint() -> tuple[str, str, str]:
    for base, env in (DIRECT, VIA_OPENROUTER):
        key = os.environ.get(env)
        if key:
            return base, key, env
    sys.exit("set TYPESAFE_API_KEY (preferred) or OPENROUTER_API_KEY")


def call(base: str, key: str | None, path: str, body: dict | None) -> tuple[int, object]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args()

    base, key, env = pick_endpoint()
    direct = base == DIRECT[0]
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"recording from {base} (key from {env}){'' if direct else '  [OpenRouter: errors and envelope are rewritten]'}"
    )

    for name, body in CASES.items():
        if args.only and name not in args.only:
            continue
        payload = {"model": args.model, **body}
        status, response = call(base, key, "/v1/systemone", payload)
        (OUT / f"{name}.json").write_text(
            json.dumps(
                {"source": base, "direct": direct, "request": payload, "status": status, "response": response}, indent=2
            )
            + "\n"
        )
        print(f"  {name:26} {status}")

    status, response = call(base, key, "/v1/models", None)
    (OUT / "models_list.json").write_text(
        json.dumps(
            {"source": base, "direct": direct, "request": None, "status": status, "response": response}, indent=2
        )
        + "\n"
    )
    print(f"  {'models_list':26} {status}")

    status, response = call(
        base,
        None,
        "/v1/systemone",
        {"model": args.model, "state": "x", "questions": {"q": {"type": "noul", "instructions": "y?"}}},
    )
    (OUT / "err_no_auth.json").write_text(
        json.dumps(
            {"source": base, "direct": direct, "request": None, "status": status, "response": response}, indent=2
        )
        + "\n"
    )
    print(f"  {'err_no_auth':26} {status}")


if __name__ == "__main__":
    main()
