"""Evaluation cases, vendored verbatim from longest.py - the run that produced the 14/15 baseline.

Kept in-tree so the eval is reproducible without reaching outside the project. Only the
case-generation half was copied; scoring runs against our own service in scripts/eval_service.py.
"""

import random
from datetime import datetime, timedelta

NEUTRAL_CUSTOMER = [
    "Thanks for the walkthrough yesterday, the team found the dashboard tour helpful.",
    "Could you resend the link to the onboarding recording? I think it went to my spam folder.",
    "We are planning to invite three more colleagues from the design team next week.",
    "Quick note: I will be out of office on Friday, so please reply to my colleague Dana if anything comes up.",
    "The documentation page on keyboard shortcuts was useful, especially the section on bulk editing.",
    "Our quarterly planning meeting moved to Thursday, so I may reply a bit slower this week.",
    "I have updated our company logo in the workspace settings and it looks great.",
    "Is there a recommended way to organise projects by client? We currently use folders per region.",
    "Thanks, that answers my question about naming conventions.",
    "We held an internal training session and most people are now comfortable with the basics.",
    "Can you confirm the webinar next Tuesday is still at 3pm CET?",
    "Our office will be closed for the regional holiday on Monday.",
    "I liked the new colour theme in the latest release.",
    "Dana asked whether there is a mobile app; she mostly works from her tablet.",
    "We are writing an internal guide for new hires and will link to your help centre.",
    "The template gallery saved us a lot of setup time for the marketing projects.",
]


NEUTRAL_AGENT = [
    "Happy to help! I have resent the recording link to your inbox.",
    "Great to hear the tour was useful. Let me know if the design team needs a separate session.",
    "Yes, the webinar is still scheduled for Tuesday at 3pm CET.",
    "Organising projects by client works well; many teams use tags for regions.",
    "Thanks for the heads up about Friday, I have noted Dana as the backup contact.",
    "The mobile app is available on both iOS and Android and syncs with the web version.",
    "Glad the templates helped. We add new ones every month.",
    "Feel free to reuse any screenshots from the help centre in your internal guide.",
    "Enjoy the holiday! We will pick things up when you are back.",
]


SALES_CUSTOMER = [
    "We are comparing the Team and Business tiers for next year.",
    "Our finance lead wants to know whether annual prepayment includes a discount.",
    "We might expand to 40 seats in Q1, so volume pricing would be interesting.",
    "Is the enterprise SSO add-on priced per seat or as a flat fee?",
    "We saw the promotion for nonprofits and wondered whether a sister organisation qualifies.",
    "Procurement will need a formal quote with our VAT number on it.",
    "Does the Business tier include priority support, or is that a separate add-on?",
]


SALES_AGENT = [
    "Annual prepayment comes with a 15% discount on both tiers.",
    "SSO is included in the Business tier at no extra cost.",
    "Priority support is part of the Business tier.",
    "Volume pricing starts at 25 seats; I can include it in the quote.",
    "Nonprofit pricing is available for registered charities; the sister organisation would need to apply separately.",
]


def build_thread(target_tokens, key_messages, seed, pool="neutral"):
    """A support thread of roughly target_tokens tokens, with key messages inserted.

    key_messages: list of (position 0..1, speaker, text). Position 1.0 = the very end.
    Token size is estimated at ~4 characters per token.
    """
    rng = random.Random(seed)
    cust = NEUTRAL_CUSTOMER + (SALES_CUSTOMER * 2 if pool == "sales" else [])
    agent = NEUTRAL_AGENT + (SALES_AGENT * 2 if pool == "sales" else [])
    key_chars = sum(len(t) + 60 for _, _, t in key_messages)
    target_chars = max(0, target_tokens * 4 - key_chars)

    msgs, total, i = [], 0, 0
    while total < target_chars:
        if i % 2 == 0:
            speaker, text = "Customer (Alex Chen)", " ".join(rng.sample(cust, rng.randint(1, 3)))
        else:
            speaker, text = "Agent (Priya, Support)", " ".join(rng.sample(agent, rng.randint(1, 2)))
        msgs.append((speaker, text))
        total += len(text) + 60
        i += 1

    for pos, speaker, text in sorted(key_messages, key=lambda k: k[0]):
        idx = len(msgs) if pos >= 1.0 else int(pos * len(msgs))
        msgs.insert(idx, (speaker, text))

    ts = datetime(2026, 9, 1, 9, 0)
    lines = []
    for speaker, text in msgs:
        ts += timedelta(minutes=rng.randint(7, 180))
        lines.append(f"[{ts:%Y-%m-%d %H:%M}] {speaker}: {text}")
    return "SUPPORT THREAD (oldest first)\n" + "\n".join(lines)


POLICY_FILLER = [
    (
        "Service availability",
        "We target 99.9% monthly uptime for the web application, excluding "
        "scheduled maintenance announced at least 48 hours in advance. Status updates are published on "
        "the status page. Credits for missed uptime targets are applied to the next invoice and are not "
        "paid out as cash.",
    ),
    (
        "Support hours",
        "Standard support is available Monday to Friday, 08:00 to 18:00 CET. Business tier "
        "customers receive priority routing. Response-time targets apply to business hours only and do not "
        "apply during public holidays in Germany.",
    ),
    (
        "Data export",
        "Customers may export reports in CSV, XLSX and PDF formats. Exports are generated "
        "asynchronously and links remain valid for seven days. Very large exports may be split into several "
        "files. Export activity is logged per billing period.",
    ),
    (
        "Security",
        "All data is encrypted in transit and at rest. Access to customer data by staff requires "
        "a support ticket and is logged. Security incidents are reported to affected customers within 72 hours.",
    ),
    (
        "Seat management",
        "Seats can be added at any time and are prorated for the remainder of the billing "
        "period. Removing seats takes effect at the next renewal; removed seats are not refunded.",
    ),
    (
        "Taxes",
        "Prices exclude VAT and other applicable taxes, which are added to invoices based on the "
        "billing address. Customers are responsible for providing a valid VAT number where applicable.",
    ),
    (
        "Account closure",
        "Customers can close their account at any time from the settings page. Data is "
        "retained for 30 days after closure and then permanently deleted, unless legal obligations require "
        "longer retention.",
    ),
    (
        "Plan changes",
        "Upgrades take effect immediately and are prorated. Downgrades take effect at the next "
        "renewal date. Changing between monthly and annual billing is treated as a downgrade or upgrade "
        "depending on the resulting price.",
    ),
]


REFUND_SECTION = (
    "Refunds",
    (
        "4.1 Monthly plans: a charge may be refunded if the refund is requested within 14 days of that charge.\n"
        "4.2 Annual plans: a charge may be refunded in full if the refund is requested within 30 days of that "
        "charge, provided that fewer than 5 report exports were performed in the current billing period.\n"
        "4.3 Accounts on promotional or discounted pricing, including the nonprofit discount, are not eligible "
        "for refunds under 4.1 or 4.2.\n"
        "4.4 Duplicate charges (the same invoice charged more than once) are always refunded, regardless of plan, "
        "pricing or timing. This rule overrides 4.1 to 4.3.\n"
        "4.5 Refunds are returned to the original payment method within 5 to 10 business days."
    ),
)


def build_policy(target_tokens, seed):
    rng = random.Random(seed)
    sections = POLICY_FILLER[:]
    rng.shuffle(sections)
    sections.insert(len(sections) // 2, REFUND_SECTION)
    text = "REFUND AND BILLING POLICY (effective 1 January 2026)\n\n"
    text += "\n\n".join(f"Section: {t}\n{b}" for t, b in sections)
    # Pad with a glossary appendix to reach the target size
    glossary, n = [], 1
    while len(text) + sum(len(g) for g in glossary) < target_tokens * 4:
        term = rng.choice(
            [
                "workspace",
                "seat",
                "project",
                "export",
                "invoice",
                "renewal date",
                "billing period",
                "admin",
                "guest user",
                "integration",
                "audit log",
            ]
        )
        glossary.append(
            f"G{n}. '{term}' has the meaning given in the main terms of service, "
            f"section {rng.randint(2, 19)}.{rng.randint(1, 9)}, and applies to all plans."
        )
        n += 1
    return text + "\n\nAPPENDIX: GLOSSARY\n" + "\n".join(glossary)


ROUTING = [
    ("billing", "Billing: payments, charges, invoices, refunds or subscriptions"),
    ("technical", "Technical: bugs, errors, outages or integration problems"),
    ("sales", "Sales: pricing, quotes, plan upgrades or new seats"),
    ("general", "General: onboarding, how-to questions or feedback with no problem to fix"),
]


URGENCY = [
    ("1", "Very low: no action needed"),
    ("2", "Low: can wait a week or more"),
    ("3", "Medium: should be handled within a few days"),
    ("4", "High: blocks important work, handle today"),
    ("5", "Critical: outage, data loss or security risk, handle immediately"),
]


def q_choice(qid, text, options, expected):
    return {"id": qid, "type": "choice", "text": text, "options": options, "expected": expected}


def q_noul(qid, text, expected):
    return {"id": qid, "type": "noul", "text": text, "options": [("yes", None), ("no", None)], "expected": expected}


def q_score(qid, text, options, min_expected):
    return {"id": qid, "type": "score", "text": text, "options": options, "min_expected": min_expected}


def build_cases():
    cases = []

    # 1. Buried fact at increasing lengths (also gives latency vs length)
    billing_msg = (
        "Separate issue that still needs attention: I just checked our company card statement and "
        "we were charged $49.00 twice on September 3 for the same invoice (INV-20931). "
        "Could someone look into this and fix it?"
    )
    for length in (500, 2000, 4000, 6500):
        state = build_thread(length, [(0.5, "Customer (Alex Chen)", billing_msg)], seed=length)
        q = q_choice(
            "routing",
            "Which team should handle the open issue in this thread that still needs action?",
            ROUTING,
            "billing",
        )
        cases.append({"case": "buried_fact", "variant": f"{length}", "state": state, "questions": [q]})
        if length == 2000:
            cases.append(
                {"case": "buried_fact", "variant": "2000-reversed", "state": state, "questions": [q], "reverse": True}
            )

    # 2. Updated facts + several questions on one state (prefix caching)
    upd_state = build_thread(
        2500,
        [
            (
                0.2,
                "Customer (Alex Chen)",
                "We were charged $49.00 twice on September 3 for invoice INV-20931. Can you refund the duplicate?",
            ),
            (
                0.3,
                "Agent (Priya, Support)",
                "Thanks for flagging this. I've refunded the duplicate $49.00 charge; "
                "it should appear on your statement within 3-5 business days.",
            ),
            (
                0.6,
                "Customer (Alex Chen)",
                "Confirming the $49.00 refund arrived today, thanks for sorting that out so quickly.",
            ),
            (
                1.0,
                "Customer (Alex Chen)",
                "One more thing, and it's urgent: since yesterday every report export to "
                "CSV fails with 'Error 500: export worker timeout'. Our month-end reporting "
                "is due Friday and this is blocking the whole finance team.",
            ),
        ],
        seed=77,
    )
    cases.append(
        {
            "case": "updated_fact_multi",
            "variant": "2500",
            "state": upd_state,
            "multi": True,
            "questions": [
                q_choice("routing", "Which team should handle the issue that is currently open?", ROUTING, "technical"),
                q_noul("billing_unresolved", "Is there still an unresolved billing problem in this thread?", "no"),
                q_score("urgency", "How urgent is the currently open issue?", URGENCY, 3.0),
                q_noul("cancel_threat", "Does the customer threaten to cancel their subscription?", "no"),
            ],
        }
    )

    # 3. Rule application: policy document + case record
    policy = build_policy(1500, seed=5)
    records = [
        (
            "annual-3-exports",
            "yes",
            "Customer: Northwind Studio. Plan: Annual, standard pricing. Charge in question: $1,188.00 annual "
            "renewal on 30 August 2026. Refund requested on 19 September 2026 (20 days after the charge). "
            "Report exports in the current billing period: 3. Reason given: the team is moving to another tool.",
        ),
        (
            "annual-7-exports",
            "no",
            "Customer: Northwind Studio. Plan: Annual, standard pricing. Charge in question: $1,188.00 annual "
            "renewal on 30 August 2026. Refund requested on 19 September 2026 (20 days after the charge). "
            "Report exports in the current billing period: 7. Reason given: the team is moving to another tool.",
        ),
        (
            "nonprofit-monthly",
            "no",
            "Customer: Riverside Food Bank. Plan: Monthly with the nonprofit discount (30% off). Charge in "
            "question: $34.30 on 12 September 2026. Refund requested on 19 September 2026 (7 days after the "
            "charge). Report exports: 0. Reason given: not using the product enough.",
        ),
        (
            "nonprofit-duplicate",
            "yes",
            "Customer: Riverside Food Bank. Plan: Monthly with the nonprofit discount (30% off). Invoice "
            "INV-30112 for $34.30 was charged twice on 12 September 2026. Refund requested on 19 September "
            "2026 for the second, duplicate charge. Report exports: 0.",
        ),
    ]
    for name, expected, record in records:
        state = policy + "\n\nCASE RECORD\n" + record
        q = q_noul(
            "eligible",
            "Under the policy above, is the customer eligible for a refund of the charge they are requesting?",
            expected,
        )
        cases.append({"case": "rules", "variant": name, "state": state, "questions": [q]})

    # 4. Distractor: lots of sales talk, but the latest request is a bug
    dis_state = build_thread(
        2000,
        [
            (
                0.35,
                "Agent (Priya, Support)",
                "I've emailed you the quote for 40 Business seats with the annual "
                "prepayment discount; let me know if you have questions.",
            ),
            (
                0.5,
                "Customer (Alex Chen)",
                "Got the quote, thanks. Finance is reviewing it and we'll decide next "
                "month, so nothing is needed from you on pricing for now.",
            ),
            (
                1.0,
                "Customer (Alex Chen)",
                "Also, while testing checkout with a discount code, the Apply button does "
                "nothing and the browser console shows: TypeError: cannot read properties "
                "of undefined (reading 'code'). Can you get this fixed?",
            ),
        ],
        seed=11,
        pool="sales",
    )
    q = q_choice("routing", "Which team should handle the customer's latest request?", ROUTING, "technical")
    cases.append({"case": "distractor", "variant": "2000", "state": dis_state, "questions": [q]})
    cases.append(
        {"case": "distractor", "variant": "2000-reversed", "state": dis_state, "questions": [q], "reverse": True}
    )
    return cases
