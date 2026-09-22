"""A graded battery: every question type, simple to complex, with known-hard cases at the end.

Case shape matches eval_cases.build_cases() so scripts/eval_service.py can run either pack.
A question may declare:
    expected        the right answer ("yes"/"no", a choice key) - omit or None to skip grading
    min_expected    score: the expected value must reach this (levels numbered by their keys)
    expect_range    score: [low, high] the expected value must fall inside
    expect_unsure   the answer should come back with confidence < 0.5 (genuinely ambiguous)
    note            why this case is here
"""

POLICY = (
    "REFUND POLICY\n"
    "1. Monthly plans: refundable in full if requested within 14 days of the charge.\n"
    "2. Annual plans: refundable in full if requested within 30 days of the charge, provided the "
    "account has made 5 or fewer report exports in the current billing period.\n"
    "3. Duplicate charges are always refundable, on any plan, with no time limit.\n"
    "4. Accounts on the nonprofit discount are not eligible for goodwill refunds, but rule 3 still applies.\n"
)


def q(qid, type_, text, **kw):
    return {"id": qid, "type": type_, "text": text, **kw}


def choice_q(qid, text, options, expected=None, **kw):
    return q(qid, "choice", text, options=options, expected=expected, **kw)


def noul_q(qid, text, expected=None, **kw):
    return q(qid, "noul", text, options=[("yes", None), ("no", None)], expected=expected, **kw)


def score_q(qid, text, options, **kw):
    return q(qid, "score", text, options=options, **kw)


SENTIMENT3 = [("1", "Calm and satisfied"), ("2", "Mildly annoyed"), ("3", "Angry")]
URGENCY5 = [
    ("1", "Very low: no action needed"),
    ("2", "Low: can wait a week or more"),
    ("3", "Medium: should be handled within a few days"),
    ("4", "High: blocks important work, handle today"),
    ("5", "Critical: outage, data loss or security risk, handle immediately"),
]
SEVERITY10 = [(str(i), f"Severity {i} of 9, where 0 is harmless and 9 is catastrophic") for i in range(10)]
ROUTING4 = [
    ("billing", "Billing: payments, charges, invoices, refunds or subscriptions"),
    ("technical", "Technical: bugs, errors, outages or integration problems"),
    ("sales", "Sales: pricing, quotes, plan upgrades or new seats"),
    ("general", "General: onboarding, how-to questions or feedback with no problem to fix"),
]
ROUTING8 = [
    *ROUTING4,
    ("security", "Security: vulnerabilities, breaches, suspicious access"),
    ("legal", "Legal: contracts, compliance, data processing agreements"),
    ("partnerships", "Partnerships: integrations with other vendors, reseller enquiries"),
    ("careers", "Careers: job applications and recruiting"),
]
DEPARTMENTS16 = [
    *ROUTING8,
    ("accessibility", "Accessibility: screen readers, contrast, keyboard navigation"),
    ("localisation", "Localisation: translations, date and currency formats"),
    ("data_export", "Data export: bulk downloads, archives, migration out"),
    ("mobile", "Mobile: the iOS and Android applications specifically"),
    ("api", "API: REST endpoints, SDKs, webhooks, rate limits"),
    ("billing_disputes", "Billing disputes: chargebacks and disputed transactions escalated by a bank"),
    ("training", "Training: paid onboarding sessions and certification"),
    ("status", "Status: questions about published uptime and incident history"),
]


def build_suite():
    cases = []

    def add(group, variant, state, questions, **kw):
        cases.append({"case": group, "variant": variant, "state": state, "questions": questions, **kw})

    # ---------------------------------------------------------------- noul, simple -> complex
    add(
        "noul",
        "1-trivial",
        "Hello there! Hope you are having a good week.",
        [noul_q("greeting", "Is this message a greeting?", "yes", note="floor: should be near certain")],
    )
    add(
        "noul",
        "2-sentiment",
        "This is the third time I have had to chase you about the same broken export.",
        [noul_q("unhappy", "Is the customer unhappy?", "yes")],
    )
    add(
        "noul",
        "3-criteria",
        "Buy cheap watches now!!! Limited offer, click here.",
        [
            noul_q(
                "spam",
                "Is this message spam?",
                "yes",
                criteria={"true": "Unsolicited advertising", "false": "A legitimate conversation"},
            )
        ],
    )
    add(
        "noul",
        "4-negation",
        "I do not want a refund, I just want the bug fixed.",
        [
            noul_q(
                "wants_refund",
                "Is the customer asking for a refund?",
                "no",
                note="explicit negation: a keyword matcher gets this wrong",
            )
        ],
    )
    add(
        "noul",
        "5-resolved",
        (
            "Customer: We were charged twice for invoice INV-20931.\n"
            "Agent: I have refunded the duplicate charge, it will land in 3-5 days.\n"
            "Customer: Confirmed, the refund arrived today, thank you.\n"
            "Customer: Separately, CSV export has been failing since yesterday."
        ),
        [
            noul_q(
                "billing_open",
                "Is there still an unresolved billing problem in this thread?",
                "no",
                note="requires noticing the billing issue was closed",
            )
        ],
    )
    add(
        "noul",
        "6-absence",
        "Customer: Can you tell me how to invite a teammate?",
        [
            noul_q(
                "threat",
                "Does the customer threaten to cancel their subscription?",
                "no",
                note="absence of evidence; models like to hedge",
            )
        ],
    )
    add(
        "noul",
        "7-policy-duplicate",
        POLICY
        + (
            "\nCASE: Riverside Food Bank, monthly plan with the nonprofit discount. Invoice INV-30112 for "
            "$34.30 was charged twice on 12 September 2026. Refund requested 19 September 2026 for the "
            "second charge."
        ),
        [
            noul_q(
                "eligible",
                "Under the policy above, is the customer eligible for a refund?",
                "yes",
                note="rule 3 overrides rule 4",
            )
        ],
    )
    add(
        "noul",
        "8-policy-datemath",
        POLICY
        + (
            "\nCASE: Northwind Studio, annual plan. $1,188.00 renewal charged 30 August 2026. Refund "
            "requested 19 September 2026. Report exports this billing period: 3."
        ),
        [
            noul_q(
                "eligible",
                "Under the policy above, is the customer eligible for a refund?",
                "yes",
                note="KNOWN BLIND SPOT: needs 30 Aug -> 19 Sep = 20 days, i.e. within 30",
            )
        ],
    )
    add(
        "noul",
        "9-policy-datemath-neg",
        POLICY
        + (
            "\nCASE: Northwind Studio, annual plan. $1,188.00 renewal charged 30 August 2026. Refund "
            "requested 19 September 2026. Report exports this billing period: 7."
        ),
        [
            noul_q(
                "eligible",
                "Under the policy above, is the customer eligible for a refund?",
                "no",
                note="same dates, fails the export test",
            )
        ],
    )
    add(
        "noul",
        "10-ambiguous",
        "The meeting is at 3.",
        [
            noul_q(
                "actionable",
                "Does this message require the support team to take an action?",
                expect_unsure=True,
                note="genuinely underdetermined; should not be confident",
            )
        ],
    )

    # ---------------------------------------------------------------- choice, simple -> complex
    add(
        "choice",
        "1-binary",
        "My card was charged twice for the same order.",
        [
            choice_q(
                "kind",
                "Is this a billing problem or a technical problem?",
                [("billing", "About money, charges or invoices"), ("technical", "About software defects")],
                "billing",
            )
        ],
    )
    add(
        "choice",
        "2-routing4",
        "Every report export fails with 'Error 500: export worker timeout'.",
        [choice_q("routing", "Which team should handle this?", ROUTING4, "technical")],
    )
    add(
        "choice",
        "3-distractor",
        (
            "Thanks for the quote for 40 Business seats, finance is reviewing it and will decide next month, "
            "so nothing is needed on pricing. Also, the Apply button on the discount code field does nothing "
            "and the console shows TypeError: cannot read properties of undefined."
        ),
        [
            choice_q(
                "routing",
                "Which team should handle the customer's latest request?",
                ROUTING4,
                "technical",
                note="most of the text is sales talk",
            )
        ],
    )
    add(
        "choice",
        "4-nulldesc",
        "I would like to upgrade us to the Business plan and add ten seats.",
        [
            choice_q(
                "routing",
                "Which team should handle this?",
                [("billing", None), ("technical", None), ("sales", None), ("general", None)],
                "sales",
                note="keys must carry the meaning on their own",
            )
        ],
    )
    add(
        "choice",
        "5-eight",
        "We found an endpoint that returns other tenants' invoices without auth.",
        [choice_q("routing", "Which team should handle this?", ROUTING8, "security")],
    )
    add(
        "choice",
        "6-sixteen",
        "Our screen reader announces the invoice table headers in the wrong order.",
        [
            choice_q(
                "routing",
                "Which team should handle this?",
                DEPARTMENTS16,
                "accessibility",
                note="16 options: stresses single-token label allocation",
            )
        ],
    )
    add(
        "choice",
        "7-reversed",
        "Every report export fails with 'Error 500: export worker timeout'.",
        [choice_q("routing", "Which team should handle this?", ROUTING4, "technical")],
        reverse=True,
    )
    add(
        "choice",
        "8-ambiguous",
        "Hi, I have a question about our account.",
        [
            choice_q(
                "routing",
                "Which team should handle this?",
                ROUTING4,
                expect_unsure=True,
                note="no signal; a flat distribution is the correct answer",
            )
        ],
    )

    # ---------------------------------------------------------------- score, simple -> complex
    add(
        "score",
        "1-two-levels",
        "Absolutely fantastic service, you fixed it in ten minutes.",
        [
            score_q(
                "satisfaction",
                "How satisfied is the customer?",
                [("0", "Dissatisfied"), ("1", "Satisfied")],
                min_expected=0.7,
            )
        ],
    )
    add(
        "score",
        "2-three-low",
        "Thanks, that answered my question perfectly.",
        [score_q("anger", "How angry is the customer?", SENTIMENT3, expect_range=[1.0, 1.6])],
    )
    add(
        "score",
        "3-three-high",
        "This is the fourth outage this month and nobody has called me back. Unacceptable.",
        [score_q("anger", "How angry is the customer?", SENTIMENT3, min_expected=2.5)],
    )
    add(
        "score",
        "4-urgency5-low",
        "Whenever you get a chance, could you send the onboarding recording again?",
        [score_q("urgency", "How urgent is this request?", URGENCY5, expect_range=[1.0, 2.6])],
    )
    add(
        "score",
        "5-urgency5-high",
        ("Production is down for all our users, checkout returns 500 for every request, we are losing orders."),
        [score_q("urgency", "How urgent is this request?", URGENCY5, min_expected=4.3)],
    )
    add(
        "score",
        "6-ten-levels",
        "A tooltip is slightly misaligned on the settings page.",
        [
            score_q(
                "severity",
                "How severe is this issue?",
                SEVERITY10,
                expect_range=[0.0, 3.0],
                note="10 levels, the documented Jev maximum",
            )
        ],
    )
    add(
        "score",
        "7-ten-levels-high",
        "An attacker can read any customer's invoices without authenticating.",
        [score_q("severity", "How severe is this issue?", SEVERITY10, min_expected=7.0)],
    )
    add(
        "score",
        "8-midpoint",
        "The product is fine. Some things are good, some are annoying.",
        [
            score_q(
                "anger",
                "How angry is the customer?",
                SENTIMENT3,
                expect_range=[1.2, 2.2],
                note="should land between levels, which is what score is for",
            )
        ],
    )

    # ---------------------------------------------------------------- mixed, one state many questions
    add(
        "mixed",
        "fanout",
        (
            "Customer: Our month-end close is Friday and CSV export has failed every time since yesterday "
            "with 'Error 500: export worker timeout'. We are on the annual Business plan. If this is not "
            "fixed I will have to escalate internally."
        ),
        [
            noul_q("is_bug", "Is the customer reporting a software defect?", "yes"),
            noul_q(
                "threat",
                "Does the customer threaten to cancel their subscription?",
                "no",
                note="'escalate internally' is not a cancellation threat",
            ),
            choice_q("routing", "Which team should handle this?", ROUTING4, "technical"),
            score_q("urgency", "How urgent is this request?", URGENCY5, min_expected=3.5),
        ],
    )
    return cases
