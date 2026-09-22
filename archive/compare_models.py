#!/usr/bin/env python3
"""
compare_models.py - compare two vLLM endpoints (Gemma 4 E4B vs Gemma 4 26B-A4B) on decision tasks.

Two modes are run on every test item:
  logit : Jev-style. One forward pass, read the probability of each option label (max_tokens=1).
  gen   : normal generation. The model writes its answer as text, which is parsed.

Test categories:
  buried            deciding fact hidden in a long support thread (several lengths and positions)
  update_multi      facts that change over time, 4 questions on one state (prefix caching)
  distractor        lots of sales talk, but the latest request is a bug
  rules_direct      refund policy + case record, one direct "eligible?" question (12 cases)
  rules_sub         the same cases broken into 6 simple yes/no checks each
  rules_decomposed  the sub-answers combined by code into an eligibility decision

Setup:
  export RUNPOD_API_KEY="your-key"
  # optional overrides (defaults are the endpoints from our conversation):
  export E4B_URL="https://h2duge74u2oxgm.api.runpod.ai"
  export B26_URL="https://jg2qfy9mbdrw0h.api.runpod.ai"

Run:
  python compare_models.py                       # everything, both modes
  python compare_models.py --modes logit         # only the Jev-style mode
  python compare_models.py --gen-thinking        # let the model reason before answering in gen mode
  python compare_models.py --lengths 500,2000,4000,6500,12000   # only if max-model-len allows it
  python compare_models.py --dry-run             # show test items and sizes, no API calls

Standard library only. Runs sequentially so that latency numbers are clean (roughly 5-10 minutes).
"""
import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINTS = [
    {"name": "E4B", "base": os.environ.get("E4B_URL", "https://h2duge74u2oxgm.api.runpod.ai"),
     "model": os.environ.get("E4B_MODEL")},
    {"name": "26B-A4B", "base": os.environ.get("B26_URL", "https://jg2qfy9mbdrw0h.api.runpod.ai"),
     "model": os.environ.get("B26_MODEL")},
]
TOP_LOGPROBS = 20
THOUGHT_PREFIX = "<|channel>thought\n<channel|>"   # Gemma 4 empty thought block (from the model card)
SYSTEM = ("You are a precise decision classifier. Read the state carefully, then answer the "
          "question about it. Reply with only the requested label and nothing else.")


class ApiError(Exception):
    pass


# ======================================================================================
# HTTP
# ======================================================================================

def http(method, url, body=None, timeout=330, retries=5, quiet=False):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}"
            retryable = e.code in (429, 500, 502, 503, 504) or "no workers" in last.lower()
            if not retryable or attempt == retries:
                raise ApiError(last)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = str(e)
            if attempt == retries:
                raise ApiError(last)
        if not quiet:
            print(f"      retry {attempt + 1}: {last[:150]}", file=sys.stderr)
        time.sleep(min(30, 5 * (attempt + 1)))


# ======================================================================================
# Test data
# ======================================================================================

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
    """Support thread of ~target_tokens (estimated at 4 chars/token) with key messages inserted.
    key_messages: list of (position 0..1, speaker, text); position 1.0 = the very end."""
    rng = random.Random(seed)
    cust = NEUTRAL_CUSTOMER + (SALES_CUSTOMER * 2 if pool == "sales" else [])
    agent = NEUTRAL_AGENT + (SALES_AGENT * 2 if pool == "sales" else [])
    target_chars = max(0, target_tokens * 4 - sum(len(t) + 60 for _, _, t in key_messages))
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
        msgs.insert(len(msgs) if pos >= 1.0 else int(pos * len(msgs)), (speaker, text))
    ts = datetime(2026, 9, 1, 9, 0)
    lines = []
    for speaker, text in msgs:
        ts += timedelta(minutes=rng.randint(7, 180))
        lines.append(f"[{ts:%Y-%m-%d %H:%M}] {speaker}: {text}")
    return "SUPPORT THREAD (oldest first)\n" + "\n".join(lines)


POLICY_FILLER = [
    ("Service availability", "We target 99.9% monthly uptime for the web application, excluding scheduled "
     "maintenance announced at least 48 hours in advance. Credits for missed uptime targets are applied to the "
     "next invoice and are not paid out as cash."),
    ("Support hours", "Standard support is available Monday to Friday, 08:00 to 18:00 CET. Business tier "
     "customers receive priority routing. Response-time targets do not apply during public holidays."),
    ("Data export", "Customers may export reports in CSV, XLSX and PDF formats. Exports are generated "
     "asynchronously and links remain valid for seven days. Export activity is logged per billing period."),
    ("Security", "All data is encrypted in transit and at rest. Staff access to customer data requires a support "
     "ticket and is logged. Security incidents are reported to affected customers within 72 hours."),
    ("Seat management", "Seats can be added at any time and are prorated for the remainder of the billing period. "
     "Removing seats takes effect at the next renewal; removed seats are not refunded."),
    ("Taxes", "Prices exclude VAT and other applicable taxes, which are added to invoices based on the billing "
     "address. Customers are responsible for providing a valid VAT number where applicable."),
    ("Account closure", "Customers can close their account at any time from the settings page. Data is retained "
     "for 30 days after closure and then permanently deleted."),
    ("Plan changes", "Upgrades take effect immediately and are prorated. Downgrades take effect at the next "
     "renewal date."),
]
REFUND_SECTION = ("Refunds", (
    "4.1 Monthly plans: a charge may be refunded if the refund is requested within 14 days of that charge.\n"
    "4.2 Annual plans: a charge may be refunded in full if the refund is requested within 30 days of that "
    "charge, provided that fewer than 5 report exports were performed in the current billing period.\n"
    "4.3 Accounts on promotional or discounted pricing, including the nonprofit discount, are not eligible "
    "for refunds under 4.1 or 4.2.\n"
    "4.4 Duplicate charges (the same invoice charged more than once) are always refunded, regardless of plan, "
    "pricing or timing. This rule overrides 4.1 to 4.3.\n"
    "4.5 Refunds are returned to the original payment method within 5 to 10 business days."))


def build_policy(target_tokens, seed):
    rng = random.Random(seed)
    sections = POLICY_FILLER[:]
    rng.shuffle(sections)
    sections.insert(len(sections) // 2, REFUND_SECTION)
    text = "REFUND AND BILLING POLICY (effective 1 January 2026)\n\n"
    text += "\n\n".join(f"Section: {t}\n{b}" for t, b in sections)
    glossary, n = [], 1
    while len(text) + sum(len(g) for g in glossary) < target_tokens * 4:
        term = rng.choice(["workspace", "seat", "project", "export", "invoice", "renewal date",
                           "billing period", "admin", "guest user", "integration", "audit log"])
        glossary.append(f"G{n}. '{term}' has the meaning given in the main terms of service, "
                        f"section {rng.randint(2, 19)}.{rng.randint(1, 9)}, and applies to all plans.")
        n += 1
    return text + "\n\nAPPENDIX: GLOSSARY\n" + "\n".join(glossary)


# Refund cases: (name, plan, days since charge, exports, discounted, duplicate)
RULE_RECORDS = [
    ("annual-20d-3exp", "annual", 20, 3, False, False),
    ("annual-20d-7exp", "annual", 20, 7, False, False),
    ("nonprofit-monthly-7d", "monthly", 7, 0, True, False),
    ("nonprofit-duplicate", "monthly", 7, 0, True, True),
    ("monthly-10d", "monthly", 10, 2, False, False),
    ("monthly-20d", "monthly", 20, 1, False, False),
    ("annual-40d-2exp", "annual", 40, 2, False, False),
    ("annual-60d-duplicate", "annual", 60, 9, False, True),
    ("annual-29d-4exp", "annual", 29, 4, False, False),
    ("annual-30d-5exp", "annual", 30, 5, False, False),
    ("nonprofit-annual-12d", "annual", 12, 1, True, False),
    ("monthly-3d", "monthly", 3, 0, False, False),
]
CUSTOMERS = ["Northwind Studio", "Riverside Food Bank", "Blue Harbor Legal", "Maple & Co", "Kestrel Labs",
             "Open Fields Trust", "Atlas Freight", "Sunline Clinics", "Pixel Forge", "Hilltop Schools",
             "Green Leaf Charity", "Orbit Analytics"]


def eligible(plan, days, exports, discounted, duplicate):
    if duplicate:
        return True
    if discounted:
        return False
    if plan == "annual":
        return days <= 30 and exports < 5
    return days <= 14


def record_text(i, plan, days, exports, discounted, duplicate):
    request = datetime(2026, 9, 19)
    charge = request - timedelta(days=days)
    amount = ("$1,188.00" if plan == "annual" else "$99.00") if not discounted else \
             ("$831.60" if plan == "annual" else "$69.30")
    pricing = "with the nonprofit discount (30% off)" if discounted else "standard pricing"
    lines = [f"Customer: {CUSTOMERS[i % len(CUSTOMERS)]}.",
             f"Plan: {plan.capitalize()}, {pricing}."]
    if duplicate:
        lines.append(f"Invoice INV-{30100 + i} for {amount} was charged twice on {charge:%d %B %Y}. "
                     f"Refund requested on {request:%d %B %Y} ({days} days after the charges) for the second, "
                     f"duplicate charge.")
    else:
        lines.append(f"Charge in question: {amount} on {charge:%d %B %Y}. Refund requested on "
                     f"{request:%d %B %Y} ({days} days after the charge).")
    lines.append(f"Report exports in the current billing period: {exports}.")
    return " ".join(lines)


SUB_QUESTIONS = [
    ("annual", "Is the customer on an annual plan?"),
    ("within14", "Was the refund requested within 14 days of the charge in question?"),
    ("within30", "Was the refund requested within 30 days of the charge in question?"),
    ("exports_lt5", "Were fewer than 5 report exports performed in the current billing period?"),
    ("discounted", "Is the account on promotional or discounted pricing, such as the nonprofit discount?"),
    ("duplicate", "Is the charge in question a duplicate charge (the same invoice charged more than once)?"),
]


def sub_expected(plan, days, exports, discounted, duplicate):
    return {"annual": plan == "annual", "within14": days <= 14, "within30": days <= 30,
            "exports_lt5": exports < 5, "discounted": discounted, "duplicate": duplicate}


def combine_subs(a):
    """Policy logic in code, applied to the model's sub-answers (True/False/None)."""
    if None in a.values():
        return None
    if a["duplicate"]:
        return True
    if a["discounted"]:
        return False
    return (a["within30"] and a["exports_lt5"]) if a["annual"] else a["within14"]


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
    return {"id": qid, "type": "noul", "text": text, "options": [("yes", None), ("no", None)],
            "expected": "yes" if expected in (True, "yes") else "no"}


def q_score(qid, text, options, min_expected):
    return {"id": qid, "type": "score", "text": text, "options": options, "min_expected": min_expected}


def build_items(lengths):
    items = []

    def add(cat, case, variant, state, q, **kw):
        items.append({"cat": cat, "case": case, "variant": variant, "state": state, "q": q, **kw})

    billing_msg = ("Separate issue that still needs attention: I just checked our company card statement and we "
                   "were charged $49.00 twice on September 3 for the same invoice (INV-20931). Could someone "
                   "look into this and fix it?")
    route_q = q_choice("routing", "Which team should handle the open issue in this thread that still needs action?",
                       ROUTING, "billing")
    longest = max(lengths)
    for length in lengths:
        positions = (0.1, 0.5, 0.9) if length == longest else (0.5,)
        for pos in positions:
            state = build_thread(length, [(pos, "Customer (Alex Chen)", billing_msg)], seed=length * 10 + int(pos * 10))
            add("buried", "buried_fact", f"{length}@{pos}", state, route_q, length=length, pos=pos)
            if length == 2000 and pos == 0.5:
                add("buried", "buried_fact", f"{length}@{pos}-reversed", state, route_q, reverse=True,
                    length=length, pos=pos)

    upd = build_thread(2500, [
        (0.2, "Customer (Alex Chen)", "We were charged $49.00 twice on September 3 for invoice INV-20931. "
                                      "Can you refund the duplicate?"),
        (0.3, "Agent (Priya, Support)", "Thanks for flagging this. I've refunded the duplicate $49.00 charge; "
                                        "it should appear on your statement within 3-5 business days."),
        (0.6, "Customer (Alex Chen)", "Confirming the $49.00 refund arrived today, thanks for sorting that out."),
        (1.0, "Customer (Alex Chen)", "One more thing, and it's urgent: since yesterday every report export to "
                                      "CSV fails with 'Error 500: export worker timeout'. Our month-end reporting "
                                      "is due Friday and this is blocking the whole finance team."),
    ], seed=77)
    for q in [q_choice("routing", "Which team should handle the issue that is currently open?", ROUTING, "technical"),
              q_noul("billing_unresolved", "Is there still an unresolved billing problem in this thread?", "no"),
              q_score("urgency", "How urgent is the currently open issue?", URGENCY, 3.0),
              q_noul("cancel_threat", "Does the customer threaten to cancel their subscription?", "no")]:
        add("update_multi", "updated_fact", "2500", upd, q)

    dis = build_thread(2000, [
        (0.35, "Agent (Priya, Support)", "I've emailed you the quote for 40 Business seats with the annual "
                                         "prepayment discount; let me know if you have questions."),
        (0.5, "Customer (Alex Chen)", "Got the quote, thanks. Finance is reviewing it and we'll decide next month, "
                                      "so nothing is needed from you on pricing for now."),
        (1.0, "Customer (Alex Chen)", "Also, while testing checkout with a discount code, the Apply button does "
                                      "nothing and the browser console shows: TypeError: cannot read properties of "
                                      "undefined (reading 'code'). Can you get this fixed?"),
    ], seed=11, pool="sales")
    dq = q_choice("routing", "Which team should handle the customer's latest request?", ROUTING, "technical")
    add("distractor", "distractor", "2000", dis, dq)
    add("distractor", "distractor", "2000-reversed", dis, dq, reverse=True)

    policy = build_policy(1500, seed=5)
    for i, (name, plan, days, exports, disc, dup) in enumerate(RULE_RECORDS):
        state = policy + "\n\nCASE RECORD\n" + record_text(i, plan, days, exports, disc, dup)
        exp = eligible(plan, days, exports, disc, dup)
        add("rules_direct", "rules", name, state,
            q_noul("eligible", "Under the policy above, is the customer eligible for a refund of the charge "
                               "they are requesting?", exp))
        subs = sub_expected(plan, days, exports, disc, dup)
        for sid, text in SUB_QUESTIONS:
            add("rules_sub", "rules_sub", name, state, q_noul(sid, text, subs[sid]), group=name, rule_expected=exp)
    return items


# ======================================================================================
# Prompting and parsing
# ======================================================================================

def labels_for(q, reverse=False):
    if q["type"] == "noul":
        return ["Yes", "No"], ["yes", "no"], [None, None]
    opts = list(reversed(q["options"])) if reverse and q["type"] == "choice" else list(q["options"])
    return [chr(ord("A") + i) for i in range(len(opts))], [o[0] for o in opts], [o[1] for o in opts]


def build_messages(state, q, labels, descs):
    lines = [f"STATE:\n{state}", "", f"QUESTION: {q['text']}"]
    if q["type"] in ("choice", "score"):
        lines.append("OPTIONS:")
        lines += [f"{lab}. {d}" for lab, d in zip(labels, descs)]
        instr = f"Reply with only the letter of the best option ({', '.join(labels)})."
    else:
        instr = "Reply with only Yes or No."
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines + ["", instr])}]


def norm_tok(tok):
    return tok.strip().lower().rstrip(".:)")


def label_probs(top, labels):
    """top: list of (token, logprob). Sums label variants, normalises over labels.
    Returns (probs per label, coverage = share of total probability on the labels)."""
    mass = {lab: 0.0 for lab in labels}
    for tok, lp in top:
        for lab in labels:
            if norm_tok(tok) == lab.lower():
                mass[lab] += math.exp(lp)
    cov = sum(mass.values())
    if cov == 0:
        return {lab: 1 / len(labels) for lab in labels}, 0.0
    return {lab: m / cov for lab, m in mass.items()}, cov


TAG_RE = re.compile(r"<\|?[A-Za-z_]+\|?>")


def extract_answer(text):
    if "<channel|>" in text:            # drop Gemma 4 thought block, keep the final answer
        text = text.rsplit("<channel|>", 1)[1]
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"^\s*thought\b", "", text)
    return text.strip()


def parse_label(text, labels):
    ans = extract_answer(text)
    if labels == ["Yes", "No"]:
        m = re.search(r"\b(yes|no)\b", ans, re.I)
        return m.group(1).capitalize() if m else None
    m = re.search(r"(?<![A-Za-z])([%s])(?![A-Za-z])" % "".join(labels), ans)
    return m.group(1) if m else None


# ======================================================================================
# Endpoint calls
# ======================================================================================

def check_endpoint(ep, wait_s):
    deadline = time.time() + wait_s
    while True:
        try:
            data, secs = http("GET", ep["base"] + "/v1/models", retries=0, timeout=60, quiet=True)
            m = data["data"][0]
            ep["model"] = ep.get("model") or m["id"]
            ep["root"], ep["max_model_len"] = m.get("root"), m.get("max_model_len")
            print(f"  {ep['name']:<8} UP   model={ep['model']}  root={ep['root']}  "
                  f"max_model_len={ep['max_model_len']}  ({secs * 1000:.0f} ms)")
            if "diffusion" in f"{ep['root']} {ep['model']}".lower():
                print(f"  WARNING: {ep['name']} looks like DiffusionGemma. The logit method does not work on it "
                      f"with stock vLLM; results for that endpoint will not be meaningful.")
            return True
        except (ApiError, KeyError, IndexError) as e:
            if time.time() > deadline:
                print(f"  {ep['name']:<8} NOT READY after {wait_s}s: {str(e)[:120]}")
                return False
            print(f"  {ep['name']:<8} waiting for a worker ({str(e)[:70]}) ...")
            time.sleep(15)


def tokenize_prompt(ep, messages):
    r, _ = http("POST", ep["base"] + "/tokenize",
                {"model": ep["model"], "messages": messages, "add_generation_prompt": True,
                 "chat_template_kwargs": {"enable_thinking": False}})
    ids = r["tokens"]
    th = ep["thought_ids"]
    return ids if ids[-len(th):] == th else ids + th


def logit_call(ep, messages, salt):
    """Returns (top [(token, logprob)], top token, prompt tokens, seconds for the decision call)."""
    if ep["strategy"] == "chat":
        body = {"model": ep["model"], "messages": messages, "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": TOP_LOGPROBS,
                "chat_template_kwargs": {"enable_thinking": False}}
        if salt:
            body["cache_salt"] = salt
        resp, secs = http("POST", ep["base"] + "/v1/chat/completions", body)
        c = resp["choices"][0]["logprobs"]["content"][0]
        top, tok = [(t["token"], t["logprob"]) for t in c["top_logprobs"]], c["token"]
    else:  # "prefix": render the chat template, append the empty thought block, score the next token
        body = {"model": ep["model"], "prompt": tokenize_prompt(ep, messages), "max_tokens": 1,
                "temperature": 0, "logprobs": TOP_LOGPROBS}
        if salt:
            body["cache_salt"] = salt
        resp, secs = http("POST", ep["base"] + "/v1/completions", body)
        lp = resp["choices"][0]["logprobs"]
        top, tok = list(lp["top_logprobs"][0].items()), lp["tokens"][0]
    return top, tok, resp.get("usage", {}).get("prompt_tokens"), secs


def gen_call(ep, messages, salt, thinking, max_tokens):
    body = {"model": ep["model"], "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "skip_special_tokens": False, "chat_template_kwargs": {"enable_thinking": thinking}}
    if salt:
        body["cache_salt"] = salt
    resp, secs = http("POST", ep["base"] + "/v1/chat/completions", body)
    msg = resp["choices"][0]["message"]
    text = (msg.get("content") or "")
    usage = resp.get("usage", {})
    return text, usage.get("completion_tokens"), usage.get("prompt_tokens"), secs


def choose_logit_strategy(ep):
    """Plain chat works for E4B. Larger Gemma 4 models emit an empty thought block first, so the first
    token is not the answer; then we append that block to the prompt ourselves ('prefix')."""
    q = q_noul("probe", "Is the sky blue on a clear day?", "yes")
    labels, _, descs = labels_for(q)
    msgs = build_messages("Weather note: it is a clear, sunny day.", q, labels, descs)
    for strat in ("chat", "prefix"):
        ep["strategy"] = strat
        try:
            if strat == "prefix" and "thought_ids" not in ep:
                r, _ = http("POST", ep["base"] + "/tokenize",
                            {"model": ep["model"], "prompt": THOUGHT_PREFIX, "add_special_tokens": False})
                ep["thought_ids"] = r["tokens"]
            top, tok, _, _ = logit_call(ep, msgs, None)
        except (ApiError, KeyError, IndexError) as e:
            print(f"    {ep['name']}: logit strategy '{strat}' failed: {str(e)[:150]}")
            continue
        _, cov = label_probs(top, labels)
        print(f"    {ep['name']}: logit strategy '{strat}': first token {tok!r}, label coverage {cov:.2f}")
        if cov >= 0.5:
            return True
    ep["strategy"] = "chat"
    print(f"    WARNING: {ep['name']}: no logit strategy put the answer first; logit results will be unreliable.")
    return False


def run_item(ep, mode, item, args, salt):
    q = item["q"]
    labels, ids, descs = labels_for(q, item.get("reverse", False))
    msgs = build_messages(item["state"], q, labels, descs)
    row = {"model": ep["name"], "mode": mode, "cat": item["cat"], "case": item["case"], "variant": item["variant"],
           "q": q["id"], "type": q["type"], "group": item.get("group"), "length": item.get("length"),
           "pos": item.get("pos")}
    try:
        if mode == "logit":
            top, tok, ptoks, secs = logit_call(ep, msgs, salt)
            pl, cov = label_probs(top, labels)
            probs = {i: pl[lab] for lab, i in zip(labels, ids)}
            pred = max(probs, key=probs.get)
            row.update(tokens=ptoks, ms=round(secs * 1000), pred=pred, p_pred=probs[pred], coverage=cov,
                       top_token=tok, probs=probs)
        else:
            text, ctoks, ptoks, secs = gen_call(ep, msgs, salt, args.gen_thinking, args.gen_max_tokens)
            lab = parse_label(text, labels)
            pred = ids[labels.index(lab)] if lab else None
            probs = None
            row.update(tokens=ptoks, ms=round(secs * 1000), pred=pred, completion_tokens=ctoks,
                       output=extract_answer(text)[:100])
        if q["type"] == "score":
            ev = sum(p * float(i) for i, p in probs.items()) if probs else (float(pred) if pred else None)
            row.update(ev=None if ev is None else round(ev, 2),
                       correct=ev is not None and ev >= q["min_expected"])
        else:
            row["correct"] = pred == q["expected"]
            row["expected"] = q["expected"]
            if probs:
                row["p_exp"] = probs[q["expected"]]
    except (ApiError, KeyError, IndexError, TypeError) as e:
        row.update(error=str(e)[:200], correct=None)
    return row


def decomposed_rows(rows):
    """Combine rules_sub answers per (model, mode, case) into an eligibility decision."""
    out, groups = [], {}
    for r in rows:
        if r["cat"] == "rules_sub":
            groups.setdefault((r["model"], r["mode"], r["group"]), []).append(r)
    rec = {name: eligible(*vals) for name, *vals in RULE_RECORDS}
    for (model, mode, group), subs in groups.items():
        answers = {r["q"]: (None if r.get("error") or r.get("pred") is None else r["pred"] == "yes") for r in subs}
        dec = combine_subs(answers)
        ms = [r.get("ms") or 0 for r in subs]
        out.append({"model": model, "mode": mode, "cat": "rules_decomposed", "case": "rules", "variant": group,
                    "q": "eligible", "type": "noul", "pred": None if dec is None else ("yes" if dec else "no"),
                    "expected": "yes" if rec[group] else "no",
                    "correct": dec is not None and dec == rec[group],
                    "ms": sum(ms), "ms_parallel": max(ms), "tokens": subs[0].get("tokens")})
    return out


# ======================================================================================
# Reporting
# ======================================================================================

CATS = ["buried", "update_multi", "distractor", "rules_direct", "rules_decomposed", "rules_sub"]


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)] if rows else \
             [len(h) for h in headers]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print("  " + line + "\n  " + "-" * len(line))
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def acc(rows):
    ok = [r for r in rows if r.get("correct") is not None]
    return (sum(r["correct"] for r in ok), len(ok))


def pct(c, n):
    return "-" if n == 0 else f"{c}/{n} ({100 * c / n:.0f}%)"


def med(values):
    v = [x for x in values if x is not None]
    return "-" if not v else f"{statistics.median(v):.0f}"


def report(rows, endpoints, modes):
    names = [ep["name"] for ep in endpoints]
    print("\n" + "=" * 90 + "\nACCURACY BY CATEGORY\n" + "=" * 90)
    headers = ["category"] + [f"{n} {m}" for m in modes for n in names]
    out = []
    for cat in CATS:
        out.append([cat] + [pct(*acc([r for r in rows if r["cat"] == cat and r["model"] == n and r["mode"] == m]))
                            for m in modes for n in names])
    main = [c for c in CATS if c not in ("rules_sub", "rules_direct")]
    out.append(["OVERALL*"] + [pct(*acc([r for r in rows if r["cat"] in main and r["model"] == n
                                         and r["mode"] == m])) for m in modes for n in names])
    out.append(["OVERALL (direct rules)"] + [pct(*acc([r for r in rows if r["cat"] in
                                                       ("buried", "update_multi", "distractor", "rules_direct")
                                                       and r["model"] == n and r["mode"] == m]))
                                             for m in modes for n in names])
    table(headers, out)
    print("  * OVERALL uses the decomposed rules decision instead of the direct rules question.")

    print("\n" + "=" * 90 + "\nLATENCY (median ms per request, includes network to RunPod)\n" + "=" * 90)
    single = [r for r in rows if r["cat"] != "rules_decomposed" and not r.get("error")]
    out = [[f"{n} {m}", med([r["ms"] for r in single if r["model"] == n and r["mode"] == m]),
            med([r.get("completion_tokens") for r in single if r["model"] == n and r["mode"] == m])
            if m == "gen" else "1",
            med([r["ms"] for r in rows if r["cat"] == "rules_decomposed" and r["model"] == n and r["mode"] == m]),
            med([r.get("ms_parallel") for r in rows if r["cat"] == "rules_decomposed" and r["model"] == n
                 and r["mode"] == m])]
           for m in modes for n in names]
    table(["model/mode", "median ms", "output tokens", "6 checks seq ms", "6 checks parallel ms"], out)

    print("\n  Latency vs prompt length (buried fact, middle position):")
    lens = sorted({r["length"] for r in rows if r["cat"] == "buried" and r.get("length")})
    out = []
    for L in lens:
        line = [L]
        for m in modes:
            for n in names:
                rr = [r for r in rows if r["cat"] == "buried" and r["length"] == L and r["pos"] == 0.5
                      and r["model"] == n and r["mode"] == m and "reversed" not in r["variant"]]
                line.append("error" if rr and rr[0].get("error") else
                            (f"{rr[0]['ms']} ms / {rr[0].get('tokens')} tok" if rr else "-"))
        out.append(line)
    table(["target"] + [f"{n} {m}" for m in modes for n in names], out)

    print("\n  Prefix caching (4 questions on one state, logit mode):")
    for n in names:
        s = [r["ms"] for r in rows if r["cat"] == "update_multi" and r["model"] == n and r["mode"] == "logit"
             and not r.get("error")]
        if s:
            print(f"    {n}: {s} ms (first {s[0]}, later median {med(s[1:])})")

    print("\n" + "=" * 90 + "\nCONTEXT (long states)\n" + "=" * 90)
    out = []
    for r in sorted([r for r in rows if r["cat"] == "buried"], key=lambda r: (r["length"], r["pos"], r["variant"])):
        pass
    keys = sorted({(r["length"], r["pos"], r["variant"]) for r in rows if r["cat"] == "buried"})
    for L, pos, var in keys:
        line = [var]
        for m in modes:
            for n in names:
                rr = [r for r in rows if r["cat"] == "buried" and r["variant"] == var and r["model"] == n
                      and r["mode"] == m]
                if not rr:
                    line.append("-")
                elif rr[0].get("error"):
                    line.append("ERR " + ("too long" if "context" in rr[0]["error"].lower()
                                          or "maximum" in rr[0]["error"].lower() else ""))
                else:
                    extra = f" p={rr[0]['p_exp']:.3f}" if rr[0].get("p_exp") is not None else ""
                    line.append(("OK" if rr[0]["correct"] else "WRONG") + extra)
        out.append(line)
    table(["length@position"] + [f"{n} {m}" for m in modes for n in names], out)

    if "logit" in modes:
        print("\n" + "=" * 90 + "\nCONFIDENCE QUALITY (logit mode)\n" + "=" * 90)
        out = []
        for n in names:
            lr = [r for r in rows if r["model"] == n and r["mode"] == "logit" and r.get("p_exp") is not None]
            if not lr:
                continue
            brier = sum((1 - r["p_exp"]) ** 2 for r in lr) / len(lr)
            right = [r["p_pred"] for r in lr if r["correct"]]
            wrong = [r["p_pred"] for r in lr if not r["correct"]]
            conf_wrong = sum(1 for p in wrong if p > 0.9)
            low_cov = sum(1 for r in lr if r.get("coverage", 1) < 0.9)
            out.append([n, f"{brier:.4f}", f"{statistics.mean(right):.4f}" if right else "-",
                        f"{statistics.mean(wrong):.4f}" if wrong else "-", f"{conf_wrong}/{len(wrong)}", low_cov])
        table(["model", "Brier (lower=better)", "mean p when right", "mean p when wrong",
               "confident wrong (p>0.9)", "coverage<0.9"], out)

    print("\n" + "=" * 90 + "\nDETAILS\n" + "=" * 90)
    for n in names:
        for m in modes:
            bad = [r for r in rows if r["model"] == n and r["mode"] == m and r.get("correct") is False
                   and r["cat"] != "rules_sub"]
            errs = [r for r in rows if r["model"] == n and r["mode"] == m and r.get("error")]
            unparsed = [r for r in rows if r["model"] == n and r["mode"] == m and m == "gen"
                        and not r.get("error") and r.get("pred") is None]
            print(f"  {n} {m}: {len(bad)} wrong, {len(errs)} errors, {len(unparsed)} unparseable outputs")
            for r in bad:
                extra = f" p={r['p_pred']:.3f}" if r.get("p_pred") is not None else \
                        (f" output={r.get('output')!r}" if m == "gen" else "")
                print(f"    wrong: {r['cat']}/{r['variant']}/{r['q']}: predicted {r.get('pred')}, "
                      f"expected {r.get('expected', 'see score')}{extra}")
            for r in errs[:3]:
                print(f"    error: {r['cat']}/{r['variant']}/{r['q']}: {r['error'][:120]}")
            for r in unparsed[:3]:
                print(f"    unparsed: {r['cat']}/{r['variant']}/{r['q']}: {r.get('output')!r}")
            subs = [r for r in rows if r["model"] == n and r["mode"] == m and r["cat"] == "rules_sub"]
            for sid, _ in SUB_QUESTIONS:
                s = [r for r in subs if r["q"] == sid]
                c, t = acc(s)
                if t and c < t:
                    print(f"    sub-check '{sid}': {c}/{t} correct")
        for a, b, label in (("2000@0.5", "2000@0.5-reversed", "buried"), ("2000", "2000-reversed", "distractor")):
            for m in modes:
                ra = [r for r in rows if r["model"] == n and r["mode"] == m and r["variant"] == a and
                      r["cat"] == label]
                rb = [r for r in rows if r["model"] == n and r["mode"] == m and r["variant"] == b and
                      r["cat"] == label]
                if ra and rb and not ra[0].get("error") and not rb[0].get("error"):
                    same = ra[0]["pred"] == rb[0]["pred"]
                    print(f"  {n} {m} option-order ({label}): {ra[0]['pred']} vs reversed {rb[0]['pred']} -> "
                          f"{'consistent' if same else 'CHANGED'}")

    if len(names) == 2:
        a, b = names
        print("\n" + "=" * 90 + f"\nVERDICT: does {b} beat {a}?\n" + "=" * 90)
        for m in modes:
            for cat in CATS + ["OVERALL"]:
                cats = main if cat == "OVERALL" else [cat]
                ca, na = acc([r for r in rows if r["cat"] in cats and r["model"] == a and r["mode"] == m])
                cb, nb = acc([r for r in rows if r["cat"] in cats and r["model"] == b and r["mode"] == m])
                if na and nb:
                    diff = 100 * (cb / nb - ca / na)
                    tag = "better" if diff > 0.5 else ("worse" if diff < -0.5 else "same")
                    print(f"  [{m}] {cat:<17} {a} {100 * ca / na:5.1f}%  vs  {b} {100 * cb / nb:5.1f}%   "
                          f"-> {b} {tag} ({diff:+.1f} pts)")
            la = [r["ms"] for r in single if r["model"] == a and r["mode"] == m]
            lb = [r["ms"] for r in single if r["model"] == b and r["mode"] == m]
            if la and lb:
                ratio = statistics.median(lb) / statistics.median(la)
                print(f"  [{m}] latency: {a} {statistics.median(la):.0f} ms vs {b} {statistics.median(lb):.0f} ms "
                      f"-> {b} is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")
        print("  Note: small hand-written test set. Treat differences of one or two items as noise.")


# ======================================================================================
# Main
# ======================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="logit,gen", help="comma list: logit,gen")
    ap.add_argument("--lengths", default="500,2000,4000,6500", help="buried-fact target lengths in tokens")
    ap.add_argument("--cats", help="comma list of categories to run (default all)")
    ap.add_argument("--gen-thinking", action="store_true", help="let the model reason before answering (gen mode)")
    ap.add_argument("--gen-max-tokens", type=int, default=None, help="default 24, or 2048 with --gen-thinking")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for endpoints to come up")
    ap.add_argument("--no-cache-salt", action="store_true",
                    help="do not isolate the prefix cache between modes (use if the server rejects cache_salt)")
    ap.add_argument("--cold", action="store_true",
                    help="unique cache salt per request: no prefix-cache reuse at all (pure cold timings)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=f"compare_{datetime.now():%Y%m%d_%H%M%S}")
    args = ap.parse_args()
    args.gen_max_tokens = args.gen_max_tokens or (2048 if args.gen_thinking else 24)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    lengths = [int(x) for x in args.lengths.split(",")]
    items = build_items(lengths)
    if args.cats:
        wanted = set(args.cats.split(","))
        items = [i for i in items if i["cat"] in wanted or (i["cat"] == "rules_sub" and "rules_decomposed" in wanted)]

    if args.dry_run:
        counts = {}
        for it in items:
            counts[it["cat"]] = counts.get(it["cat"], 0) + 1
        print("Items per category:", counts, f"-> {len(items)} requests per model per mode")
        for it in items:
            if it["cat"] in ("buried", "distractor", "update_multi") or it["q"]["id"] == "eligible":
                labels, _, descs = labels_for(it["q"], it.get("reverse", False))
                msg = build_messages(it["state"], it["q"], labels, descs)[1]["content"]
                print(f"  {it['cat']:<14} {it['variant']:<24} {it['q']['id']:<20} ~{len(msg) // 4} tokens")
        return

    if not API_KEY:
        sys.exit("Set RUNPOD_API_KEY first.")

    print("Checking endpoints (GET /v1/models)...")
    endpoints = [ep for ep in ENDPOINTS if check_endpoint(ep, args.wait)]
    if not endpoints:
        sys.exit("No endpoint is up.")

    print("\nWarming up and choosing the logit method per endpoint...")
    for ep in endpoints:
        if "logit" in modes:
            choose_logit_strategy(ep)
        if "gen" in modes:
            try:
                text, ctoks, _, secs = gen_call(ep, [{"role": "user", "content": "Reply with only Yes or No: "
                                                      "is water wet?"}], None, args.gen_thinking, args.gen_max_tokens)
                print(f"    {ep['name']}: gen warm-up output {extract_answer(text)!r} "
                      f"({ctoks} tokens, {secs * 1000:.0f} ms)")
            except ApiError as e:
                print(f"    {ep['name']}: gen warm-up failed: {str(e)[:150]}")

    run_id = datetime.now().strftime("%H%M%S")
    rows = []
    total = len(items) * len(endpoints) * len(modes)
    done = 0
    for ep in endpoints:
        for mode in modes:
            salt = None if args.no_cache_salt else f"{mode}-{run_id}"
            print(f"\nRunning {len(items)} items on {ep['name']} in {mode} mode...")
            for it in items:
                req_salt = f"{salt}-{done}" if (args.cold and salt) else salt
                row = run_item(ep, mode, it, args, req_salt)
                rows.append(row)
                done += 1
                status = "ERR" if row.get("error") else ("ok" if row["correct"] else "WRONG")
                print(f"  [{done}/{total}] {it['cat']:<13} {it['variant']:<24} {it['q']['id']:<18} "
                      f"{row.get('ms', '-'):>6} ms  {status}")
    rows += decomposed_rows(rows)

    report(rows, endpoints, modes)

    meta = {"endpoints": [{k: ep.get(k) for k in ("name", "base", "model", "root", "max_model_len", "strategy")}
                          for ep in endpoints],
            "modes": modes, "gen_thinking": args.gen_thinking, "lengths": lengths}
    with open(args.out + ".json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    fields = ["model", "mode", "cat", "case", "variant", "q", "type", "tokens", "ms", "pred", "expected", "correct",
              "p_pred", "p_exp", "coverage", "ev", "completion_tokens", "output", "top_token", "error"]
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
