#!/usr/bin/env python3
"""Evaluate our service, and optionally the real Jev on the identical cases, head to head.

    uv run python scripts/eval_service.py --suite all --compare-jev
    uv run python scripts/eval_service.py --dump runs.jsonl      # feeds fit_temperature.py

Case packs:
    baseline  the 15 questions from the original direct-endpoint run (long states, 14/15)
    battery   30 questions, every type, simple to complex, incl. known-hard cases
Reports accuracy (overall and per type), per-request latency, calibration (ECE/Brier),
and where the two systems disagree.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
import eval_cases  # noqa: E402
import eval_suite  # noqa: E402

BASELINE = ROOT / "tests" / "fixtures" / "baseline_e4b_direct.json"
JEV_URL = "https://api.typesafe.ai"


# ---------------------------------------------------------------- case -> request


def to_question(question: dict, reverse: bool) -> dict:
    options = list(question.get("options") or [])
    if reverse and question["type"] == "choice":
        options = list(reversed(options))
    if question["type"] == "noul":
        body = {"type": "noul", "instructions": question["text"]}
        if question.get("criteria"):
            body["criteria"] = question["criteria"]
        return body
    if question["type"] == "choice":
        return {"type": "choice", "instructions": question["text"], "criteria": dict(options)}
    return {"type": "score", "instructions": question["text"], "criteria": [d for _, d in options]}


def to_request(case: dict, model: str) -> dict:
    reverse = case.get("reverse", False)
    return {
        "model": model,
        "state": case["state"],
        "questions": {q["id"]: to_question(q, reverse) for q in case["questions"]},
    }


# ---------------------------------------------------------------- grading


def score_value(question: dict, answer: dict, reverse: bool) -> float:
    """Expected value in the levels' own numbering; the API's levels are always zero-based."""
    options = list(question["options"])
    if reverse:
        options = list(reversed(options))
    keys = [float(k) for k, _ in options]
    return sum(k * answer["probabilities"][str(i)] for i, k in enumerate(keys))


def grade(question: dict, answer: dict, reverse: bool) -> tuple[bool | None, float, str, str]:
    """Returns (correct or None if ungraded, p(predicted outcome), prediction, check performed)."""
    kind = question["type"]
    if kind == "noul":
        probability = answer["noul"]
        predicted, confident = ("yes" if probability >= 0.5 else "no"), max(probability, 1 - probability)
        if question.get("expect_unsure"):
            return 0.3 <= probability <= 0.7, confident, f"{probability:.2f}", "unsure"
        if question.get("expected") is None:
            return None, confident, predicted, "none"
        return predicted == question["expected"], confident, predicted, "exact"
    if kind == "choice":
        confident = max(answer["probabilities"].values())
        if question.get("expect_unsure"):
            return answer["confidence"] < 0.5, confident, f"conf={answer['confidence']:.2f}", "unsure"
        if question.get("expected") is None:
            return None, confident, answer["choice"], "none"
        return answer["choice"] == question["expected"], confident, answer["choice"], "exact"

    value = score_value(question, answer, reverse)
    if question.get("expect_unsure"):
        return answer["confidence"] < 0.5, answer["confidence"], f"{value:.2f}", "unsure"
    if question.get("expect_range"):
        low, high = question["expect_range"]
        mass = sum(p for i, p in enumerate(answer["probabilities"].values()) if low <= i <= high)
        return low <= value <= high, mass, f"{value:.2f}", f"in [{low},{high}]"
    if question.get("min_expected") is not None:
        threshold = question["min_expected"]
        options = list(question["options"])
        keys = [float(k) for k, _ in options]
        mass = sum(p for k, p in zip(keys, answer["probabilities"].values(), strict=False) if k >= threshold)
        return value >= threshold, mass, f"{value:.2f}", f">={threshold}"
    return None, 1.0, f"{value:.2f}", "none"


def calibration(rows: list[dict], bins: int = 10) -> tuple[float, float]:
    graded = [r for r in rows if r["correct"] is not None]
    if not graded:
        return 0.0, 0.0
    brier = sum((r["p"] - float(r["correct"])) ** 2 for r in graded) / len(graded)
    ece = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        bucket = [r for r in graded if (low < r["p"] <= high) or (b == 0 and r["p"] == 0)]
        if bucket:
            accuracy = sum(r["correct"] for r in bucket) / len(bucket)
            mean_p = sum(r["p"] for r in bucket) / len(bucket)
            ece += len(bucket) / len(graded) * abs(accuracy - mean_p)
    return ece, brier


# ---------------------------------------------------------------- run


def run(client: httpx.Client, url: str, headers: dict, model: str, cases: list[dict], label: str) -> dict:
    # One throwaway request so a cold worker does not land in the latency numbers.
    try:
        client.post(
            f"{url}/v1/systemone",
            headers=headers,
            timeout=600,
            json={"model": model, "state": "warm up", "questions": {"w": {"type": "noul", "instructions": "ok?"}}},
        )
    except httpx.HTTPError as error:
        print(f"  ! {label} warm-up failed: {error}")

    rows, latencies, failures = [], [], 0
    for case in cases:
        body = to_request(case, model)
        started = time.perf_counter()
        try:
            response = client.post(f"{url}/v1/systemone", json=body, headers=headers, timeout=600)
        except httpx.HTTPError as error:
            print(f"  ! {label} {case['case']}/{case['variant']}: {error}")
            failures += 1
            continue
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            print(f"  ! {label} {case['case']}/{case['variant']}: HTTP {response.status_code} {response.text[:200]}")
            failures += 1
            continue
        latencies.append(elapsed)
        payload = response.json()
        for question in case["questions"]:
            answer = payload["answers"][question["id"]]
            correct, probability, predicted, check = grade(question, answer, case.get("reverse", False))
            rows.append(
                {
                    "case": case["case"],
                    "variant": case["variant"],
                    "q": question["id"],
                    "type": question["type"],
                    "correct": correct,
                    "p": probability,
                    "pred": predicted,
                    "check": check,
                    "expected": question.get("expected")
                    or question.get("expect_range")
                    or question.get("min_expected")
                    or ("unsure" if question.get("expect_unsure") else "-"),
                    "request_ms": elapsed,
                    "questions_in_request": len(case["questions"]),
                    "answer": answer,
                    "note": question.get("note", ""),
                }
            )
    return {
        "label": label,
        "rows": rows,
        "latencies": latencies,
        "failures": failures,
        "server_timing": response.headers.get("Server-Timing") if rows else None,
    }


def summarise(result: dict) -> dict:
    rows = result["rows"]
    graded = [r for r in rows if r["correct"] is not None]
    ece, brier = calibration(rows)
    latencies = sorted(result["latencies"]) or [0.0]
    per_type = {}
    for kind in ("noul", "choice", "score"):
        subset = [r for r in graded if r["type"] == kind]
        per_type[kind] = (sum(r["correct"] for r in subset), len(subset))
    return {
        "label": result["label"],
        "correct": sum(r["correct"] for r in graded),
        "graded": len(graded),
        "per_type": per_type,
        "ece": ece,
        "brier": brier,
        "p50": statistics.median(latencies),
        "p90": latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))],
        "mean_q": statistics.mean([r["request_ms"] / r["questions_in_request"] for r in rows]) if rows else 0.0,
        "failures": result["failures"],
    }


def print_comparison(summaries: list[dict]) -> None:
    def pct(hit, total):
        return f"{hit}/{total} ({100 * hit / total:.0f}%)" if total else "-"

    width = max(len(s["label"]) for s in summaries) + 2
    print("\n" + "=" * 78)
    print(f"{'metric':22}" + "".join(f"{s['label']:>{width}}" for s in summaries))
    print("-" * 78)
    print(f"{'accuracy':22}" + "".join(f"{pct(s['correct'], s['graded']):>{width}}" for s in summaries))
    for kind in ("noul", "choice", "score"):
        print(f"{'  ' + kind:22}" + "".join(f"{pct(*s['per_type'][kind]):>{width}}" for s in summaries))
    print(f"{'latency p50 / request':22}" + "".join(f"{s['p50']:>{width - 3}.0f} ms" for s in summaries))
    print(f"{'latency p90 / request':22}" + "".join(f"{s['p90']:>{width - 3}.0f} ms" for s in summaries))
    print(f"{'latency / question':22}" + "".join(f"{s['mean_q']:>{width - 3}.0f} ms" for s in summaries))
    print(f"{'ECE (lower better)':22}" + "".join(f"{s['ece']:>{width}.3f}" for s in summaries))
    print(f"{'Brier (lower better)':22}" + "".join(f"{s['brier']:>{width}.3f}" for s in summaries))
    print(f"{'failed requests':22}" + "".join(f"{s['failures']:>{width}}" for s in summaries))
    print("=" * 78)


def print_misses(result: dict) -> None:
    misses = [r for r in result["rows"] if r["correct"] is False]
    if not misses:
        print(f"\n{result['label']}: no misses")
        return
    print(f"\n{result['label']} misses ({len(misses)}):")
    for row in misses:
        print(
            f"  {row['case']}/{row['variant']}/{row['q']:18} expected {row['expected']!s:14} "
            f"got {row['pred']!s:12} p={row['p']:.2f}"
        )
        if row["note"]:
            print(f"      {row['note']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("SO1_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--api-key", default=os.environ.get("SO1_API_KEY"))
    parser.add_argument("--suite", choices=("baseline", "battery", "all"), default="battery")
    parser.add_argument("--compare-jev", action="store_true")
    parser.add_argument("--dump", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path, help="write the full run for later analysis")
    args = parser.parse_args()

    cases = []
    if args.suite in ("baseline", "all"):
        cases += eval_cases.build_cases()
    if args.suite in ("battery", "all"):
        cases += eval_suite.build_suite()
    questions = sum(len(c["questions"]) for c in cases)
    print(f"suite={args.suite}: {len(cases)} cases, {questions} questions")

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    results = []
    with httpx.Client() as client:
        print(f"\nrunning so1 at {args.url} ({args.model}) ...")
        ours = run(client, args.url, headers, args.model, cases, "so1")
        results.append(ours)
        if args.compare_jev:
            key = os.environ.get("TYPESAFE_API_KEY")
            if not key:
                sys.exit("--compare-jev needs TYPESAFE_API_KEY")
            print(f"running jev at {JEV_URL} ...")
            results.append(run(client, JEV_URL, {"Authorization": f"Bearer {key}"}, "jev-latest", cases, "jev-latest"))

    print_comparison([summarise(r) for r in results])
    for result in results:
        print_misses(result)

    if len(results) == 2:
        ours, theirs = results
        pairs = {(r["case"], r["variant"], r["q"]): r for r in theirs["rows"]}
        both = [(a, pairs[k]) for a in ours["rows"] if (k := (a["case"], a["variant"], a["q"])) in pairs]
        disagree = [(a, b) for a, b in both if a["pred"] != b["pred"]]
        print(f"\nagreement: {len(both) - len(disagree)}/{len(both)} identical predictions")
        if disagree:
            print(f"  {'case/question':46} {'expected':14} {'so1':14} {'jev':14}")
            for a, b in disagree:
                mark = {(True, False): "so1 wins", (False, True): "jev wins"}.get((a["correct"], b["correct"]), "")
                print(
                    f"  {a['case'] + '/' + a['variant'] + '/' + a['q']:46} {a['expected']!s:14} "
                    f"{str(a['pred'])[:13]:14} {str(b['pred'])[:13]:14} {mark}"
                )

    if BASELINE.exists() and args.suite in ("baseline", "all"):
        baseline = json.load(BASELINE.open())["rows"]
        # Only the baseline pack is comparable; battery cases have no recorded counterpart.
        covered = {(r["case"], r["variant"], r["q"]) for r in baseline}
        was = {k for r in baseline if not r["correct"] and (k := (r["case"], r["variant"], r["q"]))}
        now = {
            k
            for r in results[0]["rows"]
            if r["correct"] is False and (k := (r["case"], r["variant"], r["q"])) in covered
        }
        print(f"\nvs recorded direct-endpoint baseline ({sum(r['correct'] for r in baseline)}/{len(baseline)}):")
        for key in sorted(was - now):
            print(f"  fixed:     {'/'.join(key)}")
        for key in sorted(now - was):
            print(f"  regressed: {'/'.join(key)}")
        for key in sorted(now & was):
            print(f"  unchanged miss: {'/'.join(key)}")

    if args.json:
        args.json.write_text(json.dumps([{k: v for k, v in r.items()} for r in results], indent=1, default=str))
        print(f"\nfull run -> {args.json}")

    if args.dump:
        written = 0
        with args.dump.open("w") as handle:
            for row in results[0]["rows"]:
                if row["correct"] is None or row["check"] != "exact":
                    continue
                answer = row["answer"]
                if row["type"] == "noul":
                    record = {
                        "type": "noul",
                        "probabilities": {"Yes": answer["noul"], "No": 1 - answer["noul"]},
                        "correct": "Yes" if row["expected"] == "yes" else "No",
                    }
                elif row["type"] == "choice":
                    record = {"type": "choice", "probabilities": answer["probabilities"], "correct": row["expected"]}
                else:
                    continue
                handle.write(json.dumps(record) + "\n")
                written += 1
        print(f"\nwrote {written} labelled rows -> {args.dump}")


if __name__ == "__main__":
    main()
