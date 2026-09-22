#!/usr/bin/env python3
"""Fit the per-type calibration temperature that minimises negative log-likelihood.

Input: JSONL, one labelled question per line, as written by scripts/eval_service.py --dump:

    {"type": "choice", "probabilities": [0.7, 0.2, 0.1], "correct": 0}
    {"type": "noul",   "probabilities": {"Yes": 0.9, "No": 0.1}, "correct": "Yes"}

Output: the temperature per question type, to paste into config.yaml. T > 1 flattens an
over-confident model, T < 1 sharpens an under-confident one.

    uv run python scripts/fit_temperature.py runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

TYPES = ("noul", "choice", "score")


def parse(path: pathlib.Path) -> dict[str, list[tuple[list[float], int]]]:
    rows: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        record = json.loads(line)
        kind = record.get("type", "choice")
        probabilities, correct = record["probabilities"], record["correct"]
        if isinstance(probabilities, dict):
            keys = list(probabilities)
            values = [float(probabilities[k]) for k in keys]
            index = keys.index(correct) if correct in keys else int(correct)
        else:
            values = [float(p) for p in probabilities]
            index = int(correct)
        if not 0 <= index < len(values):
            sys.exit(f"{path}:{number}: correct index {index} is outside {len(values)} options")
        rows[kind].append((values, index))
    return rows


def negative_log_likelihood(rows: list[tuple[list[float], int]], temperature: float) -> float:
    total = 0.0
    for probabilities, correct in rows:
        logits = [(math.log(p) / temperature if p > 0 else -60.0 / temperature) for p in probabilities]
        ceiling = max(logits)
        partition = math.fsum(math.exp(x - ceiling) for x in logits)
        total -= logits[correct] - ceiling - math.log(partition)
    return total / len(rows)


def fit(rows: list[tuple[list[float], int]], low: float, high: float, steps: int) -> tuple[float, float, float]:
    """Grid search, then one refinement pass around the winner."""

    def search(lo: float, hi: float) -> tuple[float, float]:
        grid = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
        scored = [(negative_log_likelihood(rows, t), t) for t in grid if t > 0]
        return min(scored)

    loss, best = search(low, high)
    window = (high - low) / (steps - 1)
    loss, best = min([(loss, best), search(max(1e-3, best - window), best + window)])
    return best, loss, negative_log_likelihood(rows, 1.0)


def cross_check(rows: list[tuple[list[float], int]], low: float, high: float, steps: int) -> tuple[float, float] | None:
    """Two-fold: fit on one half, score the other. Fitting and scoring the same rows always flatters."""
    if len(rows) < 8:
        return None
    first, second = rows[::2], rows[1::2]
    held_t, held_1 = 0.0, 0.0
    for train, test in ((first, second), (second, first)):
        temperature, _, _ = fit(train, low, high, steps)
        held_t += negative_log_likelihood(test, temperature) / 2
        held_1 += negative_log_likelihood(test, 1.0) / 2
    return held_t, held_1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=pathlib.Path)
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()

    rows = parse(args.jsonl)
    if not rows:
        sys.exit("no rows")

    print(f"{'type':8} {'n':>5} {'T':>7} {'NLL@T':>9} {'NLL@1':>9}  {'held-out':>18}  verdict")
    fitted: dict[str, float] = {}
    for kind in TYPES:
        sample = rows.get(kind)
        if not sample:
            continue
        temperature, loss, baseline = fit(sample, args.low, args.high, args.steps)
        verdict = "over-confident" if temperature > 1.05 else "under-confident" if temperature < 0.95 else "calibrated"
        held = cross_check(sample, args.low, args.high, args.steps)
        if held is None:
            note, keep = "too few rows", False
        else:
            held_t, held_1 = held
            keep = held_t < held_1
            note = f"{held_t:.4f} vs {held_1:.4f}"
            if not keep:
                verdict += " (does not generalise)"
        fitted[kind] = round(temperature, 3) if keep else 1.0
        print(f"{kind:8} {len(sample):5d} {temperature:7.3f} {loss:9.4f} {baseline:9.4f}  {note:>18}  {verdict}")

    print("\nheld-out = mean NLL on unseen rows, fitted T vs T=1. A temperature that does not beat")
    print("T=1 on held-out rows is overfitting the sample and is reported below as 1.0.")

    if len(rows.get("noul", [])) + len(rows.get("choice", [])) + len(rows.get("score", [])) < 50:
        print("\nwarning: fewer than 50 labelled rows; treat these temperatures as provisional.", file=sys.stderr)
    print("\n# paste into config.yaml")
    print("temperature:")
    for kind in TYPES:
        print(f"  {kind}: {fitted.get(kind, 1.0)}")


if __name__ == "__main__":
    main()
