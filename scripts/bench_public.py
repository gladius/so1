#!/usr/bin/env python3
"""Evaluate so1 and the real Jev on PUBLIC datasets with real human gold labels.

Everything else in results/ is synthetic and self-authored. This is the only evaluation where
neither the inputs nor the labels came from us, so it is the only unbiased accuracy and
calibration measurement available.

    uv run python scripts/bench_public.py --dataset boolq --n 250
    uv run python scripts/bench_public.py --dataset sst5  --n 250

  boolq  yes/no reading comprehension (google/boolq validation) -> noul
  sst5   5-level sentiment (SetFit/sst5 test)                    -> score, 5 ordinal levels
  yelp   5-star reviews (Yelp/yelp_review_full test)             -> score, 5 ordinal levels

Reports accuracy with a Wilson interval, McNemar between the two systems, and calibration
(ECE / Brier) against the gold labels.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import pathlib
import statistics
import urllib.parse

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = "https://datasets-server.huggingface.co/rows"
JEV_URL = "https://api.typesafe.ai"

SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
YELP_LEVELS = ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]

DATASETS = {
    "boolq": {"id": "google/boolq", "config": "default", "split": "validation", "kind": "noul"},
    "sst5": {"id": "SetFit/sst5", "config": "default", "split": "test", "kind": "score", "levels": SST5_LEVELS},
    "yelp": {"id": "Yelp/yelp_review_full", "config": "yelp_review_full", "split": "test",
             "kind": "score", "levels": YELP_LEVELS},
}


def fetch(spec: dict, n: int, offset: int) -> list[dict]:
    out: list[dict] = []
    with httpx.Client(timeout=120) as client:
        while len(out) < n:
            params = {"dataset": spec["id"], "config": spec["config"], "split": spec["split"],
                      "offset": offset + len(out), "length": min(100, n - len(out))}
            url = f"{ROWS}?{urllib.parse.urlencode(params)}"
            rows = client.get(url).raise_for_status().json()["rows"]
            if not rows:
                break
            out += [r["row"] for r in rows]
    return out[:n]


def to_item(name: str, spec: dict, row: dict) -> tuple[object, dict, int]:
    """-> (state, questions, gold). gold is 1/0 for noul, the level index for score."""
    if name == "boolq":
        answer = row["answer"]
        gold = int(answer is True or str(answer).lower() == "true")
        state = {"passage": row["passage"]}
        question = {"q": {"type": "noul", "instructions": f"Based on `passage`: {row['question']}?"}}
        return state, question, gold
    levels = spec["levels"]
    gold = int(row["label"])
    state = {"review": row["text"]}
    question = {"q": {"type": "score", "instructions": "Rate the sentiment of `review`.", "criteria": levels}}
    return state, question, gold


def ask(client: httpx.Client, url: str, headers: dict, model: str, state, questions) -> dict:
    response = client.post(f"{url}/v1/systemone", json={"model": model, "state": state, "questions": questions},
                           headers=headers, timeout=600)
    response.raise_for_status()
    return response.json()["answers"]["q"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        bucket = [(p, y) for p, y in pairs if (low < p <= high) or (b == 0 and p <= low)]
        if bucket:
            acc = sum(y for _, y in bucket) / len(bucket)
            conf = sum(p for p, _ in bucket) / len(bucket)
            total += len(bucket) / len(pairs) * abs(acc - conf)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default="boolq")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--url", default=os.environ.get("SO1_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--skip-jev", action="store_true")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    rows = fetch(spec, args.n, args.offset)
    items = [to_item(args.dataset, spec, r) for r in rows]
    print(f"{args.dataset}: {len(items)} items from {spec['id']} [{spec['split']}] "
          f"({spec['kind']}), concurrency {args.concurrency}\n")

    systems = [("so1", args.url, {}, "jev-latest")]
    if not args.skip_jev:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("set TYPESAFE_API_KEY or pass --skip-jev")
        systems.append(("jev-1.13.0", JEV_URL, {"Authorization": f"Bearer {key}"}, "jev-latest"))

    results: dict[str, list] = {}
    for name, url, headers, model in systems:
        def run_one(item, url=url, headers=headers, model=model):
            with httpx.Client() as client:
                return ask(client, url, headers, model, item[0], item[1])

        with futures.ThreadPoolExecutor(args.concurrency) as pool:
            answers = list(pool.map(run_one, items))
        graded = []
        for (_, _, gold), answer in zip((answers and items) or [], answers, strict=True):
            if spec["kind"] == "noul":
                p = answer["noul"]
                graded.append({"gold": gold, "pred": int(p >= 0.5), "p_yes": p,
                               "p_pred": max(p, 1 - p), "correct": int(p >= 0.5) == gold})
            else:
                value = sum(i * q for i, q in enumerate(answer["probabilities"].values()))
                pred = min(len(spec["levels"]) - 1, max(0, round(value)))
                graded.append({"gold": gold, "pred": pred, "value": value,
                               "p_pred": answer["probabilities"][str(pred)],
                               "correct": pred == gold, "abs_err": abs(value - gold)})
        results[name] = graded
        print(f"  {name}: done")

    print(f"\n{'metric':28}" + "".join(f"{n:>18}" for n in results))
    print("-" * (28 + 18 * len(results)))
    names = list(results)
    for label, fn in [
        ("accuracy", lambda g: f"{sum(r['correct'] for r in g)}/{len(g)} "
                               f"({100 * sum(r['correct'] for r in g) / len(g):.1f}%)"),
        ("95% CI", lambda g: "[{:.1f}, {:.1f}]".format(
            *(100 * x for x in wilson(sum(r["correct"] for r in g), len(g))))),
        ("ECE (lower better)", lambda g: f"{ece([(r['p_pred'], r['correct']) for r in g]):.3f}"),
        ("Brier (lower better)",
         lambda g: f"{statistics.mean((r['p_pred'] - r['correct']) ** 2 for r in g):.3f}"),
    ]:
        print(f"{label:28}" + "".join(f"{fn(results[n]):>18}" for n in names))
    if spec["kind"] == "score":
        for label, key in [("mean |score - gold|", "abs_err")]:
            print(f"{label:28}" + "".join(f"{statistics.mean(r[key] for r in results[n]):>18.3f}" for n in names))
        print(f"{'within 1 level':28}"
              + "".join(f"{100 * sum(r['abs_err'] <= 1 for r in results[n]) / len(results[n]):>17.1f}%"
                        for n in names))

    if len(names) == 2:
        a, b = results[names[0]], results[names[1]]
        b_only = sum(1 for x, y in zip(a, b, strict=True) if not x["correct"] and y["correct"])
        a_only = sum(1 for x, y in zip(a, b, strict=True) if x["correct"] and not y["correct"])
        n_disc = a_only + b_only
        # exact binomial two-sided p for McNemar
        p = min(1.0, 2 * sum(math.comb(n_disc, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n_disc) \
            if n_disc else 1.0
        print(f"\nMcNemar: {names[0]} only {a_only}, {names[1]} only {b_only}, p = {p:.4f}"
              f"  -> {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    out = args.out or ROOT / "results" / f"public_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": spec["id"], "split": spec["split"], "n": len(items),
                               "results": results}, indent=1) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
