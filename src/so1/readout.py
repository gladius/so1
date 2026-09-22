"""Turning one forward pass into a probability distribution over answer labels.

Confidence is NOT the entropy statistic: it was reverse-engineered from 66 live Jev
responses (see README). Choice uses the scaled peak probability; Score uses an
ordinal spread around the most likely level. They agree when K == 2.
"""

from __future__ import annotations

import math

Distribution = list[float]


def normalise_token(token: str) -> str:
    """Fold the shapes a label comes back in: ' a.', 'A)', 'yes' ..."""
    return token.strip().lower().rstrip(".:)")


def label_distribution(top: list[tuple[str, float]], labels: list[str]) -> tuple[Distribution, float]:
    """Sum probability over label variants in a top-k logprob list, then normalise.

    Returns (probabilities, coverage) where coverage is the share of the returned mass that
    landed on a label. In exact mode the server has already restricted the candidates to the
    labels, so coverage is ~1; in fallback mode it is the diagnostic that says whether the
    top-k was wide enough. Neither OpenAI-compatible endpoint returns token ids, so labels are
    matched by their decoded text.
    """
    wanted = {label.lower(): i for i, label in enumerate(labels)}
    mass = [0.0] * len(labels)
    for token, logprob in top:
        index = wanted.get(normalise_token(token))
        if index is not None:
            mass[index] += math.exp(logprob)
    coverage = math.fsum(mass)
    if coverage <= 0.0:
        return [1.0 / len(labels)] * len(labels), 0.0
    return [m / coverage for m in mass], min(coverage, 1.0)


def calibrate(probs: Distribution, temperature: float) -> Distribution:
    """Temperature-scale a distribution: logits = log p, divide, softmax."""
    if temperature == 1.0 or len(probs) < 2:
        return list(probs)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = [(math.log(p) / temperature if p > 0.0 else -math.inf) for p in probs]
    ceiling = max(logits)
    if ceiling == -math.inf:
        return [1.0 / len(probs)] * len(probs)
    exponentials = [math.exp(x - ceiling) for x in logits]
    total = math.fsum(exponentials)
    return [x / total for x in exponentials]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def choice_confidence(probs: Distribution) -> float:
    """(K * p_max - 1) / (K - 1): 1 when all mass is on one option, 0 when uniform."""
    k = len(probs)
    if k < 2:
        return 1.0
    return _clamp((k * max(probs) - 1.0) / (k - 1.0))


def _uniform_spread(k: int) -> float:
    """Mean |level - centre| of a uniform distribution over k ordered levels."""
    centre = (k - 1) / 2
    return math.fsum(abs(i - centre) for i in range(k)) / k


def score_confidence(probs: Distribution) -> float:
    """1 - E|level - mode| / uniform spread. Bimodal answers score low even with a tall peak."""
    k = len(probs)
    if k < 2:
        return 1.0
    mode = max(range(k), key=probs.__getitem__)
    spread = math.fsum(p * abs(i - mode) for i, p in enumerate(probs))
    return _clamp(1.0 - spread / _uniform_spread(k))


def expected_score(probs: Distribution) -> float:
    """Zero-based probability-weighted level."""
    return math.fsum(i * p for i, p in enumerate(probs))
