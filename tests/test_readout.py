"""Label readout, variant folding, calibration and the two confidence statistics."""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from so1.readout import (
    calibrate,
    choice_confidence,
    expected_score,
    label_distribution,
    normalise_token,
    score_confidence,
)

SAMPLES = json.loads((pathlib.Path(__file__).parent / "fixtures" / "jev_confidence_samples.json").read_text())[
    "samples"
]


def lp(p: float) -> float:
    return math.log(p) if p > 0 else -60.0


# ------------------------------------------------------------------ distribution


def test_reads_a_clean_two_label_distribution():
    probs, coverage = label_distribution([("Yes", lp(0.8)), ("No", lp(0.2))], ["Yes", "No"])
    assert probs == pytest.approx([0.8, 0.2])
    assert coverage == pytest.approx(1.0)


def test_sums_label_variants_and_renormalises():
    """' a.', 'A' and 'A)' are all the same answer."""
    top = [("A", lp(0.30)), (" a.", lp(0.20)), ("A)", lp(0.10)), ("B", lp(0.20)), ("The", lp(0.20))]
    probs, coverage = label_distribution(top, ["A", "B"])
    assert probs == pytest.approx([0.75, 0.25])
    assert coverage == pytest.approx(0.8)  # 0.2 of the mass was not a label


def test_missing_labels_score_zero():
    probs, _ = label_distribution([("A", lp(1.0))], ["A", "B", "C"])
    assert probs == pytest.approx([1.0, 0.0, 0.0])


def test_no_label_in_the_top_k_falls_back_to_uniform_with_zero_coverage():
    probs, coverage = label_distribution([("The", lp(0.9)), ("A friendly", lp(0.1))], ["A", "B"])
    assert probs == pytest.approx([0.5, 0.5])
    assert coverage == 0.0


def test_labels_that_normalise_apart_do_not_collide():
    probs, _ = label_distribution([("A", lp(0.5)), ("AA", lp(0.5))], ["A", "AA"])
    assert probs == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(("raw", "expected"), [(" A ", "a"), ("A.", "a"), ("B)", "b"), ("yes:", "yes"), ("No", "no")])
def test_token_normalisation(raw, expected):
    assert normalise_token(raw) == expected


# ------------------------------------------------------------------ calibration


def test_temperature_one_is_a_no_op():
    assert calibrate([0.7, 0.3], 1.0) == [0.7, 0.3]


def test_higher_temperature_flattens_and_lower_sharpens():
    base = [0.7, 0.2, 0.1]
    flat, sharp = calibrate(base, 2.0), calibrate(base, 0.5)
    assert max(flat) < max(base) < max(sharp)
    assert sum(flat) == pytest.approx(1.0)
    assert sum(sharp) == pytest.approx(1.0)


def test_calibration_preserves_order_and_handles_zeros():
    out = calibrate([0.6, 0.4, 0.0], 3.0)
    assert out[0] > out[1] > out[2] == 0.0
    assert sum(out) == pytest.approx(1.0)


def test_calibration_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError):
        calibrate([0.5, 0.5], 0.0)


# ------------------------------------------------------------------ confidence


def test_choice_confidence_endpoints():
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert choice_confidence([0.88, 0.12, 0.0]) == pytest.approx(0.82)


def test_score_confidence_punishes_a_split_verdict_even_with_a_tall_peak():
    """A 0.7/0.26 split across opposite ends of the rubric is not a confident answer."""
    bimodal = [0.7, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.26]
    assert score_confidence(bimodal) == pytest.approx(0.0, abs=0.02)
    assert choice_confidence(bimodal) > 0.6  # the nominal statistic would call this confident


def test_score_and_choice_confidence_agree_for_two_levels():
    for p in (0.5, 0.52, 0.75, 1.0):
        assert score_confidence([p, 1 - p]) == pytest.approx(choice_confidence([p, 1 - p]))


def test_single_option_is_certain():
    assert choice_confidence([1.0]) == 1.0
    assert score_confidence([1.0]) == 1.0


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: f"{s['kind']}-k{s['K']}")
def test_confidence_matches_the_real_jev(sample):
    """Both formulas were derived from these live responses; residual is their 2-dp rounding."""
    k = sample["K"]
    if sample["kind"] == "score":
        probs = [sample["probs"][str(i)] for i in range(k)]
        got = score_confidence(probs)
    else:
        probs = list(sample["probs"].values())
        got = choice_confidence(probs)
    assert got == pytest.approx(sample["conf"], abs=0.07)


def test_confidence_matches_the_real_jev_closely_on_average():
    errors = []
    for sample in SAMPLES:
        k = sample["K"]
        if sample["kind"] == "score":
            probs = [sample["probs"][str(i)] for i in range(k)]
            errors.append(abs(score_confidence(probs) - sample["conf"]))
        else:
            errors.append(abs(choice_confidence(list(sample["probs"].values())) - sample["conf"]))
    assert sum(errors) / len(errors) < 0.01


# ------------------------------------------------------------------ score value


def test_expected_score_is_zero_based():
    assert expected_score([1.0, 0.0, 0.0]) == 0.0
    assert expected_score([0.0, 0.0, 1.0]) == 2.0
    assert expected_score([0.0, 0.95, 0.05]) == pytest.approx(1.05)
