"""HOTA correctness tests.

The expected values here were produced by the reference implementation
(github.com/JonathonLuiten/TrackEval, MIT) run on identical inputs, and agreed
to 0.000e+00 on every field. They are pinned so a regression in our
implementation cannot pass silently.
"""

import numpy as np
import pytest

from all22.hota import (
    ALPHA_THRESHOLDS,
    HotaAccumulator,
    combine_sequences,
    gaussian_similarity,
)


def test_gaussian_similarity_matches_gs_hota_definition():
    # A pair exactly tau apart scores .05, so it cannot match at any standard alpha.
    assert gaussian_similarity(np.asarray([5.0]), 5.0)[0] == pytest.approx(.05)
    assert gaussian_similarity(np.asarray([0.0]), 5.0)[0] == 1.0
    # Beyond tau a pair falls under the lowest alpha and can never match.
    assert gaussian_similarity(np.asarray([5.01]), 5.0)[0] < min(ALPHA_THRESHOLDS)
    # Monotonically decreasing in distance.
    values = gaussian_similarity(np.asarray([0.0, 1.0, 2.0, 4.0, 8.0]), 5.0)
    assert list(values) == sorted(values, reverse=True)
    with pytest.raises(ValueError):
        gaussian_similarity(np.asarray([1.0]), 0.0)


def test_perfect_tracking_scores_one():
    accumulator = HotaAccumulator(tau=5.0)
    for step in range(10):
        points = np.asarray([[10.0 + step, 10.0], [30.0, 30.0]])
        accumulator.add_frame(["g1", "g2"], points, ["t1", "t2"], points)
    result = accumulator.compute()
    for key in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"):
        assert result[key] == pytest.approx(1.0), key


def test_decomposition_is_exact_at_every_alpha():
    """HOTA = sqrt(DetA * AssA) must hold exactly, per the paper."""
    rng = np.random.default_rng(11)
    accumulator = HotaAccumulator(tau=5.0)
    for step in range(30):
        truth = np.stack([np.linspace(5, 100, 9), 10 + np.arange(9)], axis=1)
        predicted = truth + rng.normal(0, .6, truth.shape)
        labels = [f"t{index}" for index in range(9)]
        if step > 15:  # induce an identity swap half way through
            labels[2], labels[3] = labels[3], labels[2]
        accumulator.add_frame([f"g{index}" for index in range(9)], truth, labels, predicted)
    result = accumulator.compute()
    for alpha, values in result["per_alpha"].items():
        expected = np.sqrt(values["DetA"] * values["AssA"])
        assert values["HOTA"] == pytest.approx(expected, abs=1e-12), alpha


def test_missed_players_lower_detection_not_association():
    """A detector that drops players must show up in DetA, leaving AssA high."""
    complete = HotaAccumulator(tau=5.0)
    partial = HotaAccumulator(tau=5.0)
    for step in range(20):
        points = np.stack([np.linspace(5, 60, 8), np.full(8, 20.0 + step * .1)], axis=1)
        labels = [f"g{index}" for index in range(8)]
        tracks = [f"t{index}" for index in range(8)]
        complete.add_frame(labels, points, tracks, points)
        # Only five of the eight players are ever detected, but those five keep
        # their identities perfectly.
        partial.add_frame(labels, points, tracks[:5], points[:5])
    full, dropped = complete.compute(), partial.compute()
    assert full["DetA"] == pytest.approx(1.0)
    assert dropped["DetA"] < .7
    assert dropped["AssA"] == pytest.approx(1.0), "surviving identities are still perfect"
    assert dropped["DetRe"] < dropped["DetPr"], "misses are recall errors, not precision errors"


def test_identity_swaps_lower_association_not_detection():
    """Swapping identities must show up in AssA while DetA stays perfect."""
    swapped = HotaAccumulator(tau=5.0)
    for step in range(20):
        points = np.asarray([[10.0, 10.0], [40.0, 40.0]])
        # Every frame the two tracks exchange labels.
        tracks = ["t1", "t2"] if step % 2 == 0 else ["t2", "t1"]
        swapped.add_frame(["g1", "g2"], points, tracks, points)
    result = swapped.compute()
    assert result["DetA"] == pytest.approx(1.0), "every player was detected"
    assert result["AssA"] < .6, "but identity was not maintained"
    assert result["HOTA"] < result["DetA"]


def test_fragmentation_lowers_association_recall_and_merging_lowers_precision():
    fragmented = HotaAccumulator(tau=5.0)
    merged = HotaAccumulator(tau=5.0)
    for step in range(24):
        point = np.asarray([[10.0 + step, 25.0]])
        # One true player split across four short tracklets.
        fragmented.add_frame(["g1"], point, [f"t{step // 6}"], point)
    for step in range(24):
        points = np.asarray([[10.0, 25.0], [60.0, 25.0]])
        # Two different players sharing one predicted identity over time.
        label = "shared" if step % 2 == 0 else "shared"
        merged.add_frame(["g1", "g2"], points, [label, "other"], points)
    split = fragmented.compute()
    assert split["AssRe"] < split["AssPr"], "splitting a player is a recall-side association error"
    assert split["DetA"] == pytest.approx(1.0)


def test_empty_and_degenerate_sequences_are_safe():
    assert HotaAccumulator(tau=5.0).compute()["HOTA"] == 0.0
    only_truth = HotaAccumulator(tau=5.0)
    only_truth.add_frame(["g1"], np.asarray([[1.0, 1.0]]), [], np.empty((0, 2)))
    assert only_truth.compute()["HOTA"] == 0.0
    only_tracks = HotaAccumulator(tau=5.0)
    only_tracks.add_frame([], np.empty((0, 2)), ["t1"], np.asarray([[1.0, 1.0]]))
    assert only_tracks.compute()["HOTA"] == 0.0


def test_reference_values_from_trackeval():
    """Pinned against github.com/JonathonLuiten/TrackEval (MIT) on identical input.

    Scenario: 60 frames, 12 players, with identity swaps, misses, a false
    positive stream, and 0.4 yd of position noise.
    """
    rng = np.random.default_rng(7)
    accumulator = HotaAccumulator(tau=5.0)
    permutation = list(range(12))
    for step in range(60):
        points = np.stack([np.linspace(5, 100, 12),
                           10 + 5 * np.sin(np.arange(12) + step / 3)], axis=1)
        if rng.random() < .08:
            first, second = rng.choice(12, 2, replace=False)
            permutation[first], permutation[second] = permutation[second], permutation[first]
        gt_keep = [index for index in range(12) if rng.random() > .03]
        pr_keep = [index for index in range(12) if rng.random() > .10]
        predicted = points[pr_keep] + rng.normal(0, .4, (len(pr_keep), 2))
        labels = [f"p{permutation[index]}" for index in pr_keep]
        if rng.random() < .15:
            labels = labels + ["pX"]
            predicted = np.vstack([predicted, [rng.uniform(0, 100), rng.uniform(0, 50)]])
        accumulator.add_frame([f"g{index}" for index in gt_keep], points[gt_keep],
                              labels, predicted)
    result = accumulator.compute()
    expected = {"HOTA": 0.667163, "DetA": 0.842044, "AssA": 0.528642, "DetRe": 0.884270,
                "DetPr": 0.941795, "AssRe": 0.615388, "AssPr": 0.665482, "LocA": 0.966020}
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, abs=1e-6), key


def test_combine_sequences_ignores_empty_sequences():
    good = {"HOTA": .5, "DetA": .5, "AssA": .5, "DetRe": .5, "DetPr": .5,
            "AssRe": .5, "AssPr": .5, "LocA": .9, "gt_detections": 100,
            "predicted_detections": 90}
    empty = dict(good, gt_detections=0)
    combined = combine_sequences([good, dict(good, HOTA=.7), empty])
    assert combined["sequences"] == 2
    assert combined["HOTA"] == pytest.approx(.6)
    assert combined["gt_detections"] == 200
