"""HOTA for field-coordinate tracking.

Implements Higher Order Tracking Accuracy (Luiten et al., IJCV 2021) rather than
a bespoke score. Two properties matter here:

* HOTA integrates over a localization threshold alpha, so no single distance
  gate decides whether a pair counts. A hard gate was previously doing much of
  the work in our own metric: tightening it from 8 to 2 yards moved "purity"
  from .457 to .623 with no change to the tracker at all.
* It decomposes exactly as ``HOTA = sqrt(DetA * AssA)``, which separates "players
  we failed to detect" from "identities we failed to keep". Those are different
  problems with different fixes, and a single blended number hides which one
  moved.

Detections here are points on the field, not image boxes, so IoU is unavailable.
Following SoccerNet Game State Reconstruction (arXiv:2404.11335) the similarity
is a Gaussian in field distance, calibrated so that a pair exactly ``tau`` apart
scores 0.05 and therefore cannot match at any standard alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# The standard TrackEval alpha grid: 0.05 to 0.95 in steps of 0.05.
ALPHA_THRESHOLDS = tuple(np.round(np.arange(.05, .96, .05), 2))
# Similarity at exactly tau. Matches SoccerNet GS-HOTA.
SIMILARITY_AT_TAU = .05


def gaussian_similarity(distances: np.ndarray, tau: float) -> np.ndarray:
    """Map field distances to a similarity in [0, 1].

    ``exp(ln(0.05) * d^2 / tau^2)`` so that ``similarity(tau) == 0.05``: a pair
    separated by more than tau cannot be matched at any standard alpha, while
    closer pairs survive more thresholds and so score higher.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    scaled = np.asarray(distances, dtype=float) ** 2 / float(tau) ** 2
    return np.exp(np.log(SIMILARITY_AT_TAU) * scaled)


@dataclass
class HotaAccumulator:
    """Accumulate HOTA counts over the frames of one sequence.

    Ground-truth and predicted identities are arbitrary hashable labels; the
    caller supplies per-frame positions in field coordinates.
    """

    tau: float
    alphas: Sequence[float] = ALPHA_THRESHOLDS
    gt_ids: Dict[object, int] = field(default_factory=dict)
    pr_ids: Dict[object, int] = field(default_factory=dict)
    frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)

    def _index(self, table: Dict[object, int], key: object) -> int:
        if key not in table:
            table[key] = len(table)
        return table[key]

    def add_frame(self, gt_labels: Sequence, gt_points: np.ndarray,
                  pr_labels: Sequence, pr_points: np.ndarray) -> None:
        gt_index = np.asarray([self._index(self.gt_ids, label) for label in gt_labels], dtype=int)
        pr_index = np.asarray([self._index(self.pr_ids, label) for label in pr_labels], dtype=int)
        if len(gt_index) and len(pr_index):
            gt_points = np.asarray(gt_points, dtype=float).reshape(-1, 2)
            pr_points = np.asarray(pr_points, dtype=float).reshape(-1, 2)
            distances = np.linalg.norm(gt_points[:, None, :] - pr_points[None, :, :], axis=2)
            similarity = gaussian_similarity(distances, self.tau)
        else:
            similarity = np.zeros((len(gt_index), len(pr_index)), dtype=float)
        self.frames.append((gt_index, pr_index, similarity))

    def compute(self) -> dict:
        """Score the accumulated sequence.

        Association is global over the sequence: pairwise alignment is computed
        first, then one assignment per frame is biased by it and thresholded at
        each alpha. This mirrors the reference implementation in TrackEval.
        """
        gt_count, pr_count = len(self.gt_ids), len(self.pr_ids)
        total_gt = sum(len(frame[0]) for frame in self.frames)
        total_pr = sum(len(frame[1]) for frame in self.frames)
        if not self.frames or not total_gt or not total_pr:
            return _empty_result(self.alphas)
        gt_totals = np.zeros((gt_count, 1), dtype=float)
        pr_totals = np.zeros((1, pr_count), dtype=float)
        potential = np.zeros((gt_count, pr_count), dtype=float)
        for gt_index, pr_index, similarity in self.frames:
            if similarity.size:
                denominator = (similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis]
                               - similarity)
                ratio = np.zeros_like(similarity)
                mask = denominator > np.finfo(float).eps
                ratio[mask] = similarity[mask] / denominator[mask]
                potential[gt_index[:, np.newaxis], pr_index[np.newaxis, :]] += ratio
            gt_totals[gt_index] += 1
            pr_totals[0, pr_index] += 1
        alignment = potential / np.maximum(gt_totals + pr_totals - potential, 1e-10)
        alphas = list(self.alphas)
        true_positives = np.zeros(len(alphas), dtype=float)
        false_negatives = np.zeros(len(alphas), dtype=float)
        false_positives = np.zeros(len(alphas), dtype=float)
        localization = np.zeros(len(alphas), dtype=float)
        matches = [np.zeros_like(potential) for _ in alphas]
        for gt_index, pr_index, similarity in self.frames:
            if not len(gt_index):
                false_positives += len(pr_index)
                continue
            if not len(pr_index):
                false_negatives += len(gt_index)
                continue
            score = alignment[gt_index[:, np.newaxis], pr_index[np.newaxis, :]] * similarity
            rows, columns = linear_sum_assignment(-score)
            chosen = similarity[rows, columns]
            for position, alpha in enumerate(alphas):
                keep = chosen >= alpha - np.finfo(float).eps
                count = int(keep.sum())
                true_positives[position] += count
                false_negatives[position] += len(gt_index) - count
                false_positives[position] += len(pr_index) - count
                if count:
                    localization[position] += float(chosen[keep].sum())
                    matches[position][gt_index[rows[keep]], pr_index[columns[keep]]] += 1
        per_alpha = {}
        for position, alpha in enumerate(alphas):
            count = true_positives[position]
            matched = matches[position]
            safe = max(count, 1.0)
            association = float((matched * (matched / np.maximum(
                gt_totals + pr_totals - matched, 1.0))).sum() / safe)
            association_recall = float((matched * (matched / np.maximum(gt_totals, 1.0))).sum() / safe)
            association_precision = float((matched * (matched / np.maximum(pr_totals, 1.0))).sum() / safe)
            denominator = count + false_negatives[position] + false_positives[position]
            detection = float(count / denominator) if denominator else 0.0
            per_alpha[f"{alpha:.2f}"] = {
                "HOTA": float(np.sqrt(detection * association)), "DetA": detection,
                "AssA": association,
                "DetRe": float(count / max(float(gt_totals.sum()), 1e-10)),
                "DetPr": float(count / max(float(pr_totals.sum()), 1e-10)),
                "AssRe": association_recall, "AssPr": association_precision,
                "LocA": float(max(localization[position], 1e-10) / max(count, 1e-10)),
                "true_positives": int(count), "false_negatives": int(false_negatives[position]),
                "false_positives": int(false_positives[position])}
        keys = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")
        summary = {key: float(np.mean([value[key] for value in per_alpha.values()])) for key in keys}
        summary.update({
            "gt_ids": gt_count, "predicted_ids": pr_count,
            "gt_detections": int(total_gt), "predicted_detections": int(total_pr),
            "true_positives": int(true_positives.sum() / len(alphas)),
            "tau": self.tau, "per_alpha": per_alpha,
        })
        return summary


def _empty_result(alphas: Sequence[float]) -> dict:
    return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "DetRe": 0.0, "DetPr": 0.0,
            "AssRe": 0.0, "AssPr": 0.0, "LocA": 0.0, "gt_ids": 0, "predicted_ids": 0,
            "gt_detections": 0, "predicted_detections": 0, "tau": 0.0, "per_alpha": {}}


def combine_sequences(results: Sequence[dict]) -> dict:
    """Aggregate per-sequence HOTA fields.

    Reported as an unweighted mean over sequences so one long clip cannot
    dominate, and alongside the per-sequence values rather than instead of them.
    """
    usable = [value for value in results if value.get("gt_detections")]
    if not usable:
        return _empty_result(ALPHA_THRESHOLDS)
    keys = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")
    combined = {key: float(np.mean([value[key] for value in usable])) for key in keys}
    combined["sequences"] = len(usable)
    combined["gt_detections"] = int(sum(value["gt_detections"] for value in usable))
    combined["predicted_detections"] = int(sum(value["predicted_detections"] for value in usable))
    return combined
