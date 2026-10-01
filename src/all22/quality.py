from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .models import PlayStatus, QualityReport


@dataclass(frozen=True)
class QualityThresholds:
    minimum_segmentation_confidence: float = 0.90
    minimum_alignment_confidence: float = 0.98
    max_median_field_error_yards: float = 1.0
    max_p95_field_error_yards: float = 2.0
    minimum_track_coverage: float = 0.95
    minimum_angle_agreement: float = 0.90
    minimum_identity_confidence: float = 0.97


def evaluate(play_id: str, metrics: Mapping[str, float], identities: Iterable[tuple[str, int]],
             thresholds: QualityThresholds = QualityThresholds()) -> QualityReport:
    report = QualityReport(play_id=play_id, metrics=dict(metrics))
    identity_list = list(identities)
    teams = {}
    for team, jersey in identity_list:
        teams.setdefault(team, set()).add(jersey)
    if sorted(len(values) for values in teams.values()) != [11, 11]:
        report.reject("expected exactly 11 unique jersey identities per team at snap")
    if metrics.get("segmentation_confidence", 0.0) < thresholds.minimum_segmentation_confidence:
        report.reject("snap/end segmentation confidence below threshold")
    if metrics.get("alignment_confidence", 0.0) < thresholds.minimum_alignment_confidence:
        report.reject("PFR/film alignment confidence below threshold")
    if metrics.get("median_field_error_yards", float("inf")) > thresholds.max_median_field_error_yards:
        report.reject("median field calibration error exceeded threshold")
    if metrics.get("p95_field_error_yards", float("inf")) > thresholds.max_p95_field_error_yards:
        report.reject("p95 field calibration error exceeded threshold")
    if metrics.get("track_coverage", 0.0) < thresholds.minimum_track_coverage:
        report.reject("track coverage below threshold")
    if metrics.get("angle_agreement", 0.0) < thresholds.minimum_angle_agreement:
        report.reject("sideline/endzone agreement below threshold")
    if metrics.get("identity_confidence", 0.0) < thresholds.minimum_identity_confidence:
        report.reject("identity confidence below threshold")
    if not report.reasons:
        report.status = PlayStatus.ACCEPTED
    return report
