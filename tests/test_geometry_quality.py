import numpy as np
import cv2

from all22.geometry import FIELD_WIDTH, estimate_homography, project_points, semantic_correspondences
from all22.models import PlayStatus
from all22.quality import evaluate
from all22.models import Angle, Clip
from all22.video import classify_angle, coalesce_short_fragments


def test_homography():
    image = [(0, 0), (100, 0), (100, 50), (0, 50)]
    field = [(10, 0), (20, 0), (20, 10), (10, 10)]
    calibration = estimate_homography(image, field)
    result = project_points(calibration.matrix, [(50, 25)])[0]
    assert np.allclose(result, [15, 5], atol=0.01)


def test_two_semantic_yard_lines_create_metric_local_calibration():
    image, field, metadata = semantic_correspondences([
        {"kind": "line", "image_points": [[100, 100], [100, 500]]},
        {"kind": "line", "image_points": [[300, 100], [300, 500]]},
    ])
    assert metadata == {"mode": "lines", "absolute_x": False, "line_count": 2}
    calibration = estimate_homography(image, field)
    result = project_points(calibration.matrix, [[200, 300]])[0]
    assert np.allclose(result, [60, FIELD_WIDTH / 2], atol=.05)


def test_semantic_field_numbers_infer_rows_and_yard_values():
    annotations = []
    for value, x in ((20, 30), (40, 50), (50, 60)):
        for y in (12, FIELD_WIDTH - 12):
            annotations.append({"kind": "number", "value": value, "image_point": [x * 10, y * 8]})
    image, field, metadata = semantic_correspondences(annotations)
    assert metadata["mode"] == "numbers"
    calibration = estimate_homography(image, field)
    projected = project_points(calibration.matrix, [[400, 12 * 8]])[0]
    assert np.isclose(projected[0], 40, atol=.1)
    assert any(np.isclose(projected[1], value, atol=.1) for value in (12, FIELD_WIDTH - 12))


def test_quality_is_fail_closed():
    identities = [("home", value) for value in range(1, 12)] + [("away", value) for value in range(20, 31)]
    good = evaluate("p", {"segmentation_confidence": .95, "alignment_confidence": .99,
                          "median_field_error_yards": .5, "p95_field_error_yards": 1.2,
                          "track_coverage": .98, "angle_agreement": .95, "identity_confidence": .99}, identities)
    assert good.status == PlayStatus.ACCEPTED
    bad = evaluate("p", {"median_field_error_yards": 2, "p95_field_error_yards": 3,
                         "track_coverage": .5, "angle_agreement": .5, "identity_confidence": .5}, identities[:-1])
    assert bad.status == PlayStatus.REJECTED
    assert len(bad.reasons) >= 5


def test_angle_classification_from_field_lines():
    horizontal = np.zeros((500, 900, 3), dtype=np.uint8)
    vertical = horizontal.copy()
    for y in range(80, 450, 80):
        cv2.line(horizontal, (20, y), (880, y), (255, 255, 255), 8)
    for x in range(80, 850, 100):
        cv2.line(vertical, (x, 20), (x, 480), (255, 255, 255), 8)
    assert classify_angle(horizontal)[0] == Angle.ENDZONE
    assert classify_angle(vertical)[0] == Angle.SIDELINE


def test_short_same_angle_fragments_are_coalesced():
    clips = [
        Clip("g", "a", Angle.SIDELINE, 0, 3),
        Clip("g", "b", Angle.SIDELINE, 3, 15),
        Clip("g", "c", Angle.ENDZONE, 15, 25),
    ]
    merged = coalesce_short_fragments(clips)
    assert [(clip.clip_id, clip.start_s, clip.end_s) for clip in merged] == [
        ("a", 0, 15), ("c", 15, 25)
    ]
