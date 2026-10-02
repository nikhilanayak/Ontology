import cv2
import numpy as np

from all22.field_registration import OCRNumber, propagate_registration, read_field_numbers, register_field
from all22.geometry import project_points


def test_line_detection_and_ocr_anchor_fit_field_template():
    frame = np.zeros((600, 1100, 3), np.uint8)
    frame[:] = (35, 115, 45)
    xs = list(range(150, 951, 100))
    for x in xs:
        cv2.line(frame, (x, 40), (x, 560), (245, 245, 245), 8)
    cv2.line(frame, (40, 40), (1060, 40), (245, 245, 245), 8)
    cv2.line(frame, (40, 560), (1060, 560), (245, 245, 245), 8)
    numbers = [OCRNumber(50, (550, 150), .99)]
    registration = register_field(frame, numbers=numbers)
    assert registration.absolute_x
    assert registration.diagnostics["line_count"] >= 7
    center = project_points(registration.matrix, [[550, 300]])[0]
    assert np.allclose(center, [60, 80 / 3], atol=1.0)


def test_ocr_normalizes_goal_arrows_and_upside_down_ten():
    class Reader:
        def readtext(self, *_args, **_kwargs):
            box = [[0, 0], [20, 0], [20, 10], [0, 10]]
            return [(box, "205", .9), (box, "305", .8), (box, "01", .7), (box, "22", .9)]

    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:] = (40, 120, 40)
    assert [value.value for value in read_field_numbers(frame, Reader())] == [20, 30, 10]


def test_registration_propagates_through_camera_motion():
    rng = np.random.default_rng(4)
    source = np.zeros((500, 800, 3), np.uint8);source[:] = (35, 115, 45)
    for x in range(100, 701, 100): cv2.line(source, (x, 30), (x, 470), (245, 245, 245), 6)
    for _ in range(100):
        x, y = rng.integers([30, 30], [770, 470]);cv2.circle(source, (int(x), int(y)), 3, (80, 150, 80), -1)
    registration = register_field(source, numbers=[OCRNumber(50, (400, 120), .99)])
    motion = np.float32([[1, 0, 25], [0, 1, 12]])
    target = cv2.warpAffine(source, motion, (800, 500), borderValue=(35, 115, 45))
    propagated = propagate_registration(source, target, registration)
    expected = project_points(registration.matrix, [[375, 238]])[0]
    actual = project_points(propagated.matrix, [[400, 250]])[0]
    assert np.allclose(actual, expected, atol=.5)
    assert propagated.diagnostics["temporal_inlier_ratio"] > .5


def test_automatic_registration_does_not_anchor_from_one_jersey_number():
    frame = np.zeros((600, 1100, 3), np.uint8)
    frame[:] = (35, 115, 45)
    for x in range(150, 951, 100):
        cv2.line(frame, (x, 40), (x, 560), (245, 245, 245), 8)

    class Reader:
        def readtext(self, *_args, **_kwargs):
            return [([[490, 250], [530, 250], [530, 290], [490, 290]], "50", .99)]

    registration = register_field(frame, reader=Reader())
    assert not registration.absolute_x
    assert registration.diagnostics["ocr_numbers_raw"] == 1
    assert registration.diagnostics["ocr_numbers"] == 0


def test_verification_accepts_a_plausible_broadcast_homography():
    from all22.field_registration import verify_registration
    # 20 px per yard, field origin near the top-left of a 1920x1080 frame.
    matrix = np.asarray([[1 / 20, 0, 30.0], [0, 1 / 20, 5.0], [0, 0, 1.0]])
    report = verify_registration(matrix, (1080, 1920, 3))
    assert report["verified"], report["reasons"]
    assert 15 < report["pixels_per_yard_x"] < 25


def test_verification_rejects_physically_impossible_registrations():
    """Residuals cannot catch a confident fit placed on the wrong yard lines."""
    from all22.field_registration import verify_registration
    shape = (1080, 1920, 3)
    # Absurd scale: the whole stadium inside one yard.
    tiny = np.asarray([[1 / 5000, 0, 50.0], [0, 1 / 5000, 25.0], [0, 0, 1.0]])
    assert not verify_registration(tiny, shape)["verified"]
    # Absurd scale the other way: a yard covering the frame.
    huge = np.asarray([[5.0, 0, 0.0], [0, 5.0, 0.0], [0, 0, 1.0]])
    assert not verify_registration(huge, shape)["verified"]
    # Projects far off the side of the field.
    beside = np.asarray([[1 / 20, 0, 0.0], [0, 1 / 20, 900.0], [0, 0, 1.0]])
    report = verify_registration(beside, shape)
    assert not report["verified"]
    assert any("off the side" in reason or "spans" in reason for reason in report["reasons"])
    # Wildly anisotropic: 1 yd horizontally, 40 yd vertically per pixel.
    skewed = np.asarray([[1 / 20, 0, 10.0], [0, 1 / 2, 5.0], [0, 0, 1.0]])
    assert not verify_registration(skewed, shape)["verified"]
    # Singular.
    assert not verify_registration(np.zeros((3, 3)), shape)["verified"]


def test_field_line_evidence_collapses_hough_fragments_into_fitted_lines():
    from all22.field_registration import _field_mask, detect_field_line_evidence, detect_yard_lines
    frame = np.full((500, 900, 3), (60, 140, 70), dtype=np.uint8)
    # Six thick lines: Hough sees two edges and multiple fragments for each.
    for x in range(120, 800, 120):
        cv2.line(frame, (x, 40), (x, 460), (245, 245, 245), 7)
    lines, mask, white = detect_yard_lines(frame)
    evidence = detect_field_line_evidence(frame, mask, white, lines)
    assert evidence["raw_count"] > len(evidence["fitted"])
    cross = [item for item in evidence["fitted"] if item["kind"] == "cross_field"]
    assert 4 <= len(cross) <= 8
    assert all(item["length_pixels"] > 350 for item in cross)
