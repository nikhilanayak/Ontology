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
