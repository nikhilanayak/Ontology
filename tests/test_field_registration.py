import cv2
import numpy as np

from all22.field_registration import OCRNumber, register_field
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
