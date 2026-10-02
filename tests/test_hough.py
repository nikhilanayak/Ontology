import cv2
import numpy as np
import pytest

from all22 import hough


def frame_with_lines() -> np.ndarray:
    """A dark frame with three bright straight lines: two parallel, one crossing."""
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.line(image, (60, 40), (60, 320), (255, 255, 255), 3)
    cv2.line(image, (200, 40), (200, 320), (255, 255, 255), 3)
    cv2.line(image, (40, 300), (600, 300), (255, 255, 255), 3)
    return image


def test_parameters_are_clamped_and_blur_stays_odd():
    values = hough.HoughParameters(blur=4, canny_low=200, canny_high=50, threshold=0,
                                   min_line_length=0, max_line_gap=-5).normalized()
    assert values.blur == 5
    assert values.canny_low < values.canny_high
    assert (values.threshold, values.min_line_length, values.max_line_gap) == (1, 1, 0)


def test_detect_lines_finds_the_drawn_geometry():
    segments = hough.detect_lines(frame_with_lines())
    assert len(segments) >= 3
    assert segments.shape[1] == 4
    angles = hough.segment_angles_degrees(segments)
    assert np.any(np.abs(angles - 90) < 5)
    assert np.any((angles < 5) | (angles > 175))
    assert hough.segment_lengths(segments).max() > 200


def test_detect_lines_returns_an_empty_array_on_a_blank_frame():
    segments = hough.detect_lines(np.zeros((240, 320, 3), dtype=np.uint8))
    assert segments.shape == (0, 4)
    assert hough.segment_angles_degrees(segments).shape == (0,)
    assert hough.segment_lengths(segments).shape == (0,)


def test_a_higher_minimum_length_keeps_only_longer_lines():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.line(image, (40, 100), (600, 100), (255, 255, 255), 3)
    cv2.line(image, (40, 240), (110, 240), (255, 255, 255), 3)
    strict = hough.detect_lines(image, hough.HoughParameters(min_line_length=200, threshold=60))
    assert len(strict)
    assert hough.segment_lengths(strict).min() > 150


def test_annotate_draws_without_mutating_the_source_frame():
    source = frame_with_lines()
    original = source.copy()
    annotated = hough.annotate(source, hough.detect_lines(source))
    assert annotated.shape == source.shape
    assert np.array_equal(source, original)
    channel_spread = annotated.max(axis=2).astype(int) - annotated.min(axis=2).astype(int)
    assert channel_spread.max() > 40


def test_annotate_accepts_no_segments():
    source = frame_with_lines()
    assert np.array_equal(hough.annotate(source, np.empty((0, 4), dtype=np.int32)), source)


def test_describe_reports_segments_and_diagnostics():
    payload = hough.describe(frame_with_lines())
    assert payload["width"] == 640 and payload["height"] == 360
    assert payload["count"] == len(payload["segments"]) >= 3
    assert payload["longest"] >= payload["median_length"] > 0
    assert set(payload["segments"][0]) == {"x1", "y1", "x2", "y2", "length", "angle"}
    assert payload["parameters"]["threshold"] == hough.HoughParameters().threshold


def test_yard_line_rows_keep_long_collinear_white_fragments():
    image = np.full((480, 800, 3), (55, 130, 55), dtype=np.uint8)
    for y in (100, 220, 340):
        for x1, x2 in ((40, 180), (215, 390), (430, 610), (650, 760)):
            cv2.line(image, (x1, y), (x2, y), (245, 245, 245), 4)
    cv2.rectangle(image, (300, 30), (360, 80), (245, 245, 245), 4)
    cv2.line(image, (50, 430), (120, 380), (20, 20, 220), 5)
    fragments, rows = hough.find_yard_line_rows(
        image,
        hough.HoughParameters(min_line_length=35, max_line_gap=10, threshold=30),
        hough.YardLineParameters(minimum_fragments=3, minimum_span_ratio=.4),
    )
    assert len(fragments) > len(rows)
    assert len(rows) >= 3
    segments = hough.yard_line_segments(rows)
    assert np.all((hough.segment_angles_degrees(segments) < 3) |
                  (hough.segment_angles_degrees(segments) > 177))
    assert hough.segment_lengths(segments).min() > 500
    assert all(row["fragments"] >= 3 for row in rows)


def test_yard_line_rows_reject_isolated_white_shapes():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (180, 180), (255, 255, 255), 3)
    cv2.circle(image, (400, 200), 40, (255, 255, 255), 3)
    _, rows = hough.find_yard_line_rows(
        image, yard_parameters=hough.YardLineParameters(minimum_span_ratio=.5)
    )
    assert rows == []
    assert hough.yard_line_segments(rows).shape == (0, 4)


def test_describe_includes_grouped_yard_line_diagnostics():
    payload = hough.describe(frame_with_lines())
    assert payload["white_fragment_count"] >= payload["yard_line_count"]
    assert "yard_lines" in payload
    assert payload["yard_parameters"]["minimum_fragments"] == 3


def test_encode_jpeg_produces_a_decodable_image():
    data = hough.encode_jpeg(frame_with_lines())
    assert data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)
