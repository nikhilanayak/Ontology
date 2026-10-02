"""Annotate frames with straight lines found by the probabilistic Hough transform.

This module is deliberately standalone and generic: it runs Canny edge detection
followed by `cv2.HoughLinesP` on a whole frame and draws what it finds. It makes
no assumption that the frame shows a football field, applies no green/white paint
mask, and performs no clustering, fitting, or yard-line reasoning. The output is
the raw transform, so a preview can show exactly what the operator sees before
any later stage interprets it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HoughParameters:
    """Tunable inputs for the Canny + probabilistic Hough pipeline."""

    blur: int = 5
    canny_low: int = 50
    canny_high: int = 150
    threshold: int = 45
    min_line_length: int = 60
    max_line_gap: int = 35

    def normalized(self) -> "HoughParameters":
        """Clamp values into the ranges OpenCV accepts, keeping an odd blur kernel."""
        blur = max(0, int(self.blur))
        if blur and blur % 2 == 0:
            blur += 1
        low, high = sorted((max(0, int(self.canny_low)), max(1, int(self.canny_high))))
        return HoughParameters(
            blur=blur,
            canny_low=low,
            canny_high=max(high, low + 1),
            threshold=max(1, int(self.threshold)),
            min_line_length=max(1, int(self.min_line_length)),
            max_line_gap=max(0, int(self.max_line_gap)),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def edge_map(frame: np.ndarray, parameters: HoughParameters | None = None) -> np.ndarray:
    """Return the Canny edge image the transform is run against."""
    values = (parameters or HoughParameters()).normalized()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if values.blur:
        gray = cv2.GaussianBlur(gray, (values.blur, values.blur), 0)
    return cv2.Canny(gray, values.canny_low, values.canny_high)


def detect_lines(frame: np.ndarray, parameters: HoughParameters | None = None) -> np.ndarray:
    """Return probabilistic Hough segments as an `(n, 4)` array of `x1,y1,x2,y2`."""
    values = (parameters or HoughParameters()).normalized()
    raw = cv2.HoughLinesP(
        edge_map(frame, values),
        rho=1,
        theta=np.pi / 360,
        threshold=values.threshold,
        minLineLength=values.min_line_length,
        maxLineGap=values.max_line_gap,
    )
    if raw is None:
        return np.empty((0, 4), dtype=np.int32)
    return raw[:, 0, :].astype(np.int32)


def segment_angles_degrees(segments: np.ndarray) -> np.ndarray:
    """Return each segment's orientation in `[0, 180)` degrees."""
    if not len(segments):
        return np.empty((0,), dtype=np.float64)
    vectors = segments[:, 2:4].astype(np.float64) - segments[:, 0:2].astype(np.float64)
    return np.degrees(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi))


def segment_lengths(segments: np.ndarray) -> np.ndarray:
    """Return each segment's pixel length."""
    if not len(segments):
        return np.empty((0,), dtype=np.float64)
    vectors = segments[:, 2:4].astype(np.float64) - segments[:, 0:2].astype(np.float64)
    return np.hypot(vectors[:, 0], vectors[:, 1])


def annotate(frame: np.ndarray, segments: np.ndarray, thickness: int = 2,
             show_edges: bool = False, parameters: HoughParameters | None = None) -> np.ndarray:
    """Draw `segments` over a copy of `frame`, colouring each line by orientation.

    Hue encodes angle so that families of parallel lines are separable by eye
    without labelling every segment.
    """
    canvas = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if show_edges:
        edges = edge_map(frame, parameters)
        canvas[edges > 0] = (60, 60, 60)
    if not len(segments):
        return canvas
    angles = segment_angles_degrees(segments)
    hues = np.clip(angles, 0, 179).astype(np.uint8).reshape(-1, 1, 1)
    saturation = np.full_like(hues, 255)
    colors = cv2.cvtColor(np.dstack([hues, saturation, saturation]), cv2.COLOR_HSV2BGR)
    for (x1, y1, x2, y2), color in zip(segments, colors.reshape(-1, 3)):
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                 tuple(int(channel) for channel in color), thickness, cv2.LINE_AA)
    return canvas


def describe(frame: np.ndarray, parameters: HoughParameters | None = None) -> dict:
    """Return the segments plus simple diagnostics for one frame."""
    values = (parameters or HoughParameters()).normalized()
    segments = detect_lines(frame, values)
    lengths = segment_lengths(segments)
    height, width = frame.shape[:2]
    return {
        "parameters": values.as_dict(),
        "width": int(width),
        "height": int(height),
        "count": int(len(segments)),
        "segments": [
            {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
             "length": round(float(length), 2), "angle": round(float(angle), 2)}
            for (x1, y1, x2, y2), length, angle
            in zip(segments, lengths, segment_angles_degrees(segments))
        ],
        "longest": round(float(lengths.max()), 2) if len(lengths) else 0.0,
        "median_length": round(float(np.median(lengths)), 2) if len(lengths) else 0.0,
    }


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Could not encode the annotated frame")
    return buffer.tobytes()
