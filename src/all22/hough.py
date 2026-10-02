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


@dataclass(frozen=True)
class YardLineParameters:
    brightness: int = 150
    maximum_saturation: int = 115
    angle_tolerance: float = 7.0
    alignment_tolerance: float = 12.0
    minimum_fragments: int = 3
    minimum_span_ratio: float = .50
    minimum_coverage: float = .45

    def normalized(self) -> "YardLineParameters":
        return YardLineParameters(
            brightness=min(255, max(0, int(self.brightness))),
            maximum_saturation=min(255, max(0, int(self.maximum_saturation))),
            angle_tolerance=min(30, max(1, float(self.angle_tolerance))),
            alignment_tolerance=min(100, max(1, float(self.alignment_tolerance))),
            minimum_fragments=max(2, int(self.minimum_fragments)),
            minimum_span_ratio=min(1, max(.05, float(self.minimum_span_ratio))),
            minimum_coverage=min(1, max(.05, float(self.minimum_coverage))),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _angle_distance_degrees(first: float, second: float) -> float:
    difference = abs(first - second) % 180
    return min(difference, 180 - difference)


def white_edge_map(frame: np.ndarray, hough_parameters: HoughParameters | None = None,
                   yard_parameters: YardLineParameters | None = None) -> np.ndarray:
    values = (hough_parameters or HoughParameters()).normalized()
    yard = (yard_parameters or YardLineParameters()).normalized()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = ((gray >= yard.brightness) &
             (hsv[:, :, 1] <= yard.maximum_saturation)).astype(np.uint8) * 255
    if values.blur:
        white = cv2.GaussianBlur(white, (values.blur, values.blur), 0)
    return cv2.Canny(white, values.canny_low, values.canny_high)


def detect_white_lines(frame: np.ndarray, hough_parameters: HoughParameters | None = None,
                       yard_parameters: YardLineParameters | None = None) -> np.ndarray:
    values = (hough_parameters or HoughParameters()).normalized()
    raw = cv2.HoughLinesP(
        white_edge_map(frame, values, yard_parameters), rho=1, theta=np.pi / 360,
        threshold=max(10, values.threshold // 2),
        minLineLength=max(15, values.min_line_length // 2),
        maxLineGap=values.max_line_gap,
    )
    return np.empty((0, 4), dtype=np.int32) if raw is None else raw[:, 0, :].astype(np.int32)


def _fit_fragment_row(segments: np.ndarray) -> tuple[np.ndarray, float, float]:
    points = segments.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = (float(value) for value in cv2.fitLine(
        points, cv2.DIST_L2, 0, .01, .01).reshape(-1))
    direction = np.asarray([vx, vy], dtype=np.float64)
    origin = np.asarray([x0, y0], dtype=np.float64)
    projections = (points - origin) @ direction
    start = origin + direction * projections.min()
    end = origin + direction * projections.max()
    residuals = np.abs((points - origin) @ np.asarray([-vy, vx]))
    intervals = []
    for segment in segments.astype(np.float64):
        endpoints = segment.reshape(2, 2)
        values = (endpoints - origin) @ direction
        intervals.append((float(values.min()), float(values.max())))
    merged = []
    for begin, finish in sorted(intervals):
        if not merged or begin > merged[-1][1]:
            merged.append([begin, finish])
        else:
            merged[-1][1] = max(merged[-1][1], finish)
    covered = sum(finish - begin for begin, finish in merged)
    span = max(1, float(projections.max() - projections.min()))
    return (np.rint(np.concatenate([start, end])).astype(np.int32),
            float(np.median(residuals)), covered / span)


def find_yard_line_rows(frame: np.ndarray, hough_parameters: HoughParameters | None = None,
                        yard_parameters: YardLineParameters | None = None,
                        camera_angle: str | None = None) -> tuple[np.ndarray, list[dict]]:
    yard = (yard_parameters or YardLineParameters()).normalized()
    fragments = detect_white_lines(frame, hough_parameters, yard)
    if len(fragments) < yard.minimum_fragments:
        return fragments, []
    angles = segment_angles_degrees(fragments)
    lengths = segment_lengths(fragments)
    centers = (fragments[:, :2] + fragments[:, 2:4]) / 2
    minimum_span = min(frame.shape[:2]) * yard.minimum_span_ratio
    best: list[dict] = []
    best_score = 0.0
    seeds = angles[np.argsort(lengths)[::-1][:min(80, len(angles))]]
    for seed in seeds:
        steepness = abs(np.sin(np.deg2rad(seed)))
        if camera_angle == "sideline" and steepness < .35:
            continue
        if camera_angle == "endzone" and steepness > .35:
            continue
        parallel = np.asarray([_angle_distance_degrees(value, seed) <= yard.angle_tolerance
                               for value in angles])
        if parallel.sum() < yard.minimum_fragments:
            continue
        theta = np.deg2rad(seed)
        normal = np.asarray([-np.sin(theta), np.cos(theta)])
        indices = np.flatnonzero(parallel)
        offsets = centers[indices] @ normal
        order = np.argsort(offsets)
        groups: list[list[int]] = []
        for index in indices[order]:
            offset = float(centers[index] @ normal)
            if not groups or abs(offset - np.mean([centers[item] @ normal
                                                   for item in groups[-1]])) > yard.alignment_tolerance:
                groups.append([int(index)])
            else:
                groups[-1].append(int(index))
        rows = []
        for group in groups:
            if len(group) < yard.minimum_fragments:
                continue
            fitted, residual, coverage = _fit_fragment_row(fragments[group])
            span = float(np.hypot(fitted[2] - fitted[0], fitted[3] - fitted[1]))
            fitted_angle = float(segment_angles_degrees(fitted.reshape(1, 4))[0])
            if (span < minimum_span or residual > yard.alignment_tolerance or
                    coverage < yard.minimum_coverage):
                continue
            if _angle_distance_degrees(fitted_angle, float(seed)) > yard.angle_tolerance:
                continue
            rows.append({
                "segment": fitted,
                "fragments": len(group),
                "span": span,
                "residual": residual,
                "coverage": coverage,
                "angle": fitted_angle,
            })
        score = sum(row["span"] * np.sqrt(row["fragments"]) for row in rows)
        if len(rows) >= 2:
            score *= 1 + min(1, len(rows) / 5)
        if score > best_score:
            best, best_score = rows, score
    if best:
        dominant = float(np.median([row["angle"] for row in best]))
        theta = np.deg2rad(dominant)
        normal = np.asarray([-np.sin(theta), np.cos(theta)])
        deduplicated: list[dict] = []
        for row in sorted(best, key=lambda value: value["span"], reverse=True):
            segment = row["segment"].reshape(2, 2)
            offset = float(segment.mean(axis=0) @ normal)
            if any(abs(offset - float(existing["segment"].reshape(2, 2).mean(axis=0) @ normal))
                   <= yard.alignment_tolerance * 1.5 for existing in deduplicated):
                continue
            deduplicated.append(row)
        best = deduplicated
    best.sort(key=lambda row: (row["segment"][0] + row["segment"][2],
                               row["segment"][1] + row["segment"][3]))
    return fragments, best


def yard_line_segments(rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.empty((0, 4), dtype=np.int32)
    return np.asarray([row["segment"] for row in rows], dtype=np.int32)


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


def describe(frame: np.ndarray, parameters: HoughParameters | None = None,
             yard_parameters: YardLineParameters | None = None,
             camera_angle: str | None = None) -> dict:
    """Return raw segments and grouped yard-line candidates for one frame."""
    values = (parameters or HoughParameters()).normalized()
    yard_values = (yard_parameters or YardLineParameters()).normalized()
    segments = detect_lines(frame, values)
    lengths = segment_lengths(segments)
    white_fragments, rows = find_yard_line_rows(frame, values, yard_values, camera_angle)
    height, width = frame.shape[:2]
    return {
        "parameters": values.as_dict(),
        "yard_parameters": yard_values.as_dict(),
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
        "white_fragment_count": int(len(white_fragments)),
        "yard_line_count": len(rows),
        "yard_lines": [
            {"x1": int(row["segment"][0]), "y1": int(row["segment"][1]),
             "x2": int(row["segment"][2]), "y2": int(row["segment"][3]),
             "fragments": int(row["fragments"]), "span": round(float(row["span"]), 2),
             "residual": round(float(row["residual"]), 2),
             "coverage": round(float(row["coverage"]), 3),
             "angle": round(float(row["angle"]), 2)}
            for row in rows
        ],
    }


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Could not encode the annotated frame")
    return buffer.tobytes()
