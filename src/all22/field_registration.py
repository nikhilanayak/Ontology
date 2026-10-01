from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .geometry import FIELD_WIDTH, estimate_homography


@dataclass(frozen=True)
class OCRNumber:
    value: int
    center: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class Registration:
    image_points: list[list[float]]
    field_points: list[list[float]]
    matrix: np.ndarray
    confidence: float
    absolute_x: bool
    diagnostics: dict


def _field_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (22, 25, 25), (105, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No field-colored region was detected")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * .12:
        raise ValueError("The detected field region is too small")
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(contour)], -1, 255, -1)
    return mask


def _angle_distance(first: float, second: float) -> float:
    return abs(((first - second + np.pi / 2) % np.pi) - np.pi / 2)


def _merge_family(segments: np.ndarray, angle: float, mask: np.ndarray) -> list[np.ndarray]:
    tangent = np.asarray([np.cos(angle), np.sin(angle)])
    normal = np.asarray([-tangent[1], tangent[0]])
    groups: list[list[np.ndarray]] = []
    centers: list[float] = []
    threshold = max(8.0, .009 * np.hypot(*mask.shape))
    for segment in sorted(segments, key=lambda value: float(np.dot(value.reshape(2, 2).mean(axis=0), normal))):
        points = segment.reshape(2, 2).astype(float)
        coordinate = float(np.dot(points.mean(axis=0), normal))
        if not centers or abs(coordinate - centers[-1]) > threshold:
            groups.append([points])
            centers.append(coordinate)
        else:
            groups[-1].append(points)
            centers[-1] = float(np.mean([np.dot(item.mean(axis=0), normal) for item in groups[-1]]))
    height, width = mask.shape
    diagonal = int(np.hypot(height, width))
    output = []
    for group in groups:
        points = np.concatenate(group)
        origin = points.mean(axis=0)
        samples = origin[None, :] + np.arange(-diagonal, diagonal + 1)[:, None] * tangent[None, :]
        pixels = np.rint(samples).astype(int)
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        pixels, samples = pixels[valid], samples[valid]
        inside = mask[pixels[:, 1], pixels[:, 0]] > 0
        samples = samples[inside]
        if len(samples) < 2:
            continue
        projections = samples[:, 0] * tangent[0] + samples[:, 1] * tangent[1]
        output.append(np.vstack([samples[np.argmin(projections)], samples[np.argmax(projections)]]))
    return output


def detect_yard_lines(frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    mask = _field_mask(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = ((gray > 145) & (hsv[:, :, 1] < 115) & (mask > 0)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(white, 50, 150)
    minimum = max(45, int(min(frame.shape[:2]) * .16))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=45,
                          minLineLength=minimum, maxLineGap=35)
    if raw is None:
        raise ValueError("No long painted field lines were detected")
    segments = raw[:, 0, :]
    vectors = segments[:, 2:4] - segments[:, 0:2]
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    # Try each observed orientation and keep the family producing the most
    # distinct, long, parallel lines. Yard lines repeat; sidelines do not.
    candidates = []
    for angle in angles:
        keep = np.asarray([_angle_distance(value, angle) < np.deg2rad(8) for value in angles])
        family = _merge_family(segments[keep], float(angle), mask)
        if len(family) >= 2:
            lengths = [np.linalg.norm(line[1] - line[0]) for line in family]
            candidates.append((len(family) * np.median(lengths), family))
    if not candidates:
        raise ValueError("Painted lines could not be grouped into a yard-line family")
    lines = max(candidates, key=lambda value: value[0])[1]
    # Deduplicate nearly identical alternatives and order across the field.
    direction = lines[0][1] - lines[0][0]
    direction /= np.linalg.norm(direction)
    normal = np.asarray([-direction[1], direction[0]])
    lines.sort(key=lambda line: float(np.dot(line.mean(axis=0), normal)))
    return lines, mask, white


def read_field_numbers(frame: np.ndarray, reader=None) -> list[OCRNumber]:
    if reader is None:
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("Automatic number recognition requires `pip install -e '.[calibration]'`") from error
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    masked = cv2.bitwise_and(frame, frame, mask=_field_mask(frame))
    results = reader.readtext(masked, allowlist="012345", rotation_info=[90, 180, 270],
                              text_threshold=.45, low_text=.25)
    values = []
    for polygon, text, confidence in results:
        digits = "".join(character for character in str(text) if character.isdigit())
        if not digits:
            continue
        number = None
        for candidate in (10, 20, 30, 40, 50):
            label = str(candidate)
            # The goal-pointing arrow is frequently segmented as a trailing 5,
            # while an upside-down 10 is often returned as 01.
            if label in digits or label[::-1] in digits:
                number = candidate
                break
        if number is None or float(confidence) < .2:
            continue
        center = np.asarray(polygon, dtype=float).mean(axis=0)
        values.append(OCRNumber(number, (float(center[0]), float(center[1])), float(confidence)))
    return values


def _assign_field_x(lines: list[np.ndarray], numbers: list[OCRNumber]) -> tuple[np.ndarray, bool, float]:
    direction = lines[0][1] - lines[0][0]
    direction /= np.linalg.norm(direction)
    normal = np.asarray([-direction[1], direction[0]])
    coordinates = np.asarray([np.dot(line.mean(axis=0), normal) for line in lines])
    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    if not numbers:
        values = 60 + (np.arange(len(lines)) - (len(lines) - 1) / 2) * 5
        restored = np.empty_like(values, dtype=float); restored[order] = values
        return restored, False, 0.0
    anchors = []
    for number in numbers:
        point_coordinate = float(np.dot(number.center, normal))
        index = int(np.argmin(abs(coordinates - point_coordinate)))
        options = [60.0] if number.value == 50 else [10.0 + number.value, 110.0 - number.value]
        anchors.append((index, options, number.confidence))
    best = None
    for choices in itertools.product(*(item[1] for item in anchors)):
        for step in (-5.0, 5.0):
            offsets = [choice - step * anchor[0] for anchor, choice in zip(anchors, choices)]
            offset = float(np.average(offsets, weights=[item[2] for item in anchors]))
            residual = float(np.average([abs(value - offset) for value in offsets],
                                        weights=[item[2] for item in anchors]))
            values = offset + step * np.arange(len(lines))
            outside = float(np.maximum(0, 5 - values).sum() + np.maximum(0, values - 115).sum())
            score = residual + outside
            if best is None or score < best[0]:
                best = (score, values)
    assert best is not None
    restored = np.empty_like(best[1], dtype=float);restored[order] = best[1]
    return restored, bool(best[0] < 2.0), float(best[0])


def register_field(frame: np.ndarray, reader=None, numbers: Optional[list[OCRNumber]] = None) -> Registration:
    lines, mask, white = detect_yard_lines(frame)
    recognized = numbers if numbers is not None else read_field_numbers(frame, reader)
    field_x, absolute, ocr_error = _assign_field_x(lines, recognized)
    image_points = []
    field_points = []
    for line, x in zip(lines, field_x):
        endpoints = sorted(line, key=lambda point: (point[1], point[0]))
        image_points.extend(point.tolist() for point in endpoints)
        field_points.extend([[float(x), 0.0], [float(x), FIELD_WIDTH]])
    calibration = estimate_homography(image_points, field_points)
    centers = np.asarray([line.mean(axis=0) for line in lines])
    if len(centers) > 2:
        axis = centers[-1] - centers[0];axis /= max(np.linalg.norm(axis), 1e-9)
        spacing = np.diff(centers @ axis)
        spacing_cv = float(np.std(spacing) / max(abs(np.mean(spacing)), 1e-9))
    else:
        spacing_cv = 1.0
    confidence = float(np.clip(.25 + .08 * min(len(lines), 6) + .2 * absolute - .2 * min(spacing_cv, 1), 0, 1))
    diagnostics = {"line_count": len(lines), "ocr_numbers": len(recognized),
                   "ocr_assignment_error": ocr_error, "spacing_cv": spacing_cv,
                   "absolute_x": absolute, "field_fraction": float((mask > 0).mean()),
                   "white_fraction": float((white > 0).mean())}
    return Registration(image_points, field_points, calibration.matrix, confidence, absolute, diagnostics)


def diagnostic_image(frame: np.ndarray, registration: Registration) -> np.ndarray:
    output = frame.copy()
    for index in range(0, len(registration.image_points), 2):
        points = np.rint(registration.image_points[index:index + 2]).astype(int)
        x = registration.field_points[index][0]
        cv2.line(output, tuple(points[0]), tuple(points[1]), (0, 255, 255), 3)
        cv2.putText(output, f"x={x:.1f}", tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def _read_frame(capture: cv2.VideoCapture, timestamp: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not decode calibration frame at {timestamp:.2f}s")
    return frame


def auto_calibrate_clip(db_path: Path, clip_id: str, diagnostics_dir: Path,
                        reader=None) -> list[dict]:
    from .db import connect
    from .field_tracking import save_calibration_keyframes

    if reader is None:
        try:
            import easyocr
            import torch
        except ImportError as error:
            raise RuntimeError("Automatic calibration requires `pip install -e '.[calibration]'`") from error
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    with connect(db_path) as connection:
        clip = connection.execute(
            """SELECT c.start_s,c.end_s,g.video_path FROM clips c JOIN games g ON g.game_id=c.game_id
               WHERE c.clip_id=?""", (clip_id,),
        ).fetchone()
    if not clip:
        raise ValueError(f"No camera shot found: {clip_id}")
    duration = float(clip["end_s"] - clip["start_s"])
    fractions = (.15, .5, .85) if duration >= 4 else (.2, .8)
    sample_times = [float(clip["start_s"]) + duration * fraction for fraction in fractions]
    stored_times = [float(clip["start_s"]) + duration * index / (len(sample_times) - 1)
                    for index in range(len(sample_times))]
    capture = cv2.VideoCapture(clip["video_path"])
    if not capture.isOpened():
        raise ValueError(f"Could not open {clip['video_path']}")
    payload = {"keyframes": []}
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, (sample_time, stored_time) in enumerate(zip(sample_times, stored_times)):
            selected = None
            errors = []
            for offset in (0, -.5, .5, -1.0, 1.0):
                candidate_time = min(float(clip["end_s"]) - .1,
                                     max(float(clip["start_s"]) + .1, sample_time + offset))
                try:
                    candidate_frame = _read_frame(capture, candidate_time)
                    candidate = register_field(candidate_frame, reader=reader)
                    if candidate.absolute_x and candidate.confidence >= .5:
                        selected = (candidate_time, candidate_frame, candidate)
                        break
                    errors.append(f"{candidate_time:.2f}s: unanchored/{candidate.confidence:.2f}")
                except ValueError as error:
                    errors.append(f"{candidate_time:.2f}s: {error}")
            if selected is None:
                raise ValueError("Automatic registration failed near keyframe: " + "; ".join(errors))
            candidate_time, frame, registration = selected
            output = diagnostics_dir / f"{clip_id.replace(':', '_')}-{index}.jpg"
            cv2.imwrite(str(output), diagnostic_image(frame, registration))
            payload["keyframes"].append({
                "timestamp_s": stored_time,
                "image_points": registration.image_points,
                "field_points": registration.field_points,
                "source": "automatic_lines_ocr_v1",
                "registration_confidence": registration.confidence,
                "registration_diagnostics": registration.diagnostics,
                "sample_timestamp_s": candidate_time,
                "diagnostic_image": str(output.resolve()),
            })
    finally:
        capture.release()
    return save_calibration_keyframes(db_path, clip_id, payload)
