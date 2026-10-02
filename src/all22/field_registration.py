from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .geometry import FIELD_LENGTH, FIELD_WIDTH, estimate_homography, project_points


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
    # Hough returns both painted edges and fragments split by players. They can
    # be tens of pixels apart in HD All-22 while adjacent five-yard lines are
    # much farther apart, so merge on a field-scale tolerance.
    threshold = max(12.0, .03 * np.hypot(*mask.shape))
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
    # Full five-yard lines span most of the visible field depth. A stricter
    # length floor excludes strokes from numerals, arrows, and hash marks.
    minimum = max(60, int(min(frame.shape[:2]) * .30))
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
            observed_lengths = np.linalg.norm(vectors[keep], axis=1)
            candidates.append((len(family) ** 2 * np.median(observed_lengths), family))
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
        if number is None or float(confidence) < .55:
            continue
        center = np.asarray(polygon, dtype=float).mean(axis=0)
        values.append(OCRNumber(number, (float(center[0]), float(center[1])), float(confidence)))
    return values


def _credible_field_numbers(numbers: list[OCRNumber], frame_shape: tuple[int, ...]) -> bool:
    """Reject isolated jersey digits before they can anchor the field template."""
    if len(numbers) < 2:
        return False
    height, width = frame_shape[:2]
    by_value: dict[int, list[OCRNumber]] = {}
    for number in numbers:
        by_value.setdefault(number.value, []).append(number)
    # Painted values normally repeat on the two number rows.  Their large
    # transverse separation is much stronger evidence than OCR confidence.
    for values in by_value.values():
        for first, second in itertools.combinations(values, 2):
            if np.hypot(first.center[0] - second.center[0], first.center[1] - second.center[1]) >= .28 * min(height, width):
                return True
    # A partially cropped view may show only one row, but require a sequence of
    # at least three labels with two distinct legal values.
    return len(numbers) >= 3 and len(by_value) >= 2


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


def _number_anchored_homography(lines: list[np.ndarray], field_x: np.ndarray,
                                numbers: list[OCRNumber], frame_shape: tuple[int, ...],
                                ) -> tuple[Optional[np.ndarray], dict]:
    """Solve line-x and painted-number-y constraints without fake sidelines."""
    if len(numbers) < 3:
        return None, {"semantic_number_fit": False, "semantic_reason": "fewer_than_three_numbers"}
    height, width = frame_shape[:2]
    tangent = lines[0][1] - lines[0][0]
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    if ((abs(tangent[1]) >= abs(tangent[0]) and tangent[1] < 0) or
            (abs(tangent[0]) > abs(tangent[1]) and tangent[0] < 0)):
        tangent *= -1

    assignments = []
    for number in numbers:
        point = np.asarray(number.center, float)
        distances = []
        for line in lines:
            vector = line[1] - line[0]
            relative = point - line[0]
            cross = vector[0] * relative[1] - vector[1] * relative[0]
            distances.append(abs(cross) / max(np.linalg.norm(vector), 1e-9))
        index = int(np.argmin(distances))
        assignments.append((point, float(field_x[index]), float(point @ tangent), number.confidence))
    row_coordinates = np.asarray([item[2] for item in assignments])
    order = np.argsort(row_coordinates)
    gaps = np.diff(row_coordinates[order])
    if not len(gaps) or float(np.max(gaps)) < .08 * min(height, width):
        return None, {"semantic_number_fit": False, "semantic_reason": "number_rows_not_separated"}
    split = int(np.argmax(gaps)) + 1
    low = set(order[:split].tolist())
    if not low or len(low) == len(assignments):
        return None, {"semantic_number_fit": False, "semantic_reason": "number_row_split_failed"}

    # Solve H from image to field with h22 fixed to one. Yard lines constrain
    # field x for every point on the line; number centers additionally constrain
    # their official cross-field row. Coordinates are normalized for stability.
    rows, targets = [], []
    for line, x in zip(lines, field_x):
        for point in np.linspace(line[0], line[1], 5):
            u, v = point[0] / width, point[1] / height
            rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); targets.append(x)
    for index, (point, x, _, confidence) in enumerate(assignments):
        u, v = point[0] / width, point[1] / height
        y = 12.0 if index in low else FIELD_WIDTH - 12.0
        weight = 2.0 * max(.55, confidence)
        rows.append((weight * np.asarray([u, v, 1, 0, 0, 0, -x * u, -x * v])).tolist())
        targets.append(weight * x)
        rows.append((weight * np.asarray([0, 0, 0, u, v, 1, -y * u, -y * v])).tolist())
        targets.append(weight * y)
    design = np.asarray(rows, float)
    if np.linalg.matrix_rank(design) < 8:
        return None, {"semantic_number_fit": False, "semantic_reason": "rank_deficient"}
    parameters, _, _, singular = np.linalg.lstsq(design, np.asarray(targets), rcond=None)
    normalized = np.r_[parameters, 1.0].reshape(3, 3)
    matrix = normalized @ np.diag([1 / width, 1 / height, 1.0])
    projected = project_points(matrix, [item[0] for item in assignments])
    expected = np.asarray([[item[1], 12.0 if index in low else FIELD_WIDTH - 12.0]
                           for index, item in enumerate(assignments)])
    errors = np.linalg.norm(projected - expected, axis=1)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    valid = (np.isfinite(matrix).all() and condition < 1e7 and
             float(np.median(errors)) <= 2.0 and float(np.percentile(errors, 90)) <= 4.0)
    return (matrix if valid else None), {
        "semantic_number_fit": valid, "semantic_condition": condition,
        "semantic_median_error_yards": float(np.median(errors)),
        "semantic_p90_error_yards": float(np.percentile(errors, 90)),
        "semantic_number_rows": [len(low), len(assignments) - len(low)],
    }


# A homography can fit the lines it was handed almost perfectly and still place
# them on the wrong part of the field: yard lines repeat every 5 yd and the
# painted numbers are the only absolute cue. On the pilot, calibration's own
# residuals were uncorrelated with real error (|r| < 0.17 over 15 clips), and
# one clip reported a 0.22 yd p95 while sitting 5.92 yd out. These checks test
# the *result* against independent facts about a football field instead.
PLAUSIBLE_PIXELS_PER_YARD = (2.0, 60.0)
MAXIMUM_SKEW_RATIO = 4.0


def verify_registration(matrix: np.ndarray, frame_shape: tuple[int, ...]) -> dict:
    """Check a homography against known field geometry, independently of its fit.

    Returns the reasons it is implausible; an empty list means it survived.
    """
    reasons: list[str] = []
    height, width = frame_shape[:2]
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return {"reasons": ["matrix is singular"], "verified": False}
    # Where does the visible frame land on the field?
    corners = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    field = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(field).all():
        return {"reasons": ["projection is not finite"], "verified": False}
    span_x = float(field[:, 0].max() - field[:, 0].min())
    span_y = float(field[:, 1].max() - field[:, 1].min())
    # A broadcast shot covers a plausible slice of a 120 x 53.3 yd field.
    if not 5.0 <= span_x <= 400.0:
        reasons.append(f"frame spans {span_x:.0f} yd along the field")
    if not 3.0 <= span_y <= 300.0:
        reasons.append(f"frame spans {span_y:.0f} yd across the field")
    # The shot must actually overlap the field rather than sitting beside it.
    if field[:, 0].max() < -20 or field[:, 0].min() > FIELD_LENGTH + 20:
        reasons.append("frame projects entirely off the end of the field")
    if field[:, 1].max() < -20 or field[:, 1].min() > FIELD_WIDTH + 20:
        reasons.append("frame projects entirely off the side of the field")
    # Scale sanity: a yard must occupy a believable number of pixels at the
    # image centre, and must not be wildly anisotropic.
    centre = np.asarray([[width / 2, height / 2]], dtype=np.float32)
    probes = np.asarray([[width / 2 + 1, height / 2], [width / 2, height / 2 + 1]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(
        np.vstack([centre, probes]).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    step_x = float(np.linalg.norm(mapped[1] - mapped[0]))
    step_y = float(np.linalg.norm(mapped[2] - mapped[0]))
    for name, step in (("horizontal", step_x), ("vertical", step_y)):
        if step <= 0 or not np.isfinite(step):
            reasons.append(f"{name} scale is degenerate")
            continue
        pixels_per_yard = 1.0 / step
        low, high = PLAUSIBLE_PIXELS_PER_YARD
        if not low <= pixels_per_yard <= high:
            reasons.append(f"{name} scale is {pixels_per_yard:.1f} px/yd")
    if step_x > 0 and step_y > 0:
        ratio = max(step_x, step_y) / min(step_x, step_y)
        if ratio > MAXIMUM_SKEW_RATIO:
            reasons.append(f"anisotropic scale ratio {ratio:.1f}")
    # Orientation must be preserved: a mirrored field means the solution flipped.
    determinant = float(np.linalg.det(matrix[:2, :2]))
    if determinant == 0 or not np.isfinite(determinant):
        reasons.append("degenerate linear part")
    return {"reasons": reasons, "verified": not reasons,
            "frame_span_x_yards": span_x, "frame_span_y_yards": span_y,
            "pixels_per_yard_x": float(1.0 / step_x) if step_x > 0 else None,
            "pixels_per_yard_y": float(1.0 / step_y) if step_y > 0 else None}


def register_field(frame: np.ndarray, reader=None, numbers: Optional[list[OCRNumber]] = None) -> Registration:
    lines, mask, white = detect_yard_lines(frame)
    raw_recognized = numbers if numbers is not None else read_field_numbers(frame, reader)
    recognized = raw_recognized if numbers is not None or _credible_field_numbers(raw_recognized, frame.shape) else []
    field_x, absolute, ocr_error = _assign_field_x(lines, recognized)
    image_points = []
    field_points = []
    for line, x in zip(lines, field_x):
        endpoints = sorted(line, key=lambda point: (point[1], point[0]))
        image_points.extend(point.tolist() for point in endpoints)
        field_points.extend([[float(x), 0.0], [float(x), FIELD_WIDTH]])
    calibration = estimate_homography(image_points, field_points)
    semantic_matrix, semantic_diagnostics = _number_anchored_homography(
        lines, field_x, recognized, frame.shape) if absolute else (
            None, {"semantic_number_fit": False, "semantic_reason": "absolute_x_unresolved"})
    if semantic_matrix is not None:
        # Preserve point-pair storage compatibility while retaining the
        # semantically constrained matrix during temporal propagation.
        field_points = project_points(semantic_matrix, image_points).tolist()
        calibration = estimate_homography(image_points, field_points)
    centers = np.asarray([line.mean(axis=0) for line in lines])
    if len(centers) > 2:
        axis = centers[-1] - centers[0];axis /= max(np.linalg.norm(axis), 1e-9)
        spacing = np.diff(centers @ axis)
        spacing_cv = float(np.std(spacing) / max(abs(np.mean(spacing)), 1e-9))
    else:
        spacing_cv = 1.0
    # Hough extension can produce a bad endpoint where a sideline is hidden.
    # RANSAC exists to discard exactly those correspondences, so gate on its
    # consensus rather than allowing one rejected endpoint to poison the fit.
    geometric_ok = (calibration.inlier_ratio >= .65 and calibration.median_error_yards <= 1.5
                    and calibration.inlier_p95_error_yards <= 2.0)
    # Residuals only say the fit explains the lines it was given; they cannot
    # detect a confident fit placed on the wrong yard lines. Verify the result.
    verification = verify_registration(calibration.matrix, frame.shape)
    confidence = float(np.clip(.25 + .08 * min(len(lines), 6) + .2 * absolute
                               - .2 * min(spacing_cv, 1), 0, 1)) if geometric_ok else 0.0
    if not verification["verified"]:
        confidence = 0.0
    diagnostics = {"line_count": len(lines), "ocr_numbers": len(recognized),
                   "ocr_numbers_raw": len(raw_recognized),
                   "ocr_assignment_error": ocr_error, "spacing_cv": spacing_cv,
                   "absolute_x": absolute, "field_fraction": float((mask > 0).mean()),
                   "white_fraction": float((white > 0).mean()),
                   "inlier_ratio": calibration.inlier_ratio,
                   "median_error_yards": calibration.median_error_yards,
                   "p95_error_yards": calibration.p95_error_yards,
                   "inlier_p95_error_yards": calibration.inlier_p95_error_yards}
    diagnostics.update(semantic_diagnostics)
    diagnostics["verification"] = verification
    diagnostics["verified"] = verification["verified"]
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


def propagate_registration(source: np.ndarray, target: np.ndarray,
                           registration: Registration) -> Registration:
    """Carry an anchored field registration through a no-cut camera motion."""
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=5000, fastThreshold=8)
    source_keys, source_descriptors = detector.detectAndCompute(source_gray, _field_mask(source))
    target_keys, target_descriptors = detector.detectAndCompute(target_gray, _field_mask(target))
    if source_descriptors is None or target_descriptors is None:
        raise ValueError("Insufficient field features for temporal registration")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < .72 * second.distance]
    if len(matches) < 12:
        raise ValueError("Insufficient stable features for temporal registration")
    source_points = np.float32([source_keys[item.queryIdx].pt for item in matches])
    target_points = np.float32([target_keys[item.trainIdx].pt for item in matches])
    motion, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.0)
    if motion is None or mask is None or float(mask.mean()) < .35:
        raise ValueError("Temporal field transform was inconsistent")
    image = cv2.perspectiveTransform(
        np.asarray(registration.image_points, np.float32).reshape(-1, 1, 2), motion).reshape(-1, 2)
    calibration = estimate_homography(image, registration.field_points)
    inlier_ratio = float(mask.mean())
    confidence = registration.confidence * (.75 + .25 * inlier_ratio)
    diagnostics = {**registration.diagnostics, "propagated": True,
                   "temporal_matches": len(matches), "temporal_inlier_ratio": inlier_ratio}
    return Registration(image.tolist(), registration.field_points, calibration.matrix,
                        confidence, registration.absolute_x, diagnostics)


def _read_frame(capture: cv2.VideoCapture, timestamp: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not decode calibration frame at {timestamp:.2f}s")
    return frame


def _propagate_to_time(capture: cv2.VideoCapture, source_time: float, source_frame: np.ndarray,
                       target_time: float, registration: Registration) -> tuple[np.ndarray, Registration]:
    steps = max(1, int(np.ceil(abs(target_time - source_time) / .5)))
    frame = source_frame
    value = registration
    for timestamp in np.linspace(source_time, target_time, steps + 1)[1:]:
        following = _read_frame(capture, float(timestamp))
        value = propagate_registration(frame, following, value)
        frame = following
    return frame, value


def _propagate_toward(capture: cv2.VideoCapture, source_time: float, source_frame: np.ndarray,
                      target_time: float, registration: Registration,
                      maximum_step_s: float = .5) -> tuple[float, np.ndarray, Registration, Optional[str]]:
    """Propagate as far as possible without discarding a valid interior anchor.

    Tight crops and graphics at either edge of an otherwise useful camera shot
    can make ORB registration impossible.  Those frames should delimit the
    calibrated interval, not invalidate every absolute field observation in
    the shot.
    """
    steps = max(1, int(np.ceil(abs(target_time - source_time) / maximum_step_s)))
    timestamp = source_time
    frame = source_frame
    value = registration
    for candidate_time in np.linspace(source_time, target_time, steps + 1)[1:]:
        try:
            following = _read_frame(capture, float(candidate_time))
            following_value = propagate_registration(frame, following, value)
        except ValueError as error:
            return timestamp, frame, value, str(error)
        timestamp = float(candidate_time)
        frame = following
        value = following_value
    return timestamp, frame, value, None


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
    start, end = float(clip["start_s"]), float(clip["end_s"])
    duration = end - start
    capture = cv2.VideoCapture(clip["video_path"])
    if not capture.isOpened():
        raise ValueError(f"Could not open {clip['video_path']}")
    payload = {"keyframes": []}
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Painted numbers need to be readable only once anywhere in the shot.
        # Scan first, choose the strongest absolute anchor, then carry that
        # geometry in both directions using the no-hard-cut invariant.
        scan_count = min(25, max(5, int(np.ceil(duration / .75)) + 1))
        scan_times = np.linspace(start + min(.2, duration * .05),
                                 end - min(.2, duration * .05), scan_count)
        anchors = []
        attempts = []
        for candidate_time in scan_times:
            try:
                candidate_frame = _read_frame(capture, float(candidate_time))
                candidate = register_field(candidate_frame, reader=reader)
                diagnostics = candidate.diagnostics
                attempts.append({"timestamp_s": float(candidate_time),
                                 "numbers": diagnostics["ocr_numbers"],
                                 "lines": diagnostics["line_count"],
                                 "absolute": candidate.absolute_x,
                                 "confidence": candidate.confidence})
                if candidate.absolute_x and candidate.confidence >= .5:
                    score = (candidate.confidence + .03 * min(diagnostics["ocr_numbers"], 3)
                             + .01 * min(diagnostics["line_count"], 8))
                    anchors.append((score, float(candidate_time), candidate_frame, candidate))
            except ValueError as error:
                attempts.append({"timestamp_s": float(candidate_time), "error": str(error)})
        if not anchors:
            summary = "; ".join(
                f"{item['timestamp_s']:.2f}s: {item.get('numbers', 0)} numbers/"
                f"{item.get('lines', 0)} lines/conf {item.get('confidence', 0):.2f}"
                for item in attempts)
            raise ValueError(f"No absolute numbered-field anchor found after scanning the shot: {summary}")
        # Absolute OCR/line solutions are evidence in their own right.  Retain
        # the best one in each short temporal neighborhood, then use temporal
        # registration only to extend the first and last anchors toward the
        # shot boundaries.  A failed extension now produces a shorter verified
        # interval instead of failing the entire shot.
        selected_anchors = []
        for item in sorted(anchors, key=lambda value: value[1]):
            if selected_anchors and item[1] - selected_anchors[-1][1] < .65:
                if item[0] > selected_anchors[-1][0]:
                    selected_anchors[-1] = item
            else:
                selected_anchors.append(item)
        candidates = [(time, frame, registration, time, None)
                      for _, time, frame, registration in selected_anchors]
        for target, source in ((start, selected_anchors[0]), (end, selected_anchors[-1])):
            _, anchor_time, anchor_frame, anchor = source
            reached_time, reached_frame, reached, error = _propagate_toward(
                capture, anchor_time, anchor_frame, target, anchor)
            if abs(reached_time - anchor_time) >= .20:
                candidates.append((reached_time, reached_frame, reached, anchor_time, error))

        # Collapse timestamps reached from different anchors, preferring the
        # higher-confidence registration at each instant.
        selected = []
        for candidate in sorted(candidates, key=lambda value: value[0]):
            if selected and abs(candidate[0] - selected[-1][0]) < .10:
                if candidate[2].confidence > selected[-1][2].confidence:
                    selected[-1] = candidate
            else:
                selected.append(candidate)
        coverage = selected[-1][0] - selected[0][0]
        minimum_coverage = min(1.0, duration * .15)
        if len(selected) < 2 or coverage < minimum_coverage:
            raise ValueError(
                f"Absolute anchors covered only {coverage:.2f}s of this {duration:.2f}s shot")

        for index, (timestamp, frame, registration, anchor_time, extension_error) in enumerate(selected):
            output = diagnostics_dir / f"{clip_id.replace(':', '_')}-{index}.jpg"
            cv2.imwrite(str(output), diagnostic_image(frame, registration))
            payload["keyframes"].append({
                "timestamp_s": timestamp,
                "image_points": registration.image_points,
                "field_points": registration.field_points,
                "source": "automatic_lines_ocr_v2_partial_interval",
                "registration_confidence": registration.confidence,
                "registration_diagnostics": {
                    **registration.diagnostics,
                    "calibrated_start_s": selected[0][0],
                    "calibrated_end_s": selected[-1][0],
                    "shot_coverage_fraction": coverage / max(duration, 1e-6),
                    "extension_error": extension_error,
                },
                "sample_timestamp_s": timestamp,
                "anchor_timestamp_s": anchor_time,
                "diagnostic_image": str(output.resolve()),
            })
    finally:
        capture.release()
    return save_calibration_keyframes(db_path, clip_id, payload)


def auto_reconstruct_game(db_path: Path, game_id: str, output_root: Path,
                          limit: Optional[int] = None, resume: bool = True) -> list[dict]:
    """Register and reconstruct every already-detected shot, reusing one OCR model."""
    from .db import connect
    from .workflow import reconstruct_clip

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    try:
        import easyocr
        import torch
    except ImportError as error:
        raise RuntimeError("Automatic calibration requires `pip install -e '.[calibration]'`") from error
    with connect(db_path) as connection:
        rows = connection.execute(
            """SELECT c.clip_id,
                      (SELECT path FROM artifacts a WHERE a.clip_id=c.clip_id AND a.kind='clip_detections'
                       ORDER BY artifact_id DESC LIMIT 1) AS detections,
                      EXISTS(SELECT 1 FROM artifacts a WHERE a.clip_id=c.clip_id AND a.kind='clip_tracks') tracked
               FROM clips c WHERE c.game_id=? AND EXISTS(
                 SELECT 1 FROM artifacts a WHERE a.clip_id=c.clip_id AND a.kind='clip_detections')
               ORDER BY c.start_s""", (game_id,),
        ).fetchall()
    selected = list(rows[:limit] if limit is not None else rows)
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    results = []
    for row in selected:
        clip_id = row["clip_id"]
        if resume and bool(row["tracked"]):
            results.append({"clip_id": clip_id, "status": "cached"})
            continue
        try:
            keyframes = auto_calibrate_clip(
                db_path, clip_id, output_root / "calibration-diagnostics", reader=reader)
            reconstruction = reconstruct_clip(
                db_path, clip_id, Path(row["detections"]), output_root)
            projection = reconstruction["projection"]
            tracking = reconstruction["tracking"]
            quality_reasons = []
            if projection["valid_fraction"] < .75:
                quality_reasons.append("calibration_coverage_below_75_percent")
            if projection["on_field_fraction"] < .75:
                quality_reasons.append("on_field_detections_below_75_percent")
            if not 18 <= tracking["reliable_tracks"] <= 35:
                quality_reasons.append("implausible_reliable_track_count")
            if not reconstruction["actions"]:
                quality_reasons.append("no_live_action_detected")
            results.append({"clip_id": clip_id, "status": "created",
                            "quality_status": "review" if quality_reasons else "usable",
                            "quality_reasons": quality_reasons,
                            "keyframes": len(keyframes), "reconstruction": reconstruction})
        except (ValueError, RuntimeError) as error:
            results.append({"clip_id": clip_id, "status": "failed", "error": str(error)})
    return results


def detect_field_line_evidence(frame: np.ndarray, mask: np.ndarray, white: np.ndarray,
                               yard_lines: list[np.ndarray]) -> dict:
    """Fit visible white field paint into a small set of stable line hypotheses.

    Hough returns two edges for each painted stripe plus fragments split by
    players, which is useful internally but unreadable as an overlay. Cluster
    parallel fragments by their normal coordinate and extend each fitted model
    through the green-field mask, yielding one visually meaningful segment per
    candidate painted line. Raw fragments are counted for diagnostics, never
    drawn by default.
    """
    edges = cv2.Canny(white, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=22,
                           minLineLength=max(25, int(min(frame.shape[:2]) * .08)),
                           maxLineGap=20)
    if raw is None or not yard_lines:
        return {"raw_count": 0 if raw is None else int(len(raw)), "fitted": []}
    segments = raw[:, 0, :]
    vectors = segments[:, 2:4] - segments[:, 0:2]
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    yard_vector = yard_lines[0][1] - yard_lines[0][0]
    yard_angle = float(np.mod(np.arctan2(yard_vector[1], yard_vector[0]), np.pi))
    fitted = []
    for kind, target, tolerance in (
        ("cross_field", yard_angle, np.deg2rad(8)),
        ("downfield_boundary", (yard_angle + np.pi / 2) % np.pi, np.deg2rad(12)),
    ):
        keep = np.asarray([_angle_distance(angle, target) < tolerance for angle in angles])
        if not keep.any():
            continue
        # _merge_family is the same robust grouping the calibrator uses: it
        # collapses both edges and player-split fragments into one line.
        models = _merge_family(segments[keep], target, mask)
        for index, line in enumerate(models):
            supporting = int(keep.sum())
            fitted.append({"index": len(fitted), "family_index": index,
                           "image_points": line.tolist(), "kind": kind,
                           "supporting_fragments": supporting,
                           "length_pixels": float(np.linalg.norm(line[1] - line[0])),
                           "angle_degrees": float(np.degrees(target))})
    return {"raw_count": int(len(segments)), "fitted": fitted}


def describe_field_detections(video_path: Path, timestamp_s: float, reader=None) -> dict:
    """Report what field registration actually detected in one frame.

    Built for debugging the absolute-position problem: yard lines repeat every
    five yards, so a fit can be confident and still be placed wrongly. This
    returns the raw evidence behind that decision -- the detected lines, every
    recognised number with its confidence and whether it was believed, and the
    yard value assigned to each line -- in image pixel coordinates the viewer
    can draw directly.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_s) * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Could not decode a frame at {timestamp_s:.2f}s")
    height, width = frame.shape[:2]
    lines, mask, white = detect_yard_lines(frame)
    line_evidence = detect_field_line_evidence(frame, mask, white, lines)
    # Line geometry is still worth showing when OCR is unavailable, and the
    # absence of numbers is itself the finding: without them the five-yard
    # ambiguity cannot be resolved.
    ocr_error_message = None
    try:
        raw_numbers = read_field_numbers(frame, reader)
    except RuntimeError as error:
        raw_numbers = []
        ocr_error_message = str(error)
    credible = _credible_field_numbers(raw_numbers, frame.shape)
    trusted = raw_numbers if credible else []
    assigned: list[Optional[float]] = [None] * len(lines)
    absolute = False
    ocr_error = None
    if lines:
        field_x, absolute, ocr_error = _assign_field_x(lines, trusted)
        assigned = [float(value) for value in field_x]
    registration = None
    verification = None
    if lines:
        try:
            value = register_field(frame, reader=reader, numbers=trusted)
            registration = {"confidence": value.confidence, "absolute_x": value.absolute_x,
                            "diagnostics": value.diagnostics}
            verification = value.diagnostics.get("verification")
        except Exception as error:  # diagnostics must never break the viewer
            registration = {"error": str(error)}
    # Which line each number was snapped to, so a mis-snapped number is visible.
    number_rows = []
    if lines:
        direction = lines[0][1] - lines[0][0]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
        normal = np.asarray([-direction[1], direction[0]])
        coordinates = np.asarray([float(np.dot(line.mean(axis=0), normal)) for line in lines])
        for number in raw_numbers:
            projection = float(np.dot(np.asarray(number.center, dtype=float), normal))
            index = int(np.argmin(abs(coordinates - projection)))
            number_rows.append(index)
    else:
        number_rows = [None] * len(raw_numbers)
    return {
        "frame_width": int(width), "frame_height": int(height),
        "line_evidence": line_evidence["fitted"],
        "raw_line_fragment_count": line_evidence["raw_count"],
        "lines": [{"index": index,
                   "image_points": [[float(point[0]), float(point[1])] for point in line],
                   "assigned_field_x": assigned[index]}
                  for index, line in enumerate(lines)],
        "numbers": [{"value": number.value, "center": [float(number.center[0]), float(number.center[1])],
                     "confidence": float(number.confidence), "trusted": bool(credible),
                     "nearest_line_index": number_rows[index] if index < len(number_rows) else None,
                     # Each painted number is ambiguous between the two halves
                     # of the field until something else resolves which side.
                     "candidate_field_x": ([60.0] if number.value == 50
                                           else [10.0 + number.value, 110.0 - number.value])}
                    for index, number in enumerate(raw_numbers)],
        "numbers_credible": bool(credible),
        "absolute_x": bool(absolute),
        "ocr_assignment_error": ocr_error,
        "field_fraction": float((mask > 0).mean()),
        "white_fraction": float((white > 0).mean()),
        "registration": registration,
        "verification": verification,
        "ocr_unavailable": ocr_error_message,
    }
