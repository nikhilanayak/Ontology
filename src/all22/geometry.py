from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0
NUMBER_ROW_YARDS = 12.0


@dataclass(frozen=True)
class Calibration:
    matrix: np.ndarray
    inlier_ratio: float
    median_error_yards: float
    p95_error_yards: float
    inlier_p95_error_yards: float


def estimate_homography(image_points: Iterable[Tuple[float, float]], field_points: Iterable[Tuple[float, float]]) -> Calibration:
    image = np.asarray(list(image_points), dtype=np.float32)
    field = np.asarray(list(field_points), dtype=np.float32)
    if image.shape != field.shape or image.ndim != 2 or image.shape[0] < 4 or image.shape[1] != 2:
        raise ValueError("At least four paired 2-D landmarks are required")
    matrix, mask = cv2.findHomography(image, field, cv2.RANSAC, 1.5)
    if matrix is None:
        raise ValueError("Homography estimation failed")
    projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - field, axis=1)
    inliers = errors[mask.reshape(-1).astype(bool)]
    return Calibration(matrix, float(mask.mean()), float(np.median(errors)), float(np.percentile(errors, 95)),
                       float(np.percentile(inliers, 95)) if len(inliers) else float("inf"))


def project_points(matrix: np.ndarray, points: Iterable[Tuple[float, float]]) -> np.ndarray:
    source = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(source, matrix).reshape(-1, 2)


def _semantic_lines(annotations: list[dict]) -> tuple[list[list[float]], list[list[float]], dict]:
    lines = [np.asarray(value["image_points"], dtype=float) for value in annotations if value.get("kind") == "line"]
    if len(lines) < 2 or any(value.shape != (2, 2) for value in lines):
        raise ValueError("Line calibration needs at least two yard lines, each clicked at both sidelines")
    directions = []
    for line in lines:
        vector = line[1] - line[0]
        vector /= max(float(np.linalg.norm(vector)), 1e-9)
        if directions and np.dot(vector, directions[0]) < 0:
            vector *= -1
        directions.append(vector)
    across = np.mean(directions, axis=0)
    across /= max(float(np.linalg.norm(across)), 1e-9)
    downfield = np.asarray([-across[1], across[0]])
    ordered = sorted(lines, key=lambda line: float(np.dot(line.mean(axis=0), downfield)))
    # Absolute yard-line identity is intentionally deferred. Centering the
    # clicked five-yard lines at midfield preserves metric trajectories; PBP
    # supplies the absolute offset later.
    origin = 60.0 - 2.5 * (len(ordered) - 1)
    image_points: list[list[float]] = []
    field_points: list[list[float]] = []
    for index, line in enumerate(ordered):
        endpoints = sorted(line, key=lambda point: float(np.dot(point, across)))
        yard_x = origin + 5.0 * index
        image_points.extend(point.tolist() for point in endpoints)
        field_points.extend([[yard_x, 0.0], [yard_x, FIELD_WIDTH]])
    return image_points, field_points, {"mode": "lines", "absolute_x": False, "line_count": len(ordered)}


def _number_x_sequence(values: list[int]) -> list[float]:
    candidates = [[60.0] if value == 50 else [10.0 + value, 110.0 - value] for value in values]
    paths: list[list[float]] = [[]]
    for options in candidates:
        paths = [path + [option] for path in paths for option in options
                 if not path or option >= path[-1] - 1e-6]
    if not paths:
        raise ValueError("Number clicks are not in a consistent downfield order; include the 50 to resolve midfield")
    return min(paths, key=lambda path: sum(abs((path[i] - path[i - 1])) for i in range(1, len(path))))


def _semantic_numbers(annotations: list[dict]) -> tuple[list[list[float]], list[list[float]], dict]:
    numbers = [value for value in annotations if value.get("kind") == "number"]
    if len(numbers) < 4:
        raise ValueError("Number calibration needs at least four number clicks across both painted number rows")
    values = [int(value["value"]) for value in numbers]
    if any(value not in {10, 20, 30, 40, 50} for value in values):
        raise ValueError("Field numbers must be 10, 20, 30, 40, or 50")
    image = np.asarray([value["image_point"] for value in numbers], dtype=float)
    centered = image - image.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    along = centered @ axes[0]
    across = centered @ axes[1]
    # Split the near/far painted-number rows at the largest transverse gap.
    order_across = np.argsort(across)
    gaps = np.diff(across[order_across])
    if not len(gaps) or float(np.max(gaps)) < 2:
        raise ValueError("Click matching numbers on both sides of the field")
    split = int(np.argmax(gaps)) + 1
    low = set(order_across[:split].tolist())
    row_y = [NUMBER_ROW_YARDS if index in low else FIELD_WIDTH - NUMBER_ROW_YARDS
             for index in range(len(numbers))]
    order = np.argsort(along).tolist()
    ordered_values = [values[index] for index in order]
    try:
        ordered_x = _number_x_sequence(ordered_values)
    except ValueError:
        order = list(reversed(order))
        ordered_values = [values[index] for index in order]
        ordered_x = _number_x_sequence(ordered_values)
    field_x = [0.0] * len(numbers)
    for index, value in zip(order, ordered_x):
        field_x[index] = value
    field = [[field_x[index], row_y[index]] for index in range(len(numbers))]
    return image.tolist(), field, {"mode": "numbers", "absolute_x": True, "number_count": len(numbers)}


def semantic_correspondences(annotations: list[dict]) -> tuple[list[list[float]], list[list[float]], dict]:
    """Convert human-friendly line or painted-number clicks into point pairs."""
    kinds = {value.get("kind") for value in annotations}
    if kinds == {"line"}:
        return _semantic_lines(annotations)
    if kinds == {"number"}:
        return _semantic_numbers(annotations)
    raise ValueError("Use either yard lines or field numbers within one keyframe")


def normalize_direction(x: float, y: float, play_direction: str) -> tuple[float, float]:
    if (play_direction or "").lower() == "left":
        return FIELD_LENGTH - x, FIELD_WIDTH - y
    return x, y
