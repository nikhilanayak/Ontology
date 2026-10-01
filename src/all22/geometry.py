from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0


@dataclass(frozen=True)
class Calibration:
    matrix: np.ndarray
    inlier_ratio: float
    median_error_yards: float
    p95_error_yards: float


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
    return Calibration(matrix, float(mask.mean()), float(np.median(errors)), float(np.percentile(errors, 95)))


def project_points(matrix: np.ndarray, points: Iterable[Tuple[float, float]]) -> np.ndarray:
    source = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(source, matrix).reshape(-1, 2)


def normalize_direction(x: float, y: float, play_direction: str) -> tuple[float, float]:
    if (play_direction or "").lower() == "left":
        return FIELD_LENGTH - x, FIELD_WIDTH - y
    return x, y
