from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .db import connect
from .geometry import estimate_homography, project_points


@dataclass
class ActiveTrack:
    track_id: int
    box: np.ndarray
    missed: int = 0


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


class BoxTracker:
    """Small deterministic tracker for establishing a no-training baseline."""

    def __init__(self, maximum_missed: int = 5, maximum_cost: float = 1.4):
        self.maximum_missed = maximum_missed
        self.maximum_cost = maximum_cost
        self.active: list[ActiveTrack] = []
        self.next_id = 1

    def update(self, boxes: np.ndarray, image_shape: tuple[int, int]) -> list[int]:
        boxes = np.asarray(boxes, dtype=float).reshape((-1, 4))
        assigned = [-1] * len(boxes)
        if self.active and len(boxes):
            diagonal = float(np.hypot(*image_shape))
            costs = np.empty((len(self.active), len(boxes)), dtype=float)
            for row, track in enumerate(self.active):
                old_center = (track.box[:2] + track.box[2:]) / 2
                for column, box in enumerate(boxes):
                    center = (box[:2] + box[2:]) / 2
                    costs[row, column] = (1 - _iou(track.box, box)) + np.linalg.norm(old_center - center) / diagonal
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] <= self.maximum_cost:
                    self.active[row].box = boxes[column]
                    self.active[row].missed = 0
                    assigned[column] = self.active[row].track_id
        matched_ids = set(assigned)
        for track in self.active:
            if track.track_id not in matched_ids:
                track.missed += 1
        self.active = [track for track in self.active if track.missed <= self.maximum_missed]
        for index, box in enumerate(boxes):
            if assigned[index] < 0:
                assigned[index] = self.next_id
                self.active.append(ActiveTrack(self.next_id, box))
                self.next_id += 1
        return assigned


class TorchvisionPersonDetector:
    model_version = "torchvision:fasterrcnn_resnet50_fpn_v2:coco"

    def __init__(self, device: Optional[str] = None, threshold: float = 0.35):
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transform = weights.transforms()
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()
        self.threshold = threshold

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(self.torch.from_numpy(rgb).permute(2, 0, 1)).to(self.device)
        with self.torch.inference_mode():
            result = self.model([tensor])[0]
        keep = (result["labels"] == 1) & (result["scores"] >= self.threshold)
        boxes = result["boxes"][keep].detach().cpu().numpy()
        scores = result["scores"][keep].detach().cpu().numpy()
        return boxes, scores


def _source_record(db_path: Path, play_id: str, source_order: int):
    with connect(db_path) as connection:
        row = connection.execute(
            """SELECT a.game_id,ps.angle,c.clip_id,c.start_s,c.end_s,c.snap_s,c.play_end_s,g.video_path
               FROM play_sources ps JOIN clips c ON c.clip_id=ps.clip_id
               JOIN play_alignments a ON a.play_id=ps.play_id JOIN games g ON g.game_id=a.game_id
               WHERE ps.play_id=? AND ps.source_order=?""", (play_id, source_order),
        ).fetchone()
    if not row:
        raise ValueError(f"No source {source_order} for {play_id}")
    return row


def detect_source(db_path: Path, play_id: str, source_order: int, output: Path,
                  sample_hz: float = 10.0, threshold: float = .35,
                  device: Optional[str] = None, detector=None) -> int:
    source = _source_record(db_path, play_id, source_order)
    detector = detector or TorchvisionPersonDetector(device, threshold)
    capture = cv2.VideoCapture(source["video_path"])
    if not capture.isOpened():
        raise ValueError(f"Could not open {source['video_path']}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    every = max(1, round(source_fps / sample_hz))
    capture.set(cv2.CAP_PROP_POS_MSEC, float(source["start_s"]) * 1000)
    tracker = BoxTracker()
    records = []
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp > float(source["end_s"]):
                break
            if decoded % every == 0:
                boxes, scores = detector.predict(frame)
                ids = tracker.update(boxes, frame.shape[:2])
                for track_id, box, score in zip(ids, boxes, scores):
                    x1, y1, x2, y2 = (float(value) for value in box)
                    records.append({"play_id": play_id, "clip_id": source["clip_id"],
                                    "source_order": source_order, "source_angle": source["angle"],
                                    "video_timestamp": timestamp, "track_id": f"{source_order}:{track_id}",
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                    "image_x": (x1 + x2) / 2, "image_y": y2,
                                    "confidence": float(score), "model_version": detector.model_version})
            decoded += 1
    finally:
        capture.release()
    if not records:
        raise ValueError("Person detector produced no player candidates")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return len(records)


def project_source(db_path: Path, play_id: str, source_order: int, detections: Path,
                   landmarks: Path, output: Path, video_anchor_s: Optional[float] = None,
                   bdb_anchor_frame: Optional[int] = None) -> dict:
    source = _source_record(db_path, play_id, source_order)
    if (video_anchor_s is None) != (bdb_anchor_frame is None):
        raise ValueError("video_anchor_s and bdb_anchor_frame must be provided together")
    if video_anchor_s is None and source["snap_s"] is None:
        raise ValueError("Source requires an audited snap timestamp or an explicit synchronization anchor")
    calibration_input = json.loads(landmarks.read_text(encoding="utf-8"))
    calibration = estimate_homography(calibration_input["image_points"], calibration_input["field_points"])
    frame = pd.read_parquet(detections)
    points = project_points(calibration.matrix, frame[["image_x", "image_y"]].itertuples(index=False, name=None))
    frame["x"] = points[:, 0]
    frame["y"] = points[:, 1]
    anchor_s = float(video_anchor_s if video_anchor_s is not None else source["snap_s"])
    frame["t"] = frame.video_timestamp.astype(float) - anchor_s
    frame["relative_frame"] = np.rint(frame.t * 10).astype(int)
    if bdb_anchor_frame is not None:
        # BDB 2026 input sequences end immediately before the pass. Manually
        # mark pass release in each film source and anchor it to max(frame_id).
        frame["frame_id"] = frame.relative_frame + int(bdb_anchor_frame)
    frame["calibration_inlier_ratio"] = calibration.inlier_ratio
    frame["calibration_median_error_yards"] = calibration.median_error_yards
    frame["calibration_p95_error_yards"] = calibration.p95_error_yards
    frame = frame[(frame.x.between(0, 120)) & (frame.y.between(0, 160 / 3))]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {"rows": len(frame), "anchor_video_s": anchor_s,
            "anchor_bdb_frame": bdb_anchor_frame, "inlier_ratio": calibration.inlier_ratio,
            "median_error_yards": calibration.median_error_yards,
            "p95_error_yards": calibration.p95_error_yards}


def fuse_sources(inputs: Iterable[Path], output: Path, maximum_distance: float = 3.0) -> int:
    sources = [pd.read_parquet(path) for path in inputs]
    if not sources:
        raise ValueError("At least one projected source is required")
    for source in sources:
        if not {"relative_frame", "x", "y", "track_id"}.issubset(source.columns):
            raise ValueError("Projected sources require relative_frame, x, y, and track_id")
    primary = sources[0].copy()
    fused_rows = []
    all_frames = sorted(set().union(*(set(value.relative_frame.astype(int)) for value in sources)))
    for frame_id in all_frames:
        base = primary[primary.relative_frame.astype(int) == frame_id].copy()
        if base.empty:
            base = sources[1][sources[1].relative_frame.astype(int) == frame_id].copy() if len(sources) > 1 else base
        for alternate in sources[1:]:
            other = alternate[alternate.relative_frame.astype(int) == frame_id]
            if base.empty or other.empty:
                continue
            left = base[["x", "y"]].to_numpy(float)
            right = other[["x", "y"]].to_numpy(float)
            distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            rows, columns = linear_sum_assignment(distances)
            for row, column in zip(rows, columns):
                if distances[row, column] <= maximum_distance:
                    base.iloc[row, base.columns.get_loc("x")] = (left[row, 0] + right[column, 0]) / 2
                    base.iloc[row, base.columns.get_loc("y")] = (left[row, 1] + right[column, 1]) / 2
        fused_rows.extend(base.to_dict(orient="records"))
    result = pd.DataFrame(fused_rows)
    if result.empty:
        raise ValueError("Fusion produced no trajectory points")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return len(result)
