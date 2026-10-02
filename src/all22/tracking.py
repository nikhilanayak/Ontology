from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .db import connect, transaction
from .field_tracking import config_hash
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
    """Person detector for All-22 film.

    Runs at native 1080p in fp16 and keeps a low-score tier. The two-stage
    associator needs weak detections to recover occluded players, so the
    detection threshold must stay below the tracker's birth threshold; a .35
    cut baked into the artifact cannot be undone later.
    """

    detector_revision = "r2"

    def __init__(self, device: Optional[str] = None, threshold: float = .15,
                 minimum_size: int = 1080, maximum_size: int = 1920,
                 half: bool = True, batch_size: int = 4):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = bool(half) and self.device.type == "cuda"
        self.batch_size = max(1, int(batch_size))
        self.minimum_size, self.maximum_size = int(minimum_size), int(maximum_size)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transform = weights.transforms()
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, min_size=self.minimum_size, max_size=self.maximum_size,
        ).to(self.device).eval()
        appearance_weights = ResNet18_Weights.DEFAULT
        self.appearance_transform = appearance_weights.transforms()
        self.appearance_model = resnet18(weights=appearance_weights).to(self.device).eval()
        self.appearance_model.fc = torch.nn.Identity()
        generator = np.random.default_rng(20261002)
        projection = (generator.normal(size=(512, 64)) / np.sqrt(64)).astype(np.float32)
        self.appearance_projection = torch.from_numpy(projection).to(self.device)
        self.threshold = threshold

    @property
    def model_version(self) -> str:
        precision = "fp16" if self.half else "fp32"
        return ("torchvision:fasterrcnn_resnet50_fpn_v2:coco+resnet18:imagenet:embed64"
                f":{self.detector_revision}:{self.minimum_size}x{self.maximum_size}:{precision}")

    def _autocast(self):
        if self.half:
            return self.torch.autocast("cuda", dtype=self.torch.float16)
        return contextlib.nullcontext()

    def _tensor(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(self.torch.from_numpy(rgb).permute(2, 0, 1)).to(self.device)

    def predict_batch(self, frames: list) -> list:
        """Detect people in several frames at once to keep the GPU busy."""
        if not frames:
            return []
        results = []
        for start in range(0, len(frames), self.batch_size):
            chunk = [self._tensor(frame) for frame in frames[start:start + self.batch_size]]
            with self.torch.inference_mode(), self._autocast():
                outputs = self.model(chunk)
            for output in outputs:
                keep = (output["labels"] == 1) & (output["scores"] >= self.threshold)
                results.append((output["boxes"][keep].float().detach().cpu().numpy(),
                                output["scores"][keep].float().detach().cpu().numpy()))
        return results

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_batch([frame])[0]

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        crops = []
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(float)
            left, right = int(max(0, x1)), int(min(width, x2))
            top, bottom = int(max(0, y1)), int(min(height, y1 + .8 * (y2 - y1)))
            crop = rgb[top:bottom, left:right]
            if not crop.size:
                crop = np.zeros((32, 16, 3), dtype=np.uint8)
            crops.append(self.appearance_transform(self.torch.from_numpy(crop).permute(2, 0, 1)))
        if not crops:
            return np.empty((0, 64), dtype=np.float32)
        with self.torch.inference_mode(), self._autocast():
            features = self.appearance_model(self.torch.stack(crops).to(self.device))
            features = features.float() @ self.appearance_projection
            features = self.torch.nn.functional.normalize(features, dim=1)
        return features.float().detach().cpu().numpy()


def _appearance_features(frame: np.ndarray, box: np.ndarray) -> dict:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    # The upper torso is more useful for team color than helmets, legs, or turf.
    left = max(0, min(width - 1, round(x1 + .20 * (x2 - x1))))
    right = max(left + 1, min(width, round(x1 + .80 * (x2 - x1))))
    top = max(0, min(height - 1, round(y1 + .10 * (y2 - y1))))
    bottom = max(top + 1, min(height, round(y1 + .60 * (y2 - y1))))
    crop = frame[top:bottom, left:right]
    if not crop.size:
        return {name: None for name in ("lab_l", "lab_a", "lab_b", "hsv_h", "hsv_s", "hsv_v")}
    lab = np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
    hsv = np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)
    return {"lab_l": float(lab[0]), "lab_a": float(lab[1]), "lab_b": float(lab[2]),
            "hsv_h": float(hsv[0]), "hsv_s": float(hsv[1]), "hsv_v": float(hsv[2])}


def _on_field_detections(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                         margin_pixels: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Remove people whose ground-contact point is outside the playing surface."""
    boxes = np.asarray(boxes).reshape((-1, 4))
    scores = np.asarray(scores)
    if not len(boxes):
        return boxes, scores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (22, 25, 25), (105, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return boxes, scores
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * .12:
        return boxes, scores
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(contour)], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((2 * margin_pixels + 1, 2 * margin_pixels + 1), np.uint8))
    height, width = mask.shape
    keep = []
    for box in boxes:
        x = int(np.clip(round((float(box[0]) + float(box[2])) / 2), 0, width - 1))
        y = int(np.clip(round(float(box[3])), 0, height - 1))
        keep.append(bool(mask[y, x]))
    keep = np.asarray(keep, dtype=bool)
    return boxes[keep], scores[keep]


def _clip_record(db_path: Path, clip_id: str):
    with connect(db_path) as connection:
        row = connection.execute(
            """SELECT c.*,g.video_path FROM clips c JOIN games g ON g.game_id=c.game_id
               WHERE c.clip_id=?""", (clip_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"No camera shot found: {clip_id}")
    return row


def _config_hash(value: dict) -> str:
    return config_hash(value)


def detect_clip(db_path: Path, clip_id: str, output: Path, sample_hz: float = 10.0,
                threshold: float = .15, device: Optional[str] = None, detector=None,
                batch_size: int = 4) -> int:
    """Detect people in one hard-cut-bounded shot, independent of PBP alignment."""
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    source = _clip_record(db_path, clip_id)
    detector = detector or TorchvisionPersonDetector(device, threshold, batch_size=batch_size)
    capture = cv2.VideoCapture(source["video_path"])
    if not capture.isOpened():
        raise ValueError(f"Could not open {source['video_path']}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    every = max(1, round(source_fps / sample_hz))
    capture.set(cv2.CAP_PROP_POS_MSEC, float(source["start_s"]) * 1000)
    records = []
    decoded = sample_index = 0
    pending: list[tuple[float, np.ndarray]] = []
    batched = getattr(detector, "predict_batch", None)
    width = max(1, int(batch_size) if batched else 1)

    def flush() -> None:
        nonlocal sample_index, pending
        if not pending:
            return
        frames = [frame for _, frame in pending]
        results = batched(frames) if batched else [detector.predict(frame) for frame in frames]
        for (timestamp, frame), (boxes, scores) in zip(pending, results):
            boxes, scores = _on_field_detections(frame, boxes, scores)
            embeddings = detector.encode(frame, boxes) if hasattr(detector, "encode") else None
            for detection_index, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = (float(value) for value in box)
                records.append({"game_id": source["game_id"], "clip_id": clip_id,
                                "frame_id": sample_index, "video_timestamp": timestamp,
                                "detection_id": detection_index, "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2, "contact_x": (x1 + x2) / 2,
                                "contact_y": y2, "confidence": float(score),
                                "model_version": detector.model_version,
                                **_appearance_features(frame, np.asarray(box)),
                                **({f"emb_{column:02d}": float(value)
                                    for column, value in enumerate(embeddings[detection_index])}
                                   if embeddings is not None else {})})
            sample_index += 1
        pending = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp > float(source["end_s"]):
                break
            if decoded % every == 0:
                pending.append((timestamp, frame))
                if len(pending) >= width:
                    flush()
            decoded += 1
        flush()
    finally:
        capture.release()
    if not records:
        raise ValueError(f"Person detector produced no candidates for {clip_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    config = {"sample_hz": sample_hz, "threshold": threshold,
              "model_version": detector.model_version}
    with transaction(db_path) as connection:
        connection.execute(
            """INSERT INTO artifacts(game_id,clip_id,kind,path,model_version,metadata_json,config_hash,input_revision)
               VALUES(?,?,?,?,?,?,?,?)""",
            (source["game_id"], clip_id, "clip_detections", str(output.resolve()),
             detector.model_version, json.dumps({"rows": len(records), **config}),
             _config_hash(config), f"{source['start_s']:.3f}:{source['end_s']:.3f}"),
        )
    return len(records)


def detect_clips(db_path: Path, game_id: str, output_dir: Path, sample_hz: float = 10.0,
                 threshold: float = .15, device: Optional[str] = None,
                 clip_ids: Optional[list[str]] = None, limit: Optional[int] = None,
                 resume: bool = True, detector=None, batch_size: int = 4) -> list[dict]:
    """Batch clip detection while loading the model only once."""
    with connect(db_path) as connection:
        available = connection.execute(
            "SELECT clip_id FROM clips WHERE game_id=? ORDER BY start_s", (game_id,),
        ).fetchall()
    wanted = set(clip_ids or [])
    selected = [row["clip_id"] for row in available if not wanted or row["clip_id"] in wanted]
    if wanted - set(selected):
        raise ValueError(f"Unknown clip IDs: {sorted(wanted - set(selected))}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    detector = detector or TorchvisionPersonDetector(device, threshold, batch_size=batch_size)
    results = []
    for clip_id in selected:
        output = output_dir / game_id / f"{clip_id}.parquet"
        if resume and output.exists():
            results.append({"clip_id": clip_id, "output": str(output), "status": "cached"})
            continue
        rows = detect_clip(db_path, clip_id, output, sample_hz, threshold, device, detector,
                           batch_size=batch_size)
        results.append({"clip_id": clip_id, "output": str(output), "status": "created", "rows": rows})
    return results


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
                boxes, scores = _on_field_detections(frame, boxes, scores)
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
