from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .db import connect, transaction
from .geometry import FIELD_LENGTH, FIELD_WIDTH, estimate_homography, project_points


def save_calibration_keyframes(db_path: Path, clip_id: str, payload: dict) -> list[dict]:
    keyframes = payload.get("keyframes") or []
    if not keyframes:
        raise ValueError("Calibration requires at least one keyframe")
    with connect(db_path) as connection:
        clip = connection.execute("SELECT start_s,end_s FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    if not clip:
        raise ValueError(f"No camera shot found: {clip_id}")
    stored = []
    with transaction(db_path) as connection:
        revision_row = connection.execute(
            "SELECT COALESCE(MAX(revision),0)+1 AS revision FROM shot_calibration_keyframes WHERE clip_id=?",
            (clip_id,),
        ).fetchone()
        revision = int(revision_row["revision"])
        connection.execute("DELETE FROM shot_calibration_keyframes WHERE clip_id=?", (clip_id,))
        for value in sorted(keyframes, key=lambda item: float(item["timestamp_s"])):
            timestamp = float(value["timestamp_s"])
            if not float(clip["start_s"]) <= timestamp <= float(clip["end_s"]):
                raise ValueError(f"Calibration timestamp {timestamp} is outside the camera shot")
            calibration = estimate_homography(value["image_points"], value["field_points"])
            landmarks = {"image_points": value["image_points"], "field_points": value["field_points"]}
            matrix = calibration.matrix / calibration.matrix[2, 2]
            connection.execute(
                """INSERT INTO shot_calibration_keyframes
                   (clip_id,timestamp_s,landmarks_json,matrix_json,inlier_ratio,
                    median_error_yards,p95_error_yards,revision,status)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (clip_id, timestamp, json.dumps(landmarks), json.dumps(matrix.tolist()),
                 calibration.inlier_ratio, calibration.median_error_yards,
                 calibration.p95_error_yards, revision, "verified"),
            )
            stored.append({"timestamp_s": timestamp, "revision": revision,
                           "inlier_ratio": calibration.inlier_ratio,
                           "median_error_yards": calibration.median_error_yards,
                           "p95_error_yards": calibration.p95_error_yards})
    return stored


def _calibration_rows(db_path: Path, clip_id: str):
    with connect(db_path) as connection:
        rows = connection.execute(
            """SELECT * FROM shot_calibration_keyframes WHERE clip_id=? AND status='verified'
               ORDER BY timestamp_s""", (clip_id,),
        ).fetchall()
    if not rows:
        raise ValueError(f"No verified calibration exists for {clip_id}")
    return rows


def _matrix_at(rows, timestamp: float) -> tuple[np.ndarray, bool, float]:
    times = np.asarray([float(row["timestamp_s"]) for row in rows])
    if timestamp < times[0] or timestamp > times[-1]:
        index = 0 if timestamp < times[0] else len(rows) - 1
        return np.asarray(json.loads(rows[index]["matrix_json"]), dtype=float), False, float(rows[index]["p95_error_yards"])
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right == 0 or times[right] == timestamp:
        row = rows[right]
        return np.asarray(json.loads(row["matrix_json"]), dtype=float), True, float(row["p95_error_yards"])
    left = right - 1
    fraction = (timestamp - times[left]) / (times[right] - times[left])
    first = np.asarray(json.loads(rows[left]["matrix_json"]), dtype=float)
    second = np.asarray(json.loads(rows[right]["matrix_json"]), dtype=float)
    matrix = (1 - fraction) * first + fraction * second
    matrix /= matrix[2, 2]
    p95 = (1 - fraction) * float(rows[left]["p95_error_yards"]) + fraction * float(rows[right]["p95_error_yards"])
    return matrix, True, p95


def project_clip(db_path: Path, clip_id: str, detections: Path, output: Path) -> dict:
    frame = pd.read_parquet(detections)
    if frame.empty or not {"clip_id", "video_timestamp", "contact_x", "contact_y"}.issubset(frame.columns):
        raise ValueError("Clip detections are empty or missing contact points")
    if set(frame.clip_id.astype(str)) != {clip_id}:
        raise ValueError("Detection artifact does not belong to the requested clip")
    rows = _calibration_rows(db_path, clip_id)
    projected_parts = []
    for timestamp, group in frame.groupby("video_timestamp", sort=True):
        matrix, valid, p95 = _matrix_at(rows, float(timestamp))
        points = project_points(matrix, group[["contact_x", "contact_y"]].itertuples(index=False, name=None))
        value = group.copy()
        value["field_x"] = points[:, 0]
        value["field_y"] = points[:, 1]
        value["calibration_valid"] = bool(valid)
        value["calibration_p95_yards"] = p95
        value["on_field"] = (value.field_x.between(0, FIELD_LENGTH) & value.field_y.between(0, FIELD_WIDTH))
        projected_parts.append(value)
    result = pd.concat(projected_parts, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    with transaction(db_path) as connection:
        game_id = connection.execute("SELECT game_id FROM clips WHERE clip_id=?", (clip_id,)).fetchone()["game_id"]
        connection.execute(
            """INSERT INTO artifacts(game_id,clip_id,kind,path,metadata_json,input_revision)
               VALUES(?,?,?,?,?,?)""",
            (game_id, clip_id, "clip_projection", str(output.resolve()),
             json.dumps({"rows": len(result), "valid_fraction": float(result.calibration_valid.mean()),
                         "on_field_fraction": float(result.on_field.mean())}),
             str(max(int(row["revision"]) for row in rows))),
        )
    return {"rows": len(result), "valid_fraction": float(result.calibration_valid.mean()),
            "on_field_fraction": float(result.on_field.mean())}


def assign_team_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    usable = result[["lab_a", "lab_b"]].astype(float).to_numpy()
    finite = np.isfinite(usable).all(axis=1)
    result["team"] = "unknown"
    result["team_confidence"] = 0.0
    if finite.sum() < 4:
        return result
    values = usable[finite]
    first = values[np.argmin(values[:, 0])]
    second = values[np.argmax(np.linalg.norm(values - first, axis=1))]
    centers = np.vstack([first, second])
    for _ in range(20):
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        updated = np.vstack([values[labels == index].mean(axis=0) if np.any(labels == index) else centers[index]
                             for index in range(2)])
        if np.allclose(updated, centers):
            break
        centers = updated
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    ordered = np.sort(distances, axis=1)
    confidence = 1 - ordered[:, 0] / np.maximum(ordered[:, 1], 1e-6)
    indices = np.flatnonzero(finite)
    confident = confidence >= .15
    result.loc[result.index[indices[confident]], "team"] = [f"team_{value}" for value in labels[confident]]
    result.loc[result.index[indices], "team_confidence"] = confidence
    return result


@dataclass
class FieldTrack:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    team: str
    appearance: np.ndarray
    timestamp: float
    missed: int = 0


class FieldSpaceTracker:
    def __init__(self, maximum_missed: int = 8, maximum_speed_yps: float = 15.0):
        self.maximum_missed = maximum_missed
        self.maximum_speed_yps = maximum_speed_yps
        self.active: list[FieldTrack] = []
        self.next_id = 1

    def update(self, detections: pd.DataFrame, timestamp: float) -> list[int]:
        assigned = [-1] * len(detections)
        positions = detections[["field_x", "field_y"]].to_numpy(float)
        appearances = detections[["lab_a", "lab_b"]].fillna(128).to_numpy(float)
        teams = detections.team.astype(str).tolist()
        if self.active and len(detections):
            costs = np.full((len(self.active), len(detections)), 1e6, dtype=float)
            for row, track in enumerate(self.active):
                elapsed = max(timestamp - track.timestamp, 1e-3)
                predicted = track.position + track.velocity * elapsed
                for column, position in enumerate(positions):
                    distance = float(np.linalg.norm(predicted - position))
                    if distance > self.maximum_speed_yps * elapsed + 1.5:
                        continue
                    team_penalty = 0 if track.team == "unknown" or teams[column] == "unknown" or track.team == teams[column] else 8
                    appearance = float(np.linalg.norm(track.appearance - appearances[column])) / 25
                    costs[row, column] = distance + appearance + team_penalty
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] >= 1e5:
                    continue
                track = self.active[row]
                elapsed = max(timestamp - track.timestamp, 1e-3)
                measured_velocity = (positions[column] - track.position) / elapsed
                track.velocity = .6 * track.velocity + .4 * measured_velocity
                track.position = positions[column]
                track.appearance = .8 * track.appearance + .2 * appearances[column]
                if track.team == "unknown" and teams[column] != "unknown":
                    track.team = teams[column]
                track.timestamp = timestamp
                track.missed = 0
                assigned[column] = track.track_id
        matched = set(assigned)
        for track in self.active:
            if track.track_id not in matched:
                track.missed += 1
        self.active = [track for track in self.active if track.missed <= self.maximum_missed]
        for index, position in enumerate(positions):
            if assigned[index] >= 0:
                continue
            track = FieldTrack(self.next_id, position, np.zeros(2), teams[index], appearances[index], timestamp)
            self.active.append(track)
            assigned[index] = self.next_id
            self.next_id += 1
        return assigned


def track_projected_clip(db_path: Path, clip_id: str, projected: Path, output: Path) -> dict:
    frame = assign_team_probabilities(pd.read_parquet(projected))
    frame = frame[frame.on_field & frame.calibration_valid].copy()
    if frame.empty:
        raise ValueError("No valid on-field detections are available for tracking")
    tracker = FieldSpaceTracker()
    parts = []
    for timestamp, group in frame.groupby("video_timestamp", sort=True):
        value = group.copy()
        ids = tracker.update(value, float(timestamp))
        value["track_id"] = [f"{clip_id}:t{item}" for item in ids]
        active = {track.track_id: track for track in tracker.active}
        value["vx"] = [float(active[item].velocity[0]) for item in ids]
        value["vy"] = [float(active[item].velocity[1]) for item in ids]
        value["speed"] = np.hypot(value.vx, value.vy)
        value["observed"] = True
        value["interpolated"] = False
        parts.append(value)
    result = pd.concat(parts, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    with transaction(db_path) as connection:
        game_id = connection.execute("SELECT game_id FROM clips WHERE clip_id=?", (clip_id,)).fetchone()["game_id"]
        connection.execute(
            "INSERT INTO artifacts(game_id,clip_id,kind,path,metadata_json) VALUES(?,?,?,?,?)",
            (game_id, clip_id, "clip_tracks", str(output.resolve()),
             json.dumps({"rows": len(result), "tracks": int(result.track_id.nunique()),
                         "team_labeled_fraction": float((result.team != 'unknown').mean())})),
        )
    return {"rows": len(result), "tracks": int(result.track_id.nunique()),
            "team_labeled_fraction": float((result.team != "unknown").mean())}
