from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .db import connect, transaction
from .geometry import FIELD_LENGTH, FIELD_WIDTH, estimate_homography, project_points, semantic_correspondences


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
            if value.get("annotations"):
                image_points, field_points, semantic = semantic_correspondences(value["annotations"])
            else:
                image_points, field_points = value["image_points"], value["field_points"]
                semantic = {"mode": "points", "absolute_x": True}
            calibration = estimate_homography(image_points, field_points)
            landmarks = {"image_points": image_points, "field_points": field_points,
                         "annotations": value.get("annotations", []), "semantic": semantic,
                         "source": value.get("source", "manual"),
                         "registration_confidence": value.get("registration_confidence"),
                         "registration_diagnostics": value.get("registration_diagnostics", {}),
                         "sample_timestamp_s": value.get("sample_timestamp_s"),
                         "anchor_timestamp_s": value.get("anchor_timestamp_s"),
                         "diagnostic_image": value.get("diagnostic_image")}
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
                           "semantic": semantic,
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


def assign_team_probabilities(frame: pd.DataFrame, anchor_timestamp: Optional[float] = None) -> pd.DataFrame:
    result = frame.copy()
    scale_by_column = {"lab_l": 50, "lab_a": 30, "lab_b": 30, "hsv_s": 60}
    columns = [column for column in scale_by_column if column in result]
    usable = result[columns].astype(float).to_numpy()
    finite = np.isfinite(usable).all(axis=1)
    result["team"] = "unknown"
    result["team_confidence"] = 0.0
    result["person_role"] = "unknown"
    if finite.sum() < 4:
        return result
    scales = np.asarray([scale_by_column[column] for column in columns], dtype=float)
    normalized = usable / scales
    fit_mask = finite.copy()
    if anchor_timestamp is not None and "video_timestamp" in result:
        distance = (pd.to_numeric(result.video_timestamp, errors="coerce") - anchor_timestamp).abs()
        fit_mask &= distance <= float(distance[finite].min()) + .02
    fit_values = normalized[fit_mask]
    clusters = 3 if anchor_timestamp is not None and len(fit_values) >= 9 else 2
    if len(fit_values) < clusters:
        fit_values = normalized[finite]
        clusters = 2
    # Deterministic farthest-point initialization avoids a sklearn dependency.
    first = fit_values[np.argmin(fit_values[:, columns.index("lab_a")])]
    centers = [first]
    while len(centers) < clusters:
        distance = np.min(np.linalg.norm(fit_values[:, None, :] - np.asarray(centers)[None, :, :], axis=2), axis=1)
        centers.append(fit_values[np.argmax(distance)])
    centers = np.asarray(centers)
    for _ in range(20):
        distances = np.linalg.norm(fit_values[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        updated = np.vstack([fit_values[labels == index].mean(axis=0) if np.any(labels == index) else centers[index]
                             for index in range(clusters)])
        if np.allclose(updated, centers):
            break
        centers = updated
    fit_labels = np.linalg.norm(fit_values[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    counts = np.bincount(fit_labels, minlength=clusters)
    if clusters == 3 and int(counts.min()) > max(2, int(.18 * len(fit_values))):
        # A genuine official group should be distinctly smaller than either
        # 11-player team. Three similarly sized clusters usually mean lighting
        # or uniform accents split one team, so refit as two teams.
        centers = centers[np.argsort(-counts)[:2]]
        clusters = 2
        for _ in range(20):
            distances = np.linalg.norm(fit_values[:, None, :] - centers[None, :, :], axis=2)
            labels = distances.argmin(axis=1)
            updated = np.vstack([fit_values[labels == index].mean(axis=0)
                                 if np.any(labels == index) else centers[index]
                                 for index in range(clusters)])
            if np.allclose(updated, centers):
                break
            centers = updated
        fit_labels = np.linalg.norm(fit_values[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
        counts = np.bincount(fit_labels, minlength=clusters)
    player_clusters = set(np.argsort(-counts)[:2].tolist())
    values = normalized[finite]
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    ordered = np.sort(distances, axis=1)
    confidence = 1 - ordered[:, 0] / np.maximum(ordered[:, 1], 1e-6)
    indices = np.flatnonzero(finite)
    confident = confidence >= .15
    team_names = {cluster: f"team_{order}" for order, cluster in enumerate(sorted(player_clusters))}
    names = [team_names.get(int(value), "official") for value in labels[confident]]
    result.loc[result.index[indices[confident]], "team"] = names
    result.loc[result.index[indices[confident]], "person_role"] = [
        "player" if name.startswith("team_") else "official" for name in names]
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
    hits: int = 1
    confidence: float = 1.0
    box_shape: Optional[np.ndarray] = None


class FieldSpaceTracker:
    """Associate detections after camera motion has been removed by field projection.

    The two association passes follow the useful part of ByteTrack's design:
    confident detections establish the assignment and weaker detections may only
    recover tracks that remain unmatched.  This prevents a weak duplicate from
    stealing an established identity while still bridging brief detector misses.
    """

    appearance_columns = ("lab_l", "lab_a", "lab_b", "hsv_h", "hsv_s", "hsv_v")
    appearance_scale = np.asarray([50, 30, 30, 45, 60, 60], dtype=float)

    def __init__(self, maximum_missed: int = 15, maximum_speed_yps: float = 15.0,
                 high_confidence: float = .55, low_confidence: float = .15,
                 crowd_multiplier: float = 1.6, crowd_floor: int = 32):
        self.maximum_missed = maximum_missed
        self.maximum_speed_yps = maximum_speed_yps
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.crowd_multiplier = crowd_multiplier
        self.crowd_floor = crowd_floor
        self.nominal_detection_counts: list[int] = []
        self.active: list[FieldTrack] = []
        self.next_id = 1
        self.last_crowd_burst = False

    def _appearance(self, detections: pd.DataFrame) -> np.ndarray:
        values = np.full((len(detections), len(self.appearance_columns)), np.nan, dtype=float)
        for column_index, column in enumerate(self.appearance_columns):
            if column in detections:
                values[:, column_index] = pd.to_numeric(detections[column], errors="coerce")
        defaults = np.asarray([128, 128, 128, 90, 80, 128], dtype=float)
        return np.where(np.isfinite(values), values, defaults)

    def _associate(self, track_indices: list[int], detection_indices: list[int],
                   positions: np.ndarray, appearances: np.ndarray, teams: list[str],
                   confidences: np.ndarray, box_shapes: np.ndarray, timestamp: float,
                   assigned: list[int]) -> set[int]:
        if not track_indices or not detection_indices:
            return set()
        costs = np.full((len(track_indices), len(detection_indices)), 1e6, dtype=float)
        for cost_row, track_index in enumerate(track_indices):
            track = self.active[track_index]
            elapsed = max(timestamp - track.timestamp, 1e-3)
            predicted = track.position + track.velocity * elapsed
            # Allow modest registration noise as well as physically plausible motion.
            maximum_distance = self.maximum_speed_yps * elapsed + 1.5 + .15 * track.missed
            for cost_column, detection_index in enumerate(detection_indices):
                distance = float(np.linalg.norm(predicted - positions[detection_index]))
                if distance > maximum_distance:
                    continue
                candidate_team = teams[detection_index]
                team_penalty = (3.0 if track.team != "unknown" and candidate_team != "unknown"
                                and track.team != candidate_team else 0.0)
                appearance = float(np.linalg.norm(
                    (track.appearance - appearances[detection_index]) / self.appearance_scale))
                shape_penalty = 0.0
                if track.box_shape is not None and np.all(box_shapes[detection_index] > 0):
                    shape_change = np.abs(np.log(box_shapes[detection_index] / track.box_shape))
                    if float(shape_change.max()) > 1.0:
                        continue
                    shape_penalty = .6 * float(np.linalg.norm(shape_change))
                confidence_penalty = .5 * (1 - confidences[detection_index])
                costs[cost_row, cost_column] = (distance + .8 * appearance + shape_penalty
                                                + team_penalty + confidence_penalty)
        rows, columns = linear_sum_assignment(costs)
        matched_tracks = set()
        for cost_row, cost_column in zip(rows, columns):
            if costs[cost_row, cost_column] >= 1e5:
                continue
            track_index = track_indices[cost_row]
            detection_index = detection_indices[cost_column]
            track = self.active[track_index]
            elapsed = max(timestamp - track.timestamp, 1e-3)
            measured_velocity = (positions[detection_index] - track.position) / elapsed
            track.velocity = .7 * track.velocity + .3 * measured_velocity
            track.position = positions[detection_index]
            track.appearance = .85 * track.appearance + .15 * appearances[detection_index]
            if track.team == "unknown" and teams[detection_index] != "unknown":
                track.team = teams[detection_index]
            track.timestamp = timestamp
            track.missed = 0
            track.hits += 1
            track.confidence = .8 * track.confidence + .2 * confidences[detection_index]
            if np.all(box_shapes[detection_index] > 0):
                track.box_shape = (.8 * track.box_shape + .2 * box_shapes[detection_index]
                                   if track.box_shape is not None else box_shapes[detection_index].copy())
            assigned[detection_index] = track.track_id
            matched_tracks.add(track_index)
        return matched_tracks

    def update(self, detections: pd.DataFrame, timestamp: float) -> list[int]:
        assigned = [-1] * len(detections)
        positions = detections[["field_x", "field_y"]].to_numpy(float)
        appearances = self._appearance(detections)
        teams = detections.team.astype(str).tolist()
        if {"x1", "y1", "x2", "y2"}.issubset(detections.columns):
            box_shapes = np.column_stack([
                detections.x2.to_numpy(float) - detections.x1.to_numpy(float),
                detections.y2.to_numpy(float) - detections.y1.to_numpy(float),
            ])
        else:
            box_shapes = np.ones((len(detections), 2), dtype=float)
        confidences = (pd.to_numeric(detections["confidence"], errors="coerce").fillna(1).to_numpy(float)
                       if "confidence" in detections else np.ones(len(detections), dtype=float))
        confidences = np.clip(confidences, 0, 1)
        if self.active and len(detections):
            all_tracks = list(range(len(self.active)))
            high = np.flatnonzero(confidences >= self.high_confidence).tolist()
            matched_tracks = self._associate(
                all_tracks, high, positions, appearances, teams, confidences, box_shapes, timestamp, assigned)
            remaining_tracks = [index for index in all_tracks if index not in matched_tracks]
            low = np.flatnonzero((confidences >= self.low_confidence)
                                 & (confidences < self.high_confidence)).tolist()
            matched_tracks |= self._associate(
                remaining_tracks, low, positions, appearances, teams, confidences, box_shapes, timestamp, assigned)
        else:
            matched_tracks = set()
        for track_index, track in enumerate(self.active):
            if track_index not in matched_tracks:
                track.missed += 1
        self.active = [track for track in self.active if track.missed <= self.maximum_missed]
        nominal = float(np.median(self.nominal_detection_counts)) if self.nominal_detection_counts else len(detections)
        crowd_burst = len(detections) > max(self.crowd_floor, self.crowd_multiplier * nominal)
        self.last_crowd_burst = crowd_burst
        if not crowd_burst:
            self.nominal_detection_counts.append(len(detections))
            self.nominal_detection_counts = self.nominal_detection_counts[-30:]
        # Detector outputs are normally score ordered, but make identity births
        # deterministic. During a sudden bench/crowd burst, existing players may
        # still match while unrelated people cannot spawn new trajectories.
        for index in np.argsort(-confidences):
            position = positions[index]
            if assigned[index] >= 0:
                continue
            # Very weak detections may recover an existing track but do not create
            # new identities of their own.
            if confidences[index] < self.high_confidence:
                continue
            if crowd_burst:
                continue
            track = FieldTrack(self.next_id, position, np.zeros(2), teams[index], appearances[index],
                               timestamp, confidence=float(confidences[index]),
                               box_shape=box_shapes[index].copy())
            self.active.append(track)
            assigned[index] = self.next_id
            self.next_id += 1
        return assigned


def track_projected_clip(db_path: Path, clip_id: str, projected: Path, output: Path) -> dict:
    with connect(db_path) as connection:
        action = connection.execute(
            "SELECT snap_s FROM action_windows WHERE clip_id=? AND snap_s IS NOT NULL "
            "ORDER BY (status='verified') DESC, confidence DESC LIMIT 1", (clip_id,),
        ).fetchone()
    frame = assign_team_probabilities(
        pd.read_parquet(projected), float(action["snap_s"]) if action else None)
    frame = frame[frame.on_field & frame.calibration_valid].copy()
    if frame.empty:
        raise ValueError("No valid on-field detections are available for tracking")
    tracker = FieldSpaceTracker()
    parts = []
    for timestamp, group in frame.groupby("video_timestamp", sort=True):
        value = group.copy()
        ids = tracker.update(value, float(timestamp))
        value = value.loc[[item >= 0 for item in ids]].copy()
        ids = [item for item in ids if item >= 0]
        if value.empty:
            continue
        value["track_id"] = [f"{clip_id}:t{item}" for item in ids]
        value["crowd_burst"] = tracker.last_crowd_burst
        active = {track.track_id: track for track in tracker.active}
        value["team"] = [active[item].team for item in ids]
        value["person_role"] = ["official" if active[item].team == "official" else
                                ("player" if active[item].team.startswith("team_") else "unknown")
                                for item in ids]
        value["vx"] = [float(active[item].velocity[0]) for item in ids]
        value["vy"] = [float(active[item].velocity[1]) for item in ids]
        value["speed"] = np.hypot(value.vx, value.vy)
        value["observed"] = True
        value["interpolated"] = False
        parts.append(value)
    result = pd.concat(parts, ignore_index=True)
    total_frames = max(1, int(result.video_timestamp.nunique()))
    observations = result.groupby("track_id").size()
    coverage = (observations / total_frames).clip(upper=1)
    result["track_observations"] = result.track_id.map(observations).astype(int)
    result["track_coverage"] = result.track_id.map(coverage).astype(float)
    # Keep short tracklets in the artifact for diagnosis, but mark the durable
    # trajectories that should drive action discovery and the tactical map.
    result["track_reliable"] = result.track_coverage >= .35
    reliable = result[result.track_reliable]
    burst_times = result.loc[result.crowd_burst, "video_timestamp"]
    roster_pool = reliable
    if action and len(reliable):
        distance = (reliable.video_timestamp - float(action["snap_s"])).abs()
        snap_rows = reliable[distance <= float(distance.min()) + .02]
        snap_ids = set(snap_rows.track_id)
        if len(snap_ids) >= 12:
            roster_pool = reliable[reliable.track_id.isin(snap_ids)]
    elif len(burst_times):
        roster_pool = reliable[reliable.video_timestamp < float(burst_times.min())]
    player_pool = roster_pool[roster_pool.person_role == "player"]
    if player_pool.track_id.nunique() >= 12:
        roster_pool = player_pool
    roster_scores = roster_pool.groupby("track_id").agg(
        observations=("track_id", "size"),
        mean_confidence=("confidence", "mean") if "confidence" in reliable else ("track_id", "size"),
        median_speed=("speed", "median"),
    )
    roster_scores["score"] = (roster_scores.observations / total_frames
                              + .15 * roster_scores.mean_confidence
                              + .05 * np.minimum(roster_scores.median_speed / 3, 1))
    roster_ids = set(roster_scores.nlargest(22, "score").index)
    result["roster_candidate"] = result.track_id.isin(roster_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    with transaction(db_path) as connection:
        game_id = connection.execute("SELECT game_id FROM clips WHERE clip_id=?", (clip_id,)).fetchone()["game_id"]
        connection.execute(
            "INSERT INTO artifacts(game_id,clip_id,kind,path,metadata_json) VALUES(?,?,?,?,?)",
            (game_id, clip_id, "clip_tracks", str(output.resolve()),
             json.dumps({"rows": len(result), "tracks": int(result.track_id.nunique()),
                         "reliable_tracks": int(result[result.track_reliable].track_id.nunique()),
                         "roster_tracks": int(result[result.roster_candidate].track_id.nunique()),
                         "team_labeled_fraction": float((result.team != 'unknown').mean())})),
        )
    return {"rows": len(result), "tracks": int(result.track_id.nunique()),
            "reliable_tracks": int(result[result.track_reliable].track_id.nunique()),
            "roster_tracks": int(result[result.roster_candidate].track_id.nunique()),
            "team_labeled_fraction": float((result.team != "unknown").mean())}
