from __future__ import annotations

import hashlib
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
        # Players legitimately stand just outside the painted field: split
        # receivers and gunners line up on or beyond the numbers, and the
        # sideline itself is a yard wide. A hard boundary discarded them -- on
        # one pilot clip 98% of dropped detections failed only on field_y, with
        # a median of 2.2 yd outside the sideline -- which removed exactly the
        # wide route runners we most want to follow. Admit a margin scaled by
        # how well this frame is calibrated, and record how far outside each
        # detection sits so later stages can judge it.
        margin = min(MAXIMUM_OUT_OF_BOUNDS_YARDS, max(OUT_OF_BOUNDS_MARGIN_YARDS, p95))
        value["out_of_bounds_yards"] = np.maximum(
            np.maximum(-value.field_x, value.field_x - FIELD_LENGTH),
            np.maximum(-value.field_y, value.field_y - FIELD_WIDTH)).clip(lower=0)
        value["on_field"] = (value.field_x.between(-margin, FIELD_LENGTH + margin)
                             & value.field_y.between(-margin, FIELD_WIDTH + margin))
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


# How far outside the painted field a detection may sit and still be tracked.
# Scaled by the frame's calibration p95 so a well-registered shot stays tight.
OUT_OF_BOUNDS_MARGIN_YARDS = 3.0
MAXIMUM_OUT_OF_BOUNDS_YARDS = 8.0

OFFICIAL_CLUSTER_FRACTION = .18
OFFICIAL_CLUSTER_FLOOR = 2
TEAM_CONFIDENCE_MINIMUM = .15
SNAP_SEPARATION_MINIMUM_PLAYERS = 12
SNAP_SEPARATION_MINIMUM_GAP_YARDS = .75


def snap_team_split(snap_rows: pd.DataFrame) -> Optional[dict]:
    """Split the snap formation across the line of scrimmage.

    At the snap the two teams occupy disjoint ranges of ``field_x``: in BDB truth
    a single x threshold separates them perfectly. That is a far stronger signal
    than jersey colour under stadium lighting, so it is used to supervise the
    colour clustering rather than the other way round.
    """
    if snap_rows.empty or "field_x" not in snap_rows:
        return None
    values = pd.to_numeric(snap_rows.field_x, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) < SNAP_SEPARATION_MINIMUM_PLAYERS:
        return None
    ordered = np.sort(values)
    gaps = np.diff(ordered)
    # Only consider splits that leave a plausible unit on each side.
    margin = max(4, int(.25 * len(ordered)))
    interior = slice(margin - 1, len(gaps) - margin + 1)
    if interior.start >= interior.stop:
        return None
    offset = int(np.argmax(gaps[interior])) + interior.start
    gap = float(gaps[offset])
    if gap < SNAP_SEPARATION_MINIMUM_GAP_YARDS:
        return None
    threshold = float((ordered[offset] + ordered[offset + 1]) / 2)
    return {"threshold": threshold, "gap_yards": gap,
            "left": int((values < threshold).sum()), "right": int((values > threshold).sum())}


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
    if clusters == 3 and int(counts.min()) > max(OFFICIAL_CLUSTER_FLOOR,
                                                 int(OFFICIAL_CLUSTER_FRACTION * len(fit_values))):
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
    confident = confidence >= TEAM_CONFIDENCE_MINIMUM
    team_names = {cluster: f"team_{order}" for order, cluster in enumerate(sorted(player_clusters))}
    names = [team_names.get(int(value), "official") for value in labels[confident]]
    result.loc[result.index[indices[confident]], "team"] = names
    result.loc[result.index[indices[confident]], "person_role"] = [
        "player" if name.startswith("team_") else "official" for name in names]
    result.loc[result.index[indices], "team_confidence"] = confidence
    return result


def _track_appearance(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One appearance vector per track, robust to a few bad crops."""
    if not columns:
        return pd.DataFrame(index=frame.track_id.unique())
    return frame.groupby("track_id")[columns].median()


def resolve_track_teams(frame: pd.DataFrame, snap_timestamp: Optional[float]) -> tuple[pd.DataFrame, dict]:
    """Assign one team per track, deciding sides by geometry and spreading them by appearance.

    Jersey colour is a poor primary signal: it depends on the two teams playing,
    the stadium lighting, shadow across the field, and white-balance, so any
    clustering tuned on one matchup transfers badly. Formation geometry does
    not: at the snap the two units occupy disjoint ranges of ``field_x``, which
    separates them perfectly in BDB truth regardless of who is playing.

    So the line of scrimmage *defines* the two sides, and appearance is learned
    per clip from those sides to label players who are absent or hidden at the
    snap. Nothing here is specific to a team, a colour, or a competition.
    """
    result = frame.copy()
    diagnostics: dict = {"method": "colour_vote"}
    if result.empty or "track_id" not in result:
        return result, diagnostics
    roles = (result.assign(_role=result.person_role)
             .groupby("track_id")._role.agg(lambda values: values.mode().iloc[0]))
    official_tracks = set(roles[roles == "official"].index)
    players = result[~result.track_id.isin(official_tracks)]
    votes = (players[players.team.str.startswith("team_")]
             .groupby(["track_id", "team"]).size().unstack(fill_value=0))
    assignment = dict(votes.idxmax(axis=1)) if len(votes) else {}
    if snap_timestamp is None or players.empty:
        return _apply_team_assignment(result, assignment, official_tracks, diagnostics)
    distance = (players.video_timestamp - float(snap_timestamp)).abs()
    snap_rows = players[distance <= float(distance.min()) + .02]
    snap_positions = snap_rows.groupby("track_id")[["field_x"]].median().reset_index()
    split = snap_team_split(snap_positions)
    if not split:
        return _apply_team_assignment(result, assignment, official_tracks, diagnostics)
    left = set(snap_positions.loc[snap_positions.field_x < split["threshold"], "track_id"])
    right = set(snap_positions.loc[snap_positions.field_x > split["threshold"], "track_id"])
    for track in left:
        assignment[track] = "team_0"
    for track in right:
        assignment[track] = "team_1"
    diagnostics = {"method": "snap_line_of_scrimmage", "threshold_x": split["threshold"],
                   "gap_yards": split["gap_yards"], "left": split["left"], "right": split["right"],
                   "snap_tracks": int(len(snap_positions))}
    # Learn this clip's two uniforms from the players the geometry just labelled,
    # then label whoever the snap could not see. The model is fitted per clip, so
    # it carries no assumption about which colours the teams wear.
    columns = [column for column in FieldSpaceTracker.appearance_columns if column in players]
    appearance = _track_appearance(players, columns)
    scale_lookup = dict(zip(FieldSpaceTracker.appearance_columns, FieldSpaceTracker.appearance_scale))
    scale = np.asarray([scale_lookup[column] for column in columns]) if columns else np.zeros(0)
    unlabelled = [track for track in appearance.index if track not in assignment]
    if len(scale) and unlabelled:
        centroids = {}
        for name, members in (("team_0", left), ("team_1", right)):
            known = appearance.loc[[track for track in members if track in appearance.index]]
            if len(known):
                centroids[name] = known.to_numpy(float).mean(axis=0)
        if len(centroids) == 2:
            propagated = 0
            for track in unlabelled:
                vector = appearance.loc[track].to_numpy(float)
                if not np.isfinite(vector).all():
                    continue
                distances = {name: float(np.linalg.norm((vector - centre) / scale))
                             for name, centre in centroids.items()}
                nearest = min(distances, key=distances.get)
                other = max(distances, key=distances.get)
                # Only accept a clear winner; an ambiguous uniform stays unknown
                # rather than inventing a side.
                if distances[other] > 0 and distances[nearest] / distances[other] <= .8:
                    assignment[track] = nearest
                    propagated += 1
            diagnostics["appearance_propagated"] = propagated
    return _apply_team_assignment(result, assignment, official_tracks, diagnostics)


def _apply_team_assignment(result: pd.DataFrame, assignment: dict, official_tracks: set,
                           diagnostics: dict) -> tuple[pd.DataFrame, dict]:
    for track in official_tracks:
        assignment[track] = "official"
    resolved = result.track_id.map(assignment).fillna("unknown")
    result = result.copy()
    result["team"] = resolved
    result["person_role"] = np.where(resolved == "official", "official",
                                     np.where(resolved.str.startswith("team_"), "player", "unknown"))
    counts = result.groupby("team").track_id.nunique().to_dict()
    diagnostics["tracks_by_team"] = {str(key): int(value) for key, value in counts.items()}
    return result, diagnostics


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

    embedding_columns = tuple(f"emb_{index:02d}" for index in range(64))
    appearance_columns = ("lab_l", "lab_a", "lab_b", "hsv_h", "hsv_s", "hsv_v") + embedding_columns
    color_scale = (50.0, 30.0, 30.0, 45.0, 60.0, 60.0)
    embedding_scale = 1.0
    appearance_scale = np.asarray(list(color_scale) + [embedding_scale] * 64, dtype=float)
    # Association cost weights and gates.  Keep these named so a config hash
    # can be recorded with every tracks artifact.
    appearance_weight = .8
    appearance_memory, appearance_update = .85, .15
    velocity_memory, velocity_update = .7, .3
    distance_slack_yards = 1.5
    missed_slack_yards = .15
    team_penalty = 3.0
    shape_gate = 1.0
    shape_weight = .6
    confidence_weight = .5
    # Lifecycle defaults live on the class so a sweep can override them in one
    # place; passing None to the constructor keeps the class value.
    maximum_missed = 15
    maximum_speed_yps = 15.0
    high_confidence = .55
    low_confidence = .15
    crowd_multiplier = 1.6
    crowd_floor = 32

    def __init__(self, maximum_missed: Optional[int] = None, maximum_speed_yps: Optional[float] = None,
                 high_confidence: Optional[float] = None, low_confidence: Optional[float] = None,
                 crowd_multiplier: Optional[float] = None, crowd_floor: Optional[int] = None,
                 use_box_shape: bool = True):
        if maximum_missed is not None:
            self.maximum_missed = maximum_missed
        if maximum_speed_yps is not None:
            self.maximum_speed_yps = maximum_speed_yps
        if high_confidence is not None:
            self.high_confidence = high_confidence
        if low_confidence is not None:
            self.low_confidence = low_confidence
        if crowd_multiplier is not None:
            self.crowd_multiplier = crowd_multiplier
        if crowd_floor is not None:
            self.crowd_floor = crowd_floor
        self.use_box_shape = use_box_shape
        self.nominal_detection_counts: list[int] = []
        self.active: list[FieldTrack] = []
        self.next_id = 1
        self.last_crowd_burst = False

    def config(self) -> dict:
        return {
            "maximum_missed": self.maximum_missed, "maximum_speed_yps": self.maximum_speed_yps,
            "high_confidence": self.high_confidence, "low_confidence": self.low_confidence,
            "crowd_multiplier": self.crowd_multiplier, "crowd_floor": self.crowd_floor,
            "use_box_shape": self.use_box_shape,
            "color_scale": list(self.color_scale), "embedding_scale": self.embedding_scale,
            "embedding_dimensions": len(self.embedding_columns),
            "appearance_weight": self.appearance_weight,
            "appearance_memory": self.appearance_memory, "appearance_update": self.appearance_update,
            "velocity_memory": self.velocity_memory, "velocity_update": self.velocity_update,
            "distance_slack_yards": self.distance_slack_yards, "missed_slack_yards": self.missed_slack_yards,
            "team_penalty": self.team_penalty, "shape_gate": self.shape_gate, "shape_weight": self.shape_weight,
            "confidence_weight": self.confidence_weight,
        }

    def _appearance(self, detections: pd.DataFrame) -> np.ndarray:
        values = np.full((len(detections), len(self.appearance_columns)), np.nan, dtype=float)
        for column_index, column in enumerate(self.appearance_columns):
            if column in detections:
                values[:, column_index] = pd.to_numeric(detections[column], errors="coerce")
        defaults = np.asarray([128, 128, 128, 90, 80, 128] + [0.0] * 64, dtype=float)
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
            maximum_distance = (self.maximum_speed_yps * elapsed + self.distance_slack_yards
                                + self.missed_slack_yards * track.missed)
            for cost_column, detection_index in enumerate(detection_indices):
                distance = float(np.linalg.norm(predicted - positions[detection_index]))
                if distance > maximum_distance:
                    continue
                candidate_team = teams[detection_index]
                team_penalty = (self.team_penalty if track.team != "unknown" and candidate_team != "unknown"
                                and track.team != candidate_team else 0.0)
                appearance = float(np.linalg.norm(
                    (track.appearance - appearances[detection_index]) / self.appearance_scale))
                shape_penalty = 0.0
                if self.use_box_shape and track.box_shape is not None and np.all(box_shapes[detection_index] > 0):
                    shape_change = np.abs(np.log(box_shapes[detection_index] / track.box_shape))
                    if float(shape_change.max()) > self.shape_gate:
                        continue
                    shape_penalty = self.shape_weight * float(np.linalg.norm(shape_change))
                confidence_penalty = self.confidence_weight * (1 - confidences[detection_index])
                costs[cost_row, cost_column] = (distance + self.appearance_weight * appearance + shape_penalty
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
            track.velocity = (self.velocity_memory * track.velocity
                              + self.velocity_update * measured_velocity)
            track.position = positions[detection_index]
            track.appearance = (self.appearance_memory * track.appearance
                                + self.appearance_update * appearances[detection_index])
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


RELINK_PARAMETERS = {
    "maximum_gap_s": .8,
    "distance_gate_yps": 12.0, "distance_slack_yards": 1.5,
    "appearance_gate": 2.25, "appearance_weight": .8,
    "shape_gate": 1.0, "shape_weight": .5,
}
RELIABLE_COVERAGE_MINIMUM = .35


def tracking_config(tracker: "FieldSpaceTracker", relink: Optional[dict] = None) -> dict:
    return {"tracker": tracker.config(), "relink": dict(relink or RELINK_PARAMETERS),
            "reliable_coverage_minimum": RELIABLE_COVERAGE_MINIMUM,
            "reliability_window": "clip",
            "global_association": dict(GLOBAL_ASSOCIATION_PARAMETERS),
            "team": {"official_cluster_fraction": OFFICIAL_CLUSTER_FRACTION,
                     "official_cluster_floor": OFFICIAL_CLUSTER_FLOOR,
                     "confidence_minimum": TEAM_CONFIDENCE_MINIMUM,
                     "snap_minimum_players": SNAP_SEPARATION_MINIMUM_PLAYERS,
                     "snap_minimum_gap_yards": SNAP_SEPARATION_MINIMUM_GAP_YARDS,
                     "resolution": "per_track_snap_los"}}


def config_hash(config: dict) -> str:
    """Short, order-independent digest shared by detections and tracks artifacts."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def relink_tracklets(frame: pd.DataFrame, maximum_gap_s: Optional[float] = None,
                     gates: Optional[dict] = None) -> tuple[pd.DataFrame, int]:
    """Join short, non-overlapping fragments only when several cues agree."""
    gates = dict(RELINK_PARAMETERS if gates is None else gates)
    if maximum_gap_s is not None:
        gates["maximum_gap_s"] = maximum_gap_s
    maximum_gap_s = gates["maximum_gap_s"]
    if frame.empty or frame.track_id.nunique() < 2:
        return frame.copy(), 0
    summaries = {}
    appearance_columns = [column for column in FieldSpaceTracker.appearance_columns if column in frame]
    for track_id, rows in frame.groupby("track_id"):
        rows = rows.sort_values("video_timestamp")
        first, last = rows.iloc[0], rows.iloc[-1]
        team_values = rows.loc[rows.team != "unknown", "team"] if "team" in rows else pd.Series(dtype=str)
        summaries[track_id] = {
            "start": float(first.video_timestamp), "end": float(last.video_timestamp),
            "start_position": first[["field_x", "field_y"]].to_numpy(float),
            "end_position": last[["field_x", "field_y"]].to_numpy(float),
            "velocity": np.asarray([float(last.get("vx", 0) or 0),
                                    float(last.get("vy", 0) or 0)]),
            "appearance": rows[appearance_columns].median().to_numpy(float),
            "team": str(team_values.mode().iloc[0]) if len(team_values) else "unknown",
            "shape": np.asarray([float(last.x2 - last.x1), float(last.y2 - last.y1)])
                     if {"x1", "y1", "x2", "y2"}.issubset(rows.columns) else None,
            "start_shape": np.asarray([float(first.x2 - first.x1), float(first.y2 - first.y1)])
                           if {"x1", "y1", "x2", "y2"}.issubset(rows.columns) else None,
        }
    scale_lookup = dict(zip(FieldSpaceTracker.appearance_columns, FieldSpaceTracker.appearance_scale))
    appearance_scale = np.asarray([scale_lookup[column] for column in appearance_columns])
    candidates = []
    for predecessor, left in summaries.items():
        for successor, right in summaries.items():
            gap = right["start"] - left["end"]
            if predecessor == successor or not 0 < gap <= maximum_gap_s:
                continue
            if (left["team"] != "unknown" and right["team"] != "unknown"
                    and left["team"] != right["team"]):
                continue
            predicted = left["end_position"] + left["velocity"] * gap
            distance = float(np.linalg.norm(predicted - right["start_position"]))
            if distance > gates["distance_gate_yps"] * gap + gates["distance_slack_yards"]:
                continue
            appearance = float(np.linalg.norm(
                (left["appearance"] - right["appearance"]) / appearance_scale)) if len(appearance_scale) else 0
            if not np.isfinite(appearance) or appearance > gates["appearance_gate"]:
                continue
            shape = 0.0
            if left["shape"] is not None and np.all(left["shape"] > 0) and np.all(right["start_shape"] > 0):
                shape = float(np.linalg.norm(np.log(right["start_shape"] / left["shape"])))
                if shape > gates["shape_gate"]:
                    continue
            candidates.append((distance + gates["appearance_weight"] * appearance
                               + gates["shape_weight"] * shape, predecessor, successor))
    used_left, used_right, links = set(), set(), {}
    for _, predecessor, successor in sorted(candidates):
        if predecessor in used_left or successor in used_right:
            continue
        links[successor] = predecessor
        used_left.add(predecessor); used_right.add(successor)
    def root(track_id):
        seen = set()
        while track_id in links and track_id not in seen:
            seen.add(track_id); track_id = links[track_id]
        return track_id
    result = frame.copy()
    result["track_id"] = result.track_id.map(root)
    return result, len(links)


def _run_tracker(frame: pd.DataFrame, clip_id: str, angle: Optional[str]) -> tuple[pd.DataFrame, "FieldSpaceTracker"]:
    tracker = FieldSpaceTracker(use_box_shape=angle != "endzone")
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
    if not parts:
        raise ValueError("Tracking produced no trajectories")
    return pd.concat(parts, ignore_index=True), tracker


def estimate_play_window(tracks: pd.DataFrame, clip_start: float, clip_end: float) -> Optional[tuple[float, float]]:
    """Derive the live-action span from motion alone, without reading action_windows.

    Tracking must not depend on `action_windows`, because those rows are written
    by action discovery *after* tracking. Reading them made the first run of a
    clip behave differently from every later run.
    """
    from .actions import discover_action_candidates
    candidates = discover_action_candidates(tracks, clip_start, clip_end)
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item.dead_s - item.snap_s) * item.confidence)
    return float(best.snap_s), float(best.dead_s)


def track_projected_clip(db_path: Path, clip_id: str, projected: Path, output: Path) -> dict:
    with connect(db_path) as connection:
        clip = connection.execute(
            "SELECT angle,start_s,end_s FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    angle = clip["angle"] if clip else None
    projection = pd.read_parquet(projected)
    valid = projection[projection.on_field & projection.calibration_valid].copy()
    if valid.empty:
        raise ValueError("No valid on-field detections are available for tracking")
    clip_start = float(clip["start_s"]) if clip else float(valid.video_timestamp.min())
    clip_end = float(clip["end_s"]) if clip else float(valid.video_timestamp.max())
    # Pass 1: unanchored tracking purely to locate the live-action span.
    scout_frame = assign_team_probabilities(valid, None)
    scout, _ = _run_tracker(scout_frame, clip_id, angle)
    scout_frames = max(1, int(scout.video_timestamp.nunique()))
    scout = scout.assign(
        track_coverage=scout.track_id.map(scout.groupby("track_id").size() / scout_frames))
    scout["track_reliable"] = scout.track_coverage >= RELIABLE_COVERAGE_MINIMUM
    window = estimate_play_window(scout, clip_start, clip_end)
    snap_s = window[0] if window else None
    # Pass 2: the real run, anchored on the motion-derived snap.
    frame = assign_team_probabilities(valid, snap_s)
    result, tracker = _run_tracker(frame, clip_id, angle)
    relink_gates = dict(RELINK_PARAMETERS)
    result, tracklet_links = relink_tracklets(result, gates=relink_gates)
    global_parameters = dict(GLOBAL_ASSOCIATION_PARAMETERS)
    global_diagnostics = {"links": 0}
    if global_parameters.get("enabled"):
        result, global_diagnostics = associate_tracklets_globally(result, global_parameters)
    result, team_diagnostics = resolve_track_teams(result, snap_s)
    # Durability is judged over the whole shot. A play-span denominator was
    # measured on the frozen protocol and made identity worse (+3.5 switches/100):
    # it admits marginal tracklets whose fragmentation costs more than the
    # recovered part-time players gain.
    span = result
    span_frames = max(1, int(span.video_timestamp.nunique()))
    observations = span.groupby("track_id").size()
    coverage = (observations / span_frames).clip(upper=1)
    result["track_observations"] = result.track_id.map(observations).fillna(0).astype(int)
    result["track_coverage"] = result.track_id.map(coverage).fillna(0.0).astype(float)
    # Keep short tracklets in the artifact for diagnosis, but mark the durable
    # trajectories that should drive action discovery and the tactical map.
    result["track_reliable"] = result.track_coverage >= RELIABLE_COVERAGE_MINIMUM
    reliable = result[result.track_reliable]
    burst_times = result.loc[result.crowd_burst, "video_timestamp"]
    roster_pool = reliable
    if snap_s is not None and len(reliable):
        distance = (reliable.video_timestamp - float(snap_s)).abs()
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
    roster_scores["score"] = (roster_scores.observations / span_frames
                              + .15 * roster_scores.mean_confidence
                              + .05 * np.minimum(roster_scores.median_speed / 3, 1))
    roster_ids = set(roster_scores.nlargest(22, "score").index)
    result["roster_candidate"] = result.track_id.isin(roster_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    config = tracking_config(tracker, relink_gates)
    digest = config_hash(config)
    summary = {"rows": len(result), "tracks": int(result.track_id.nunique()),
               "reliable_tracks": int(result[result.track_reliable].track_id.nunique()),
               "roster_tracks": int(result[result.roster_candidate].track_id.nunique()),
               "tracklet_links": tracklet_links,
               "team_labeled_fraction": float((result.team != "unknown").mean()),
               "estimated_window": list(window) if window else None,
               "global_links": global_diagnostics.get("links", 0),
               "team_assignment": team_diagnostics,
               "config_hash": digest}
    with transaction(db_path) as connection:
        game_id = connection.execute("SELECT game_id FROM clips WHERE clip_id=?", (clip_id,)).fetchone()["game_id"]
        # Carry the calibration revision forward from the projection this run consumed.
        projection = connection.execute(
            """SELECT input_revision FROM artifacts WHERE clip_id=? AND kind='clip_projection' AND path=?
               ORDER BY artifact_id DESC LIMIT 1""", (clip_id, str(Path(projected).resolve()))).fetchone()
        connection.execute(
            """INSERT INTO artifacts(game_id,clip_id,kind,path,metadata_json,config_hash,input_revision)
               VALUES(?,?,?,?,?,?,?)""",
            (game_id, clip_id, "clip_tracks", str(output.resolve()),
             json.dumps({**summary, "config": config}), digest,
             projection["input_revision"] if projection else None),
        )
    return summary


GLOBAL_ASSOCIATION_PARAMETERS = {
    "enabled": True,
    "maximum_gap_s": 2.5,
    "maximum_speed_yps": 11.0,
    "position_slack_yards": 2.0,
    "appearance_weight": .5,
    "appearance_gate": 2.0,
    "minimum_observations": 3,
}


def _tracklet_summaries(frame: pd.DataFrame, columns: list[str]) -> dict:
    summaries = {}
    for track_id, rows in frame.groupby("track_id"):
        rows = rows.sort_values("video_timestamp")
        first, last = rows.iloc[0], rows.iloc[-1]
        team_values = rows.loc[rows.team != "unknown", "team"] if "team" in rows else pd.Series(dtype=str)
        summaries[track_id] = {
            "start": float(first.video_timestamp), "end": float(last.video_timestamp),
            "start_position": first[["field_x", "field_y"]].to_numpy(float),
            "end_position": last[["field_x", "field_y"]].to_numpy(float),
            "velocity": np.asarray([float(last.get("vx", 0) or 0), float(last.get("vy", 0) or 0)]),
            "appearance": rows[columns].median().to_numpy(float) if columns else np.zeros(0),
            "team": str(team_values.mode().iloc[0]) if len(team_values) else "unknown",
            "observations": int(len(rows)),
        }
    return summaries


def associate_tracklets_globally(frame: pd.DataFrame,
                                 parameters: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """Chain tracklets into whole-play trajectories using offline constraints.

    This pipeline is batch, so the entire clip is available at once. Two facts
    make the problem far more constrained than general multi-object tracking:
    tracklets that overlap in time cannot belong to the same player, and a
    football play contains a known, small number of participants. Chains are
    built greedily under those hard constraints, cheapest compatible link first.
    """
    parameters = dict(GLOBAL_ASSOCIATION_PARAMETERS if parameters is None else parameters)
    diagnostics = {"links": 0, "tracks_before": 0, "tracks_after": 0}
    if frame.empty or "track_id" not in frame or frame.track_id.nunique() < 2:
        return frame.copy(), diagnostics
    columns = [column for column in FieldSpaceTracker.appearance_columns if column in frame]
    scale_lookup = dict(zip(FieldSpaceTracker.appearance_columns, FieldSpaceTracker.appearance_scale))
    appearance_scale = np.asarray([scale_lookup[column] for column in columns]) if columns else np.zeros(0)
    summaries = _tracklet_summaries(frame, columns)
    diagnostics["tracks_before"] = len(summaries)
    eligible = {track: value for track, value in summaries.items()
                if value["observations"] >= parameters["minimum_observations"]}
    candidates = []
    for predecessor, left in eligible.items():
        for successor, right in eligible.items():
            if predecessor == successor:
                continue
            gap = right["start"] - left["end"]
            # Hard constraint: overlapping tracklets are different players.
            if gap <= 0 or gap > parameters["maximum_gap_s"]:
                continue
            if (left["team"] != "unknown" and right["team"] != "unknown"
                    and left["team"] != right["team"]):
                continue
            predicted = left["end_position"] + left["velocity"] * gap
            reachable = parameters["maximum_speed_yps"] * gap + parameters["position_slack_yards"]
            distance = float(np.linalg.norm(predicted - right["start_position"]))
            straight = float(np.linalg.norm(right["start_position"] - left["end_position"]))
            if min(distance, straight) > reachable:
                continue
            appearance = 0.0
            if len(appearance_scale):
                appearance = float(np.linalg.norm(
                    (left["appearance"] - right["appearance"]) / appearance_scale))
                if not np.isfinite(appearance) or appearance > parameters["appearance_gate"]:
                    continue
            cost = min(distance, straight) + parameters["appearance_weight"] * appearance
            candidates.append((cost, gap, predecessor, successor))
    # Greedy cheapest-first chaining. Each tracklet keeps at most one
    # predecessor and one successor, so chains stay simple paths.
    successor_of: dict = {}
    predecessor_of: dict = {}
    def chain_end(track):
        seen = set()
        while track in successor_of and track not in seen:
            seen.add(track)
            track = successor_of[track]
        return track
    def chain_start(track):
        seen = set()
        while track in predecessor_of and track not in seen:
            seen.add(track)
            track = predecessor_of[track]
        return track
    for cost, gap, predecessor, successor in sorted(candidates):
        if predecessor in successor_of or successor in predecessor_of:
            continue
        if chain_start(predecessor) == chain_start(successor):
            continue
        # Reject if the merged chain would overlap itself in time.
        head, tail = chain_start(predecessor), chain_end(successor)
        if summaries[tail]["end"] < summaries[head]["start"]:
            continue
        successor_of[predecessor] = successor
        predecessor_of[successor] = predecessor
    identity = {}
    for track in summaries:
        identity[track] = chain_start(track)
    result = frame.copy()
    result["global_track_id"] = result.track_id.map(identity)
    result["track_id"] = result.global_track_id
    diagnostics["links"] = len(successor_of)
    diagnostics["tracks_after"] = int(result.track_id.nunique())
    return result.drop(columns=["global_track_id"]), diagnostics
