from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json
import re
import subprocess

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import bdb
from .action_alignment import ActionUnit, match_cost
from .actions import signature_for_action
from .db import connect
from .geometry import normalize_direction


# Bump this whenever the BDB comparison logic (alignment search, refinement,
# identity matching, or aggregation) changes.  Results produced by different
# evaluator versions are not comparable and must not be reported side by side.
EVALUATOR_VERSION = "bdb-audited-v2"
PROTOCOL_VERSION = 1

EVALUATOR_PARAMETERS: dict = {
    "spatial_confidence_minimum": .55,
    "coarse_offset_range_s": [-1.5, 1.5],
    "coarse_offset_step_s": .2,
    "coverage_penalty_weight": 2.0,
    "fine_offset_half_range_s": .2,
    "fine_offset_step_s": .05,
    "yard_line_translation_step_yards": 5.0,
    "yard_line_translation_limit_yards": 50.0,
    "refinement_gates_yards": [15.0, 10.0, 7.0],
    "refinement_ransac_threshold_yards": 2.0,
    "refinement_minimum_pairs": 30,
    "identity_maximum_error_yards": 8.0,
}


def _git(*arguments: str) -> Optional[str]:
    try:
        output = subprocess.run(
            ["git", *arguments], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return output.stdout


def _git_state() -> dict:
    """Commit plus a dirty flag so a result can never be attributed to a clean commit it did not run from."""
    commit = (_git("rev-parse", "HEAD") or "").strip() or None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status.strip())}


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SupervisionPair:
    game_id: str
    play_id: str
    video_path: str
    sideline_snap_s: float
    endzone_snap_s: float
    snap_frame_id: int
    samples: int


def bdb_play_dataframe(db_path: Path, game_id: str, play_id: str) -> pd.DataFrame:
    with connect(db_path) as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM bdb_tracking WHERE game_id=? AND play_id=? ORDER BY frame_id,nfl_id",
            connection, params=(game_id, play_id),
        )
    if frame.empty:
        return frame
    normalized = frame.apply(
        lambda row: normalize_direction(float(row.x), float(row.y), row.play_direction), axis=1,
        result_type="expand",
    )
    frame["x_normalized"] = normalized[0]
    frame["y_normalized"] = normalized[1]
    snap_rows = frame[frame.event.fillna("").str.lower() == "ball_snap"]
    if not snap_rows.empty:
        snap = int(snap_rows.frame_id.min())
        frame["t"] = (frame.frame_id.astype(int) - snap) / 10.0
    return frame


def export_labels(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["game_id", "play_id", "frame_id", "t", "nfl_id", "team", "jersey_number",
               "x_normalized", "y_normalized", "speed", "acceleration", "direction", "orientation", "event"]
    frame[[column for column in columns if column in frame.columns]].to_parquet(output, index=False)


def create_pair(db_path: Path, manifest: Path, output_dir: Path, game_id: str, play_id: str,
                angle: str, video_path: Path, video_snap_s: float) -> Dict:
    if angle not in {"sideline", "endzone"}:
        raise ValueError("angle must be sideline or endzone")
    if not video_path.exists():
        raise ValueError(f"Video does not exist: {video_path}")
    frame = bdb_play_dataframe(db_path, game_id, play_id)
    if frame.empty or "t" not in frame:
        raise ValueError("BDB play has no ball_snap event; pairing is not allowed")
    labels_path = output_dir / game_id / f"{play_id}.parquet"
    export_labels(frame, labels_path)
    record = {
        "game_id": str(game_id), "play_id": str(play_id), "angle": angle,
        "video_path": str(video_path.resolve()), "snap_s": float(video_snap_s),
        "labels_path": str(labels_path.resolve()), "samples": int(len(frame)),
    }
    existing = []
    if manifest.exists():
        existing = pd.read_json(manifest, lines=True).to_dict(orient="records")
    existing = [row for row in existing if not (
        str(row["game_id"]) == str(game_id) and str(row["play_id"]) == str(play_id) and row["angle"] == angle
    )]
    existing.append(record)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(existing).to_json(manifest, orient="records", lines=True)
    return record


def _trajectory_errors(tracks: pd.DataFrame, truth: pd.DataFrame, video_snap_s: float,
                       bdb_snap_frame: int, offset_s: float, flip_x: bool,
                       flip_y: bool, shift_x: float = 0.0) -> tuple[list[float], int, int]:
    answers = {int(frame_id): group[["x", "y"]].to_numpy(float)
               for frame_id, group in truth.groupby("frame_id")}
    errors: list[float] = []
    matched = possible = 0
    for timestamp, group in tracks.groupby("video_timestamp"):
        frame_id = bdb_snap_frame + round((float(timestamp) - video_snap_s + offset_s) * 10)
        actual = answers.get(frame_id)
        if actual is None:
            continue
        predicted = group[["field_x", "field_y"]].to_numpy(float).copy()
        if flip_x:
            predicted[:, 0] = 120.0 - predicted[:, 0]
        if flip_y:
            predicted[:, 1] = 160.0 / 3.0 - predicted[:, 1]
        predicted[:, 0] += shift_x
        distances = np.linalg.norm(actual[:, None, :] - predicted[None, :, :], axis=2)
        answer_index, prediction_index = linear_sum_assignment(distances)
        errors.extend(distances[answer_index, prediction_index].tolist())
        matched += len(answer_index)
        possible += len(actual)
    return errors, matched, possible


def _five_yard_x_correction(tracks: pd.DataFrame, truth: pd.DataFrame, video_snap_s: float,
                            bdb_snap_frame: int, offset_s: float, flip_x: bool,
                            step_yards: float = 5.0, limit_yards: float = 50.0) -> float:
    timestamps = tracks.video_timestamp.drop_duplicates().to_numpy(float)
    if not len(timestamps):
        return 0.0
    # offset maps video time to BDB time, so this is the video frame aligned to
    # the tagged BDB snap.  Restrict the correction to repeated five-yard-line
    # ambiguity; do not fit a free transform to the answers.
    target = video_snap_s - offset_s
    timestamp = float(timestamps[np.argmin(abs(timestamps - target))])
    predicted = tracks[tracks.video_timestamp == timestamp].field_x.to_numpy(float)
    actual = truth[truth.frame_id == bdb_snap_frame].x.to_numpy(float)
    if not len(predicted) or not len(actual):
        return 0.0
    if flip_x:
        predicted = 120.0 - predicted
    delta = float(np.median(actual) - np.median(predicted))
    return float(np.clip(round(delta / step_yards) * step_yards, -limit_yards, limit_yards))


def refine_with_bdb_tracks(tracks: pd.DataFrame, truth: pd.DataFrame, video_snap_s: float,
                           bdb_snap_frame: int, offset_s: float, flip_x: bool,
                           flip_y: bool, shift_x: float,
                           gates_yards: tuple = (15.0, 10.0, 7.0), ransac_threshold: float = 2.0,
                           minimum_pairs: int = 30) -> dict:
    """Fit on alternating frames and score the correction on held-out frames.

    This is an answer-key supervision stage, not an inference-time shortcut.
    Anonymous players are assigned one-to-one at each frame, while one shared
    projective correction must explain the entire action.
    """
    value = tracks.copy()
    base = value[["field_x", "field_y"]].to_numpy(float)
    if flip_x:
        base[:, 0] = 120.0 - base[:, 0]
    if flip_y:
        base[:, 1] = 160.0 / 3.0 - base[:, 1]
    base[:, 0] += shift_x
    value[["base_x", "base_y"]] = base
    answers = {int(frame_id): group[["x", "y"]].to_numpy(float)
               for frame_id, group in truth.groupby("frame_id")}
    groups = []
    for index, (timestamp, group) in enumerate(value.groupby("video_timestamp")):
        frame_id = bdb_snap_frame + round((float(timestamp) - video_snap_s + offset_s) * 10)
        if frame_id in answers:
            groups.append((index, group[["base_x", "base_y"]].to_numpy(float), answers[frame_id]))
    correction = np.eye(3)
    fit_pairs = fit_inliers = 0
    for maximum_distance in gates_yards:
        source, target = [], []
        for index, predicted, actual in groups:
            if index % 2:
                continue
            corrected = cv2.perspectiveTransform(
                predicted.astype(np.float32).reshape(-1, 1, 2), correction).reshape(-1, 2)
            distances = np.linalg.norm(actual[:, None, :] - corrected[None, :, :], axis=2)
            actual_index, predicted_index = linear_sum_assignment(distances)
            keep = distances[actual_index, predicted_index] < maximum_distance
            source.extend(predicted[predicted_index[keep]])
            target.extend(actual[actual_index[keep]])
        fit_pairs = len(source)
        if fit_pairs < minimum_pairs:
            break
        fitted, mask = cv2.findHomography(
            np.asarray(source, np.float32), np.asarray(target, np.float32), cv2.RANSAC, ransac_threshold)
        if fitted is None or mask is None:
            break
        correction = fitted / fitted[2, 2]
        fit_inliers = int(mask.sum())
    errors = []
    matched = possible = 0
    for index, predicted, actual in groups:
        if index % 2 == 0:
            continue
        corrected = cv2.perspectiveTransform(
            predicted.astype(np.float32).reshape(-1, 1, 2), correction).reshape(-1, 2)
        distances = np.linalg.norm(actual[:, None, :] - corrected[None, :, :], axis=2)
        actual_index, predicted_index = linear_sum_assignment(distances)
        errors.extend(distances[actual_index, predicted_index].tolist())
        matched += len(actual_index)
        possible += len(actual)
    if not errors:
        raise ValueError("BDB refinement has no held-out frames")
    return {"correction_matrix": correction.tolist(), "fit_pairs": fit_pairs,
            "fit_inliers": fit_inliers, "held_out_player_frames": matched,
            "held_out_coverage": matched / max(possible, 1),
            "held_out_median_error_yards": float(np.median(errors)),
            "held_out_p90_error_yards": float(np.percentile(errors, 90))}


def identity_metrics(tracks: pd.DataFrame, truth: pd.DataFrame, video_snap_s: float,
                     bdb_snap_frame: int, offset_s: float, flip_x: bool, flip_y: bool,
                     shift_x: float, correction_matrix: list, maximum_error: float = 8.0) -> dict:
    """Measure anonymous track consistency after spatial/time alignment to BDB."""
    if "track_id" not in tracks or "nfl_id" not in truth:
        return {"matched_assignments": 0, "track_purity": 0.0, "id_switches": 0,
                "switches_per_100_assignments": 0.0, "mean_fragments_per_player": 0.0}
    value = tracks.copy()
    predicted = value[["field_x", "field_y"]].to_numpy(float)
    if flip_x:
        predicted[:, 0] = 120.0 - predicted[:, 0]
    if flip_y:
        predicted[:, 1] = 160.0 / 3.0 - predicted[:, 1]
    predicted[:, 0] += shift_x
    correction = np.asarray(correction_matrix, dtype=float)
    predicted = cv2.perspectiveTransform(
        predicted.astype(np.float32).reshape(-1, 1, 2), correction).reshape(-1, 2)
    value[["aligned_x", "aligned_y"]] = predicted
    answers = {int(frame_id): group for frame_id, group in truth.groupby("frame_id")}
    assignments = []
    for timestamp, group in value.groupby("video_timestamp", sort=True):
        frame_id = bdb_snap_frame + round((float(timestamp) - video_snap_s + offset_s) * 10)
        actual = answers.get(frame_id)
        if actual is None or actual.empty:
            continue
        distances = np.linalg.norm(
            actual[["x", "y"]].to_numpy(float)[:, None, :]
            - group[["aligned_x", "aligned_y"]].to_numpy(float)[None, :, :], axis=2)
        actual_index, predicted_index = linear_sum_assignment(distances)
        for left, right in zip(actual_index, predicted_index):
            error = float(distances[left, right])
            if error <= maximum_error:
                assignments.append((float(timestamp), str(group.iloc[right].track_id),
                                    str(actual.iloc[left].nfl_id), error))
    if not assignments:
        return {"matched_assignments": 0, "track_purity": 0.0, "id_switches": 0,
                "switches_per_100_assignments": 0.0, "mean_fragments_per_player": 0.0}
    matched = pd.DataFrame(assignments, columns=["timestamp", "track_id", "nfl_id", "error"])
    majority = matched.groupby("track_id").nfl_id.value_counts().groupby(level=0).max()
    purity = float(majority.sum() / len(matched))
    switches = 0
    for _, rows in matched.sort_values("timestamp").groupby("nfl_id"):
        identities = rows.track_id.to_numpy()
        switches += int(np.sum(identities[1:] != identities[:-1]))
    fragments = matched.groupby("nfl_id").track_id.nunique()
    return {"matched_assignments": len(matched), "track_purity": purity,
            "id_switches": switches, "switches_per_100_assignments": 100 * switches / len(matched),
            "mean_fragments_per_player": float(fragments.mean()),
            "median_assignment_error_yards": float(matched.error.median())}


def _audited_source_rows(db_path: Path, game_id: str) -> tuple[str, dict[str, list]]:
    """Return the BDB external ID and audited (clip -> action window rows) mapping."""
    with connect(db_path) as connection:
        external = connection.execute(
            "SELECT external_id FROM game_external_ids WHERE game_id=? AND provider='bdb'", (game_id,),
        ).fetchone()
        rows = connection.execute(
            """SELECT DISTINCT ps.clip_id,c.angle,p.source_row_id,aw.action_id,aw.action_order,
                              aw.snap_s,aw.dead_s
               FROM play_sources ps JOIN play_alignments pa ON pa.play_id=ps.play_id
               JOIN pbp_plays p ON p.game_id=pa.game_id AND p.ordinal=pa.pbp_ordinal
               JOIN alignment_audits aa ON aa.play_id=pa.play_id
               JOIN clips c ON c.clip_id=ps.clip_id
               JOIN action_windows aw ON aw.clip_id=ps.clip_id
               WHERE pa.game_id=? AND aa.mapping_correct=1 AND aa.sources_correct=1
               ORDER BY ps.clip_id,aw.action_order""", (game_id,),
        ).fetchall()
    if not external:
        raise ValueError("Set the game's BDB external ID before evaluation")
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["clip_id"], []).append(dict(row))
    return str(external["external_id"]), grouped


def _bdb_play_id(source_row_id: Optional[str]) -> Optional[str]:
    match = re.search(r":(\d+)$", source_row_id or "")
    return match.group(1) if match else None


def _clip_play_id(clip_id: str, candidates: list[dict], expected: dict,
                  skipped: list[dict]) -> Optional[str]:
    """Resolve the single BDB play a clip is audited against, or record why it cannot be."""
    play_ids = {_bdb_play_id(row["source_row_id"]) for row in candidates}
    if len(play_ids) != 1:
        skipped.append({"clip_id": clip_id, "reason": "clip is a source for multiple plays",
                        "bdb_play_ids": sorted(str(item) for item in play_ids)})
        return None
    play_id = play_ids.pop()
    if not play_id or play_id not in expected:
        skipped.append({"clip_id": clip_id, "reason": "no BDB play signature", "bdb_play_id": play_id})
        return None
    return play_id


def _latest_artifact(connection, clip_id: str, kind: str):
    return connection.execute(
        """SELECT path,model_version,config_hash,input_revision,created_at FROM artifacts
           WHERE clip_id=? AND kind=? ORDER BY artifact_id DESC LIMIT 1""", (clip_id, kind),
    ).fetchone()


def _calibration_revision(connection, clip_id: str) -> Optional[int]:
    row = connection.execute(
        "SELECT MAX(revision) AS revision FROM shot_calibration_keyframes WHERE clip_id=?", (clip_id,),
    ).fetchone()
    return int(row["revision"]) if row and row["revision"] is not None else None


def _source_inputs(connection, clip_id: str) -> dict:
    """Describe the frozen inputs (detections, calibration) that feed a clip's tracks."""
    detections = _latest_artifact(connection, clip_id, "clip_detections")
    detections_path = Path(detections["path"]) if detections else None
    return {
        "detections": {
            "path": str(detections_path) if detections_path else None,
            "sha256": _file_sha256(detections_path) if detections_path else None,
            "model_version": detections["model_version"] if detections else None,
            "config_hash": detections["config_hash"] if detections else None,
        },
        "calibration_revision": _calibration_revision(connection, clip_id),
    }


def _select_action(candidates: list[dict], tracks: pd.DataFrame, answer) -> tuple[float, dict]:
    ranked = []
    for row in candidates:
        signature = signature_for_action(row, tracks, row["angle"])
        unit = ActionUnit("", row["action_id"], signature.duration, 1.0,
                          speed_profile=signature.speed_profile, formation=signature.formation)
        ranked.append((match_cost(
            # Only BDB-derived fields participate in this local choice.
            type("Play", (), {"description": "", "play_type": answer.category})(),
            unit, answer), row))
    return min(ranked, key=lambda value: value[0])


def _load_truth(connection, external_id: str, play_id: str) -> tuple[pd.DataFrame, Optional[int]]:
    truth = pd.read_sql_query(
        """SELECT frame_id,nfl_id,x,y,event FROM bdb_tracking
           WHERE game_id=? AND play_id=? AND nfl_id IS NOT NULL ORDER BY frame_id""",
        connection, params=(external_id, play_id),
    )
    snap_rows = truth[truth.event.fillna("").str.lower().isin(("ball_snap", "autoevent_ballsnap"))]
    if truth.empty or snap_rows.empty:
        return truth, None
    return truth, int(snap_rows.frame_id.min())


def _evaluate_source(all_tracks: pd.DataFrame, truth: pd.DataFrame, bdb_snap: int,
                     snap_s: float, dead_s: float,
                     parameters: Optional[dict] = None) -> tuple[Optional[dict], Optional[str]]:
    """Align one clip's tracks to BDB over a fixed video window and score them.

    Returns ``(result, None)`` or ``(None, reason)`` when the source cannot be scored.
    """
    parameters = EVALUATOR_PARAMETERS if parameters is None else parameters
    identity_tracks = all_tracks
    if "track_reliable" in identity_tracks and bool(identity_tracks.track_reliable.any()):
        identity_tracks = identity_tracks[identity_tracks.track_reliable].copy()
    spatial_tracks = all_tracks
    if "confidence" in spatial_tracks:
        spatial_tracks = spatial_tracks[spatial_tracks.confidence >= parameters["spatial_confidence_minimum"]]
    window = spatial_tracks[(spatial_tracks.video_timestamp >= snap_s) &
                            (spatial_tracks.video_timestamp <= dead_s)].copy()
    identity_window = identity_tracks[(identity_tracks.video_timestamp >= snap_s) &
                                      (identity_tracks.video_timestamp <= dead_s)].copy()
    yard_step = parameters["yard_line_translation_step_yards"]
    yard_limit = parameters["yard_line_translation_limit_yards"]
    low, high = parameters["coarse_offset_range_s"]
    searches = []
    for offset in np.arange(low, high + 1e-3, parameters["coarse_offset_step_s"]):
        for flip_x in (False, True):
            for flip_y in (False, True):
                shift_x = _five_yard_x_correction(window, truth, snap_s, bdb_snap, float(offset), flip_x,
                                                  yard_step, yard_limit)
                errors, matched, possible = _trajectory_errors(
                    window, truth, snap_s, bdb_snap, float(offset), flip_x, flip_y, shift_x)
                if errors:
                    score = (float(np.median(errors))
                             + parameters["coverage_penalty_weight"] * (1 - matched / max(possible, 1)))
                    searches.append((score, float(offset), flip_x, flip_y))
    if not searches:
        return None, "no overlapping BDB frames in window"
    _, coarse_offset, flip_x, flip_y = min(searches)
    refined = []
    half = parameters["fine_offset_half_range_s"]
    for offset in np.arange(coarse_offset - half, coarse_offset + half + 1e-3, parameters["fine_offset_step_s"]):
        candidate_shift = _five_yard_x_correction(window, truth, snap_s, bdb_snap, float(offset), flip_x,
                                                  yard_step, yard_limit)
        errors, matched, possible = _trajectory_errors(
            window, truth, snap_s, bdb_snap, float(offset), flip_x, flip_y, candidate_shift)
        if errors:
            refined.append((float(np.median(errors)), float(offset), candidate_shift, errors, matched, possible))
    median, offset, shift_x, errors, matched, possible = min(refined)
    try:
        bdb_refinement = refine_with_bdb_tracks(
            window, truth, snap_s, bdb_snap, offset, flip_x, flip_y, shift_x,
            tuple(parameters["refinement_gates_yards"]), parameters["refinement_ransac_threshold_yards"],
            parameters["refinement_minimum_pairs"])
    except ValueError as error:
        return None, str(error)
    identities = identity_metrics(
        identity_window, truth, snap_s, bdb_snap, offset, flip_x, flip_y, shift_x,
        bdb_refinement["correction_matrix"], maximum_error=parameters["identity_maximum_error_yards"])
    return {
        "window": {"snap_s": snap_s, "dead_s": dead_s},
        "spatial_track_rows": int(len(window)), "identity_track_rows": int(len(identity_window)),
        "snap_adjustment_s": offset, "flip_x": flip_x, "flip_y": flip_y,
        "yard_line_translation_x": shift_x,
        "matched_player_frames": matched, "possible_player_frames": possible,
        "player_coverage": matched / max(possible, 1),
        "median_error_yards": median, "p90_error_yards": float(np.percentile(errors, 90)),
        "bdb_refinement": bdb_refinement,
        "identity_metrics": identities,
    }, None


def _tracks_provenance(connection, clip_id: str, path: Path) -> dict:
    """Describe the exact tracks file scored, attributing a config hash only when the artifact matches it."""
    artifact = connection.execute(
        """SELECT config_hash,input_revision FROM artifacts WHERE clip_id=? AND kind='clip_tracks' AND path=?
           ORDER BY artifact_id DESC LIMIT 1""", (clip_id, str(path.resolve())),
    ).fetchone()
    return {"path": str(path), "sha256": _file_sha256(path),
            "config_hash": artifact["config_hash"] if artifact else None,
            "calibration_revision": artifact["input_revision"] if artifact else None,
            "artifact_matched": artifact is not None}


def _summarize(game_id: str, results: list[dict], skipped: list[dict], provenance: dict) -> dict:
    if not results:
        raise ValueError("No audited reconstructed sources overlap BDB plays: "
                         + json.dumps(skipped))
    return {
        "game_id": game_id, "sources": len(results),
        "median_source_error_yards": float(np.median([row["median_error_yards"] for row in results])),
        "median_source_coverage": float(np.median([row["player_coverage"] for row in results])),
        "median_refined_held_out_error_yards": float(np.median([
            row["bdb_refinement"]["held_out_median_error_yards"] for row in results])),
        "median_track_purity": float(np.median([
            row["identity_metrics"]["track_purity"] for row in results])),
        "median_switches_per_100_assignments": float(np.median([
            row["identity_metrics"]["switches_per_100_assignments"] for row in results])),
        "skipped": skipped,
        "provenance": provenance,
        "results": results,
    }


def _base_provenance(tracks_dir: Path) -> dict:
    return {"evaluator_version": EVALUATOR_VERSION, "evaluator_parameters": dict(EVALUATOR_PARAMETERS),
            **_git_state(), "evaluated_at": _utc_now(), "tracks_dir": str(tracks_dir)}


def _reliable_tracks(all_tracks: pd.DataFrame) -> pd.DataFrame:
    if "track_reliable" in all_tracks and bool(all_tracks.track_reliable.any()):
        return all_tracks[all_tracks.track_reliable].copy()
    return all_tracks


def freeze_evaluation_protocol(db_path: Path, game_id: str, tracks_dir: Path, output: Path,
                               clip_ids: Optional[list[str]] = None) -> dict:
    """Freeze the audited source set, windows, and inputs for repeatable comparison.

    The action window is selected once here, using the tracks present at freeze
    time, so later tracker changes cannot move the evaluated time span.  The
    protocol embeds BDB-derived identifiers and must stay under ignored ``data/``.
    """
    external_id, grouped = _audited_source_rows(db_path, game_id)
    expected = bdb.play_signatures(db_path, external_id)
    requested = set(clip_ids or [])
    unknown = sorted(requested - set(grouped))
    if unknown:
        raise ValueError(f"Requested clips are not audited sources of {game_id}: {unknown}")
    sources, skipped = [], []
    with connect(db_path) as connection:
        for clip_id, candidates in grouped.items():
            if requested and clip_id not in requested:
                continue
            play_id = _clip_play_id(clip_id, candidates, expected, skipped)
            if play_id is None:
                continue
            path = tracks_dir / f"{clip_id}.parquet"
            if not path.exists():
                skipped.append({"clip_id": clip_id, "reason": "tracks parquet missing", "path": str(path)})
                continue
            all_tracks = pd.read_parquet(path)
            selection_cost, selected = _select_action(candidates, _reliable_tracks(all_tracks), expected[play_id])
            truth, bdb_snap = _load_truth(connection, external_id, play_id)
            if bdb_snap is None:
                skipped.append({"clip_id": clip_id, "reason": "BDB play has no snap event", "bdb_play_id": play_id})
                continue
            scored, reason = _evaluate_source(all_tracks, truth, bdb_snap,
                                              float(selected["snap_s"]), float(selected["dead_s"]))
            if scored is None:
                # Freeze only sources that are scorable at freeze time so every later
                # run evaluates the same complete set.
                skipped.append({"clip_id": clip_id, "reason": reason, "action_id": selected["action_id"]})
                continue
            sources.append({
                "clip_id": clip_id, "bdb_play_id": play_id, "angle": selected["angle"],
                "action_id": selected["action_id"], "candidate_actions": len(candidates),
                "selection_cost": float(selection_cost),
                "snap_s": float(selected["snap_s"]), "dead_s": float(selected["dead_s"]),
                "bdb_snap_frame": bdb_snap, "bdb_rows": int(len(truth)),
                "tracks_at_freeze": _tracks_provenance(connection, clip_id, path),
                **_source_inputs(connection, clip_id),
            })
    if not sources:
        raise ValueError("No audited reconstructed sources can be frozen: " + json.dumps(skipped))
    protocol = {
        "protocol_version": PROTOCOL_VERSION, "evaluator_version": EVALUATOR_VERSION,
        "evaluator_parameters": dict(EVALUATOR_PARAMETERS),
        "game_id": game_id, "bdb_game_id": external_id,
        "frozen_at": _utc_now(), **_git_state(),
        "tracks_dir_at_freeze": str(tracks_dir),
        "sources": sources, "skipped": skipped,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def _protocol_drift(frozen: dict, current: dict) -> list[str]:
    reasons = []
    for key in ("sha256", "model_version", "config_hash"):
        before = frozen["detections"].get(key)
        after = current["detections"].get(key)
        if before != after:
            reasons.append(f"detections.{key}: {before} -> {after}")
    if frozen.get("calibration_revision") != current.get("calibration_revision"):
        reasons.append(f"calibration_revision: {frozen.get('calibration_revision')} -> "
                       f"{current.get('calibration_revision')}")
    return reasons


def evaluate_with_protocol(db_path: Path, protocol_path: Path, tracks_dir: Path,
                           allow_drift: bool = False, drift_note: Optional[str] = None) -> dict:
    """Evaluate the frozen source set with frozen windows; fail closed on input drift.

    ``allow_drift`` only covers detections/calibration inputs.  A changed BDB
    answer key or a missing/unscorable frozen source always fails.
    """
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {protocol.get('protocol_version')}")
    if protocol.get("evaluator_version") != EVALUATOR_VERSION:
        raise ValueError(f"Protocol was frozen for evaluator {protocol.get('evaluator_version')}; "
                         f"this code is {EVALUATOR_VERSION}. Re-freeze or run the matching revision.")
    if protocol.get("evaluator_parameters") != EVALUATOR_PARAMETERS:
        raise ValueError("Protocol evaluator parameters differ from this code; re-freeze the protocol")
    game_id, external_id = protocol["game_id"], protocol["bdb_game_id"]
    results, failures, drift = [], [], []
    with connect(db_path) as connection:
        for source in protocol["sources"]:
            clip_id = source["clip_id"]
            path = tracks_dir / f"{clip_id}.parquet"
            if not path.exists():
                failures.append({"clip_id": clip_id, "reason": "tracks parquet missing", "path": str(path)})
                continue
            truth, bdb_snap = _load_truth(connection, external_id, source["bdb_play_id"])
            if bdb_snap is None or bdb_snap != source["bdb_snap_frame"] or len(truth) != source["bdb_rows"]:
                failures.append({"clip_id": clip_id, "reason": "BDB answer key changed since freeze",
                                 "frozen": {"snap": source["bdb_snap_frame"], "rows": source["bdb_rows"]},
                                 "current": {"snap": bdb_snap, "rows": int(len(truth))}})
                continue
            current_inputs = _source_inputs(connection, clip_id)
            reasons = _protocol_drift(source, current_inputs)
            if reasons:
                drift.append({"clip_id": clip_id, "reasons": reasons})
            scored, reason = _evaluate_source(pd.read_parquet(path), truth, bdb_snap,
                                              float(source["snap_s"]), float(source["dead_s"]))
            if scored is None:
                failures.append({"clip_id": clip_id, "reason": reason})
                continue
            results.append({
                "clip_id": clip_id, "bdb_play_id": source["bdb_play_id"], "action_id": source["action_id"],
                "candidate_actions": source["candidate_actions"], "selection_cost": source["selection_cost"],
                "tracks": _tracks_provenance(connection, clip_id, path),
                "inputs": current_inputs, "input_drift": reasons,
                **scored,
            })
    if failures:
        # The frozen set must be evaluated in full; a partial set is not comparable.
        raise ValueError("Frozen protocol sources are missing or unscorable: " + json.dumps(failures))
    if drift and not allow_drift:
        raise ValueError("Protocol inputs drifted since freeze (pass allow_drift to override): "
                         + json.dumps(drift))
    provenance = {**_base_provenance(tracks_dir),
                  "protocol": {"path": str(protocol_path), "sha256": _file_sha256(protocol_path),
                               "frozen_at": protocol.get("frozen_at"),
                               "git_commit_at_freeze": protocol.get("git_commit"),
                               "git_dirty_at_freeze": protocol.get("git_dirty")},
                  "input_drift": drift, "drift_allowed": bool(allow_drift), "drift_note": drift_note}
    return _summarize(game_id, results, [], provenance)


def evaluate_audited_sources(db_path: Path, game_id: str, tracks_dir: Path) -> dict:
    """Compare reconstructed, human-verified sources with matching BDB truth.

    This exploratory mode re-selects action windows from the tracks under test
    and uses whatever sources are present.  Use ``freeze_evaluation_protocol``
    and ``evaluate_with_protocol`` for comparisons across tracker changes.
    """
    external_id, grouped = _audited_source_rows(db_path, game_id)
    expected = bdb.play_signatures(db_path, external_id)
    results, skipped = [], []
    with connect(db_path) as connection:
        for clip_id, candidates in grouped.items():
            play_id = _clip_play_id(clip_id, candidates, expected, skipped)
            if play_id is None:
                continue
            path = tracks_dir / f"{clip_id}.parquet"
            if not path.exists():
                skipped.append({"clip_id": clip_id, "reason": "tracks parquet missing", "path": str(path)})
                continue
            all_tracks = pd.read_parquet(path)
            selection_cost, selected = _select_action(candidates, _reliable_tracks(all_tracks), expected[play_id])
            truth, bdb_snap = _load_truth(connection, external_id, play_id)
            if bdb_snap is None:
                skipped.append({"clip_id": clip_id, "reason": "BDB play has no snap event", "bdb_play_id": play_id})
                continue
            scored, reason = _evaluate_source(all_tracks, truth, bdb_snap,
                                              float(selected["snap_s"]), float(selected["dead_s"]))
            if scored is None:
                skipped.append({"clip_id": clip_id, "reason": reason, "action_id": selected["action_id"]})
                continue
            results.append({
                "clip_id": clip_id, "bdb_play_id": play_id, "action_id": selected["action_id"],
                "candidate_actions": len(candidates), "selection_cost": float(selection_cost),
                "tracks": _tracks_provenance(connection, clip_id, path),
                "inputs": _source_inputs(connection, clip_id),
                **scored,
            })
    provenance = {**_base_provenance(tracks_dir), "protocol": None}
    return _summarize(game_id, results, skipped, provenance)
