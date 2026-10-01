from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import bdb
from .action_alignment import ActionUnit, match_cost
from .actions import signature_for_action
from .db import connect
from .geometry import normalize_direction


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
                       flip_y: bool) -> tuple[list[float], int, int]:
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
        distances = np.linalg.norm(actual[:, None, :] - predicted[None, :, :], axis=2)
        answer_index, prediction_index = linear_sum_assignment(distances)
        errors.extend(distances[answer_index, prediction_index].tolist())
        matched += len(answer_index)
        possible += len(actual)
    return errors, matched, possible


def evaluate_audited_sources(db_path: Path, game_id: str, tracks_dir: Path) -> dict:
    """Compare reconstructed, human-verified sources with matching BDB truth."""
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
    expected = bdb.play_signatures(db_path, external["external_id"])
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["clip_id"], []).append(row)
    results = []
    with connect(db_path) as connection:
        for clip_id, candidates in grouped.items():
            match = re.search(r":(\d+)$", candidates[0]["source_row_id"] or "")
            if not match or match.group(1) not in expected:
                continue
            play_id = match.group(1)
            answer = expected[play_id]
            path = tracks_dir / f"{clip_id}.parquet"
            if not path.exists():
                continue
            tracks = pd.read_parquet(path)
            if "track_reliable" in tracks and bool(tracks.track_reliable.any()):
                tracks = tracks[tracks.track_reliable].copy()
            ranked = []
            for row in candidates:
                signature = signature_for_action(dict(row), tracks, row["angle"])
                unit = ActionUnit("", row["action_id"], signature.duration, 1.0,
                                  speed_profile=signature.speed_profile, formation=signature.formation)
                ranked.append((match_cost(
                    # Only BDB-derived fields participate in this local choice.
                    type("Play", (), {"description": "", "play_type": answer.category})(),
                    unit, answer), row))
            selection_cost, selected = min(ranked, key=lambda value: value[0])
            window = tracks[(tracks.video_timestamp >= float(selected["snap_s"])) &
                            (tracks.video_timestamp <= float(selected["dead_s"]))].copy()
            truth = pd.read_sql_query(
                """SELECT frame_id,nfl_id,x,y,event FROM bdb_tracking
                   WHERE game_id=? AND play_id=? AND nfl_id IS NOT NULL ORDER BY frame_id""",
                connection, params=(external["external_id"], play_id),
            )
            snap_rows = truth[truth.event.fillna("").str.lower().isin(("ball_snap", "autoevent_ballsnap"))]
            bdb_snap = int(snap_rows.frame_id.min())
            searches = []
            for offset in np.arange(-1.5, 1.501, .2):
                for flip_x in (False, True):
                    for flip_y in (False, True):
                        errors, matched, possible = _trajectory_errors(
                            window, truth, float(selected["snap_s"]), bdb_snap,
                            float(offset), flip_x, flip_y)
                        if errors:
                            score = float(np.median(errors)) + 2 * (1 - matched / max(possible, 1))
                            searches.append((score, float(offset), flip_x, flip_y))
            if not searches:
                continue
            _, coarse_offset, flip_x, flip_y = min(searches)
            refined = []
            for offset in np.arange(coarse_offset - .2, coarse_offset + .201, .05):
                errors, matched, possible = _trajectory_errors(
                    window, truth, float(selected["snap_s"]), bdb_snap,
                    float(offset), flip_x, flip_y)
                if errors:
                    refined.append((float(np.median(errors)), float(offset), errors, matched, possible))
            median, offset, errors, matched, possible = min(refined)
            results.append({
                "clip_id": clip_id, "bdb_play_id": play_id, "action_id": selected["action_id"],
                "candidate_actions": len(candidates), "selection_cost": float(selection_cost),
                "snap_adjustment_s": offset, "flip_x": flip_x, "flip_y": flip_y,
                "matched_player_frames": matched, "possible_player_frames": possible,
                "player_coverage": matched / max(possible, 1),
                "median_error_yards": median, "p90_error_yards": float(np.percentile(errors, 90)),
            })
    if not results:
        raise ValueError("No audited reconstructed sources overlap BDB plays")
    return {
        "game_id": game_id, "sources": len(results),
        "median_source_error_yards": float(np.median([row["median_error_yards"] for row in results])),
        "median_source_coverage": float(np.median([row["player_coverage"] for row in results])),
        "results": results,
    }
