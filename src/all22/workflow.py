from __future__ import annotations

import json
from pathlib import Path

from . import actions, field_tracking
from .db import connect


def reconstruct_clip(db_path: Path, clip_id: str, detections: Path, output_root: Path) -> dict:
    """Run the deterministic post-detection stages for one calibrated shot."""
    projected = output_root / "projected" / f"{clip_id}.parquet"
    tracks = output_root / "clip-tracks" / f"{clip_id}.parquet"
    projection = field_tracking.project_clip(db_path, clip_id, detections, projected)
    tracking = field_tracking.track_projected_clip(db_path, clip_id, projected, tracks)
    windows = actions.discover_clip_actions(db_path, clip_id, tracks)
    return {"clip_id": clip_id, "projection": projection, "tracking": tracking,
            "actions": windows, "projected": str(projected), "tracks": str(tracks)}


def pilot_report(db_path: Path, game_id: str) -> dict:
    with connect(db_path) as connection:
        game = connection.execute("SELECT 1 FROM games WHERE game_id=?", (game_id,)).fetchone()
        if not game:
            raise ValueError(f"Game is not registered: {game_id}")
        shots = connection.execute(
            """SELECT COUNT(*) total,
                      SUM(EXISTS(SELECT 1 FROM shot_calibration_keyframes k WHERE k.clip_id=c.clip_id)) calibrated,
                      SUM(EXISTS(SELECT 1 FROM artifacts ar WHERE ar.clip_id=c.clip_id AND ar.kind='clip_tracks')) tracked
               FROM clips c WHERE c.game_id=?""", (game_id,),
        ).fetchone()
        actions_row = connection.execute(
            """SELECT COUNT(*) total,SUM(status='verified') verified,SUM(status='review') review
               FROM action_windows a JOIN clips c ON c.clip_id=a.clip_id WHERE c.game_id=?""", (game_id,),
        ).fetchone()
        pairs = connection.execute(
            """SELECT COUNT(*) total,SUM(ap.status='paired') paired,SUM(ap.status='ambiguous') ambiguous
               FROM action_pairs ap JOIN action_windows a ON a.action_id=ap.primary_action_id
               JOIN clips c ON c.clip_id=a.clip_id WHERE c.game_id=?""", (game_id,),
        ).fetchone()
        operations = connection.execute(
            """SELECT operation,COUNT(*) count FROM action_alignment_operations
               WHERE game_id=? GROUP BY operation""", (game_id,),
        ).fetchall()
    action_total = int(actions_row["total"] or 0)
    verified = int(actions_row["verified"] or 0)
    result = {
        "game_id": game_id,
        "shots": {key: int(shots[key] or 0) for key in ("total", "calibrated", "tracked")},
        "actions": {key: int(actions_row[key] or 0) for key in ("total", "verified", "review")},
        "pairs": {key: int(pairs[key] or 0) for key in ("total", "paired", "ambiguous")},
        "alignment_operations": {row["operation"]: int(row["count"]) for row in operations},
        "milestone": {
            "target_verified_actions": 10,
            "verified_actions": verified,
            "ready": verified >= 10,
            "remaining": max(0, 10 - verified),
        },
    }
    result["status"] = "ready_for_pilot_evaluation" if result["milestone"]["ready"] else "needs_manual_validation"
    return result


def write_pilot_report(db_path: Path, game_id: str, output: Path | None = None) -> dict:
    result = pilot_report(db_path, game_id)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
