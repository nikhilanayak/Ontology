from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

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
