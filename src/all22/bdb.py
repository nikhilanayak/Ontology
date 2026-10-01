from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from .db import transaction


ALIASES = {
    "game_id": ("game_id", "gameId"),
    "play_id": ("play_id", "playId"),
    "frame_id": ("frame_id", "frameId"),
    "nfl_id": ("nfl_id", "nflId"),
    "team": ("team", "club", "player_side"),
    "jersey_number": ("jersey_number", "jerseyNumber"),
    "x": ("x",),
    "y": ("y",),
    "speed": ("s", "speed"),
    "acceleration": ("a", "acceleration"),
    "direction": ("dir", "direction"),
    "orientation": ("o", "orientation"),
    "event": ("event",),
    "play_direction": ("play_direction", "playDirection"),
}


@dataclass(frozen=True)
class BDBPlaySignature:
    play_id: str
    duration: float
    category: str
    speed_profile: tuple[float, ...]
    formation: tuple[float, ...]
    events: tuple[str, ...]


def _value(row: Dict[str, str], field: str) -> Optional[str]:
    for candidate in ALIASES[field]:
        value = row.get(candidate)
        if value not in (None, "", "NA", "NaN", "nan"):
            return value
    return None


def _number(value: Optional[str], cast=float):
    if value is None:
        return None
    return cast(float(value)) if cast is int else cast(value)


def iter_tracking(path: Path) -> Iterator[tuple]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("game_id", "play_id", "frame_id", "x", "y")
        missing = [field for field in required if not any(alias in (reader.fieldnames or []) for alias in ALIASES[field])]
        if missing:
            raise ValueError(f"{path} is missing BDB fields: {', '.join(missing)}")
        for row in reader:
            yield (
                _value(row, "game_id"), _value(row, "play_id"), _number(_value(row, "frame_id"), int),
                _value(row, "nfl_id"), _value(row, "team"), _number(_value(row, "jersey_number"), int),
                _number(_value(row, "x")), _number(_value(row, "y")), _number(_value(row, "speed")),
                _number(_value(row, "acceleration")), _number(_value(row, "direction")),
                _number(_value(row, "orientation")), _value(row, "event"), _value(row, "play_direction"),
            )


def import_tracking(db_path: Path, files: Iterable[Path], batch_size: int = 10_000) -> int:
    sql = """INSERT OR REPLACE INTO bdb_tracking
      (game_id,play_id,frame_id,nfl_id,team,jersey_number,x,y,speed,acceleration,direction,orientation,event,play_direction)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    imported = 0
    with transaction(db_path) as connection:
        for path in files:
            batch = []
            for record in iter_tracking(path):
                batch.append(record)
                if len(batch) >= batch_size:
                    connection.executemany(sql, batch)
                    imported += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(sql, batch)
                imported += len(batch)
    return imported


def snap_frame(db_path: Path, game_id: str, play_id: str) -> Optional[int]:
    from .db import connect
    with connect(db_path) as connection:
        row = connection.execute(
            """SELECT MIN(frame_id) AS frame_id FROM bdb_tracking
               WHERE game_id=? AND play_id=? AND lower(event)='ball_snap'""",
            (game_id, play_id),
        ).fetchone()
        return row["frame_id"] if row and row["frame_id"] is not None else None


def play_signatures(db_path: Path, game_id: str) -> dict[str, BDBPlaySignature]:
    """Build compact, angle-independent answer-key features for one game."""
    from .db import connect
    with connect(db_path) as connection:
        frame = pd.read_sql_query(
            """SELECT play_id,frame_id,nfl_id,team,x,y,speed,event FROM bdb_tracking
               WHERE game_id=? ORDER BY CAST(play_id AS INTEGER),frame_id""",
            connection, params=(game_id,),
        )
    output: dict[str, BDBPlaySignature] = {}
    for play_id, play in frame.groupby("play_id", sort=False):
        event_values = play.event.fillna("").astype(str).str.lower()
        snap_rows = play[event_values.isin(("ball_snap", "autoevent_ballsnap"))]
        if snap_rows.empty:
            continue
        snap = int(snap_rows.frame_id.min())
        post = play[play.frame_id >= snap].copy()
        duration = max(.1, (int(post.frame_id.max()) - snap) / 10.0)
        players = post[post.nfl_id.notna()]
        at_snap = players[players.frame_id == snap]
        points = at_snap[["x", "y"]].to_numpy(float)
        if len(points):
            centered = points - np.median(points, axis=0)
            formation = tuple(np.round(np.sort(np.hypot(centered[:, 0], centered[:, 1])), 3))
        else:
            formation = ()
        edges = np.linspace(snap, int(post.frame_id.max()) + 1e-6, 9)
        post["bin"] = np.clip(np.digitize(post.frame_id, edges) - 1, 0, 7)
        speeds = (post[post.nfl_id.notna()].groupby("bin").speed.median().reindex(range(8))
                  .interpolate(limit_direction="both").fillna(0.0))
        events = tuple(sorted(set(value for value in event_values if value)))
        if any(value in events for value in ("pass_forward", "autoevent_passforward")):
            category = "pass"
        elif any(value in events for value in ("handoff", "run")):
            category = "run"
        else:
            category = "unknown"
        output[str(play_id)] = BDBPlaySignature(
            str(play_id), duration, category, tuple(speeds.astype(float)), formation, events)
    return output
