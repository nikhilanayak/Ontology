from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

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
