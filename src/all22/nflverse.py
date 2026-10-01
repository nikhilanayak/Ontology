from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .models import PlayByPlay


ELIGIBLE_TYPES = {"pass", "run"}


def _optional_int(value) -> Optional[int]:
    if pd.isna(value):
        return None
    return int(value)


def _optional_text(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def source_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    return f"{season}_{week:02d}_{away_team.upper()}_{home_team.upper()}"


def parse_parquet(path: Path, game_id: str, nflverse_game_id: str) -> List[PlayByPlay]:
    columns = [
        "game_id", "play_id", "qtr", "time", "posteam", "down", "ydstogo",
        "yrdln", "desc", "play_type", "qb_kneel", "qb_spike", "play_deleted",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["game_id"] == nflverse_game_id].copy()
    if frame.empty:
        raise ValueError(f"No nflverse game named {nflverse_game_id} in {path}")

    plays: List[PlayByPlay] = []
    for row in frame.itertuples(index=False):
        description = _optional_text(row.desc)
        if not description:
            continue
        play_type = (_optional_text(row.play_type) or "administrative").lower()
        deleted = bool(row.play_deleted) if not pd.isna(row.play_deleted) else False
        kneel = bool(row.qb_kneel) if not pd.isna(row.qb_kneel) else False
        spike = bool(row.qb_spike) if not pd.isna(row.qb_spike) else False
        eligible = play_type in ELIGIBLE_TYPES and not (deleted or kneel or spike)
        plays.append(
            PlayByPlay(
                game_id=game_id,
                ordinal=len(plays) + 1,
                quarter=_optional_int(row.qtr),
                clock=_optional_text(row.time),
                possession=_optional_text(row.posteam),
                down=_optional_int(row.down),
                distance=_optional_int(row.ydstogo),
                yard_line=_optional_text(row.yrdln),
                description=description,
                play_type=play_type,
                eligible=eligible,
                source_row_id=f"nflverse:{nflverse_game_id}:{_optional_int(row.play_id)}",
            )
        )
    return plays
