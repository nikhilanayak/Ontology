from pathlib import Path

import pandas as pd

from all22.nflverse import parse_parquet, source_game_id


def test_source_game_id():
    assert source_game_id(2025, 1, "dal", "phi") == "2025_01_DAL_PHI"


def test_parse_nflverse_filters_non_scrimmage_and_kneels(tmp_path: Path):
    path = tmp_path / "pbp.parquet"
    pd.DataFrame([
        {"game_id": "2025_01_DAL_PHI", "play_id": 1, "qtr": 1, "time": "15:00",
         "posteam": None, "down": None, "ydstogo": 0, "yrdln": None, "desc": "GAME",
         "play_type": None, "qb_kneel": 0, "qb_spike": 0, "play_deleted": 0},
        {"game_id": "2025_01_DAL_PHI", "play_id": 71, "qtr": 1, "time": "14:54",
         "posteam": "DAL", "down": 1, "ydstogo": 10, "yrdln": "DAL 47",
         "desc": "Runner left tackle for 7 yards", "play_type": "run", "qb_kneel": 0,
         "qb_spike": 0, "play_deleted": 0},
        {"game_id": "2025_01_DAL_PHI", "play_id": 99, "qtr": 4, "time": "00:30",
         "posteam": "PHI", "down": 1, "ydstogo": 10, "yrdln": "PHI 40",
         "desc": "Quarterback kneels", "play_type": "run", "qb_kneel": 1,
         "qb_spike": 0, "play_deleted": 0},
    ]).to_parquet(path)

    plays = parse_parquet(path, "local-game", "2025_01_DAL_PHI")
    assert len(plays) == 3
    assert [play.eligible for play in plays] == [False, True, False]
    assert plays[1].source_row_id == "nflverse:2025_01_DAL_PHI:71"
