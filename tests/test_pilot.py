from pathlib import Path

import pandas as pd

from all22.db import initialize, transaction
from all22.pilot import audit_summary, create_audit_sample, evaluate_trajectories, set_external_id


def seed_game(database: Path, play_count: int = 40):
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,season,week,home_team,away_team) VALUES('g',2022,1,'HOME','AWAY')")
        for ordinal in range(1, play_count + 1):
            quarter = min(4, (ordinal - 1) // 10 + 1)
            kind = "run" if ordinal % 2 else "pass"
            connection.execute(
                """INSERT INTO pbp_plays(game_id,ordinal,quarter,description,play_type,eligible,source_row_id)
                   VALUES('g',?,?,?,?,1,?)""",
                (ordinal, quarter, f"Play {ordinal}", kind, f"nflverse:2022_01_AWAY_HOME:{ordinal}"),
            )
            connection.execute(
                """INSERT INTO play_alignments(play_id,game_id,pbp_ordinal,score,status)
                   VALUES(?,?,?,0,'pending')""", (f"g:{ordinal:04d}", "g", ordinal),
            )


def test_balanced_audit_sample_and_summary(tmp_path: Path):
    database = tmp_path / "all22.sqlite3"
    initialize(database)
    seed_game(database)
    assert len(create_audit_sample(database, "g", 40)) == 40
    assert audit_summary(database, "g") == {"selected": 40, "reviewed": 0,
                                             "mapping_accuracy": None, "source_accuracy": None,
                                             "timing_accuracy": None}


def test_anonymous_trajectory_evaluation(tmp_path: Path):
    database = tmp_path / "all22.sqlite3"
    initialize(database)
    seed_game(database, 1)
    set_external_id(database, "g", "bdb", "100")
    with transaction(database) as connection:
        connection.executemany(
            """INSERT INTO bdb_tracking(game_id,play_id,frame_id,nfl_id,team,x,y)
               VALUES('100','1',1,?,?,?,?)""",
            [("1", "HOME", 10.0, 20.0), ("2", "AWAY", 30.0, 40.0)],
        )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pd.DataFrame([
        {"frame_id": 1, "track_id": "a", "team": "home", "x": 10.5, "y": 20.0},
        {"frame_id": 1, "track_id": "b", "team": "away", "x": 30.0, "y": 41.0},
    ]).to_parquet(predictions / "g:0001.parquet")
    result = evaluate_trajectories(database, "g", predictions)
    assert result["coverage"] == 1
    assert result["within_3_yards"] == 1
    assert result["median_error_yards"] == .75
