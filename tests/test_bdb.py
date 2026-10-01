from pathlib import Path

import numpy as np
import pandas as pd

from all22.bdb import import_tracking, play_signatures, snap_frame
from all22.db import initialize
from all22.supervision import bdb_play_dataframe, refine_with_bdb_tracks


def test_bdb_aliases_and_snap(tmp_path: Path):
    csv_path = tmp_path / "tracking.csv"
    csv_path.write_text(
        "gameId,playId,frameId,nflId,club,jerseyNumber,x,y,s,a,dir,o,event,playDirection\n"
        "1,10,5,100,PHI,1,20,10,0,0,90,90,ball_snap,left\n"
        "1,10,6,100,PHI,1,19,11,1,1,90,90,,left\n", encoding="utf-8",
    )
    database = tmp_path / "test.sqlite3"
    initialize(database)
    assert import_tracking(database, [csv_path]) == 2
    assert snap_frame(database, "1", "10") == 5
    frame = bdb_play_dataframe(database, "1", "10")
    assert list(frame.t) == [0.0, 0.1]
    assert list(frame.x_normalized) == [100.0, 101.0]


def test_bdb_play_signature_contains_answer_key_features(tmp_path: Path):
    csv_path = tmp_path / "tracking.csv"
    rows = ["gameId,playId,frameId,nflId,club,jerseyNumber,x,y,s,a,dir,o,event,playDirection"]
    for frame_id, event in ((5, "ball_snap"), (6, ""), (7, "pass_forward")):
        for player in range(1, 5):
            rows.append(f"1,10,{frame_id},{player},PHI,{player},{20 + player + frame_id / 10},"
                        f"{10 + player},{player / 2},0,90,90,{event},right")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    database = tmp_path / "test.sqlite3"
    initialize(database)
    import_tracking(database, [csv_path])
    signature = play_signatures(database, "1")["10"]
    assert signature.duration == .2
    assert signature.category == "pass"
    assert len(signature.speed_profile) == 8
    assert len(signature.formation) == 4


def test_multiframe_bdb_refinement_generalizes_to_held_out_frames():
    track_rows, truth_rows = [], []
    points = np.asarray([[20, 8], [22, 16], [24, 24], [26, 32],
                         [28, 40], [30, 12], [32, 28], [34, 44]], float)
    for frame in range(10):
        actual = points + [frame * .2, 0]
        predicted = actual * [1.08, .9] + [5, -2]
        for player, (guess, answer) in enumerate(zip(predicted, actual)):
            track_rows.append({"video_timestamp": frame / 10, "track_id": player,
                               "field_x": guess[0], "field_y": guess[1]})
            truth_rows.append({"frame_id": frame + 5, "nfl_id": player,
                               "x": answer[0], "y": answer[1]})
    result = refine_with_bdb_tracks(pd.DataFrame(track_rows), pd.DataFrame(truth_rows),
                                    0.0, 5, 0.0, False, False, 0.0)
    assert result["fit_inliers"] >= 30
    assert result["held_out_median_error_yards"] < .05
