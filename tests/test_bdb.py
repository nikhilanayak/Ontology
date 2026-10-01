from pathlib import Path

from all22.bdb import import_tracking, play_signatures, snap_frame
from all22.db import initialize
from all22.supervision import bdb_play_dataframe


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
