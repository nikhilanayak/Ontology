from pathlib import Path

from all22.bdb import import_tracking, snap_frame
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
