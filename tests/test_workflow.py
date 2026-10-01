from pathlib import Path

from all22.db import initialize, transaction
from all22.workflow import pilot_report


def test_pilot_report_requires_ten_verified_actions(tmp_path: Path):
    database = tmp_path / "db.sqlite3"
    initialize(database)
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id) VALUES('g')")
        connection.execute("INSERT INTO clips(clip_id,game_id,angle,start_s,end_s) VALUES('c','g','sideline',0,20)")
        for index in range(3):
            connection.execute(
                """INSERT INTO action_windows(action_id,clip_id,action_order,snap_s,dead_s,status)
                   VALUES(?,?,?,?,?,?)""", (f"a{index}", "c", index + 1, index * 2, index * 2 + 1, "verified"),
            )
    report = pilot_report(database, "g")
    assert report["milestone"]["verified_actions"] == 3
    assert report["milestone"]["remaining"] == 7
    assert not report["milestone"]["ready"]
