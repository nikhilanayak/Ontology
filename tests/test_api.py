from pathlib import Path

from fastapi.testclient import TestClient

from all22.api import create_app
from all22.db import initialize, transaction


def test_game_and_static_endpoints(tmp_path: Path):
    database = tmp_path / "test.sqlite3"
    initialize(database)
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,season,video_path) VALUES('g',2025,NULL)")
        connection.execute(
            """INSERT INTO pbp_plays(game_id,ordinal,description,play_type,eligible)
               VALUES('g',1,'A.Player pass short right','pass',1)"""
        )
        connection.execute(
            """INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence)
               VALUES('c1','g','sideline',10,22,.9)"""
        )
        connection.execute(
            """INSERT INTO play_alignments(play_id,game_id,pbp_ordinal,score,status)
               VALUES('p1','g',1,.1,'pending')"""
        )
        connection.execute(
            """INSERT INTO play_sources(play_id,clip_id,source_order,angle)
               VALUES('p1','c1',0,'sideline')"""
        )
    static = Path(__file__).parents[1] / "web"
    client = TestClient(create_app(database, tmp_path / "trajectories", static))
    assert client.get("/").status_code == 200
    games = client.get("/api/games").json()
    assert games[0]["game_id"] == "g"
    plays = client.get("/api/games/g/plays").json()
    assert plays[0]["description"] == "A.Player pass short right"
    assert plays[0]["processing_status"] == "awaiting_reconstruction"
    assert plays[0]["sources"][0]["angle"] == "sideline"
    assert client.get("/api/plays/p1/sources").json()[0]["start_s"] == 10
