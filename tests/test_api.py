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
            """INSERT INTO pbp_plays(game_id,ordinal,description,play_type,eligible)
               VALUES('g',2,'B.Player run middle','run',1)"""
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
    assert len(plays) == 2
    assert plays[0]["description"] == "A.Player pass short right"
    assert plays[0]["processing_status"] == "awaiting_reconstruction"
    assert plays[0]["sources"][0]["angle"] == "sideline"
    assert client.get("/api/plays/p1/sources").json()[0]["start_s"] == 10
    queue = client.get("/api/games/g/audit-queue").json()
    assert [item["play_id"] for item in queue] == ["p1"]
    assert "1 source; expected 2" in queue[0]["review_reasons"]
    response = client.put("/api/plays/p1/audit", json={
        "mapping_correct": True, "sources_correct": False,
        "timing_correct": True, "notes": "extra replay",
    })
    assert response.status_code == 200
    audit = client.get("/api/games/g/plays").json()[0]["audit"]
    assert audit == {"selected": True, "mapping_correct": 1, "sources_correct": 0,
                     "timing_correct": 1, "notes": "extra replay"}
    assert client.get("/api/games/g/audit-queue").json() == []
    response = client.put("/api/clips/c1/timing", json={"snap_s": 12.5, "play_end_s": 20.5})
    assert response.status_code == 200
    source = client.get("/api/plays/p1/sources").json()[0]
    assert source["snap_s"] == 12.5
    assert source["play_end_s"] == 20.5
