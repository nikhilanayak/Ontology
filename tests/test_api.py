from pathlib import Path

import pytest
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
            """INSERT INTO action_windows(action_id,clip_id,action_order,formation_start_s,snap_s,
               dead_s,playback_end_s,confidence,status) VALUES('a1','c1',1,11,12,19,20,.8,'candidate')"""
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
    shots = client.get("/api/games/g/shots").json()
    assert shots[0]["action_count"] == 1
    assert client.get("/api/clips/c1/actions").json()[0]["action_id"] == "a1"
    response = client.put("/api/actions/a1/timing", json={
        "formation_start_s": 11.5, "snap_s": 12.5, "dead_s": 18.5, "playback_end_s": 20,
    })
    assert response.status_code == 200
    assert client.get("/api/clips/c1/actions").json()[0]["status"] == "verified"
    response = client.put("/api/clips/c1/calibration", json={"keyframes": [{
        "timestamp_s": 12,
        "image_points": [[0, 0], [100, 0], [0, 100], [100, 100]],
        "field_points": [[10, 0], [20, 0], [10, 20], [20, 20]],
    }]})
    assert response.status_code == 200
    calibration = client.get("/api/clips/c1/calibration").json()
    assert calibration["keyframes"][0]["field_points"][1] == [20, 0]


def test_field_detection_debug_endpoint_reports_lines_numbers_and_ambiguity(tmp_path: Path):
    """The debug view must expose the evidence behind absolute field position."""
    import cv2
    import numpy as np

    database = tmp_path / "debug.sqlite3"
    initialize(database)
    # A synthetic field: green turf with white five-yard lines.
    height, width = 360, 640
    frame = np.full((height, width, 3), (60, 140, 70), dtype=np.uint8)
    for x in range(80, width - 40, 90):
        cv2.line(frame, (x, 30), (x, height - 30), (245, 245, 245), 3)
    video = tmp_path / "film.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    for _ in range(20):
        writer.write(frame)
    writer.release()
    if not video.exists():  # codec unavailable in this environment
        return
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,video_path) VALUES('g',?)", (str(video),))
        connection.execute(
            """INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence)
               VALUES('c1','g','sideline',0,1.5,.9)""")
    client = TestClient(create_app(database, tmp_path / "trajectories", None))
    response = client.get("/api/clips/c1/field-detections", params={"timestamp_s": .5})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["frame_width"] == width and payload["frame_height"] == height
    assert payload["lines"], "the detector should find the painted yard lines"
    assert payload["line_evidence"], "the debug view must include fitted field-line candidates"
    assert payload["raw_line_fragment_count"] >= len(payload["line_evidence"])
    assert {"image_points", "length_pixels", "angle_degrees", "kind", "supporting_fragments"} <= set(payload["line_evidence"][0])
    assert {item["kind"] for item in payload["line_evidence"]} <= {
        "cross_field", "downfield_boundary", "other", "unclassified"}
    for line in payload["lines"]:
        assert len(line["image_points"]) == 2
    # No painted numbers exist here, so absolute position must stay unresolved
    # rather than being asserted from the repeating lines alone.
    assert payload["numbers"] == []
    assert payload["numbers_credible"] is False
    assert payload["absolute_x"] is False
    # A frame image is served for the overlay.
    image = client.get("/api/clips/c1/frame.jpg", params={"timestamp_s": .5})
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    # Timestamps outside the shot are refused rather than silently clamped.
    assert client.get("/api/clips/c1/field-detections", params={"timestamp_s": 99}).status_code == 422
    assert client.get("/api/clips/missing/field-detections").status_code == 404


def test_calibration_exposes_matrix_and_detections_endpoint(tmp_path: Path):
    """The viewer needs the homography to reproject the field onto the film."""
    import pandas as pd

    database = tmp_path / "overlay.sqlite3"
    initialize(database)
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,video_path) VALUES('g',NULL)")
        connection.execute(
            """INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence)
               VALUES('c1','g','sideline',0,20,.9)""")
    detections = tmp_path / "detections.parquet"
    pd.DataFrame([
        {"clip_id": "c1", "video_timestamp": 1.0, "x1": 10.0, "y1": 20.0, "x2": 30.0,
         "y2": 70.0, "confidence": .91, "detection_id": 0},
        {"clip_id": "c1", "video_timestamp": 1.1, "x1": 12.0, "y1": 21.0, "x2": 32.0,
         "y2": 71.0, "confidence": .44, "detection_id": 0},
    ]).to_parquet(detections, index=False)
    with transaction(database) as connection:
        connection.execute(
            """INSERT INTO artifacts(game_id,clip_id,kind,path) VALUES('g','c1','clip_detections',?)""",
            (str(detections),))
    client = TestClient(create_app(database, tmp_path / "trajectories", None))
    client.put("/api/clips/c1/calibration", json={"keyframes": [{
        "timestamp_s": 5,
        "image_points": [[0, 0], [100, 0], [0, 100], [100, 100]],
        "field_points": [[10, 0], [20, 0], [10, 20], [20, 20]],
    }]})
    keyframe = client.get("/api/clips/c1/calibration").json()["keyframes"][0]
    matrix = keyframe["matrix"]
    assert len(matrix) == 3 and len(matrix[0]) == 3
    assert matrix[2][2] == pytest.approx(1.0)
    rows = client.get("/api/clips/c1/detections").json()
    assert len(rows) == 2
    assert rows[0]["confidence"] == pytest.approx(.91)
    assert {"x1", "y1", "x2", "y2", "video_timestamp"} <= set(rows[0])
    assert client.get("/api/clips/c1/detections", params={"stride": 2}).json() == [rows[0]]
    assert client.get("/api/clips/missing/detections").status_code == 404
