from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import json

from all22.db import initialize, transaction
from all22.tracking import BoxTracker, _on_field_detections, detect_clip, fuse_sources, project_source


class FakeDetector:
    model_version = "fake:person:v1"

    def predict(self, frame):
        return np.asarray([[10, 5, 30, 45]], dtype=float), np.asarray([.9], dtype=float)


def test_clip_detection_is_independent_of_play_alignment(tmp_path: Path):
    import cv2

    database = tmp_path / "db.sqlite3"
    initialize(database)
    video = tmp_path / "film.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(20):
        writer.write(np.full((48, 64, 3), (30, 120, 40), dtype=np.uint8))
    writer.release()
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,video_path) VALUES('g',?)", (str(video),))
        connection.execute(
            "INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence) VALUES('c','g','unknown',0,1,0)"
        )
    output = tmp_path / "detections.parquet"
    rows = detect_clip(database, "c", output, sample_hz=2, detector=FakeDetector())
    result = pd.read_parquet(output)
    assert rows == len(result) >= 2
    assert set(result.clip_id) == {"c"}
    assert {"contact_x", "contact_y", "lab_l", "hsv_h"}.issubset(result.columns)
    with transaction(database) as connection:
        artifact = connection.execute(
            "SELECT kind,clip_id,model_version FROM artifacts WHERE clip_id='c'"
        ).fetchone()
    assert tuple(artifact) == ("clip_detections", "c", "fake:person:v1")


def test_box_tracker_preserves_nearby_tracks():
    tracker = BoxTracker()
    first = tracker.update(np.array([[10, 10, 20, 30], [100, 10, 110, 30]]), (100, 200))
    second = tracker.update(np.array([[12, 10, 22, 30], [98, 10, 108, 30]]), (100, 200))
    assert first == second


def test_people_outside_field_region_are_filtered():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, 50:150] = (30, 120, 40)
    boxes = np.asarray([[80, 20, 100, 80], [0, 20, 20, 80]], float)
    kept, scores = _on_field_detections(frame, boxes, np.asarray([.9, .8]), margin_pixels=2)
    assert kept.shape == (1, 4)
    assert scores.tolist() == [.9]


def test_people_beyond_painted_sidelines_are_filtered():
    frame = np.full((120, 240, 3), (30, 120, 40), np.uint8)
    cv2.line(frame, (0, 25), (239, 25), (245, 245, 245), 4)
    cv2.line(frame, (0, 95), (239, 95), (245, 245, 245), 4)
    boxes = np.asarray([[100, 35, 120, 70], [30, 0, 50, 15], [180, 97, 200, 115]], float)
    kept, scores = _on_field_detections(frame, boxes, np.asarray([.9, .8, .7]), margin_pixels=2)
    assert kept.tolist() == [[100, 35, 120, 70]]
    assert scores.tolist() == [.9]


def test_fuse_sources_averages_nearby_field_positions(tmp_path: Path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    output = tmp_path / "fused.parquet"
    pd.DataFrame([{"relative_frame": 0, "track_id": "s:1", "x": 10.0, "y": 20.0}]).to_parquet(left)
    pd.DataFrame([{"relative_frame": 0, "track_id": "e:8", "x": 12.0, "y": 20.0}]).to_parquet(right)
    assert fuse_sources([left, right], output) == 1
    result = pd.read_parquet(output)
    assert result.iloc[0].x == 11.0
    assert result.iloc[0].track_id == "s:1"


def test_projection_supports_explicit_pass_release_anchor(tmp_path: Path):
    database = tmp_path / "db.sqlite3"
    initialize(database)
    video = tmp_path / "film.mkv"
    video.touch()
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,video_path) VALUES('g',?)", (str(video),))
        connection.execute("INSERT INTO pbp_plays(game_id,ordinal,description,play_type,eligible) VALUES('g',1,'p','pass',1)")
        connection.execute("INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence) VALUES('c','g','sideline',0,20,1)")
        connection.execute("INSERT INTO play_alignments(play_id,game_id,pbp_ordinal,score,status) VALUES('p','g',1,0,'pending')")
        connection.execute("INSERT INTO play_sources(play_id,clip_id,source_order,angle) VALUES('p','c',0,'sideline')")
    detections = tmp_path / "detections.parquet"
    pd.DataFrame([{"image_x": 0., "image_y": 0., "video_timestamp": 9.9, "track_id": "1"}]).to_parquet(detections)
    landmarks = tmp_path / "landmarks.json"
    landmarks.write_text(json.dumps({"image_points": [[0, 0], [10, 0], [0, 10], [10, 10]],
                                     "field_points": [[20, 20], [30, 20], [20, 30], [30, 30]]}))
    output = tmp_path / "projected.parquet"
    project_source(database, "p", 0, detections, landmarks, output, 10.0, 28)
    row = pd.read_parquet(output).iloc[0]
    assert row.frame_id == 27
