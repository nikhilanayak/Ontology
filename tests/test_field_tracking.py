from pathlib import Path

import pandas as pd

from all22.db import initialize, transaction
from all22.field_tracking import project_clip, save_calibration_keyframes, track_projected_clip


def prepared_db(tmp_path: Path) -> tuple[Path, str]:
    database = tmp_path / "db.sqlite3"
    initialize(database)
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,video_path) VALUES('g','film.mkv')")
        connection.execute(
            "INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence) VALUES('c','g','sideline',0,3,1)"
        )
    return database, "c"


def test_keyframe_projection_and_field_tracking(tmp_path: Path):
    database, clip_id = prepared_db(tmp_path)
    identity = {
        "image_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]],
        "field_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]],
    }
    saved = save_calibration_keyframes(database, clip_id, {
        "keyframes": [{"timestamp_s": 1.0, **identity}, {"timestamp_s": 2.0, **identity}]
    })
    assert len(saved) == 2
    detections = tmp_path / "detections.parquet"
    rows = []
    for frame_id, timestamp in enumerate((1.0, 1.5, 2.0)):
        rows.extend([
            {"clip_id": clip_id, "frame_id": frame_id, "video_timestamp": timestamp,
             "contact_x": 20 + frame_id, "contact_y": 20, "lab_a": 80, "lab_b": 80},
            {"clip_id": clip_id, "frame_id": frame_id, "video_timestamp": timestamp,
             "contact_x": 80 - frame_id, "contact_y": 30, "lab_a": 180, "lab_b": 180},
        ])
    pd.DataFrame(rows).to_parquet(detections)
    projected = tmp_path / "projected.parquet"
    metrics = project_clip(database, clip_id, detections, projected)
    assert metrics["valid_fraction"] == 1
    assert metrics["on_field_fraction"] == 1
    tracked = tmp_path / "tracked.parquet"
    result = track_projected_clip(database, clip_id, projected, tracked)
    frame = pd.read_parquet(tracked)
    assert result["tracks"] == 2
    assert frame.groupby("track_id").size().tolist() == [3, 3]
    assert set(frame.team) == {"team_0", "team_1"}


def test_projection_marks_extrapolated_frames_invalid(tmp_path: Path):
    database, clip_id = prepared_db(tmp_path)
    save_calibration_keyframes(database, clip_id, {"keyframes": [{
        "timestamp_s": 1.0,
        "image_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]],
        "field_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]],
    }]})
    path = tmp_path / "detections.parquet"
    pd.DataFrame([{"clip_id": clip_id, "frame_id": 0, "video_timestamp": .5,
                   "contact_x": 20, "contact_y": 20, "lab_a": 80, "lab_b": 80}]).to_parquet(path)
    output = tmp_path / "projected.parquet"
    project_clip(database, clip_id, path, output)
    assert not bool(pd.read_parquet(output).iloc[0].calibration_valid)
