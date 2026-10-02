import json
from pathlib import Path

import pandas as pd
import pytest

from all22.db import connect, initialize, transaction
from all22.field_tracking import (
    FieldSpaceTracker,
    associate_tracklets_globally,
    assign_team_probabilities,
    config_hash,
    project_clip,
    resolve_track_teams,
    snap_team_split,
    save_calibration_keyframes,
    track_projected_clip,
    tracking_config,
    relink_tracklets,
)


def test_default_tracking_config_hash_is_pinned():
    """Changing any association weight or gate must be a deliberate, recorded decision."""
    assert config_hash(tracking_config(FieldSpaceTracker())) == "66e9682a0209126f"
    assert config_hash(tracking_config(FieldSpaceTracker(use_box_shape=False))) == "291dec2f8e9e5526"


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
    # The tracks artifact must carry the association config and calibration revision
    # so evaluations can be attributed to exact tracker settings.
    with connect(database) as connection:
        artifact = connection.execute(
            "SELECT config_hash,input_revision,metadata_json FROM artifacts WHERE kind='clip_tracks'").fetchone()
    assert artifact["config_hash"] == result["config_hash"]
    assert len(artifact["config_hash"]) == 16
    assert artifact["input_revision"] == "1"
    assert json.loads(artifact["metadata_json"])["config"]["tracker"]["appearance_weight"] == .8


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


def test_tracker_gives_confident_detection_first_claim():
    tracker = FieldSpaceTracker()
    columns = {"field_y": [20.0], "lab_l": [100], "lab_a": [100], "lab_b": [100],
               "hsv_h": [40], "hsv_s": [80], "hsv_v": [100], "team": ["team_0"]}
    first = pd.DataFrame({"field_x": [10.0], "confidence": [.9], **columns})
    assert tracker.update(first, 0.0) == [1]
    # A weak duplicate is geometrically closer, but it must not steal the track.
    second = pd.DataFrame({
        "field_x": [10.4, 10.1], "field_y": [20.0, 20.0], "confidence": [.9, .3],
        "lab_l": [100, 100], "lab_a": [100, 100], "lab_b": [100, 100],
        "hsv_h": [40, 40], "hsv_s": [80, 80], "hsv_v": [100, 100],
        "team": ["team_0", "team_0"],
    })
    assert tracker.update(second, .1) == [1, -1]


def test_tracker_uses_velocity_across_brief_miss():
    tracker = FieldSpaceTracker(maximum_missed=3)
    def detection(x):
        return pd.DataFrame({"field_x": [x], "field_y": [20.0], "confidence": [.9],
                             "lab_a": [100], "lab_b": [100], "team": ["team_0"]})
    assert tracker.update(detection(10), 0.0) == [1]
    assert tracker.update(detection(11), .1) == [1]
    assert tracker.update(detection(13), .3) == [1]


def test_tracker_does_not_spawn_identities_during_a_sideline_burst():
    tracker = FieldSpaceTracker(crowd_multiplier=1.5, crowd_floor=4)
    base = pd.DataFrame({
        "field_x": range(3), "field_y": [20.0] * 3, "confidence": [.99, .98, .97],
        "lab_a": [100] * 3, "lab_b": [100] * 3, "team": ["team_0"] * 3,
    })
    assert tracker.update(base, 0.0) == [1, 2, 3]
    burst = pd.DataFrame({
        "field_x": range(8), "field_y": [20.0] * 8, "confidence": [.99] * 8,
        "lab_a": [100] * 8, "lab_b": [100] * 8, "team": ["team_0"] * 8,
    })
    ids = tracker.update(burst, .1)
    assert sum(value >= 0 for value in ids) == 3
    assert tracker.next_id == 4


def test_snap_color_clusters_separate_small_official_group():
    rows = []
    for lab, count in [((80, 165, 90, 180), 5), ((210, 125, 135, 25), 5), ((135, 128, 128, 5), 2)]:
        for _ in range(count):
            rows.append({"video_timestamp": 1.0, "lab_l": lab[0], "lab_a": lab[1],
                         "lab_b": lab[2], "hsv_s": lab[3]})
    classified = assign_team_probabilities(pd.DataFrame(rows), anchor_timestamp=1.0)
    assert classified.person_role.value_counts().to_dict() == {"player": 10, "official": 2}
    assert set(classified.team) == {"team_0", "team_1", "official"}


def test_relink_tracklets_joins_compatible_non_overlapping_fragments():
    rows = []
    for track_id, times, xs, color in [
        ("a", [0.0, .1], [10.0, 10.5], 90),
        ("b", [.3, .4], [11.5, 12.0], 91),
        ("other", [.3, .4], [30.0, 30.5], 180),
    ]:
        for timestamp, x in zip(times, xs):
            rows.append({"track_id": track_id, "video_timestamp": timestamp,
                         "field_x": x, "field_y": 20.0, "vx": 5.0, "vy": 0.0,
                         "team": "team_0", "lab_a": color, "lab_b": color})
    linked, count = relink_tracklets(pd.DataFrame(rows))
    assert count == 1
    assert linked[linked.video_timestamp < .2].track_id.iloc[0] == linked[linked.field_x == 12].track_id.iloc[0]
    assert linked.track_id.nunique() == 2


def test_snap_team_split_separates_formation_across_line_of_scrimmage():
    rows = [{"field_x": 48.0 + .1 * index} for index in range(11)]
    rows += [{"field_x": 52.0 + .1 * index} for index in range(11)]
    split = snap_team_split(pd.DataFrame(rows))
    assert split is not None
    assert 49.0 < split["threshold"] < 52.0
    assert (split["left"], split["right"]) == (11, 11)
    # A single clustered blob has no defensible line of scrimmage.
    blob = pd.DataFrame([{"field_x": 50 + .05 * index} for index in range(22)])
    assert snap_team_split(blob) is None
    # Too few players at the snap is not enough evidence.
    assert snap_team_split(pd.DataFrame([{"field_x": float(i)} for i in range(6)])) is None


def test_resolve_track_teams_assigns_one_team_per_track_from_snap_geometry():
    rows = []
    for index in range(22):
        side = 48.0 if index < 11 else 52.0
        # Colour vote is deliberately wrong for one detection of each track.
        for step, team in enumerate(("team_0", "team_1" if index % 7 == 0 else "team_0")):
            rows.append({"track_id": f"t{index}", "video_timestamp": 10.0 + .1 * step,
                         "field_x": side + .1 * index, "field_y": 20.0,
                         "team": team, "person_role": "player"})
    rows.append({"track_id": "ref", "video_timestamp": 10.0, "field_x": 30.0, "field_y": 5.0,
                 "team": "official", "person_role": "official"})
    resolved, diagnostics = resolve_track_teams(pd.DataFrame(rows), 10.0)
    assert diagnostics["method"] == "snap_line_of_scrimmage"
    per_track = resolved.groupby("track_id").team.nunique()
    assert set(per_track) == {1}, "each track must carry exactly one team"
    players = resolved[resolved.person_role == "player"]
    assert players.groupby("team").track_id.nunique().to_dict() == {"team_0": 11, "team_1": 11}
    assert resolved[resolved.track_id == "ref"].team.iloc[0] == "official"


def test_tracking_does_not_read_action_windows(tmp_path: Path):
    """Tracking must be deterministic: action windows are written after it runs."""
    database, clip_id = prepared_db(tmp_path)
    identity = {"image_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]],
                "field_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]]}
    save_calibration_keyframes(database, clip_id, {
        "keyframes": [{"timestamp_s": 0.0, **identity}, {"timestamp_s": 3.0, **identity}]})
    rows = []
    for frame_id, timestamp in enumerate([round(.1 * i, 1) for i in range(31)]):
        for player in range(4):
            rows.append({"clip_id": clip_id, "frame_id": frame_id, "video_timestamp": timestamp,
                         "contact_x": 20 + 4 * player + 2.0 * timestamp, "contact_y": 20 + player,
                         "lab_a": 80 if player < 2 else 180, "lab_b": 80 if player < 2 else 180,
                         "confidence": .9})
    detections = tmp_path / "detections.parquet"
    pd.DataFrame(rows).to_parquet(detections)
    projected = tmp_path / "projected.parquet"
    project_clip(database, clip_id, detections, projected)
    first = track_projected_clip(database, clip_id, projected, tmp_path / "a.parquet")
    with transaction(database) as connection:
        connection.execute(
            """INSERT INTO action_windows(action_id,clip_id,action_order,snap_s,dead_s,confidence)
               VALUES('c:a01','c',1,1.0,2.0,.9)""")
    second = track_projected_clip(database, clip_id, projected, tmp_path / "b.parquet")
    assert first["config_hash"] == second["config_hash"]
    assert first["tracks"] == second["tracks"]
    assert first["estimated_window"] == second["estimated_window"]
    assert pd.read_parquet(tmp_path / "a.parquet").equals(pd.read_parquet(tmp_path / "b.parquet"))


def test_global_association_chains_tracklets_but_never_merges_overlapping_ones():
    """Two tracklets overlapping in time cannot be the same player."""
    rows = []
    # a -> b: a clean temporal gap along a consistent heading.
    for step in range(6):
        rows.append({"track_id": "a", "video_timestamp": 10.0 + .1 * step,
                     "field_x": 20.0 + step, "field_y": 25.0, "vx": 10.0, "vy": 0.0,
                     "team": "team_0", "lab_l": 50, "lab_a": 10, "lab_b": 10,
                     "hsv_h": 90, "hsv_s": 80, "hsv_v": 128})
    for step in range(6):
        rows.append({"track_id": "b", "video_timestamp": 11.0 + .1 * step,
                     "field_x": 30.0 + step, "field_y": 25.0, "vx": 10.0, "vy": 0.0,
                     "team": "team_0", "lab_l": 50, "lab_a": 10, "lab_b": 10,
                     "hsv_h": 90, "hsv_s": 80, "hsv_v": 128})
    # c overlaps a exactly and must stay separate however similar it looks.
    for step in range(6):
        rows.append({"track_id": "c", "video_timestamp": 10.0 + .1 * step,
                     "field_x": 20.3 + step, "field_y": 25.4, "vx": 10.0, "vy": 0.0,
                     "team": "team_0", "lab_l": 50, "lab_a": 10, "lab_b": 10,
                     "hsv_h": 90, "hsv_s": 80, "hsv_v": 128})
    frame = pd.DataFrame(rows)
    result, diagnostics = associate_tracklets_globally(frame)
    assert diagnostics["links"] == 1
    assert result[result.index.isin(frame.index[frame.track_id == "a"])].track_id.nunique() == 1
    merged = dict(zip(frame.track_id, result.track_id))
    assert merged["a"] == merged["b"], "the gapped pair should chain"
    assert merged["c"] != merged["a"], "time-overlapping tracklets must not merge"
    assert result.track_id.nunique() == 2


def test_global_association_respects_reachable_distance_and_team():
    rows = []
    for track, x, team in (("a", 20.0, "team_0"), ("b", 95.0, "team_0"), ("d", 24.0, "team_1")):
        start = 10.0 if track == "a" else 11.0
        for step in range(5):
            rows.append({"track_id": track, "video_timestamp": start + .1 * step,
                         "field_x": x + .1 * step, "field_y": 25.0, "vx": 0.0, "vy": 0.0,
                         "team": team, "lab_l": 50, "lab_a": 10, "lab_b": 10,
                         "hsv_h": 90, "hsv_s": 80, "hsv_v": 128})
    result, diagnostics = associate_tracklets_globally(pd.DataFrame(rows))
    # b is 75 yd away (unreachable in 0.6 s) and d is the other team.
    assert diagnostics["links"] == 0
    assert result.track_id.nunique() == 3


def test_sideline_players_survive_projection(tmp_path: Path):
    """Split receivers and gunners stand on or just past the sideline."""
    database, clip_id = prepared_db(tmp_path)
    identity = {"image_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]],
                "field_points": [[0, 0], [120, 0], [0, 53.333], [120, 53.333]]}
    save_calibration_keyframes(database, clip_id, {
        "keyframes": [{"timestamp_s": 1.0, **identity}, {"timestamp_s": 2.0, **identity}]})
    rows = []
    for frame_id, timestamp in enumerate((1.0, 1.5, 2.0)):
        rows.extend([
            {"clip_id": clip_id, "frame_id": frame_id, "video_timestamp": timestamp,
             "contact_x": 60.0, "contact_y": 26.0},            # midfield
            {"clip_id": clip_id, "frame_id": frame_id, "video_timestamp": timestamp,
             "contact_x": 60.0, "contact_y": -1.5},            # just outside a sideline
            {"clip_id": clip_id, "frame_id": frame_id, "video_timestamp": timestamp,
             "contact_x": 60.0, "contact_y": 55.0},            # just past the far sideline
            {"clip_id": clip_id, "frame_id": frame_id, "video_timestamp": timestamp,
             "contact_x": 60.0, "contact_y": -40.0},           # genuinely off field: a bench
        ])
    detections = tmp_path / "detections.parquet"
    pd.DataFrame(rows).to_parquet(detections)
    projected = tmp_path / "projected.parquet"
    project_clip(database, clip_id, detections, projected)
    frame = pd.read_parquet(projected)
    kept = frame[frame.on_field]
    assert set(kept.field_y.round(1)) == {26.0, -1.5, 55.0}, "sideline players must survive"
    assert (frame[~frame.on_field].field_y == -40.0).all(), "the bench must still be excluded"
    # How far outside is recorded so later stages can weigh it.
    assert frame.out_of_bounds_yards.max() == pytest.approx(40.0)
    assert frame.loc[frame.field_y == 26.0, "out_of_bounds_yards"].iloc[0] == 0.0


def test_team_resolution_is_generic_and_not_colour_dependent():
    """Sides come from formation geometry, so any uniform pairing works."""
    def build(colour_a, colour_b):
        rows = []
        for index in range(22):
            left = index < 11
            colour = colour_a if left else colour_b
            for step in range(3):
                rows.append({"track_id": f"t{index}", "video_timestamp": 10.0 + .1 * step,
                             "field_x": (48.0 if left else 52.0) + .05 * index, "field_y": 20.0,
                             "team": "team_0", "person_role": "player",
                             "lab_l": colour[0], "lab_a": colour[1], "lab_b": colour[2],
                             "hsv_h": 90, "hsv_s": 80, "hsv_v": 128})
        return pd.DataFrame(rows)

    # Three unrelated matchups, including two visually similar light uniforms.
    for colour_a, colour_b in (((200, 10, 10), (40, 200, 200)),
                               ((90, 150, 60), (95, 60, 150)),
                               ((210, 128, 128), (205, 130, 126))):
        resolved, diagnostics = resolve_track_teams(build(colour_a, colour_b), 10.0)
        assert diagnostics["method"] == "snap_line_of_scrimmage"
        counts = resolved[resolved.person_role == "player"].groupby("team").track_id.nunique()
        assert counts.to_dict() == {"team_0": 11, "team_1": 11}, (colour_a, colour_b)


def test_team_resolution_propagates_to_players_absent_at_the_snap():
    rows = []
    for index in range(22):
        left = index < 11
        for step in range(3):
            rows.append({"track_id": f"t{index}", "video_timestamp": 10.0 + .1 * step,
                         "field_x": (48.0 if left else 52.0) + .05 * index, "field_y": 20.0,
                         "team": "team_0", "person_role": "player",
                         "lab_l": 200 if left else 40, "lab_a": 10 if left else 200,
                         "lab_b": 10 if left else 200, "hsv_h": 90, "hsv_s": 80, "hsv_v": 128})
    # A player the snap frame never saw, wearing the left side's uniform.
    for step in range(3):
        rows.append({"track_id": "late", "video_timestamp": 11.0 + .1 * step,
                     "field_x": 70.0, "field_y": 30.0, "team": "unknown", "person_role": "player",
                     "lab_l": 200, "lab_a": 10, "lab_b": 10, "hsv_h": 90, "hsv_s": 80, "hsv_v": 128})
    resolved, diagnostics = resolve_track_teams(pd.DataFrame(rows), 10.0)
    assert diagnostics["appearance_propagated"] >= 1
    assert resolved[resolved.track_id == "late"].team.iloc[0] == "team_0"
