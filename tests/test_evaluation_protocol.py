import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from all22 import supervision
from all22.db import initialize, transaction


PLAYERS = 8
SNAP_FRAME = 100
FRAMES = 41  # 4.0 s of 10 Hz BDB tracking after the snap
VIDEO_SNAP_S = 10.0


def _positions(step: int) -> np.ndarray:
    base = np.asarray([[30 + 4 * index, 10 + 4 * index] for index in range(PLAYERS)], dtype=float)
    drift = np.column_stack([np.full(PLAYERS, .3 * step), .1 * step * ((-1.0) ** np.arange(PLAYERS))])
    return base + drift


def _write_tracks(tracks_dir: Path, clip_id: str, offset_s: float = 0.0, flip_x: bool = False,
                  shift_x: float = 0.0) -> Path:
    """Synthetic tracks: BDB positions after an inverse video->field misregistration."""
    rows = []
    for step in range(FRAMES):
        for index, (x, y) in enumerate(_positions(step)):
            field_x = x - shift_x
            if flip_x:
                field_x = 120.0 - field_x
            rows.append({"video_timestamp": VIDEO_SNAP_S + step / 10.0 - offset_s,
                         "track_id": f"{clip_id}:t{index + 1}",
                         "field_x": float(field_x), "field_y": float(y), "confidence": .9,
                         "track_reliable": True, "team": "team_0" if index < 4 else "team_1", "speed": 3.0})
    path = tracks_dir / f"{clip_id}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def add_second_source(database: Path) -> None:
    with transaction(database) as connection:
        connection.execute(
            "INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence) VALUES('d','g','endzone',40,70,1)")
        connection.execute("INSERT INTO play_sources(play_id,clip_id,source_order,angle) VALUES('p','d',2,'endzone')")
        connection.execute(
            """INSERT INTO action_windows(action_id,clip_id,action_order,snap_s,dead_s,confidence)
               VALUES('d:a01','d',1,50.0,54.0,.9)""")


def prepared_case(tmp_path: Path) -> tuple[Path, Path, Path]:
    database = tmp_path / "db.sqlite3"
    initialize(database)
    tracks_dir = tmp_path / "clip-tracks"
    tracks_dir.mkdir()
    detections = tmp_path / "detections.parquet"
    pd.DataFrame({"clip_id": ["c"], "video_timestamp": [VIDEO_SNAP_S], "contact_x": [1.0],
                  "contact_y": [1.0]}).to_parquet(detections, index=False)
    with transaction(database) as connection:
        connection.execute("INSERT INTO games(game_id,video_path) VALUES('g','film.mkv')")
        connection.execute("INSERT INTO game_external_ids(game_id,provider,external_id) VALUES('g','bdb','2022090800')")
        connection.execute(
            "INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,confidence) VALUES('c','g','sideline',0,30,1)")
        connection.execute(
            """INSERT INTO pbp_plays(game_id,ordinal,description,play_type,eligible,source_row_id)
               VALUES('g',1,'pass','pass',1,'2022090800:56')""")
        connection.execute(
            "INSERT INTO play_alignments(play_id,game_id,pbp_ordinal,score,status) VALUES('p','g',1,1.0,'aligned')")
        connection.execute("INSERT INTO play_sources(play_id,clip_id,source_order,angle) VALUES('p','c',1,'sideline')")
        connection.execute(
            "INSERT INTO alignment_audits(play_id,mapping_correct,sources_correct) VALUES('p',1,1)")
        connection.execute(
            """INSERT INTO action_windows(action_id,clip_id,action_order,snap_s,dead_s,confidence)
               VALUES('c:a01','c',1,?,?,.9)""", (VIDEO_SNAP_S, VIDEO_SNAP_S + 4.0))
        connection.execute(
            """INSERT INTO shot_calibration_keyframes(clip_id,timestamp_s,landmarks_json,matrix_json,inlier_ratio,
               median_error_yards,p95_error_yards,revision) VALUES('c',10.0,'[]',?,1,0,0,3)""",
            (json.dumps(np.eye(3).tolist()),))
        connection.execute(
            """INSERT INTO artifacts(game_id,clip_id,kind,path,model_version,config_hash)
               VALUES('g','c','clip_detections',?,'detector:v1','abc123')""", (str(detections),))
        for step in range(FRAMES):
            for index, (x, y) in enumerate(_positions(step)):
                connection.execute(
                    """INSERT INTO bdb_tracking(game_id,play_id,frame_id,nfl_id,team,x,y,speed,event,play_direction)
                       VALUES('2022090800','56',?,?,?,?,?,?,?,'right')""",
                    (SNAP_FRAME + step, f"n{index}", "BUF" if index < 4 else "LA", float(x), float(y),
                     3.0, "ball_snap" if step == 0 else None))
    _write_tracks(tracks_dir, "c")
    return database, tracks_dir, detections


def test_exploratory_evaluation_reports_provenance_and_perfect_alignment(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    result = supervision.evaluate_audited_sources(database, "g", tracks_dir)
    assert result["sources"] == 1
    assert result["skipped"] == []
    assert result["median_source_error_yards"] < .05
    assert result["median_source_coverage"] == 1.0
    assert result["median_track_purity"] == 1.0
    source = result["results"][0]
    assert source["window"] == {"snap_s": VIDEO_SNAP_S, "dead_s": VIDEO_SNAP_S + 4.0}
    assert source["tracks"]["sha256"]
    assert source["inputs"]["detections"]["sha256"]
    assert source["inputs"]["calibration_revision"] == 3
    provenance = result["provenance"]
    assert provenance["evaluator_version"] == supervision.EVALUATOR_VERSION
    assert provenance["evaluator_parameters"] == supervision.EVALUATOR_PARAMETERS
    assert provenance["protocol"] is None


def test_exploratory_evaluation_reports_skipped_sources_instead_of_dropping_them(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    (tracks_dir / "c.parquet").unlink()
    with pytest.raises(ValueError, match="tracks parquet missing"):
        supervision.evaluate_audited_sources(database, "g", tracks_dir)


def test_frozen_protocol_pins_sources_window_and_inputs(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = supervision.freeze_evaluation_protocol(database, "g", tracks_dir, protocol_path)
    stored = json.loads(protocol_path.read_text())
    assert stored == protocol
    assert protocol["evaluator_version"] == supervision.EVALUATOR_VERSION
    assert [item["clip_id"] for item in protocol["sources"]] == ["c"]
    source = protocol["sources"][0]
    assert source["bdb_play_id"] == "56"
    assert (source["snap_s"], source["dead_s"]) == (VIDEO_SNAP_S, VIDEO_SNAP_S + 4.0)
    assert source["bdb_snap_frame"] == SNAP_FRAME
    assert source["bdb_rows"] == PLAYERS * FRAMES
    assert source["calibration_revision"] == 3
    assert source["detections"]["model_version"] == "detector:v1"
    assert source["detections"]["sha256"]

    # A later action-window rewrite must not move the evaluated time span.
    with transaction(database) as connection:
        connection.execute("UPDATE action_windows SET snap_s=snap_s+1.0, dead_s=dead_s+1.0 WHERE clip_id='c'")
    result = supervision.evaluate_with_protocol(database, protocol_path, tracks_dir)
    assert result["sources"] == 1
    assert result["results"][0]["window"] == {"snap_s": VIDEO_SNAP_S, "dead_s": VIDEO_SNAP_S + 4.0}
    assert result["results"][0]["input_drift"] == []
    assert result["provenance"]["protocol"]["sha256"] == supervision._file_sha256(protocol_path)
    assert result["provenance"]["input_drift"] == []


def test_frozen_protocol_rejects_unknown_requested_clips(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    with pytest.raises(ValueError, match="not audited sources"):
        supervision.freeze_evaluation_protocol(database, "g", tracks_dir, tmp_path / "p.json", ["missing"])


def test_protocol_evaluation_fails_closed_on_input_drift(tmp_path: Path):
    database, tracks_dir, detections = prepared_case(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    supervision.freeze_evaluation_protocol(database, "g", tracks_dir, protocol_path)
    pd.DataFrame({"clip_id": ["c"], "video_timestamp": [VIDEO_SNAP_S], "contact_x": [2.0],
                  "contact_y": [2.0]}).to_parquet(detections, index=False)
    with pytest.raises(ValueError, match="drifted"):
        supervision.evaluate_with_protocol(database, protocol_path, tracks_dir)
    result = supervision.evaluate_with_protocol(database, protocol_path, tracks_dir, allow_drift=True)
    assert result["provenance"]["drift_allowed"] is True
    assert result["provenance"]["input_drift"][0]["clip_id"] == "c"
    assert any(reason.startswith("detections.sha256") for reason in result["provenance"]["input_drift"][0]["reasons"])


def test_protocol_evaluation_fails_closed_on_calibration_drift_and_missing_tracks(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    supervision.freeze_evaluation_protocol(database, "g", tracks_dir, protocol_path)
    with transaction(database) as connection:
        connection.execute("UPDATE shot_calibration_keyframes SET revision=4 WHERE clip_id='c'")
    with pytest.raises(ValueError, match="calibration_revision: 3 -> 4"):
        supervision.evaluate_with_protocol(database, protocol_path, tracks_dir)
    (tracks_dir / "c.parquet").unlink()
    with pytest.raises(ValueError, match="missing or unscorable"):
        supervision.evaluate_with_protocol(database, protocol_path, tracks_dir, allow_drift=True)


def test_protocol_evaluation_rejects_other_evaluator_versions(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = supervision.freeze_evaluation_protocol(database, "g", tracks_dir, protocol_path)
    protocol["evaluator_version"] = "bdb-audited-v1"
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="frozen for evaluator"):
        supervision.evaluate_with_protocol(database, protocol_path, tracks_dir)
    protocol["evaluator_version"] = supervision.EVALUATOR_VERSION
    protocol["evaluator_parameters"] = {**protocol["evaluator_parameters"], "identity_maximum_error_yards": 1.0}
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="parameters differ"):
        supervision.evaluate_with_protocol(database, protocol_path, tracks_dir)


def test_evaluator_does_not_repair_absolute_yard_line_phase_from_truth(tmp_path: Path):
    """Absolute field position is an inference output, not an evaluator free parameter."""
    database, tracks_dir, _ = prepared_case(tmp_path)
    _write_tracks(tracks_dir, "c", offset_s=.4, flip_x=True, shift_x=10.0)
    result = supervision.evaluate_audited_sources(database, "g", tracks_dir)
    source = result["results"][0]
    assert source["flip_x"] is True and source["flip_y"] is False
    assert source["yard_line_translation_x"] == 0.0
    # The ten-yard phase error must remain visible rather than being repaired
    # from BDB truth, or calibration work cannot be measured.
    assert source["median_error_yards"] > 2.0
    assert source["hota"]["DetA"] < .7
    json.dumps(result)


def test_exploratory_evaluation_scores_present_sources_and_lists_skipped_ones(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    add_second_source(database)
    result = supervision.evaluate_audited_sources(database, "g", tracks_dir)
    assert [item["clip_id"] for item in result["results"]] == ["c"]
    assert result["skipped"] == [{"clip_id": "d", "reason": "tracks parquet missing",
                                  "path": str(tracks_dir / "d.parquet")}]


def test_protocol_freeze_skips_unscorable_sources_and_evaluation_requires_full_set(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    add_second_source(database)
    protocol_path = tmp_path / "protocol.json"
    protocol = supervision.freeze_evaluation_protocol(database, "g", tracks_dir, protocol_path)
    assert [item["clip_id"] for item in protocol["sources"]] == ["c"]
    assert protocol["skipped"][0]["clip_id"] == "d"
    # Restricting to explicit clips is honoured.
    restricted = supervision.freeze_evaluation_protocol(database, "g", tracks_dir, tmp_path / "r.json", ["c"])
    assert [item["clip_id"] for item in restricted["sources"]] == ["c"]
    json.dumps(protocol)


def test_protocol_evaluation_always_fails_when_bdb_answer_key_changes(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    supervision.freeze_evaluation_protocol(database, "g", tracks_dir, protocol_path)
    with transaction(database) as connection:
        connection.execute("DELETE FROM bdb_tracking WHERE nfl_id='n7' AND frame_id=140")
    with pytest.raises(ValueError, match="BDB answer key changed"):
        supervision.evaluate_with_protocol(database, protocol_path, tracks_dir)
    # Accepting input drift must never accept an answer-key change.
    with pytest.raises(ValueError, match="BDB answer key changed"):
        supervision.evaluate_with_protocol(database, protocol_path, tracks_dir, allow_drift=True, drift_note="x")


def test_tracks_config_hash_is_attributed_only_to_the_matching_artifact(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    with transaction(database) as connection:
        connection.execute(
            """INSERT INTO artifacts(game_id,clip_id,kind,path,config_hash,input_revision)
               VALUES('g','c','clip_tracks',?,'deadbeefdeadbeef','3')""", (str((tracks_dir / "c.parquet").resolve()),))
        connection.execute(
            """INSERT INTO artifacts(game_id,clip_id,kind,path,config_hash,input_revision)
               VALUES('g','c','clip_tracks',?,'0123456789abcdef','3')""", (str(tmp_path / "elsewhere.parquet"),))
    result = supervision.evaluate_audited_sources(database, "g", tracks_dir)
    tracks = result["results"][0]["tracks"]
    assert tracks["config_hash"] == "deadbeefdeadbeef"
    assert tracks["calibration_revision"] == "3"
    assert tracks["artifact_matched"] is True
    other_dir = tmp_path / "other-tracks"
    other_dir.mkdir()
    _write_tracks(other_dir, "c")
    result = supervision.evaluate_audited_sources(database, "g", other_dir)
    assert result["results"][0]["tracks"]["config_hash"] is None
    assert result["results"][0]["tracks"]["artifact_matched"] is False


def test_provenance_records_git_state(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    result = supervision.evaluate_audited_sources(database, "g", tracks_dir)
    provenance = result["provenance"]
    assert set(provenance) >= {"git_commit", "git_dirty", "evaluated_at", "tracks_dir"}
    assert provenance["git_dirty"] in (True, False, None)


def test_protocol_assigns_stable_validation_and_test_splits(tmp_path: Path):
    """Hyperparameters are chosen on validation; test is reported once."""
    database, tracks_dir, _ = prepared_case(tmp_path)
    add_second_source(database)
    _write_tracks(tracks_dir, "d", offset_s=-40.0)
    with transaction(database) as connection:
        connection.execute("UPDATE action_windows SET snap_s=50.0, dead_s=54.0 WHERE clip_id='d'")
    protocol_path = tmp_path / "protocol.json"
    protocol = supervision.freeze_evaluation_protocol(database, "g", tracks_dir, protocol_path)
    splits = {item["clip_id"]: item["split"] for item in protocol["sources"]}
    assert set(splits.values()) <= {"validation", "test"}
    # The assignment is a pure function of the clip id, so it is reproducible.
    again = supervision.freeze_evaluation_protocol(database, "g", tracks_dir, tmp_path / "p2.json")
    assert {item["clip_id"]: item["split"] for item in again["sources"]} == splits
    assert supervision._split_for("c") == supervision._split_for("c")


def test_evaluation_reports_hota_and_separates_detection_from_association(tmp_path: Path):
    database, tracks_dir, _ = prepared_case(tmp_path)
    result = supervision.evaluate_audited_sources(database, "g", tracks_dir)
    # Tracks were generated from the truth, so detection and association are perfect.
    assert result["hota"] == pytest.approx(1.0, abs=1e-6)
    assert result["det_a"] == pytest.approx(1.0, abs=1e-6)
    assert result["ass_a"] == pytest.approx(1.0, abs=1e-6)
    assert result["loc_a"] > .99
    assert result["hota"] == pytest.approx((result["det_a"] * result["ass_a"]) ** .5, abs=1e-6)
    source = result["results"][0]
    assert source["hota"]["gt_ids"] == PLAYERS
    assert source["hota"]["tau"] == supervision.EVALUATOR_PARAMETERS["hota_tau_yards"]
    json.dumps(result)


def test_hota_detects_an_identity_swap_that_purity_understates(tmp_path: Path):
    """An induced swap must lower AssA while leaving DetA untouched."""
    database, tracks_dir, _ = prepared_case(tmp_path)
    frame = pd.read_parquet(tracks_dir / "c.parquet")
    half = frame.video_timestamp.median()
    swapped = frame.copy()
    first, second = "c:t1", "c:t2"
    late = swapped.video_timestamp > half
    swapped.loc[late & (frame.track_id == first), "track_id"] = "tmp"
    swapped.loc[late & (frame.track_id == second), "track_id"] = first
    swapped.loc[swapped.track_id == "tmp", "track_id"] = second
    swapped.to_parquet(tracks_dir / "c.parquet", index=False)
    result = supervision.evaluate_audited_sources(database, "g", tracks_dir)
    assert result["det_a"] == pytest.approx(1.0, abs=1e-6), "no detection was lost"
    assert result["ass_a"] < 1.0, "but identity was broken"
    assert result["hota"] < result["det_a"]
