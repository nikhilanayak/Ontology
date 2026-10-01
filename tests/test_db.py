from pathlib import Path

from all22.db import connect, initialize


def test_v2_schema_is_idempotent_and_preserves_clip_artifact_keys(tmp_path: Path):
    database = tmp_path / "all22.sqlite3"
    initialize(database)
    initialize(database)
    with connect(database) as connection:
        tables = {row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(artifacts)")}
    assert {"action_windows", "shot_calibration_keyframes", "action_pairs",
            "pbp_action_alignments"}.issubset(tables)
    assert {"clip_id", "action_id", "config_hash", "input_revision"}.issubset(columns)
