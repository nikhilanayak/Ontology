from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS games (
  game_id TEXT PRIMARY KEY,
  season INTEGER,
  week INTEGER,
  home_team TEXT,
  away_team TEXT,
  game_date TEXT,
  video_path TEXT,
  pfr_url TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pbp_plays (
  game_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  quarter INTEGER,
  clock TEXT,
  possession TEXT,
  down_no INTEGER,
  distance INTEGER,
  yard_line TEXT,
  description TEXT NOT NULL,
  play_type TEXT NOT NULL,
  eligible INTEGER NOT NULL,
  source_row_id TEXT,
  PRIMARY KEY (game_id, ordinal),
  FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS clips (
  clip_id TEXT PRIMARY KEY,
  game_id TEXT NOT NULL,
  angle TEXT NOT NULL,
  start_s REAL NOT NULL,
  end_s REAL NOT NULL,
  snap_s REAL,
  play_end_s REAL,
  confidence REAL NOT NULL DEFAULT 0,
  FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS play_alignments (
  play_id TEXT PRIMARY KEY,
  game_id TEXT NOT NULL,
  pbp_ordinal INTEGER NOT NULL,
  sideline_clip_id TEXT,
  endzone_clip_id TEXT,
  score REAL NOT NULL,
  status TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY (game_id, pbp_ordinal) REFERENCES pbp_plays(game_id, ordinal)
);
CREATE TABLE IF NOT EXISTS play_sources (
  play_id TEXT NOT NULL,
  clip_id TEXT NOT NULL,
  source_order INTEGER NOT NULL,
  angle TEXT NOT NULL,
  PRIMARY KEY (play_id, source_order),
  UNIQUE (play_id, clip_id),
  FOREIGN KEY (play_id) REFERENCES play_alignments(play_id) ON DELETE CASCADE,
  FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_play_sources_clip ON play_sources(clip_id);
CREATE TABLE IF NOT EXISTS bdb_tracking (
  game_id TEXT NOT NULL,
  play_id TEXT NOT NULL,
  frame_id INTEGER NOT NULL,
  nfl_id TEXT,
  team TEXT,
  jersey_number INTEGER,
  x REAL NOT NULL,
  y REAL NOT NULL,
  speed REAL,
  acceleration REAL,
  direction REAL,
  orientation REAL,
  event TEXT,
  play_direction TEXT,
  PRIMARY KEY (game_id, play_id, frame_id, nfl_id)
);
CREATE INDEX IF NOT EXISTS idx_bdb_play ON bdb_tracking(game_id, play_id, frame_id);
CREATE TABLE IF NOT EXISTS quality_reports (
  play_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  reasons_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT,
  play_id TEXT,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  model_version TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS action_windows (
  action_id TEXT PRIMARY KEY,
  clip_id TEXT NOT NULL,
  action_order INTEGER NOT NULL,
  formation_start_s REAL,
  snap_s REAL,
  dead_s REAL,
  playback_end_s REAL,
  confidence REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'candidate',
  diagnostics_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE (clip_id, action_order),
  FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_action_windows_clip ON action_windows(clip_id, action_order);
CREATE TABLE IF NOT EXISTS shot_calibration_keyframes (
  clip_id TEXT NOT NULL,
  timestamp_s REAL NOT NULL,
  landmarks_json TEXT NOT NULL,
  matrix_json TEXT NOT NULL,
  inlier_ratio REAL NOT NULL,
  median_error_yards REAL NOT NULL,
  p95_error_yards REAL NOT NULL,
  revision INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'verified',
  PRIMARY KEY (clip_id, timestamp_s),
  FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS action_pairs (
  pair_id TEXT PRIMARY KEY,
  primary_action_id TEXT NOT NULL UNIQUE,
  alternate_action_id TEXT UNIQUE,
  score REAL NOT NULL,
  status TEXT NOT NULL,
  synchronization_json TEXT NOT NULL DEFAULT '{}',
  diagnostics_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY (primary_action_id) REFERENCES action_windows(action_id) ON DELETE CASCADE,
  FOREIGN KEY (alternate_action_id) REFERENCES action_windows(action_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS pbp_action_alignments (
  game_id TEXT NOT NULL,
  pbp_ordinal INTEGER NOT NULL,
  pair_id TEXT,
  action_id TEXT,
  score REAL NOT NULL,
  status TEXT NOT NULL,
  operation TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (game_id, pbp_ordinal),
  FOREIGN KEY (game_id, pbp_ordinal) REFERENCES pbp_plays(game_id, ordinal) ON DELETE CASCADE,
  FOREIGN KEY (pair_id) REFERENCES action_pairs(pair_id) ON DELETE SET NULL,
  FOREIGN KEY (action_id) REFERENCES action_windows(action_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS action_alignment_operations (
  operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  sequence_no INTEGER NOT NULL,
  operation TEXT NOT NULL,
  pbp_ordinal INTEGER,
  pair_id TEXT,
  score REAL NOT NULL,
  status TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE (game_id, sequence_no),
  FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE,
  FOREIGN KEY (game_id, pbp_ordinal) REFERENCES pbp_plays(game_id, ordinal) ON DELETE CASCADE,
  FOREIGN KEY (pair_id) REFERENCES action_pairs(pair_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS game_external_ids (
  game_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  external_id TEXT NOT NULL,
  PRIMARY KEY (game_id, provider),
  UNIQUE (provider, external_id),
  FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS alignment_audits (
  play_id TEXT PRIMARY KEY,
  selected INTEGER NOT NULL DEFAULT 1,
  mapping_correct INTEGER,
  sources_correct INTEGER,
  timing_correct INTEGER,
  notes TEXT,
  reviewed_at TEXT,
  FOREIGN KEY (play_id) REFERENCES play_alignments(play_id) ON DELETE CASCADE
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(path: Path) -> None:
    with connect(path) as connection:
        connection.executescript(SCHEMA)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(artifacts)")}
        for name, declaration in (
            ("clip_id", "TEXT"), ("action_id", "TEXT"),
            ("config_hash", "TEXT"), ("input_revision", "TEXT"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE artifacts ADD COLUMN {name} {declaration}")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_clip_kind ON artifacts(clip_id,kind)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_action_kind ON artifacts(action_id,kind)")


@contextmanager
def transaction(path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def upsert_quality(path: Path, report) -> None:
    with transaction(path) as connection:
        connection.execute(
            """INSERT INTO quality_reports(play_id,status,reasons_json,metrics_json,updated_at)
               VALUES(?,?,?,?,CURRENT_TIMESTAMP)
               ON CONFLICT(play_id) DO UPDATE SET status=excluded.status,
                 reasons_json=excluded.reasons_json, metrics_json=excluded.metrics_json,
                 updated_at=CURRENT_TIMESTAMP""",
            (report.play_id, report.status.value, json.dumps(report.reasons), json.dumps(report.metrics)),
        )
