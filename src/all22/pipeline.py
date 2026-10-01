from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .alignment import align_monotonic, group_angle_sources
from .db import connect, transaction
from .models import Angle, Clip, PlayByPlay
from .video import analyze_clips, coalesce_short_fragments, probe, source_cut_candidates


def register_game(db_path: Path, game_id: str, video_path: Path, season: int = None,
                  week: int = None, home_team: str = None, away_team: str = None,
                  pfr_url: str = None) -> None:
    probe(video_path)
    with transaction(db_path) as connection:
        connection.execute(
            """INSERT INTO games(game_id,season,week,home_team,away_team,video_path,pfr_url)
               VALUES(?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
               season=COALESCE(excluded.season,games.season),week=COALESCE(excluded.week,games.week),
               home_team=COALESCE(excluded.home_team,games.home_team),away_team=COALESCE(excluded.away_team,games.away_team),
               video_path=excluded.video_path,pfr_url=COALESCE(excluded.pfr_url,games.pfr_url)""",
            (game_id, season, week, home_team, away_team, str(video_path.resolve()), pfr_url),
        )


def segment_game(db_path: Path, game_id: str, sample_fps: float = 4.0,
                 scene_threshold: float = 0.10) -> List[Clip]:
    with connect(db_path) as connection:
        row = connection.execute("SELECT video_path FROM games WHERE game_id=?", (game_id,)).fetchone()
    if not row:
        raise ValueError(f"Game {game_id} is not registered")
    path = Path(row["video_path"])
    cuts = source_cut_candidates(path, sample_fps, scene_threshold)
    clips = coalesce_short_fragments(analyze_clips(path, game_id, cuts))
    with transaction(db_path) as connection:
        connection.execute("DELETE FROM clips WHERE game_id=?", (game_id,))
        connection.executemany(
            """INSERT INTO clips(clip_id,game_id,angle,start_s,end_s,snap_s,play_end_s,confidence)
               VALUES(?,?,?,?,?,?,?,?)""",
            [(clip.clip_id, clip.game_id, clip.angle.value, clip.start_s, clip.end_s,
              clip.snap_s, clip.play_end_s, clip.confidence) for clip in clips],
        )
    return clips


def align_game(db_path: Path, game_id: str) -> int:
    with connect(db_path) as connection:
        play_rows = connection.execute("SELECT * FROM pbp_plays WHERE game_id=? ORDER BY ordinal", (game_id,)).fetchall()
        clip_rows = connection.execute("SELECT * FROM clips WHERE game_id=? ORDER BY start_s", (game_id,)).fetchall()
    plays = [PlayByPlay(game_id, row["ordinal"], row["quarter"], row["clock"], row["possession"],
                        row["down_no"], row["distance"], row["yard_line"], row["description"],
                        row["play_type"], bool(row["eligible"]), row["source_row_id"]) for row in play_rows]
    clips = [Clip(game_id, row["clip_id"], Angle(row["angle"]), row["start_s"], row["end_s"],
                  row["snap_s"], row["play_end_s"], row["confidence"]) for row in clip_rows]
    alignments = align_monotonic(plays, group_angle_sources(clips), include_all_filmed_plays=True)
    with transaction(db_path) as connection:
        connection.execute("DELETE FROM play_alignments WHERE game_id=?", (game_id,))
        for item in alignments:
            play_id = f"{game_id}:{item.play.ordinal:04d}"
            sources = item.source_group.sources if item.source_group else ()
            diagnostics = {"operation": item.operation, "source_count": len(sources),
                           "angles": [source.angle.value for source in sources]}
            connection.execute(
                """INSERT INTO play_alignments(play_id,game_id,pbp_ordinal,sideline_clip_id,endzone_clip_id,score,status,diagnostics_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (play_id, game_id, item.play.ordinal,
                 item.source_group.sideline.clip_id if item.source_group and item.source_group.sideline else None,
                 item.source_group.endzone.clip_id if item.source_group and item.source_group.endzone else None,
                 item.score, "pending" if item.source_group else "rejected", json.dumps(diagnostics)),
            )
            connection.executemany(
                "INSERT INTO play_sources(play_id,clip_id,source_order,angle) VALUES(?,?,?,?)",
                [(play_id, source.clip_id, order, source.angle.value)
                 for order, source in enumerate(sources)],
            )
    return len(alignments)
