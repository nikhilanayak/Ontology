from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .db import connect


def create_app(db_path: Path, trajectories_dir: Path, static_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="All-22 Reconstruction", version="0.1.0")

    @app.get("/api/games")
    def games():
        with connect(db_path) as connection:
            rows = connection.execute(
                """SELECT g.*, COUNT(p.ordinal) AS play_count
                   FROM games g LEFT JOIN pbp_plays p ON p.game_id=g.game_id AND p.eligible=1
                   GROUP BY g.game_id ORDER BY g.game_date DESC, g.game_id"""
            ).fetchall()
            return [dict(row) for row in rows]

    @app.get("/api/games/{game_id}/plays")
    def plays(game_id: str):
        with connect(db_path) as connection:
            rows = connection.execute(
                """SELECT p.*, a.play_id, a.status AS alignment_status, q.status AS quality_status,
                          q.reasons_json, q.metrics_json
                   FROM pbp_plays p
                   LEFT JOIN play_alignments a ON a.game_id=p.game_id AND a.pbp_ordinal=p.ordinal
                   LEFT JOIN quality_reports q ON q.play_id=a.play_id
                   WHERE p.game_id=? AND p.eligible=1 ORDER BY p.ordinal""", (game_id,),
            ).fetchall()
            source_rows = connection.execute(
                """SELECT ps.play_id,ps.source_order,ps.angle,c.clip_id,c.start_s,c.end_s,c.snap_s,c.play_end_s,c.confidence
                   FROM play_sources ps JOIN clips c ON c.clip_id=ps.clip_id
                   JOIN play_alignments a ON a.play_id=ps.play_id
                   WHERE a.game_id=? ORDER BY ps.play_id,ps.source_order""", (game_id,),
            ).fetchall()
        sources_by_play = {}
        for source in source_rows:
            value = dict(source)
            sources_by_play.setdefault(value.pop("play_id"), []).append(value)
        values = []
        for row in rows:
            value = dict(row)
            reasons = json.loads(value.pop("reasons_json") or "[]")
            value["metrics"] = json.loads(value.pop("metrics_json") or "{}")
            if value["play_id"] is None:
                value["processing_status"] = "not_aligned"
                reasons = ["Film segmentation and play alignment have not been run for this play."]
            elif value["alignment_status"] == "rejected":
                value["processing_status"] = "alignment_rejected"
                reasons = reasons or ["No acceptable sideline/end-zone clip pair was found for this play."]
            elif value["quality_status"] is None:
                value["processing_status"] = "awaiting_reconstruction"
                reasons = ["The play is aligned, but tracking and quality evaluation have not been run."]
            else:
                value["processing_status"] = value["quality_status"]
            value["reasons"] = reasons
            value["sources"] = sources_by_play.get(value["play_id"], [])
            values.append(value)
        return values

    @app.get("/api/plays/{play_id}/sources")
    def play_sources(play_id: str):
        with connect(db_path) as connection:
            rows = connection.execute(
                """SELECT ps.source_order,ps.angle,c.clip_id,c.start_s,c.end_s,c.snap_s,c.play_end_s,c.confidence
                   FROM play_sources ps JOIN clips c ON c.clip_id=ps.clip_id
                   WHERE ps.play_id=? ORDER BY ps.source_order""", (play_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @app.get("/api/plays/{play_id}")
    def play(play_id: str):
        with connect(db_path) as connection:
            row = connection.execute(
                """SELECT a.*, p.quarter,p.clock,p.possession,p.down_no,p.distance,p.yard_line,p.description,
                          s.start_s AS sideline_start_s,s.snap_s AS sideline_snap_s,
                          e.start_s AS endzone_start_s,e.snap_s AS endzone_snap_s,
                          q.status AS quality_status,q.reasons_json,q.metrics_json,g.video_path
                   FROM play_alignments a JOIN pbp_plays p ON p.game_id=a.game_id AND p.ordinal=a.pbp_ordinal
                   LEFT JOIN clips s ON s.clip_id=a.sideline_clip_id LEFT JOIN clips e ON e.clip_id=a.endzone_clip_id
                   LEFT JOIN quality_reports q ON q.play_id=a.play_id JOIN games g ON g.game_id=a.game_id
                   WHERE a.play_id=?""", (play_id,),
            ).fetchone()
        if not row:
            raise HTTPException(404, "Play not found")
        value = dict(row)
        value["reasons"] = json.loads(value.pop("reasons_json") or "[]")
        value["metrics"] = json.loads(value.pop("metrics_json") or "{}")
        return value

    @app.get("/api/plays/{play_id}/tracks")
    def tracks(play_id: str, stride: int = Query(1, ge=1, le=10)):
        path = trajectories_dir / f"{play_id}.parquet"
        if not path.exists():
            raise HTTPException(404, "Trajectory artifact not found")
        frame = pd.read_parquet(path)
        if stride > 1:
            frame = frame[frame.frame_id.astype(int) % stride == 0]
        return frame.where(pd.notnull(frame), None).to_dict(orient="records")

    @app.get("/api/video/{game_id}")
    def video(game_id: str):
        with connect(db_path) as connection:
            row = connection.execute("SELECT video_path FROM games WHERE game_id=?", (game_id,)).fetchone()
        if not row or not row["video_path"]:
            raise HTTPException(404, "Video not registered")
        path = Path(row["video_path"])
        if not path.exists():
            raise HTTPException(404, "Video file missing")
        return FileResponse(path)

    if static_dir and static_dir.exists():
        app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

        @app.get("/")
        def index():
            return FileResponse(static_dir / "index.html")

    return app
