from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import connect, transaction


class AuditUpdate(BaseModel):
    mapping_correct: bool
    sources_correct: bool
    timing_correct: bool
    notes: str = ""


class SourceTimingUpdate(BaseModel):
    snap_s: float
    play_end_s: float


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
                          q.reasons_json, q.metrics_json, aa.selected AS audit_selected,
                          aa.mapping_correct,aa.sources_correct,aa.timing_correct,aa.notes AS audit_notes
                   FROM pbp_plays p
                   LEFT JOIN play_alignments a ON a.game_id=p.game_id AND a.pbp_ordinal=p.ordinal
                   LEFT JOIN quality_reports q ON q.play_id=a.play_id
                   LEFT JOIN alignment_audits aa ON aa.play_id=a.play_id
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
            value["audit"] = {
                "selected": bool(value.pop("audit_selected") or 0),
                "mapping_correct": value.pop("mapping_correct"),
                "sources_correct": value.pop("sources_correct"),
                "timing_correct": value.pop("timing_correct"),
                "notes": value.pop("audit_notes") or "",
            }
            values.append(value)
        return values

    @app.put("/api/plays/{play_id}/audit")
    def save_audit(play_id: str, audit: AuditUpdate):
        with transaction(db_path) as connection:
            exists = connection.execute("SELECT 1 FROM play_alignments WHERE play_id=?", (play_id,)).fetchone()
            if not exists:
                raise HTTPException(404, "Play not found")
            connection.execute(
                """INSERT INTO alignment_audits
                     (play_id,selected,mapping_correct,sources_correct,timing_correct,notes,reviewed_at)
                   VALUES(?,1,?,?,?,?,CURRENT_TIMESTAMP)
                   ON CONFLICT(play_id) DO UPDATE SET selected=1,
                     mapping_correct=excluded.mapping_correct,sources_correct=excluded.sources_correct,
                     timing_correct=excluded.timing_correct,notes=excluded.notes,reviewed_at=CURRENT_TIMESTAMP""",
                (play_id, int(audit.mapping_correct), int(audit.sources_correct),
                 int(audit.timing_correct), audit.notes.strip()),
            )
        return {"play_id": play_id, **audit.model_dump()}

    @app.put("/api/clips/{clip_id}/timing")
    def save_source_timing(clip_id: str, timing: SourceTimingUpdate):
        with transaction(db_path) as connection:
            clip = connection.execute("SELECT start_s,end_s FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
            if not clip:
                raise HTTPException(404, "Source clip not found")
            if not (clip["start_s"] <= timing.snap_s < timing.play_end_s <= clip["end_s"]):
                raise HTTPException(422, "Timing must satisfy source start <= snap < play end <= source end")
            connection.execute("UPDATE clips SET snap_s=?,play_end_s=? WHERE clip_id=?",
                               (timing.snap_s, timing.play_end_s, clip_id))
        return {"clip_id": clip_id, **timing.model_dump()}

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
