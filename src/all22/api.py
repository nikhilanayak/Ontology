from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import field_registration, field_tracking, hough, workflow
from .db import connect, transaction


class AuditUpdate(BaseModel):
    mapping_correct: bool
    sources_correct: bool
    timing_correct: bool
    notes: str = ""


class SourceTimingUpdate(BaseModel):
    snap_s: float
    play_end_s: float


class CalibrationUpdate(BaseModel):
    keyframes: list[dict]


class ActionTimingUpdate(BaseModel):
    formation_start_s: float
    snap_s: float
    dead_s: float
    playback_end_s: float


def _review_priority(play: dict) -> tuple[float, list[str]]:
    """Rank plays where human review can resolve real uncertainty."""
    sources = play.get("sources") or []
    score = 0.0
    reasons: list[str] = []
    if len(sources) != 2:
        score += 50
        reasons.append(f"{len(sources)} source{'s' if len(sources) != 1 else ''}; expected 2")
    missing_timing = sum(source.get("snap_s") is None or source.get("play_end_s") is None
                         for source in sources)
    if missing_timing:
        score += 45 + 5 * missing_timing
        reasons.append("missing action timing")
    confidences = [float(source.get("confidence") or 0) for source in sources]
    if confidences and min(confidences) < .5:
        minimum = min(confidences)
        score += 20 + 40 * (.5 - minimum)
        reasons.append(f"low source confidence ({minimum:.2f})")
    durations = [float(source["play_end_s"]) - float(source["snap_s"]) for source in sources
                 if source.get("snap_s") is not None and source.get("play_end_s") is not None]
    if len(durations) == 2 and min(durations) > 0:
        ratio = max(durations) / min(durations)
        if ratio >= 2:
            score += 30 + min(30, 10 * (ratio - 2))
            reasons.append(f"angle timings disagree ({ratio:.1f}×)")
    # Sparse checkpoints catch monotonic alignment drift even when local
    # confidence metrics look healthy.
    if not reasons and int(play.get("ordinal") or 0) % 12 == 0:
        score += 10
        reasons.append("alignment checkpoint")
    return score, reasons


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
                          a.score AS alignment_score,q.reasons_json,q.metrics_json,aa.selected AS audit_selected,
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

    @app.get("/api/games/{game_id}/audit-queue")
    def audit_queue(game_id: str, limit: int = Query(30, ge=1, le=100)):
        candidates = []
        for play in plays(game_id):
            audit = play.get("audit") or {}
            if not play.get("play_id") or audit.get("mapping_correct") is not None:
                continue
            priority, reasons = _review_priority(play)
            if priority <= 0:
                continue
            play["review_priority"] = round(priority, 2)
            play["review_reasons"] = reasons
            # Queue membership authorizes the audit form before a row exists.
            play["audit"]["selected"] = True
            candidates.append(play)
        candidates.sort(key=lambda value: (-value["review_priority"], value["ordinal"]))
        return candidates[:limit]

    @app.get("/api/games/{game_id}/shots")
    def shots(game_id: str):
        with connect(db_path) as connection:
            rows = connection.execute(
                """SELECT c.*,COUNT(DISTINCT a.action_id) AS action_count,
                          COUNT(DISTINCT k.timestamp_s) AS calibration_keyframes
                   FROM clips c LEFT JOIN action_windows a ON a.clip_id=c.clip_id
                   LEFT JOIN shot_calibration_keyframes k ON k.clip_id=c.clip_id
                   WHERE c.game_id=? GROUP BY c.clip_id ORDER BY c.start_s""", (game_id,),
            ).fetchall()
            artifacts = connection.execute(
                """SELECT clip_id,kind,path,metadata_json,created_at FROM artifacts
                   WHERE game_id=? AND clip_id IS NOT NULL ORDER BY artifact_id""", (game_id,),
            ).fetchall()
        by_clip: dict[str, list[dict]] = {}
        for row in artifacts:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            by_clip.setdefault(item.pop("clip_id"), []).append(item)
        values = []
        for row in rows:
            item = dict(row)
            item["artifacts"] = by_clip.get(item["clip_id"], [])
            values.append(item)
        return values

    @app.get("/api/clips/{clip_id}/actions")
    def clip_actions(clip_id: str):
        with connect(db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM action_windows WHERE clip_id=? ORDER BY action_order", (clip_id,),
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["diagnostics"] = json.loads(item.pop("diagnostics_json") or "{}")
            values.append(item)
        return values

    @app.put("/api/clips/{clip_id}/calibration")
    def save_calibration(clip_id: str, update: CalibrationUpdate):
        try:
            values = field_tracking.save_calibration_keyframes(db_path, clip_id, update.model_dump())
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {"clip_id": clip_id, "keyframes": values}

    @app.get("/api/clips/{clip_id}/calibration")
    def calibration(clip_id: str):
        with connect(db_path) as connection:
            rows = connection.execute(
                """SELECT timestamp_s,landmarks_json,matrix_json,inlier_ratio,median_error_yards,
                          p95_error_yards,revision,status FROM shot_calibration_keyframes
                   WHERE clip_id=? ORDER BY timestamp_s""",
                (clip_id,),
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            # The image-to-field matrix lets the viewer reproject the field
            # template back onto the film, which is how a five-yard placement
            # error becomes visible rather than merely numerical.
            item["matrix"] = json.loads(item.pop("matrix_json") or "null")
            item.update(json.loads(item.pop("landmarks_json")))
            values.append(item)
        return {"clip_id": clip_id, "keyframes": values}

    @app.post("/api/clips/{clip_id}/densify-calibration")
    def densify_clip_calibration(clip_id: str, step_s: float = Query(.1, gt=0, le=.5)):
        try:
            values = field_registration.densify_calibration(db_path, clip_id, step_s)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error
        return {"clip_id": clip_id, "keyframes": values}

    @app.post("/api/clips/{clip_id}/auto-calibrate")
    def auto_calibrate(clip_id: str):
        try:
            values = field_registration.auto_calibrate_clip(
                db_path, clip_id, db_path.parent / "calibration-diagnostics")
            with connect(db_path) as connection:
                artifact = connection.execute(
                    """SELECT path FROM artifacts WHERE clip_id=? AND kind='clip_detections'
                       ORDER BY artifact_id DESC LIMIT 1""", (clip_id,),
                ).fetchone()
            reconstruction = (workflow.reconstruct_clip(db_path, clip_id, Path(artifact["path"]), db_path.parent)
                              if artifact and Path(artifact["path"]).exists() else None)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error
        return {"clip_id": clip_id, "keyframes": values, "reconstruction": reconstruction}

    @app.put("/api/actions/{action_id}/timing")
    def save_action_timing(action_id: str, timing: ActionTimingUpdate):
        with transaction(db_path) as connection:
            row = connection.execute(
                """SELECT c.start_s,c.end_s FROM action_windows a JOIN clips c ON c.clip_id=a.clip_id
                   WHERE a.action_id=?""", (action_id,),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Action not found")
            values = [timing.formation_start_s, timing.snap_s, timing.dead_s, timing.playback_end_s]
            if values != sorted(values) or not (row["start_s"] <= values[0] and values[-1] <= row["end_s"]):
                raise HTTPException(422, "Timing must be ordered and remain inside its camera shot")
            connection.execute(
                """UPDATE action_windows SET formation_start_s=?,snap_s=?,dead_s=?,playback_end_s=?,
                   status='verified' WHERE action_id=?""", (*values, action_id),
            )
        return {"action_id": action_id, **timing.model_dump(), "status": "verified"}

    @app.get("/api/clips/{clip_id}/tracks")
    def clip_tracks(clip_id: str, stride: int = Query(1, ge=1, le=20)):
        with connect(db_path) as connection:
            row = connection.execute(
                """SELECT path FROM artifacts WHERE clip_id=? AND kind='clip_tracks'
                   ORDER BY artifact_id DESC LIMIT 1""", (clip_id,),
            ).fetchone()
        if not row or not Path(row["path"]).exists():
            raise HTTPException(404, "Clip trajectory artifact not found")
        frame = pd.read_parquet(row["path"])
        if "roster_candidate" in frame:
            frame = frame[frame.roster_candidate]
        elif "track_reliable" in frame:
            frame = frame[frame.track_reliable]
        timestamps = sorted(frame.video_timestamp.unique())[::stride]
        frame = frame[frame.video_timestamp.isin(timestamps)]
        return frame.where(pd.notnull(frame), None).to_dict(orient="records")

    @app.get("/api/games/{game_id}/action-alignments")
    def action_alignments(game_id: str):
        with connect(db_path) as connection:
            rows = connection.execute(
                """SELECT o.*,p.description,p.play_type FROM action_alignment_operations o
                   LEFT JOIN pbp_plays p ON p.game_id=o.game_id AND p.ordinal=o.pbp_ordinal
                   WHERE o.game_id=? ORDER BY o.sequence_no""", (game_id,),
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["diagnostics"] = json.loads(item.pop("diagnostics_json") or "{}")
            values.append(item)
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

    @app.get("/api/clips/{clip_id}/field-detections")
    def field_detections(clip_id: str, timestamp_s: Optional[float] = None):
        """Expose raw yard-line and painted-number detections for one frame.

        Calibration can fit the lines it was handed almost perfectly and still
        place them on the wrong yard lines, so this reports what the detector
        actually saw -- every line, every OCR number with its confidence, and
        the yard value each line was assigned -- rather than only the result.
        """
        with connect(db_path) as connection:
            clip = connection.execute(
                """SELECT c.start_s,c.end_s,c.angle,g.video_path FROM clips c
                   JOIN games g ON g.game_id=c.game_id WHERE c.clip_id=?""", (clip_id,),
            ).fetchone()
        if not clip:
            raise HTTPException(404, "Source clip not found")
        video_path = Path(clip["video_path"] or "")
        if not video_path.exists():
            raise HTTPException(404, "Video file missing")
        start, end = float(clip["start_s"]), float(clip["end_s"])
        when = start + min(.2, (end - start) * .05) if timestamp_s is None else float(timestamp_s)
        if not start <= when <= end:
            raise HTTPException(422, f"Timestamp must lie inside the shot ({start:.2f}-{end:.2f}s)")
        try:
            payload = field_registration.describe_field_detections(video_path, when)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error
        return {"clip_id": clip_id, "angle": clip["angle"], "start_s": start, "end_s": end,
                "timestamp_s": when, **payload}

    @app.get("/api/clips/{clip_id}/detections")
    def clip_detections(clip_id: str, stride: int = Query(1, ge=1, le=20)):
        """Raw detector output, before projection and tracking filtered it."""
        with connect(db_path) as connection:
            row = connection.execute(
                """SELECT path FROM artifacts WHERE clip_id=? AND kind='clip_detections'
                   ORDER BY artifact_id DESC LIMIT 1""", (clip_id,),
            ).fetchone()
        if not row or not Path(row["path"]).exists():
            raise HTTPException(404, "Clip detection artifact not found")
        frame = pd.read_parquet(row["path"])
        keep = [column for column in
                ("video_timestamp", "x1", "y1", "x2", "y2", "confidence", "detection_id")
                if column in frame.columns]
        frame = frame[keep]
        if stride > 1:
            timestamps = sorted(frame.video_timestamp.unique())[::stride]
            frame = frame[frame.video_timestamp.isin(timestamps)]
        return frame.where(pd.notnull(frame), None).to_dict(orient="records")

    @app.get("/api/clips/{clip_id}/frame.jpg")
    def clip_frame(clip_id: str, timestamp_s: Optional[float] = None):
        """Return a single decoded frame so detections can be drawn over it."""
        import cv2
        from fastapi.responses import Response

        with connect(db_path) as connection:
            clip = connection.execute(
                """SELECT c.start_s,c.end_s,g.video_path FROM clips c
                   JOIN games g ON g.game_id=c.game_id WHERE c.clip_id=?""", (clip_id,),
            ).fetchone()
        if not clip:
            raise HTTPException(404, "Source clip not found")
        path = Path(clip["video_path"] or "")
        if not path.exists():
            raise HTTPException(404, "Video file missing")
        start, end = float(clip["start_s"]), float(clip["end_s"])
        when = start + min(.2, (end - start) * .05) if timestamp_s is None else float(timestamp_s)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise HTTPException(422, "Could not open the film")
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise HTTPException(422, f"Could not decode a frame at {when:.2f}s")
        encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not encoded:
            raise HTTPException(500, "Could not encode the frame")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    def _hough_clip_frame(clip_id: str, timestamp_s: Optional[float]):
        """Decode one frame of a shot for the Hough preview, with its shot bounds."""
        import cv2

        with connect(db_path) as connection:
            clip = connection.execute(
                """SELECT c.start_s,c.end_s,c.angle,g.video_path FROM clips c
                   JOIN games g ON g.game_id=c.game_id WHERE c.clip_id=?""", (clip_id,),
            ).fetchone()
        if not clip:
            raise HTTPException(404, "Source clip not found")
        if not clip["video_path"]:
            raise HTTPException(404, "Video not registered")
        path = Path(clip["video_path"])
        if not path.is_file():
            raise HTTPException(404, "Video file missing")
        start, end = float(clip["start_s"]), float(clip["end_s"])
        when = start if timestamp_s is None else float(timestamp_s)
        if not start <= when <= end:
            raise HTTPException(422, f"Timestamp must lie inside the shot ({start:.2f}-{end:.2f}s)")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise HTTPException(422, "Could not open the film")
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise HTTPException(422, f"Could not decode a frame at {when:.2f}s")
        return frame, {"clip_id": clip_id, "angle": clip["angle"], "start_s": start,
                       "end_s": end, "timestamp_s": when}

    def _hough_parameters(blur: int, canny_low: int, canny_high: int, threshold: int,
                          min_line_length: int, max_line_gap: int) -> "hough.HoughParameters":
        return hough.HoughParameters(
            blur=blur, canny_low=canny_low, canny_high=canny_high, threshold=threshold,
            min_line_length=min_line_length, max_line_gap=max_line_gap,
        ).normalized()

    @app.get("/api/hough/clips")
    def hough_clips(game_id: Optional[str] = None, limit: int = Query(2000, ge=1, le=2000)):
        """List shots that the Hough preview can draw frames from."""
        with connect(db_path) as connection:
            if game_id:
                rows = connection.execute(
                    """SELECT c.clip_id,c.game_id,c.angle,c.start_s,c.end_s FROM clips c
                       JOIN games g ON g.game_id=c.game_id
                       WHERE c.game_id=? AND g.video_path IS NOT NULL
                       ORDER BY c.start_s LIMIT ?""", (game_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT c.clip_id,c.game_id,c.angle,c.start_s,c.end_s FROM clips c
                       JOIN games g ON g.game_id=c.game_id WHERE g.video_path IS NOT NULL
                       ORDER BY c.game_id,c.start_s LIMIT ?""", (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    @app.get("/api/hough/clips/{clip_id}/lines")
    def hough_lines(clip_id: str, timestamp_s: Optional[float] = None,
                    blur: int = Query(5, ge=0, le=31),
                    canny_low: int = Query(50, ge=0, le=500),
                    canny_high: int = Query(150, ge=1, le=1000),
                    threshold: int = Query(45, ge=1, le=500),
                    min_line_length: int = Query(60, ge=1, le=2000),
                    max_line_gap: int = Query(35, ge=0, le=500)):
        """Return raw Hough segments for one frame, with no field interpretation."""
        frame, meta = _hough_clip_frame(clip_id, timestamp_s)
        parameters = _hough_parameters(blur, canny_low, canny_high, threshold,
                                       min_line_length, max_line_gap)
        return {**meta, **hough.describe(frame, parameters)}

    @app.get("/api/hough/clips/{clip_id}/annotated.jpg")
    def hough_annotated(clip_id: str, timestamp_s: Optional[float] = None,
                        blur: int = Query(5, ge=0, le=31),
                        canny_low: int = Query(50, ge=0, le=500),
                        canny_high: int = Query(150, ge=1, le=1000),
                        threshold: int = Query(45, ge=1, le=500),
                        min_line_length: int = Query(60, ge=1, le=2000),
                        max_line_gap: int = Query(35, ge=0, le=500),
                        thickness: int = Query(2, ge=1, le=10),
                        show_edges: bool = False, draw_lines: bool = True):
        """Return the frame with Hough segments drawn over it as a JPEG."""
        frame, _ = _hough_clip_frame(clip_id, timestamp_s)
        parameters = _hough_parameters(blur, canny_low, canny_high, threshold,
                                       min_line_length, max_line_gap)
        segments = hough.detect_lines(frame, parameters) if draw_lines \
            else np.empty((0, 4), dtype=np.int32)
        image = hough.annotate(frame, segments, thickness=thickness,
                               show_edges=show_edges, parameters=parameters)
        return Response(content=hough.encode_jpeg(image), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

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
            return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})

        @app.get("/hough")
        def hough_preview():
            return FileResponse(static_dir / "hough.html", headers={"Cache-Control": "no-store"})

    return app
