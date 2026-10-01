from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import action_alignment, actions, alignment, bdb, db, field_registration, field_tracking, nflverse, pfr, pilot, remote_download, supervision, tracking, video, workflow
from . import pipeline


ROOT = Path.cwd()
DEFAULT_DB = ROOT / "data" / "all22.sqlite3"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="all22")
    root.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("init-db")
    receive = commands.add_parser("receive-download")
    receive.add_argument("--output-root", type=Path, default=ROOT / "downloads")
    receive.add_argument("--metadata-root", type=Path, default=ROOT / "data" / "downloads")
    register = commands.add_parser("register-game")
    register.add_argument("--game-id", required=True)
    register.add_argument("--season", type=int)
    register.add_argument("--week", type=int)
    register.add_argument("--home-team")
    register.add_argument("--away-team")
    register.add_argument("--pfr-url")
    register.add_argument("video", type=Path)
    external = commands.add_parser("set-external-id")
    external.add_argument("--game-id", required=True)
    external.add_argument("--provider", required=True)
    external.add_argument("--external-id", required=True)
    import_bdb = commands.add_parser("import-bdb")
    import_bdb.add_argument("files", type=Path, nargs="+")

    import_pfr = commands.add_parser("import-pfr")
    import_pfr.add_argument("--game-id", required=True)
    import_pfr.add_argument("html", type=Path)

    import_nflverse = commands.add_parser("import-nflverse")
    import_nflverse.add_argument("--game-id", required=True)
    import_nflverse.add_argument("--source-game-id", help="For example: 2025_01_DAL_PHI")
    import_nflverse.add_argument("parquet", type=Path)

    probe = commands.add_parser("probe-video")
    probe.add_argument("video", type=Path)

    cuts = commands.add_parser("detect-cuts")
    cuts.add_argument("--game-id", required=True)
    cuts.add_argument("--sample-fps", type=float, default=2.0)
    cuts.add_argument("video", type=Path)

    actions = commands.add_parser("scan-actions")
    actions.add_argument("--sample-fps", type=float, default=5.0)
    actions.add_argument("--start-s", type=float, default=0.0)
    actions.add_argument("--end-s", type=float)
    actions.add_argument("--output", type=Path)
    actions.add_argument("video", type=Path)

    segment = commands.add_parser("segment-game")
    segment.add_argument("--game-id", required=True)
    segment.add_argument("--sample-fps", type=float, default=4.0)
    segment.add_argument("--scene-threshold", type=float, default=0.10)

    inspect_sources = commands.add_parser("inspect-sources")
    inspect_sources.add_argument("--game-id", required=True)
    inspect_sources.add_argument("--start-s", type=float, required=True)
    inspect_sources.add_argument("--end-s", type=float, required=True)
    inspect_sources.add_argument("--sample-fps", type=float, default=4.0)
    inspect_sources.add_argument("--scene-threshold", type=float, default=0.10)
    inspect_sources.add_argument("video", type=Path)

    align = commands.add_parser("align-game")
    align.add_argument("--game-id", required=True)

    select_pilot = commands.add_parser("select-pilot-games")
    select_pilot.add_argument("--games", type=Path, required=True)
    select_pilot.add_argument("--plays", type=Path, required=True)
    select_pilot.add_argument("--output", type=Path, default=ROOT / "data" / "pilot-games.json")

    audit = commands.add_parser("create-audit-sample")
    audit.add_argument("--game-id", required=True)
    audit.add_argument("--count", type=int, default=40)
    audit_summary = commands.add_parser("audit-summary")
    audit_summary.add_argument("--game-id", required=True)

    evaluate = commands.add_parser("evaluate-trajectories")
    evaluate.add_argument("--game-id", required=True)
    evaluate.add_argument("--predictions-dir", type=Path, default=ROOT / "data" / "trajectories")
    evaluate.add_argument("--output", type=Path)

    detect = commands.add_parser("detect-source")
    detect.add_argument("--play-id", required=True)
    detect.add_argument("--source-order", type=int, required=True)
    detect.add_argument("--sample-hz", type=float, default=10.0)
    detect.add_argument("--threshold", type=float, default=.35)
    detect.add_argument("--device")
    detect.add_argument("--output", type=Path, required=True)

    detect_clips = commands.add_parser("detect-clips")
    detect_clips.add_argument("--game-id", required=True)
    detect_clips.add_argument("--clip-id", action="append", dest="clip_ids")
    detect_clips.add_argument("--sample-hz", type=float, default=10.0)
    detect_clips.add_argument("--threshold", type=float, default=.35)
    detect_clips.add_argument("--device")
    detect_clips.add_argument("--limit", type=int)
    detect_clips.add_argument("--no-resume", action="store_true")
    detect_clips.add_argument("--output-dir", type=Path, default=ROOT / "data" / "detections")

    project = commands.add_parser("project-source")
    project.add_argument("--play-id", required=True)
    project.add_argument("--source-order", type=int, required=True)
    project.add_argument("--detections", type=Path, required=True)
    project.add_argument("--landmarks", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--video-anchor-s", type=float)
    project.add_argument("--bdb-anchor-frame", type=int)

    calibrate_clip = commands.add_parser("calibrate-clip")
    calibrate_clip.add_argument("--clip-id", required=True)
    calibrate_clip.add_argument("landmarks", type=Path)

    auto_calibrate = commands.add_parser("auto-calibrate-clip")
    auto_calibrate.add_argument("--clip-id", required=True)
    auto_calibrate.add_argument("--diagnostics-dir", type=Path,
                                default=ROOT / "data" / "calibration-diagnostics")

    auto_game = commands.add_parser("auto-reconstruct-game")
    auto_game.add_argument("--game-id", required=True)
    auto_game.add_argument("--limit", type=int)
    auto_game.add_argument("--no-resume", action="store_true")
    auto_game.add_argument("--output-root", type=Path, default=ROOT / "data")

    project_clip = commands.add_parser("project-clip")
    project_clip.add_argument("--clip-id", required=True)
    project_clip.add_argument("--detections", type=Path, required=True)
    project_clip.add_argument("--output", type=Path, required=True)

    track_clip = commands.add_parser("track-clip")
    track_clip.add_argument("--clip-id", required=True)
    track_clip.add_argument("--projected", type=Path, required=True)
    track_clip.add_argument("--output", type=Path, required=True)

    discover_actions = commands.add_parser("discover-actions")
    discover_actions.add_argument("--clip-id", required=True)
    discover_actions.add_argument("--tracks", type=Path, required=True)

    pair_actions = commands.add_parser("pair-actions")
    pair_actions.add_argument("--game-id", required=True)
    pair_actions.add_argument("--tracks-dir", type=Path, default=ROOT / "data" / "clip-tracks")

    align_actions = commands.add_parser("align-actions")
    align_actions.add_argument("--game-id", required=True)

    reconstruct = commands.add_parser("reconstruct-clip")
    reconstruct.add_argument("--clip-id", required=True)
    reconstruct.add_argument("--detections", type=Path, required=True)
    reconstruct.add_argument("--output-root", type=Path, default=ROOT / "data")

    trajectory_report = commands.add_parser("trajectory-pilot-report")
    trajectory_report.add_argument("--game-id", required=True)
    trajectory_report.add_argument("--output", type=Path)

    fuse = commands.add_parser("fuse-sources")
    fuse.add_argument("--output", type=Path, required=True)
    fuse.add_argument("inputs", type=Path, nargs="+")

    labels = commands.add_parser("export-bdb-play")
    labels.add_argument("--game-id", required=True)
    labels.add_argument("--play-id", required=True)
    labels.add_argument("--output", type=Path, required=True)

    pair = commands.add_parser("create-supervision-pair")
    pair.add_argument("--game-id", required=True)
    pair.add_argument("--play-id", required=True)
    pair.add_argument("--angle", choices=("sideline", "endzone"), required=True)
    pair.add_argument("--video-snap", type=float, required=True)
    pair.add_argument("--manifest", type=Path, default=ROOT / "data" / "supervision.jsonl")
    pair.add_argument("--output-dir", type=Path, default=ROOT / "data" / "bdb-labels")
    pair.add_argument("video", type=Path)

    evaluate_sources = commands.add_parser("evaluate-audited-sources")
    evaluate_sources.add_argument("--game-id", required=True)
    evaluate_sources.add_argument("--tracks-dir", type=Path, default=ROOT / "data" / "clip-tracks")
    evaluate_sources.add_argument("--output", type=Path)

    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return root


def main() -> None:
    args = parser().parse_args()
    db.initialize(args.db)
    if args.command == "init-db":
        print(args.db)
    elif args.command == "receive-download":
        job = remote_download.read_job()
        print(json.dumps(remote_download.receive_job(job, args.output_root, args.metadata_root)))
    elif args.command == "register-game":
        pipeline.register_game(args.db, args.game_id, args.video, args.season, args.week,
                               args.home_team, args.away_team, args.pfr_url)
        print(json.dumps({"game_id": args.game_id, "video": str(args.video.resolve())}))
    elif args.command == "set-external-id":
        pilot.set_external_id(args.db, args.game_id, args.provider, args.external_id)
        print(json.dumps({"game_id": args.game_id, "provider": args.provider, "external_id": args.external_id}))
    elif args.command == "import-bdb":
        print(json.dumps({"rows": bdb.import_tracking(args.db, args.files)}))
    elif args.command == "import-pfr":
        plays = pfr.parse_html(args.html, args.game_id)
        print(json.dumps({"plays": pfr.store_plays(args.db, plays), "eligible": sum(play.eligible for play in plays)}))
    elif args.command == "import-nflverse":
        with db.connect(args.db) as connection:
            game = connection.execute(
                "SELECT season,week,home_team,away_team FROM games WHERE game_id=?", (args.game_id,)
            ).fetchone()
        if not game:
            raise SystemExit(f"Game is not registered: {args.game_id}")
        source_id = args.source_game_id
        if not source_id:
            required = (game["season"], game["week"], game["home_team"], game["away_team"])
            if any(value is None for value in required):
                raise SystemExit("Game needs season, week, home team, and away team, or pass --source-game-id")
            source_id = nflverse.source_game_id(
                game["season"], game["week"], game["away_team"], game["home_team"]
            )
        plays = nflverse.parse_parquet(args.parquet, args.game_id, source_id)
        print(json.dumps({"source_game_id": source_id, "plays": pfr.store_plays(args.db, plays),
                          "eligible": sum(play.eligible for play in plays)}))
    elif args.command == "probe-video":
        print(json.dumps(video.probe(args.video).__dict__, default=str, indent=2))
    elif args.command == "detect-cuts":
        info = video.probe(args.video)
        cuts = video.visual_cut_candidates(args.video, args.sample_fps)
        print(json.dumps({"video": str(args.video), "cuts": cuts,
                          "clips": [clip.__dict__ for clip in video.clips_from_cuts(args.game_id, info.duration, cuts)]},
                         default=str))
    elif args.command == "scan-actions":
        samples = video.camera_compensated_motion(args.video, args.sample_fps, args.start_s, args.end_s)
        windows = video.action_windows_from_motion(samples, args.sample_fps)
        payload = {"video": str(args.video), "samples": [item.__dict__ for item in samples],
                   "windows": [item.__dict__ for item in windows]}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        print(json.dumps({"samples": len(samples), "camera_cuts": sum(item.camera_cut for item in samples),
                          "windows": [item.__dict__ for item in windows]}, indent=2))
    elif args.command == "segment-game":
        clips = pipeline.segment_game(args.db, args.game_id, args.sample_fps, args.scene_threshold)
        print(json.dumps({"clips": len(clips), "sideline": sum(c.angle.value == "sideline" for c in clips),
                          "endzone": sum(c.angle.value == "endzone" for c in clips)}))
    elif args.command == "inspect-sources":
        cuts = video.source_cut_candidates(args.video, args.sample_fps, args.scene_threshold,
                                           args.start_s, args.end_s)
        clips = video.coalesce_short_fragments(
            video.analyze_clips(args.video, args.game_id, cuts, args.start_s, args.end_s)
        )
        groups = alignment.group_angle_sources(clips)
        print(json.dumps({
            "cuts": cuts,
            "clips": [{**clip.__dict__, "angle": clip.angle.value} for clip in clips],
            "groups": [[source.clip_id for source in group.sources] for group in groups],
        }, default=str, indent=2))
    elif args.command == "align-game":
        print(json.dumps({"alignments": pipeline.align_game(args.db, args.game_id)}))
    elif args.command == "select-pilot-games":
        values = pilot.select_pilot_games(args.games, args.plays, args.output)
        print(json.dumps({"output": str(args.output), "games": values}, indent=2))
    elif args.command == "create-audit-sample":
        values = pilot.create_audit_sample(args.db, args.game_id, args.count)
        print(json.dumps({"game_id": args.game_id, "selected": len(values), "play_ids": values}))
    elif args.command == "audit-summary":
        print(json.dumps(pilot.audit_summary(args.db, args.game_id), indent=2))
    elif args.command == "evaluate-trajectories":
        result = pilot.evaluate_trajectories(args.db, args.game_id, args.predictions_dir)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "detect-source":
        rows = tracking.detect_source(args.db, args.play_id, args.source_order, args.output,
                                      args.sample_hz, args.threshold, args.device)
        print(json.dumps({"rows": rows, "output": str(args.output)}))
    elif args.command == "detect-clips":
        results = tracking.detect_clips(
            args.db, args.game_id, args.output_dir, args.sample_hz, args.threshold,
            args.device, args.clip_ids, args.limit, not args.no_resume,
        )
        print(json.dumps({"game_id": args.game_id, "clips": len(results), "results": results}, indent=2))
    elif args.command == "project-source":
        result = tracking.project_source(args.db, args.play_id, args.source_order,
                                         args.detections, args.landmarks, args.output,
                                         args.video_anchor_s, args.bdb_anchor_frame)
        print(json.dumps({**result, "output": str(args.output)}, indent=2))
    elif args.command == "calibrate-clip":
        payload = json.loads(args.landmarks.read_text(encoding="utf-8"))
        print(json.dumps({"clip_id": args.clip_id,
                          "keyframes": field_tracking.save_calibration_keyframes(args.db, args.clip_id, payload)},
                         indent=2))
    elif args.command == "auto-calibrate-clip":
        values = field_registration.auto_calibrate_clip(args.db, args.clip_id, args.diagnostics_dir)
        print(json.dumps({"clip_id": args.clip_id, "keyframes": values}, indent=2))
    elif args.command == "auto-reconstruct-game":
        values = field_registration.auto_reconstruct_game(
            args.db, args.game_id, args.output_root, args.limit, not args.no_resume)
        print(json.dumps({"game_id": args.game_id, "results": values,
                          "created": sum(item["status"] == "created" for item in values),
                          "failed": sum(item["status"] == "failed" for item in values),
                          "cached": sum(item["status"] == "cached" for item in values)}, indent=2))
    elif args.command == "project-clip":
        result = field_tracking.project_clip(args.db, args.clip_id, args.detections, args.output)
        print(json.dumps({**result, "output": str(args.output)}, indent=2))
    elif args.command == "track-clip":
        result = field_tracking.track_projected_clip(args.db, args.clip_id, args.projected, args.output)
        print(json.dumps({**result, "output": str(args.output)}, indent=2))
    elif args.command == "discover-actions":
        values = actions.discover_clip_actions(args.db, args.clip_id, args.tracks)
        print(json.dumps({"clip_id": args.clip_id, "actions": values}, indent=2))
    elif args.command == "pair-actions":
        values = actions.pair_game_actions(args.db, args.game_id, args.tracks_dir)
        print(json.dumps({"game_id": args.game_id, "pairs": values}, indent=2))
    elif args.command == "align-actions":
        values = action_alignment.align_game_actions(args.db, args.game_id)
        print(json.dumps({"game_id": args.game_id, "operations": values}, indent=2))
    elif args.command == "reconstruct-clip":
        print(json.dumps(workflow.reconstruct_clip(args.db, args.clip_id, args.detections,
                                                   args.output_root), indent=2))
    elif args.command == "trajectory-pilot-report":
        print(json.dumps(workflow.write_pilot_report(args.db, args.game_id, args.output), indent=2))
    elif args.command == "fuse-sources":
        rows = tracking.fuse_sources(args.inputs, args.output)
        print(json.dumps({"rows": rows, "output": str(args.output)}))
    elif args.command == "export-bdb-play":
        frame = supervision.bdb_play_dataframe(args.db, args.game_id, args.play_id)
        if frame.empty:
            raise SystemExit("No BDB samples found for that game/play")
        supervision.export_labels(frame, args.output)
        print(json.dumps({"rows": len(frame), "output": str(args.output)}))
    elif args.command == "create-supervision-pair":
        record = supervision.create_pair(args.db, args.manifest, args.output_dir, args.game_id,
                                         args.play_id, args.angle, args.video, args.video_snap)
        print(json.dumps(record, indent=2))
    elif args.command == "evaluate-audited-sources":
        value = supervision.evaluate_audited_sources(args.db, args.game_id, args.tracks_dir)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(value, indent=2))
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        app = create_app(args.db, ROOT / "data" / "trajectories", ROOT / "web")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
