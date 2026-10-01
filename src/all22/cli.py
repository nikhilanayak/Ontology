from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import alignment, bdb, db, nflverse, pfr, pilot, remote_download, supervision, tracking, video
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

    project = commands.add_parser("project-source")
    project.add_argument("--play-id", required=True)
    project.add_argument("--source-order", type=int, required=True)
    project.add_argument("--detections", type=Path, required=True)
    project.add_argument("--landmarks", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--video-anchor-s", type=float)
    project.add_argument("--bdb-anchor-frame", type=int)

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
    elif args.command == "project-source":
        result = tracking.project_source(args.db, args.play_id, args.source_order,
                                         args.detections, args.landmarks, args.output,
                                         args.video_anchor_s, args.bdb_anchor_frame)
        print(json.dumps({**result, "output": str(args.output)}, indent=2))
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
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        app = create_app(args.db, ROOT / "data" / "trajectories", ROOT / "web")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
