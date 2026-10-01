from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .db import connect, transaction


PILOT_GAMES = (
    {"away": "BUF", "home": "LA", "role": "development", "environment": "indoor-night"},
    {"away": "PIT", "home": "CIN", "role": "development", "environment": "outdoor-day"},
    {"away": "TB", "home": "DAL", "role": "held-out", "environment": "indoor-night"},
)


def select_pilot_games(games_csv: Path, plays_csv: Path, output: Path) -> list[dict]:
    games = pd.read_csv(games_csv, dtype={"gameId": str})
    plays = pd.read_csv(plays_csv, dtype={"gameId": str, "playId": str})
    required_games = {"gameId", "season", "week", "homeTeamAbbr", "visitorTeamAbbr"}
    required_plays = {"gameId", "playId"}
    if not required_games.issubset(games.columns) or not required_plays.issubset(plays.columns):
        raise ValueError("BDB games.csv or plays.csv is missing required identifiers")
    games = games[(games["season"] == 2022) & (games["week"].between(1, 9))]
    records = []
    for desired in PILOT_GAMES:
        # BDB uses LA while nflverse/NFL slugs commonly use LAR.
        matches = games[(games.homeTeamAbbr == desired["home"]) &
                        (games.visitorTeamAbbr == desired["away"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one 2022 {desired['away']} at {desired['home']} BDB game; found {len(matches)}")
        row = matches.iloc[0]
        game_id = str(row.gameId)
        records.append({**desired, "bdb_game_id": game_id, "week": int(row.week),
                        "bdb_play_count": int((plays.gameId == game_id).sum())})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"season": 2022, "games": records}, indent=2) + "\n", encoding="utf-8")
    return records


def set_external_id(db_path: Path, game_id: str, provider: str, external_id: str) -> None:
    if not re.fullmatch(r"[a-z0-9_-]+", provider):
        raise ValueError("Provider must be a lowercase identifier")
    with transaction(db_path) as connection:
        if not connection.execute("SELECT 1 FROM games WHERE game_id=?", (game_id,)).fetchone():
            raise ValueError(f"Game is not registered: {game_id}")
        connection.execute(
            """INSERT INTO game_external_ids(game_id,provider,external_id) VALUES(?,?,?)
               ON CONFLICT(game_id,provider) DO UPDATE SET external_id=excluded.external_id""",
            (game_id, provider, external_id),
        )


def _balanced_sample(rows: list, count: int) -> list:
    buckets: dict[tuple[int, str], list] = {}
    for row in rows:
        buckets.setdefault((row["quarter"] or 0, row["play_type"]), []).append(row)
    selected = []
    ordered_keys = sorted(buckets)
    while len(selected) < min(count, len(rows)):
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                # Taking the middle item spreads selection over game time without randomness.
                selected.append(bucket.pop(len(bucket) // 2))
                progressed = True
                if len(selected) == min(count, len(rows)):
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda row: row["ordinal"])


def create_audit_sample(db_path: Path, game_id: str, count: int = 40) -> list[str]:
    if count < 1:
        raise ValueError("Audit count must be positive")
    with connect(db_path) as connection:
        rows = connection.execute(
            """SELECT p.ordinal,p.quarter,p.play_type,a.play_id
               FROM pbp_plays p JOIN play_alignments a
                 ON a.game_id=p.game_id AND a.pbp_ordinal=p.ordinal
               WHERE p.game_id=? AND p.eligible=1 AND a.status!='rejected'
               ORDER BY p.ordinal""", (game_id,),
        ).fetchall()
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} aligned eligible plays are available; requested {count}")
    selected = [row["play_id"] for row in _balanced_sample(list(rows), count)]
    with transaction(db_path) as connection:
        connection.execute(
            "DELETE FROM alignment_audits WHERE play_id IN (SELECT play_id FROM play_alignments WHERE game_id=?)",
            (game_id,),
        )
        connection.executemany("INSERT INTO alignment_audits(play_id,selected) VALUES(?,1)",
                               [(play_id,) for play_id in selected])
    return selected


def audit_summary(db_path: Path, game_id: str) -> dict:
    with connect(db_path) as connection:
        rows = connection.execute(
            """SELECT aa.* FROM alignment_audits aa JOIN play_alignments a ON a.play_id=aa.play_id
               WHERE a.game_id=? AND aa.selected=1""", (game_id,),
        ).fetchall()
    reviewed = [row for row in rows if row["mapping_correct"] is not None]
    def rate(field: str) -> Optional[float]:
        values = [row[field] for row in reviewed if row[field] is not None]
        return sum(values) / len(values) if values else None
    return {"selected": len(rows), "reviewed": len(reviewed), "mapping_accuracy": rate("mapping_correct"),
            "source_accuracy": rate("sources_correct"), "timing_accuracy": rate("timing_correct")}


def _play_number(source_row_id: str | None) -> str | None:
    if not source_row_id:
        return None
    match = re.search(r":(\d+)$", source_row_id)
    return match.group(1) if match else None


def evaluate_trajectories(db_path: Path, game_id: str, predictions_dir: Path) -> dict:
    with connect(db_path) as connection:
        external = connection.execute(
            "SELECT external_id FROM game_external_ids WHERE game_id=? AND provider='bdb'", (game_id,),
        ).fetchone()
        plays = connection.execute(
            """SELECT a.play_id,p.source_row_id FROM play_alignments a JOIN pbp_plays p
               ON p.game_id=a.game_id AND p.ordinal=a.pbp_ordinal
               WHERE a.game_id=? AND p.eligible=1""", (game_id,),
        ).fetchall()
        teams = connection.execute("SELECT home_team,away_team FROM games WHERE game_id=?", (game_id,)).fetchone()
    if not external:
        raise ValueError("Set the game's bdb external ID before evaluation")
    team_map = {"home": teams["home_team"], "away": teams["away_team"]} if teams else {}
    all_errors: list[float] = []
    truth_total = 0
    within_three = 0
    evaluated_frames = 0
    evaluated_plays = 0
    with connect(db_path) as connection:
        for play in plays:
            prediction_path = predictions_dir / f"{play['play_id']}.parquet"
            bdb_play_id = _play_number(play["source_row_id"])
            if not prediction_path.exists() or not bdb_play_id:
                continue
            predicted = pd.read_parquet(prediction_path)
            if "frame_id" not in predicted.columns and "relative_frame" in predicted.columns:
                snap = connection.execute(
                    """SELECT MIN(frame_id) AS frame_id FROM bdb_tracking
                       WHERE game_id=? AND play_id=? AND lower(event)='ball_snap'""",
                    (external["external_id"], bdb_play_id),
                ).fetchone()
                if not snap or snap["frame_id"] is None:
                    continue
                predicted["frame_id"] = predicted.relative_frame.astype(int) + int(snap["frame_id"])
            required = {"frame_id", "x", "y"}
            if not required.issubset(predicted.columns):
                raise ValueError(f"{prediction_path} is missing {sorted(required - set(predicted.columns))}")
            truth = pd.read_sql_query(
                """SELECT frame_id,team,x,y FROM bdb_tracking
                   WHERE game_id=? AND play_id=? AND nfl_id IS NOT NULL ORDER BY frame_id""",
                connection, params=(external["external_id"], bdb_play_id),
            )
            if truth.empty:
                continue
            evaluated_plays += 1
            for frame_id, answer_rows in truth.groupby("frame_id"):
                guess_rows = predicted[predicted.frame_id.astype(int) == int(frame_id)]
                truth_total += len(answer_rows)
                if guess_rows.empty:
                    continue
                frame_errors = []
                constrained = "team" in guess_rows.columns and guess_rows.team.notna().any()
                truth_teams = list(answer_rows.team.dropna().unique()) if constrained else [None]
                for team in truth_teams:
                    answers = answer_rows if team is None else answer_rows[answer_rows.team == team]
                    if team is None:
                        guesses = guess_rows
                    else:
                        guesses = guess_rows[guess_rows.team.replace(team_map) == team]
                    if answers.empty or guesses.empty:
                        continue
                    a = answers[["x", "y"]].to_numpy(float)
                    g = guesses[["x", "y"]].to_numpy(float)
                    distances = np.linalg.norm(a[:, None, :] - g[None, :, :], axis=2)
                    answer_index, guess_index = linear_sum_assignment(distances)
                    frame_errors.extend(distances[answer_index, guess_index].tolist())
                if frame_errors:
                    evaluated_frames += 1
                    all_errors.extend(frame_errors)
                    within_three += sum(value <= 3.0 for value in frame_errors)
    if not all_errors:
        raise ValueError("No prediction frames could be matched to BDB answers")
    values = np.asarray(all_errors)
    return {"game_id": game_id, "evaluated_plays": evaluated_plays, "evaluated_frames": evaluated_frames,
            "matched_player_frames": len(all_errors), "truth_player_frames": truth_total,
            "median_error_yards": float(np.median(values)), "p90_error_yards": float(np.percentile(values, 90)),
            "coverage": len(all_errors) / truth_total, "within_3_yards": within_three / truth_total}
