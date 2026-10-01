from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import bdb
from .db import connect, transaction
from .models import PlayByPlay


@dataclass(frozen=True)
class ActionUnit:
    pair_id: str
    action_id: str
    duration: float
    confidence: float
    category: str = "unknown"
    speed_profile: tuple[float, ...] = ()
    formation: tuple[float, ...] = ()


@dataclass(frozen=True)
class AlignmentOperation:
    operation: str
    play: Optional[PlayByPlay]
    unit: Optional[ActionUnit]
    score: float
    status: str


_UNFILMED = re.compile(r"\b(timeout|end of (?:quarter|game)|two.minute warning|coin toss)\b", re.I)


def is_filmed_event(play: PlayByPlay) -> bool:
    if _UNFILMED.search(play.description):
        return False
    if play.play_type in {"run", "pass", "punt", "kickoff", "field_goal", "extra_point",
                          "qb_kneel", "qb_spike", "no_play", "penalty"}:
        return True
    # PFR collapses kicks and many pre-snap penalties into "excluded".
    return bool(re.search(r"\b(punt|kickoff|field goal|extra point|penalty|false start|offsides|no play)\b",
                          play.description, re.I))


def _category(play: PlayByPlay) -> str:
    text = play.description.lower()
    if "punt" in text:
        return "punt"
    if "kickoff" in text:
        return "kickoff"
    if "field goal" in text or "extra point" in text:
        return "place_kick"
    if "false start" in text or "no play" in text:
        return "no_play"
    return play.play_type if play.play_type in {"run", "pass"} else "unknown"


def _shape_distance(first: Sequence[float], second: Sequence[float], normalize: bool = False) -> float:
    if not first or not second:
        return 0.0
    count = min(11, len(first), len(second))
    quantiles = np.linspace(0, 1, count)
    a = np.quantile(np.asarray(first, float), quantiles)
    b = np.quantile(np.asarray(second, float), quantiles)
    if normalize:
        a /= max(float(np.max(np.abs(a))), 1.0)
        b /= max(float(np.max(np.abs(b))), 1.0)
    return float(np.mean(np.abs(a - b)))


def match_cost(play: PlayByPlay, unit: ActionUnit,
               expected: Optional[bdb.BDBPlaySignature] = None) -> float:
    category = _category(play)
    if expected and expected.category != "unknown":
        category = expected.category
    category_cost = 0.0 if unit.category == "unknown" or category == "unknown" else (
        0.0 if unit.category == category else 3.0)
    expected_duration = expected.duration if expected else {
        "run": 6.0, "pass": 7.5, "punt": 8.0, "kickoff": 9.0,
        "place_kick": 5.0, "no_play": 2.0,
    }.get(category, 6.5)
    duration_cost = min(2.5, abs(unit.duration - expected_duration) / max(expected_duration, 2))
    confidence_cost = max(0.0, .7 - unit.confidence)
    if expected:
        profile_cost = .55 * _shape_distance(unit.speed_profile, expected.speed_profile, normalize=True)
        formation_cost = .035 * _shape_distance(unit.formation, expected.formation)
    else:
        profile_cost = formation_cost = 0.0
    return category_cost + duration_cost + confidence_cost + profile_cost + formation_cost


def align_actions_to_pbp(plays: Sequence[PlayByPlay], units: Sequence[ActionUnit],
                         skip_play_cost: float = 1.1, skip_action_cost: float = 1.1,
                         maximum_match_cost: float = 1.75,
                         expected_signatures: Optional[dict[int, bdb.BDBPlaySignature]] = None,
                         ) -> list[AlignmentOperation]:
    """Monotonic alignment with explicit gaps and a hard incompatible-match gate."""
    events = [play for play in plays if is_filmed_event(play)]
    n, m = len(events), len(units)
    costs = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    costs[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            value = costs[i][j]
            if value == inf:
                continue
            if i < n and value + skip_play_cost < costs[i + 1][j]:
                costs[i + 1][j] = value + skip_play_cost
                back[i + 1][j] = (i, j, "missing_film", skip_play_cost)
            if j < m and value + skip_action_cost < costs[i][j + 1]:
                costs[i][j + 1] = value + skip_action_cost
                back[i][j + 1] = (i, j, "extra_film", skip_action_cost)
            if i < n and j < m:
                expected = (expected_signatures or {}).get(events[i].ordinal)
                candidate = match_cost(events[i], units[j], expected)
                if candidate <= maximum_match_cost and value + candidate < costs[i + 1][j + 1]:
                    costs[i + 1][j + 1] = value + candidate
                    back[i + 1][j + 1] = (i, j, "match", candidate)
    output = []
    i, j = n, m
    while i or j:
        previous = back[i][j]
        if previous is None:
            raise ValueError("No action/PBP alignment path exists")
        pi, pj, operation, score = previous
        play = events[pi] if operation in {"match", "missing_film"} else None
        unit = units[pj] if operation in {"match", "extra_film"} else None
        if operation == "match":
            # Trajectories give us an excellent ordering cue, but without ball or
            # event classification this must remain reviewable rather than truth.
            status = "candidate" if score <= .65 else "review"
        else:
            status = "review"
        output.append(AlignmentOperation(operation, play, unit, score, status))
        i, j = pi, pj
    return list(reversed(output))


def align_game_actions(db_path: Path, game_id: str) -> list[dict]:
    with connect(db_path) as connection:
        play_rows = connection.execute("SELECT * FROM pbp_plays WHERE game_id=? ORDER BY ordinal",
                                       (game_id,)).fetchall()
        unit_rows = connection.execute(
            """SELECT ap.pair_id,ap.primary_action_id,ap.score,ap.status,ap.diagnostics_json,
                      aw.snap_s,aw.dead_s,aw.confidence,c.start_s
               FROM action_pairs ap JOIN action_windows aw ON aw.action_id=ap.primary_action_id
               JOIN clips c ON c.clip_id=aw.clip_id WHERE c.game_id=?
               ORDER BY c.start_s,aw.action_order""", (game_id,),
        ).fetchall()
        external = connection.execute(
            "SELECT external_id FROM game_external_ids WHERE game_id=? AND provider='bdb'", (game_id,),
        ).fetchone()
    plays = [PlayByPlay(row["game_id"], row["ordinal"], row["quarter"], row["clock"],
                        row["possession"], row["down_no"], row["distance"], row["yard_line"],
                        row["description"], row["play_type"], bool(row["eligible"]), row["source_row_id"])
             for row in play_rows]
    units = []
    for row in unit_rows:
        diagnostics = json.loads(row["diagnostics_json"] or "{}")
        signature = diagnostics.get("primary_signature") or {}
        units.append(ActionUnit(
            row["pair_id"], row["primary_action_id"],
            float(row["dead_s"] - row["snap_s"]), float(row["confidence"]),
            speed_profile=tuple(signature.get("speed_profile") or ()),
            formation=tuple(signature.get("formation") or ()),
        ))
    expected_signatures = {}
    if external:
        answers = bdb.play_signatures(db_path, external["external_id"])
        for play in plays:
            match = re.search(r":(\d+)$", play.source_row_id or "")
            if match and match.group(1) in answers:
                expected_signatures[play.ordinal] = answers[match.group(1)]
    operations = align_actions_to_pbp(plays, units, expected_signatures=expected_signatures)
    with transaction(db_path) as connection:
        connection.execute("DELETE FROM action_alignment_operations WHERE game_id=?", (game_id,))
        connection.execute("DELETE FROM pbp_action_alignments WHERE game_id=?", (game_id,))
        for sequence, item in enumerate(operations, 1):
            ordinal = item.play.ordinal if item.play else None
            pair_id = item.unit.pair_id if item.unit else None
            expected = expected_signatures.get(item.play.ordinal) if item.play else None
            diagnostics = {"method": "monotonic_trajectory_bdb_v2" if expected else "monotonic_trajectory_v1",
                           "bdb_play_id": expected.play_id if expected else None,
                           "bdb_duration_s": expected.duration if expected else None}
            connection.execute(
                """INSERT INTO action_alignment_operations(game_id,sequence_no,operation,pbp_ordinal,
                   pair_id,score,status,diagnostics_json) VALUES(?,?,?,?,?,?,?,?)""",
                (game_id, sequence, item.operation, ordinal, pair_id, item.score, item.status,
                 json.dumps(diagnostics)),
            )
            if item.play:
                connection.execute(
                    """INSERT INTO pbp_action_alignments(game_id,pbp_ordinal,pair_id,action_id,score,status,
                       operation,diagnostics_json) VALUES(?,?,?,?,?,?,?,?)""",
                    (game_id, ordinal, pair_id, item.unit.action_id if item.unit else None,
                     item.score, item.status, item.operation, json.dumps(diagnostics)),
                )
    return [{"sequence": index, "operation": item.operation,
             "pbp_ordinal": item.play.ordinal if item.play else None,
             "pair_id": item.unit.pair_id if item.unit else None,
             "score": item.score, "status": item.status}
            for index, item in enumerate(operations, 1)]
