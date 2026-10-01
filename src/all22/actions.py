from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .db import connect, transaction


@dataclass(frozen=True)
class ActionCandidate:
    start_s: float
    snap_s: float
    dead_s: float
    end_s: float
    confidence: float
    boundary_collision: bool = False


@dataclass(frozen=True)
class ActionSignature:
    action_id: str
    clip_id: str
    angle: str
    order: int
    start_s: float
    duration: float
    team_counts: tuple[int, int]
    displacement: float
    speed_profile: tuple[float, ...]
    formation: tuple[float, ...]


@dataclass(frozen=True)
class ProposedPair:
    primary_action_id: str
    alternate_action_id: Optional[str]
    cost: float
    status: str
    diagnostics: dict


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def discover_action_candidates(tracks: pd.DataFrame, clip_start: float, clip_end: float,
                               minimum_active_s: float = 1.2, bridge_s: float = .8) -> list[ActionCandidate]:
    """Find zero or more complete live-action intervals inside one hard-cut shot.

    This deliberately uses only projected player motion. It never invents a play
    when the evidence is flat and never allows an interval to cross the shot.
    """
    required = {"video_timestamp", "track_id", "field_x", "field_y"}
    if tracks.empty or not required.issubset(tracks.columns):
        return []
    frame = tracks.sort_values(["track_id", "video_timestamp"]).copy()
    if "track_reliable" in frame and bool(frame.track_reliable.any()):
        frame = frame[frame.track_reliable].copy()
    if "speed" not in frame or not np.isfinite(pd.to_numeric(frame["speed"], errors="coerce")).any():
        elapsed = frame.groupby("track_id").video_timestamp.diff()
        dx = frame.groupby("track_id").field_x.diff()
        dy = frame.groupby("track_id").field_y.diff()
        frame["speed"] = np.hypot(dx, dy) / elapsed.replace(0, np.nan)
    timeline = frame.groupby("video_timestamp").agg(
        moving=("speed", lambda values: float((pd.to_numeric(values, errors="coerce") > 1.15).mean())),
        median_speed=("speed", "median"), players=("track_id", "nunique"),
    ).reset_index().sort_values("video_timestamp")
    if len(timeline) < 4:
        return []
    times = timeline.video_timestamp.to_numpy(float)
    step = float(np.median(np.diff(times)))
    smooth_n = max(1, int(round(.5 / max(step, .01))))
    activity = timeline.moving.rolling(smooth_n, center=True, min_periods=1).mean().to_numpy()
    median_speed = timeline.median_speed.rolling(smooth_n, center=True, min_periods=1).median().to_numpy()
    # Pre-snap shifts move a handful of players slowly; live action produces a
    # coordinated rise in both participation and median field-space speed.
    active = (activity >= .28) & (median_speed >= 1.8) & (timeline.players.to_numpy() >= 6)
    # Bridge brief pauses caused by occlusion or a tackle pile.
    for left, right in _runs(~active):
        if left and right < len(active) and times[right - 1] - times[left] <= bridge_s:
            active[left:right] = True
    values: list[ActionCandidate] = []
    for left, right in _runs(active):
        snap = float(times[left])
        dead = float(times[right - 1])
        if dead - snap < minimum_active_s:
            continue
        prior = times[max(0, left - max(1, int(round(2.5 / step)))):left]
        following = times[right:min(len(times), right + max(1, int(round(1.5 / step))))]
        start = float(prior[0]) if len(prior) else snap
        end = float(following[-1]) if len(following) else dead
        start, end = max(clip_start, start), min(clip_end, end)
        collision = snap - clip_start < .35 or clip_end - dead < .35
        confidence = min(.99, .45 + .45 * float(np.mean(activity[left:right])))
        if collision:
            confidence *= .6
        values.append(ActionCandidate(start, snap, dead, end, confidence, collision))
    return values


def discover_clip_actions(db_path: Path, clip_id: str, tracks_path: Path) -> list[dict]:
    tracks = pd.read_parquet(tracks_path)
    with connect(db_path) as connection:
        clip = connection.execute("SELECT game_id,start_s,end_s FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    if not clip:
        raise ValueError(f"No camera shot found: {clip_id}")
    candidates = discover_action_candidates(tracks, float(clip["start_s"]), float(clip["end_s"]))
    rows = []
    with transaction(db_path) as connection:
        connection.execute("DELETE FROM action_windows WHERE clip_id=?", (clip_id,))
        for order, item in enumerate(candidates, 1):
            action_id = f"{clip_id}:a{order:02d}"
            status = "review" if item.boundary_collision else "candidate"
            diagnostics = {"boundary_collision": item.boundary_collision, "method": "field_motion_v1"}
            connection.execute(
                """INSERT INTO action_windows(action_id,clip_id,action_order,formation_start_s,snap_s,
                   dead_s,playback_end_s,confidence,status,diagnostics_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (action_id, clip_id, order, item.start_s, item.snap_s, item.dead_s, item.end_s,
                 item.confidence, status, json.dumps(diagnostics)),
            )
            rows.append({"action_id": action_id, **item.__dict__, "status": status})
        connection.execute(
            "INSERT INTO artifacts(game_id,clip_id,kind,path,metadata_json) VALUES(?,?,?,?,?)",
            (clip["game_id"], clip_id, "action_windows", str(tracks_path.resolve()),
             json.dumps({"actions": len(rows), "method": "field_motion_v1"})),
        )
    return rows


def signature_for_action(action: dict, tracks: pd.DataFrame, angle: str) -> ActionSignature:
    window = tracks[(tracks.video_timestamp >= float(action["snap_s"])) &
                    (tracks.video_timestamp <= float(action["dead_s"]))].copy()
    duration = max(.01, float(action["dead_s"]) - float(action["snap_s"]))
    if window.empty:
        return ActionSignature(action["action_id"], action["clip_id"], angle, int(action["action_order"]),
                               float(action["snap_s"]), duration, (0, 0), 0, (), ())
    counts = window.groupby("team").track_id.nunique().sort_values(ascending=False).tolist() if "team" in window else []
    team_counts = tuple((counts + [0, 0])[:2])
    endpoints = window.sort_values("video_timestamp").groupby("track_id").agg(
        x0=("field_x", "first"), x1=("field_x", "last"), y0=("field_y", "first"), y1=("field_y", "last"))
    displacement = float(np.median(np.hypot(endpoints.x1 - endpoints.x0, endpoints.y1 - endpoints.y0)))
    bins = np.linspace(float(action["snap_s"]), float(action["dead_s"]), 9)
    window["bin"] = np.clip(np.digitize(window.video_timestamp, bins) - 1, 0, 7)
    speeds = window.groupby("bin").speed.median().reindex(range(8)).interpolate(limit_direction="both").fillna(0)
    snap_rows = window[window.video_timestamp <= float(action["snap_s"]) + .5]
    points = snap_rows.groupby("track_id")[["field_x", "field_y"]].median().to_numpy()
    if len(points):
        centered = points - np.median(points, axis=0)
        formation = tuple(float(item) for item in np.round(
            np.sort(np.hypot(centered[:, 0], centered[:, 1]))[:22], 2))
    else:
        formation = ()
    return ActionSignature(action["action_id"], action["clip_id"], angle, int(action["action_order"]),
                           float(action["snap_s"]), duration, team_counts, displacement,
                           tuple(float(item) for item in speeds), formation)


def _profile_distance(first: Sequence[float], second: Sequence[float]) -> float:
    if not first or not second:
        return 2.0
    a, b = np.asarray(first, float), np.asarray(second, float)
    a /= max(np.max(a), 1)
    b /= max(np.max(b), 1)
    return float(np.mean(np.abs(a - b)))


def signature_cost(first: ActionSignature, second: ActionSignature) -> float:
    duration = abs(first.duration - second.duration) / max(first.duration, second.duration, 1)
    displacement = abs(first.displacement - second.displacement) / max(first.displacement, second.displacement, 2)
    counts = sum(abs(a - b) for a, b in zip(first.team_counts, second.team_counts)) / 22
    profile = _profile_distance(first.speed_profile, second.speed_profile)
    angle_penalty = 0 if first.angle != second.angle and "unknown" not in (first.angle, second.angle) else .35
    return 1.6 * duration + displacement + counts + profile + angle_penalty


def pair_ordered_actions(signatures: Sequence[ActionSignature], maximum_cost: float = 1.55,
                         minimum_margin: float = .20, lookahead: int = 4) -> list[ProposedPair]:
    """Pair repeated presentations conservatively; ambiguous actions remain single."""
    values = list(signatures)
    used: set[int] = set()
    output: list[ProposedPair] = []
    for index, first in enumerate(values):
        if index in used:
            continue
        candidates = []
        for other in range(index + 1, min(len(values), index + lookahead + 1)):
            second = values[other]
            if other in used or first.clip_id == second.clip_id:
                continue
            candidates.append((signature_cost(first, second), other))
        candidates.sort()
        if candidates and candidates[0][0] <= maximum_cost:
            best, other = candidates[0]
            margin = (candidates[1][0] - best) if len(candidates) > 1 else 99.0
            if margin >= minimum_margin:
                used.update((index, other))
                output.append(ProposedPair(first.action_id, values[other].action_id, best, "paired",
                                           {"margin": margin, "method": "trajectory_signature_v1"}))
                continue
            status = "ambiguous"
            diagnostics = {"best_cost": best, "margin": margin, "method": "trajectory_signature_v1"}
        else:
            status, diagnostics = "single", {"method": "trajectory_signature_v1"}
        used.add(index)
        output.append(ProposedPair(first.action_id, None, candidates[0][0] if candidates else maximum_cost,
                                   status, diagnostics))
    return output


def pair_game_actions(db_path: Path, game_id: str, tracks_dir: Path) -> list[dict]:
    with connect(db_path) as connection:
        rows = connection.execute(
            """SELECT a.*,c.angle,c.start_s FROM action_windows a JOIN clips c ON c.clip_id=a.clip_id
               WHERE c.game_id=? ORDER BY c.start_s,a.action_order""", (game_id,)).fetchall()
    signatures = []
    for row in rows:
        path = tracks_dir / f"{row['clip_id']}.parquet"
        if path.exists():
            signatures.append(signature_for_action(dict(row), pd.read_parquet(path), row["angle"]))
    proposals = pair_ordered_actions(signatures)
    signature_by_id = {value.action_id: value for value in signatures}

    def signature_payload(action_id: Optional[str]) -> Optional[dict]:
        if not action_id or action_id not in signature_by_id:
            return None
        value = signature_by_id[action_id]
        return {"duration": float(value.duration),
                "team_counts": [int(item) for item in value.team_counts],
                "displacement": float(value.displacement),
                "speed_profile": [float(item) for item in value.speed_profile],
                "formation": [float(item) for item in value.formation], "angle": value.angle}

    with transaction(db_path) as connection:
        old = connection.execute(
            """SELECT pair_id FROM action_pairs WHERE primary_action_id IN
               (SELECT action_id FROM action_windows a JOIN clips c ON c.clip_id=a.clip_id WHERE c.game_id=?)""",
            (game_id,),
        ).fetchall()
        connection.executemany("DELETE FROM action_pairs WHERE pair_id=?", [(row["pair_id"],) for row in old])
        for index, pair in enumerate(proposals, 1):
            pair_id = f"{game_id}:pair:{index:04d}"
            diagnostics = {**pair.diagnostics,
                           "primary_signature": signature_payload(pair.primary_action_id),
                           "alternate_signature": signature_payload(pair.alternate_action_id)}
            connection.execute(
                """INSERT INTO action_pairs(pair_id,primary_action_id,alternate_action_id,score,status,
                   synchronization_json,diagnostics_json) VALUES(?,?,?,?,?,?,?)""",
                (pair_id, pair.primary_action_id, pair.alternate_action_id, pair.cost, pair.status, "{}",
                 json.dumps(diagnostics)),
            )
    return [{"pair_id": f"{game_id}:pair:{index:04d}", **pair.__dict__}
            for index, pair in enumerate(proposals, 1)]
