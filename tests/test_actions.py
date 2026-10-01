from pathlib import Path

import numpy as np
import pandas as pd

from all22.actions import ActionSignature, discover_action_candidates, pair_ordered_actions


def _tracks(pulses: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    times = np.arange(0, 30, .2)
    for track in range(12):
        x = float(track)
        for timestamp in times:
            speed = 4.0 if any(start <= timestamp <= end for start, end in pulses) else .05
            x += speed * .2
            rows.append({"video_timestamp": timestamp, "track_id": f"t{track}",
                         "field_x": x, "field_y": float(track * 2), "speed": speed})
    return pd.DataFrame(rows)


def test_multiple_actions_can_be_discovered_inside_one_shot():
    values = discover_action_candidates(_tracks([(5, 9), (16, 21)]), 0, 30)
    assert len(values) == 2
    assert 4.5 <= values[0].snap_s <= 5.5
    assert 15.5 <= values[1].snap_s <= 16.5
    assert all(0 <= value.start_s < value.snap_s < value.dead_s < value.end_s <= 30 for value in values)


def test_quiet_shot_produces_no_action():
    assert discover_action_candidates(_tracks([]), 0, 30) == []


def _signature(name: str, clip: str, angle: str, duration: float, displacement: float) -> ActionSignature:
    return ActionSignature(name, clip, angle, 1, 0, duration, (11, 11), displacement,
                           (0, .3, .8, 1, .8, .5, .2, 0), (1, 2, 3))


def test_pairing_uses_trajectory_similarity_and_preserves_singles():
    values = [
        _signature("a", "c1", "sideline", 6, 5),
        _signature("b", "c2", "endzone", 6.1, 5.1),
        _signature("c", "c3", "sideline", 14, 20),
    ]
    pairs = pair_ordered_actions(values)
    assert pairs[0].primary_action_id == "a"
    assert pairs[0].alternate_action_id == "b"
    assert pairs[0].status == "paired"
    assert pairs[1].primary_action_id == "c"
    assert pairs[1].status == "single"
