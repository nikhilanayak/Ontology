from pathlib import Path

import numpy as np
import pandas as pd

from all22.tracking import BoxTracker, fuse_sources


def test_box_tracker_preserves_nearby_tracks():
    tracker = BoxTracker()
    first = tracker.update(np.array([[10, 10, 20, 30], [100, 10, 110, 30]]), (100, 200))
    second = tracker.update(np.array([[12, 10, 22, 30], [98, 10, 108, 30]]), (100, 200))
    assert first == second


def test_fuse_sources_averages_nearby_field_positions(tmp_path: Path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    output = tmp_path / "fused.parquet"
    pd.DataFrame([{"relative_frame": 0, "track_id": "s:1", "x": 10.0, "y": 20.0}]).to_parquet(left)
    pd.DataFrame([{"relative_frame": 0, "track_id": "e:8", "x": 12.0, "y": 20.0}]).to_parquet(right)
    assert fuse_sources([left, right], output) == 1
    result = pd.read_parquet(output)
    assert result.iloc[0].x == 11.0
    assert result.iloc[0].track_id == "s:1"
