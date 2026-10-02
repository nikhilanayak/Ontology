import numpy as np
import pandas as pd

from all22.supervision import identity_metrics


def test_identity_metrics_reports_track_switches_and_purity():
    tracks = pd.DataFrame([
        {"video_timestamp": 0.0, "track_id": "a", "field_x": 10, "field_y": 10},
        {"video_timestamp": 0.0, "track_id": "b", "field_x": 20, "field_y": 20},
        {"video_timestamp": 0.1, "track_id": "c", "field_x": 11, "field_y": 10},
        {"video_timestamp": 0.1, "track_id": "b", "field_x": 21, "field_y": 20},
    ])
    truth = pd.DataFrame([
        {"frame_id": 10, "nfl_id": 1, "x": 10, "y": 10},
        {"frame_id": 10, "nfl_id": 2, "x": 20, "y": 20},
        {"frame_id": 11, "nfl_id": 1, "x": 11, "y": 10},
        {"frame_id": 11, "nfl_id": 2, "x": 21, "y": 20},
    ])
    result = identity_metrics(tracks, truth, 0.0, 10, 0.0, False, False, 0.0, np.eye(3).tolist())
    assert result["matched_assignments"] == 4
    assert result["track_purity"] == 1.0
    assert result["id_switches"] == 1
    assert result["mean_fragments_per_player"] == 1.5
