from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol

import cv2
import pandas as pd


@dataclass(frozen=True)
class PlayerPrediction:
    box: tuple[float, float, float, float]
    confidence: float
    team: Optional[str] = None
    jersey_number: Optional[int] = None


class PlayerDetector(Protocol):
    model_version: str

    def predict(self, frame) -> List[PlayerPrediction]: ...


class PairedVideoTrackingDataset:
    """Streams film frames with BDB field-coordinate answer keys.

    Alignment manifests are deliberately explicit: training cannot silently infer
    a snap offset or accept an unverified game/play pairing.
    """

    def __init__(self, manifest: Path, sample_hz: float = 10.0):
        self.rows = pd.read_json(manifest, lines=True)
        required = {"game_id", "play_id", "angle", "video_path", "snap_s", "labels_path"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"Alignment manifest is missing: {', '.join(sorted(missing))}")
        self.sample_hz = sample_hz

    def examples(self) -> Iterator[Dict]:
        for row in self.rows.itertuples(index=False):
            labels = pd.read_parquet(row.labels_path)
            capture = cv2.VideoCapture(str(row.video_path))
            if not capture.isOpened():
                raise ValueError(f"Could not open {row.video_path}")
            try:
                for frame_id, group in labels.groupby("frame_id", sort=True):
                    t = float(group.t.iloc[0])
                    timestamp = float(row.snap_s) + t
                    if timestamp < 0:
                        continue
                    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Decode failed for {row.game_id}/{row.play_id} at {timestamp:.3f}s")
                    yield {
                        "game_id": str(row.game_id), "play_id": str(row.play_id), "angle": str(row.angle),
                        "timestamp": timestamp, "frame": frame, "answers": group.reset_index(drop=True),
                    }
            finally:
                capture.release()


def write_alignment_manifest(records: List[Dict], path: Path) -> None:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Refusing to write an empty alignment manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(path, orient="records", lines=True)
