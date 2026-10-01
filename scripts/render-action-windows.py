#!/usr/bin/env python3
"""Render start/middle/end frames for visual QA of scan-actions output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("scan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    windows = json.loads(args.scan.read_text(encoding="utf-8"))["windows"][: args.limit]
    capture = cv2.VideoCapture(str(args.video))
    cells = []
    try:
        for number, window in enumerate(windows, start=1):
            times = (window["start_s"], (window["start_s"] + window["end_s"]) / 2, window["end_s"])
            row = []
            for label, timestamp in zip(("START", "MID", "END"), times):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read {args.video} at {timestamp:.2f}s")
                frame = cv2.resize(frame, (384, 216))
                cv2.rectangle(frame, (0, 0), (384, 30), (0, 0, 0), -1)
                cv2.putText(frame, f"{number:02d} {label} {timestamp:.1f}s", (8, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                row.append(frame)
            cells.append(np.hstack(row))
    finally:
        capture.release()
    if not cells:
        raise SystemExit("No action windows found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.vstack(cells)):
        raise RuntimeError(f"Could not write {args.output}")


if __name__ == "__main__":
    main()
