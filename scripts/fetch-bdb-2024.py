#!/usr/bin/env python3
"""Download and safely extract an NFL Big Data Bowl competition archive.

The Kaggle account must accept the competition rules in the browser first.
Credentials are read from ~/.kaggle/kaggle.json and are never printed.
"""

from __future__ import annotations

import base64
import argparse
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPETITION = "nfl-big-data-bowl-2024"


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if not args.competition.startswith("nfl-big-data-bowl-") or not args.competition.replace("-", "").isalnum():
        raise SystemExit("Competition must be an NFL Big Data Bowl slug")
    destination = (args.destination or ROOT / "data" / "raw" / args.competition).resolve()
    archive_path = destination / f"{args.competition}.zip"
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{args.competition}"
    credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    token = base64.b64encode(f"{credentials['username']}:{credentials['key']}".encode()).decode()
    request = urllib.request.Request(url, headers={"Authorization": f"Basic {token}", "User-Agent": "Ontology/0.1"})
    destination.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_suffix(".zip.partial")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        if error.code in (401, 403):
            raise SystemExit("Kaggle denied the download. Accept the BDB 2024 competition rules, then retry.")
        raise
    os.replace(partial, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"Downloaded archive failed its CRC check at {bad}")
        safe_extract(archive, destination)
    print(json.dumps({"competition": args.competition, "archive": str(archive_path),
                      "destination": str(destination)}))


if __name__ == "__main__":
    main()
