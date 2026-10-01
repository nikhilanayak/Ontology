#!/usr/bin/env python3
"""Download and safely extract the BDB 2024 competition archive.

The Kaggle account must accept the competition rules in the browser first.
Credentials are read from ~/.kaggle/kaggle.json and are never printed.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "bdb-2024"
ARCHIVE = DESTINATION / "nfl-big-data-bowl-2024.zip"
URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/nfl-big-data-bowl-2024"


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    token = base64.b64encode(f"{credentials['username']}:{credentials['key']}".encode()).decode()
    request = urllib.request.Request(URL, headers={"Authorization": f"Basic {token}", "User-Agent": "Ontology/0.1"})
    DESTINATION.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(".zip.partial")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        if error.code in (401, 403):
            raise SystemExit("Kaggle denied the download. Accept the BDB 2024 competition rules, then retry.")
        raise
    os.replace(partial, ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"Downloaded archive failed its CRC check at {bad}")
        safe_extract(archive, DESTINATION)
    print(json.dumps({"archive": str(ARCHIVE), "destination": str(DESTINATION)}))


if __name__ == "__main__":
    main()
