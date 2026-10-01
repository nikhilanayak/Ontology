from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .video import probe


ALLOWED_HOST_SUFFIX = ".lura.live"
SAFE_GAME_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,119}$")


def validate_manifest_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ValueError("Manifest URL must use HTTPS")
    if hostname != "lura.live" and not hostname.endswith(ALLOWED_HOST_SUFFIX):
        raise ValueError("Manifest URL host is not an allowed Lura CDN")
    if not parsed.path.lower().endswith(".m3u8"):
        raise ValueError("Manifest URL must identify an m3u8 playlist")
    if re.search(r"(?:^|/)(?:subs?|subtitles?)[^/]*\.m3u8$", parsed.path, re.I):
        raise ValueError("Subtitle playlists are not valid video inputs")
    return urllib.parse.urlunsplit(parsed)


def sanitized_manifest(value: str) -> dict[str, str]:
    parsed = urllib.parse.urlsplit(validate_manifest_url(value))
    fingerprint = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return {"host": parsed.hostname or "", "path": parsed.path, "fingerprint": fingerprint}


def _playlist_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Ontology/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def select_best_variant(url: str) -> str:
    """Return the highest-bandwidth video child, or the media playlist itself."""
    url = validate_manifest_url(url)
    text = _playlist_text(url)
    lines = [line.strip() for line in text.splitlines()]
    variants: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF:"):
            continue
        attributes = line.split(":", 1)[1]
        bandwidth = re.search(r"(?:AVERAGE-)?BANDWIDTH=(\d+)", attributes)
        resolution = re.search(r"RESOLUTION=(\d+)x(\d+)", attributes)
        child = next((candidate for candidate in lines[index + 1:] if candidate and not candidate.startswith("#")), None)
        if child:
            pixels = int(resolution.group(1)) * int(resolution.group(2)) if resolution else 0
            variants.append((pixels, int(bandwidth.group(1)) if bandwidth else 0,
                             urllib.parse.urljoin(url, child)))
    if not variants:
        return url
    return validate_manifest_url(max(variants, key=lambda item: (item[0], item[1]))[2])


def _redact(text: str) -> str:
    return re.sub(r"(https://[^\s?'\"]+)(?:\?[^\s'\"]+)?", r"\1?[redacted]", text)


def _run_ffmpeg(manifest: str, partial: Path) -> None:
    process = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning", "-y", "-i", manifest,
         "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-f", "matroska", str(partial)],
        text=True, capture_output=True, check=False,
    )
    if process.returncode:
        detail = _redact(process.stderr.strip())[-1500:]
        raise RuntimeError(f"ffmpeg exited with {process.returncode}: {detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receive_job(job: dict[str, Any], output_root: Path, metadata_root: Path | None = None) -> dict[str, Any]:
    game_id = str(job.get("game_id", ""))
    if not SAFE_GAME_ID.fullmatch(game_id):
        raise ValueError("game_id must be a lowercase NFL game slug")
    manifests = job.get("manifests")
    if not isinstance(manifests, list) or not manifests or len(manifests) > 5:
        raise ValueError("Job must contain between one and five manifests")
    urls = [validate_manifest_url(str(value)) for value in manifests]
    expected_duration = float(job.get("expected_duration_s", 0))
    if expected_duration < 3000:
        raise ValueError("Expected All-22 duration must be at least 3000 seconds")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / f"{game_id}.mkv"
    partial = output_root / f"{game_id}.mkv.partial"
    if final.exists():
        info = probe(final)
        if abs(info.duration - expected_duration) <= 120:
            return {"status": "reused", "game_id": game_id, "path": str(final),
                    "duration_s": info.duration, "width": info.width, "height": info.height}
        raise ValueError(f"Existing output has an unexpected duration: {final}")

    errors: list[str] = []
    chosen: str | None = None
    info = None
    partial.unlink(missing_ok=True)
    for candidate in urls:
        try:
            chosen = select_best_variant(candidate)
            _run_ffmpeg(chosen, partial)
            info = probe(partial)
            if info.duration < 3000 or abs(info.duration - expected_duration) > 120:
                raise ValueError(f"Downloaded duration {info.duration:.1f}s did not match {expected_duration:.1f}s")
            os.replace(partial, final)
            break
        except Exception as error:
            partial.unlink(missing_ok=True)
            errors.append(_redact(str(error)))
            chosen = None
    if chosen is None or info is None or not final.exists():
        raise RuntimeError("No candidate produced valid All-22 video: " + " | ".join(errors))

    result = {
        "status": "downloaded", "game_id": game_id, "path": str(final),
        "duration_s": info.duration, "width": info.width, "height": info.height,
        "fps": info.fps, "bytes": final.stat().st_size, "sha256": _sha256(final),
        "manifest": sanitized_manifest(chosen),
    }
    metadata_root = (metadata_root or output_root.parent / "data" / "downloads").resolve()
    metadata_root.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_root / f"{game_id}.json"
    temporary = metadata_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    return result


def read_job(stream=sys.stdin) -> dict[str, Any]:
    payload = stream.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError("Download job exceeded 1 MB")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Download job must be a JSON object")
    return value
