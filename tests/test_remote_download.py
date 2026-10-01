from pathlib import Path

import pytest

from all22 import remote_download
from all22.video import VideoInfo


def test_manifest_validation_and_redaction():
    url = "https://dcs-vod.mp.lura.live/vod/p/master.m3u8?Signature=secret"
    assert remote_download.sanitized_manifest(url)["path"] == "/vod/p/master.m3u8"
    assert "secret" not in str(remote_download.sanitized_manifest(url))
    with pytest.raises(ValueError):
        remote_download.validate_manifest_url("https://example.com/master.m3u8")
    with pytest.raises(ValueError):
        remote_download.validate_manifest_url("https://dcs-vod.mp.lura.live/vod/p/subs.m3u8")


def test_best_variant_prefers_resolution_then_bandwidth(monkeypatch):
    monkeypatch.setattr(remote_download, "_playlist_text", lambda _: """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=640x360
low/prog.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=4000000,RESOLUTION=1920x1080
best/prog.m3u8
""")
    result = remote_download.select_best_variant("https://cdn.lura.live/vod/master.m3u8?token=x")
    assert result == "https://cdn.lura.live/vod/best/prog.m3u8"


def test_receive_job_is_atomic_and_writes_sanitized_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(remote_download, "select_best_variant", lambda value: value)
    monkeypatch.setattr(remote_download, "_run_ffmpeg", lambda manifest, output: output.write_bytes(b"video"))
    monkeypatch.setattr(remote_download, "probe", lambda path: VideoInfo(path, 3600, 1920, 1080, 30))
    url = "https://cdn.lura.live/vod/prog.m3u8?Signature=secret"
    result = remote_download.receive_job(
        {"game_id": "buf-at-lar-2022-reg-1", "expected_duration_s": 3600, "manifests": [url]},
        tmp_path / "downloads", tmp_path / "metadata",
    )
    assert result["status"] == "downloaded"
    assert (tmp_path / "downloads" / "buf-at-lar-2022-reg-1.mkv").read_bytes() == b"video"
    metadata = (tmp_path / "metadata" / "buf-at-lar-2022-reg-1.json").read_text()
    assert "secret" not in metadata
    assert not list(tmp_path.rglob("*.partial"))
