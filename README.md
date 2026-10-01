# NFL All-22 reconstruction

This repository has two connected systems:

1. An authenticated Chrome/CDP downloader for NFL All-22 film.
2. A BDB-supervised computer-vision pipeline that aligns film to PFR play-by-play and reconstructs player trajectories in field coordinates.

The reconstruction pipeline is deliberately fail-closed. A play is not published unless clip pairing, PFR alignment, field calibration, 22-player identity assignment, track coverage, and cross-angle agreement all pass their thresholds.

## Install

Node dependencies support film and PFR acquisition:

```bash
npm install
```

Python dependencies live in a project-local environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Install the optional model-training stack only once paired BDB/video examples are ready:

```bash
.venv/bin/pip install -e '.[training]'
```

## Download verified All-22 film

Log in once, then run the CDP collector. `--limit` stops both discovery and downloading after the requested number of games.

```bash
npm run login
npm run batch:cdp -- 2025 --limit 1
```

The collector requires the visible All-22 control, a 50+ minute player duration, a matching HLS playlist duration, and a valid video stream before saving.

## Initialize and register a game

```bash
.venv/bin/all22 init-db
.venv/bin/all22 register-game \
  --game-id cowboys-at-eagles-2025-reg-1 \
  --season 2025 --week 1 --home-team PHI --away-team DAL \
  downloads/cowboys-at-eagles-2025-reg-1.mkv
```

## Import play-by-play

The primary source is the structured nflverse season release. It avoids PFR's
Cloudflare page and provides explicit play types instead of requiring prose
classification.

```bash
mkdir -p data/nflverse
curl -L --fail -o data/nflverse/play_by_play_2025.parquet \
  https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet

.venv/bin/all22 import-nflverse \
  --game-id cowboys-at-eagles-2025-reg-1 \
  data/nflverse/play_by_play_2025.parquet
```

The importer derives nflverse's game ID from the registered season, week, away
team, and home team. `--source-game-id 2025_01_DAL_PHI` can override it.

### Optional PFR fallback

PFR rejects plain HTTP clients, so acquisition uses the existing real Chrome profile and caches the page locally.

```bash
npm run fetch:pfr -- \
  cowboys-at-eagles-2025-reg-1 \
  'https://www.pro-football-reference.com/boxscores/BOX_SCORE_ID.htm'

.venv/bin/all22 import-pfr \
  --game-id cowboys-at-eagles-2025-reg-1 \
  data/pfr/cowboys-at-eagles-2025-reg-1.html
```

The parser supports PFR's normal and HTML-comment-wrapped play-by-play tables. Only scrimmage runs and dropbacks are eligible in the first milestone.

## Import Big Data Bowl answers

The importer accepts both snake_case and the camelCase column names used by different BDB releases.

```bash
.venv/bin/all22 import-bdb path/to/tracking_week_1.csv path/to/tracking_week_2.csv
```

Create a rigidly synchronized supervision pair after verifying the snap timestamp in matching All-22 film:

```bash
.venv/bin/all22 create-supervision-pair \
  --game-id 2023090700 --play-id 101 \
  --angle sideline --video-snap 123.40 \
  downloads/matching-bdb-game.mkv
```

This writes normalized 10 Hz Parquet answers and updates `data/supervision.jsonl`. A BDB play without a `ball_snap` event is rejected.

## Segment and align film

```bash
.venv/bin/all22 segment-game --game-id cowboys-at-eagles-2025-reg-1
.venv/bin/all22 align-game --game-id cowboys-at-eagles-2025-reg-1
```

Segmentation combines scene changes with stable camera-orientation changes,
classifies sideline/end-zone angles from painted-field geometry, and preserves
each angle as an independent source range. Alignment groups an ordered
sideline source with its following end-zone/alternate sources and globally
aligns those groups to every filmed nflverse event. Including punts, kickoffs,
and field goals in the alignment prevents special teams from shifting all
later scrimmage plays; the viewer still lists only eligible runs and passes.

## Viewer

```bash
npm run cv:serve
```

Open `http://127.0.0.1:8000`. The viewer displays PFR plays, local film, quality diagnostics, and any accepted trajectory Parquet files stored as `data/trajectories/<play-id>.parquet`.

## Verification

```bash
npm run cv:test
```

Tests cover PFR parsing, BDB schema aliases and snap normalization, monotonic alignment, homography recovery, rigid rejection gates, camera-angle classification, and API/static application behavior.

## Current implementation boundary

The data contracts, BDB answer ingestion, synchronization manifest, baseline video segmentation, alignment, calibration utilities, quality gates, API, and viewer are operational. The learned field-keypoint, RT-DETR, ByteTrack, jersey-OCR, and cross-angle fusion models require the matching BDB film and annotations before they can be trained; their training dataset interface is implemented in `src/all22/training.py`.

Keep NFL footage, browser profiles, signed URLs, BDB-derived artifacts, and reconstructed trajectories private and comply with the applicable source licenses and service terms.
