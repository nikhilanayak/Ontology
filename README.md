# NFL All-22 reconstruction

This repository has two connected systems:

1. An authenticated Chrome/CDP downloader for NFL All-22 film.
2. A BDB-evaluated computer-vision pipeline that aligns film to nflverse play-by-play and reconstructs anonymous player trajectories in field coordinates.

The reconstruction pipeline is deliberately fail-closed. A play is not published unless source grouping, play alignment, field calibration, track coverage, and cross-angle agreement pass their thresholds.

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

## Download verified All-22 film directly to production

Log in once, then run the CDP collector. `--limit` stops both discovery and downloading after the requested number of games.

```bash
npm run login
npm run batch:cdp -- 2025 --limit 1
```

The collector requires the visible All-22 control, a 50+ minute player duration, a matching HLS playlist duration, and a valid video stream before saving.

For the pilot, Chrome and the authenticated NFL profile remain local while
ffmpeg runs directly on the production box. The signed manifest is sent only
through SSH stdin and is never written to Git, command arguments, or metadata:

```bash
npm run batch:cdp -- 2022 \
  --game-id bills-at-rams-2022-reg-1 --limit 1 \
  --remote nikhil@50.39.98.5:2222 \
  --remote-root /home/nikhil/fast/Ontology
```

The remote receiver allowlists HTTPS Lura manifests, chooses the largest video
rendition, rejects non-video or short downloads, writes through a `.partial`
file, and records only a sanitized manifest fingerprint. Start with BUF–LAR;
do not acquire the remaining games until its alignment audit passes.

## Prepare the 2022 BDB pilot

The NFL host removed the 2024 competition payload after the contest. Use the
still-available, account-authorized 2026 analytics release for pass-route
trajectory supervision; keep the 2022 pilot film for segmentation/alignment.
Run on the box:

```bash
cd /home/nikhil/fast/Ontology
source scripts/production-env.sh
python scripts/fetch-bdb-2024.py \
  --competition nfl-big-data-bowl-2026-analytics

.venv/bin/all22 import-bdb \
  data/raw/nfl-big-data-bowl-2026-analytics/train/input_2023_w01.csv
```

The 2026 input files provide official 2023 tracking before the pass and their
output counterparts provide the ball-flight continuation. This makes the
first BDB-scored experiment pass-only; run plays remain part of the independent
film/PBP alignment audit.

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

For each pilot game, connect its local slug to the numeric BDB ID emitted by
`select-pilot-games`, then generate the deterministic, quarter/play-type
balanced 40-play audit set:

```bash
.venv/bin/all22 set-external-id \
  --game-id bills-at-rams-2022-reg-1 --provider bdb --external-id BDB_GAME_ID
.venv/bin/all22 create-audit-sample \
  --game-id bills-at-rams-2022-reg-1 --count 40
.venv/bin/all22 audit-summary --game-id bills-at-rams-2022-reg-1
```

The viewer exposes audit controls and independent snap/end corrections for
each source. Check the two presentations separately; extra fragments and
missing angles are retained rather than forced into an exactly-two schema.
The gate is at least 95% exact play mapping and at least 90% of audited plays
with two correct source presentations.

## Frozen detector/tracker baseline

The pilot does **not** train an end-to-end recognition model. It runs a frozen
Torchvision Faster R-CNN COCO person detector, associates boxes over time, and
uses the bottom-center of each box as the player's contact point:

The current trajectory path is camera-shot centric. Hard cuts establish the
only mandatory media boundaries; a shot may contain zero, one, or several
complete actions. Detection therefore runs before play alignment:

```bash
.venv/bin/all22 detect-clips \
  --game-id bills-at-rams-2022-reg-1 --device cuda --limit 4
```

For each shot, add one or more calibration keyframes. Every keyframe contains
at least four image/field point pairs; multiple keyframes let the homography
follow a pan or zoom without treating it as a cut:

```json
{
  "keyframes": [
    {
      "timestamp_s": 123.4,
      "image_points": [[310,170],[1610,180],[180,900],[1740,920]],
      "field_points": [[40,0],[60,0],[40,53.333],[60,53.333]]
    }
  ]
}
```

```bash
.venv/bin/all22 calibrate-clip --clip-id SHOT_ID data/landmarks/SHOT_ID.json
.venv/bin/all22 reconstruct-clip \
  --clip-id SHOT_ID --detections data/detections/SHOT_ID.parquet
```

`reconstruct-clip` projects each bottom-center contact point, assigns anonymous
team clusters, tracks in field coordinates, and discovers every motion-defined
action wholly contained in that shot. Calibration extrapolation is marked
invalid rather than silently used. Review the action boundaries in the viewer,
then pair repeated presentations and align them to play-by-play:

```bash
.venv/bin/all22 pair-actions \
  --game-id bills-at-rams-2022-reg-1 --tracks-dir data/clip-tracks
.venv/bin/all22 align-actions --game-id bills-at-rams-2022-reg-1
.venv/bin/all22 trajectory-pilot-report \
  --game-id bills-at-rams-2022-reg-1 --output data/reports/trajectory-pilot.json
```

Pairing uses duration, player counts, formation geometry, displacement, and a
normalized speed profile. It leaves ambiguous repetitions single. The PBP
aligner uses explicit `missing_film` and `extra_film` operations, ignores
timeouts, retains filmed penalties, and keeps uncertain matches reviewable.
The first experiment is ready only after 10 action windows are manually marked
`verified`; it does not require jersey identities or ball tracking.

The legacy play-source command remains available for comparison:

```bash
.venv/bin/all22 detect-source \
  --play-id bills-at-rams-2022-reg-1:0001 --source-order 0 \
  --device cuda --output data/detections/play-0001-sideline.parquet
```

Projection requires an audited snap timestamp and a landmark JSON file with at
least four non-collinear image/field correspondences, in yards on the standard
120 by 53.333 yard coordinate system:

```json
{
  "image_points": [[310, 170], [1610, 180], [180, 900], [1740, 920]],
  "field_points": [[40, 0], [60, 0], [40, 53.333], [60, 53.333]]
}
```

```bash
.venv/bin/all22 project-source \
  --play-id bills-at-rams-2022-reg-1:0001 --source-order 0 \
  --detections data/detections/play-0001-sideline.parquet \
  --landmarks data/landmarks/play-0001-sideline.json \
  --output data/projected/play-0001-sideline.parquet

.venv/bin/all22 fuse-sources \
  --output data/trajectories/bills-at-rams-2022-reg-1:0001.parquet \
  data/projected/play-0001-sideline.parquet \
  data/projected/play-0001-endzone.parquet

.venv/bin/all22 evaluate-trajectories \
  --game-id bills-at-rams-2022-reg-1 \
  --output data/reports/buf-lar.json
```

For BDB 2026, mark the pass-release time in the independent source and anchor
it to that play's maximum input `frame_id`:

```bash
.venv/bin/all22 project-source \
  --play-id lions-at-chiefs-2023-reg-1:PLAY --source-order 0 \
  --video-anchor-s PASS_RELEASE_VIDEO_SECONDS \
  --bdb-anchor-frame MAX_INPUT_FRAME \
  --detections data/detections/PLAY-sideline.parquet \
  --landmarks data/landmarks/PLAY-sideline.json \
  --output data/projected/PLAY-sideline.parquet
```

The evaluator snap-aligns results to BDB, performs anonymous Hungarian
matching, and reports median/p90 position error and player-frame coverage. The
expansion target is median error at most 2 yards, p90 at most 5 yards, and at
least 80% of BDB player-frames matched within 3 yards.

## Viewer

```bash
npm run cv:serve
```

Open `http://127.0.0.1:8000`. The default trajectory workspace lists independent
camera shots and overlays anonymous boxes on video and player positions on the
2D field. It also accepts calibration keyframe JSON and exposes discovered
action windows. Switch to **Alignment audit** for the earlier PBP review queue.

## Verification

```bash
npm run cv:test
```

Tests cover PFR parsing, BDB schema aliases and snap normalization, monotonic
alignment with gaps, multi-action shots, repeat pairing, homography recovery,
field-space tracking, calibration validity, rigid rejection gates,
camera-angle classification, and API/static application behavior.

## Current implementation boundary

The data contracts, direct-to-production acquisition, BDB ingestion, independent
shot detection, keyframed calibration, field-space tracking, multi-action
discovery, conservative repeated-view pairing, gap-aware PBP alignment, pilot
report, API, and viewer are operational. Team labels are anonymous color
clusters. Automated field-keypoint initialization, learned football-specific
detection, ball tracking, and targeted fine-tuning remain later experiments;
jersey OCR and player identity are explicitly outside this pilot.

Keep NFL footage, browser profiles, signed URLs, BDB-derived artifacts, and reconstructed trajectories private and comply with the applicable source licenses and service terms.
