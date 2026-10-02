# Project status — 2026-10-02

## Objective

Reconstruct anonymous NFL player trajectories in field coordinates from authenticated All-22 film, align independent camera presentations to nflverse play-by-play, and evaluate the result against Big Data Bowl (BDB) tracking. The pipeline is fail-closed: uncertain source grouping, alignment, calibration, coverage, or cross-angle agreement must remain reviewable rather than being published as a confident result.

Do not run the full season yet. The immediate objective is to improve and validate the bounded Bills-Rams pilot without trading away spatial track quality for identity continuity.

## Current state

- Deployed commit: **`14ffe8f`** (`Freeze a reproducible BDB evaluation protocol`) on production, GitHub `origin/main`, and local. Remote suite: **65 passed** (`.venv/bin/python -m pytest -q`, 2026-10-02). Viewer restarted and `GET /api/games` verified.
- Handoff documents were added and pushed in `dc0d9a1` (`Document agent workflow and project handoff`). Future agents must run `git rev-parse HEAD` and compare local, GitHub, and production before changing or deploying code.
- Local tooling: Python 3.13 venv at `/Users/nnayak/Ontology/.venv` (ignored); `gh` installed via Homebrew and authenticated as `nikhilanayak`; `origin` is HTTPS with `gh auth setup-git` credentials. `npm`/`node` are not installed locally, so `npm run cv:test` must be replaced by `.venv/bin/python -m pytest -q` here.
- Implemented: semantic field calibration, keyframed field tracking, crowd gating, role-aware tracking, camera-angle scale gating, tracklet relinking, identity evaluation, 64-dimensional ResNet18 appearance embeddings, separation of spatial observations from durable identity evaluation, and a **frozen evaluation protocol** (see below).
- Frozen evaluation protocol (deployed):
  - `all22 freeze-evaluation-protocol --game-id ... [--clip-id ...]` writes `data/evaluation-protocols/<game>.json` pinning: audited clip list, BDB play id, selected action window (`snap_s`/`dead_s`), BDB snap frame and truth row count, detections artifact sha256/model/config hash, calibration revision, tracks file sha256 and tracker `config_hash` at freeze, `EVALUATOR_VERSION` (`bdb-audited-v2`), `EVALUATOR_PARAMETERS`, git commit and dirty flag.
  - `all22 evaluate-audited-sources --protocol <path>` scores exactly that set over the frozen windows and fails closed on: missing tracks, unscorable source, BDB answer-key change, evaluator version/parameter mismatch, and detections/calibration drift (overridable only with `--allow-input-drift --drift-note`, which marks the output `drift_allowed: true`).
  - Every evaluation JSON now includes `provenance` (evaluator version/parameters, git commit, dirty flag, tracks dir, protocol sha256) and per-source `tracks.sha256`, `tracks.config_hash`, `inputs.detections`, `inputs.calibration_revision`, and the evaluated `window`.
  - `clip_tracks` artifacts now record `config_hash` (hash of all tracker/relink weights and gates, pinned as `287e337ef3ec9122` sideline / `c84e66fef7200bd4` endzone in `tests/test_field_tracking.py`) and `input_revision` (calibration revision from the consumed projection).
  - Exploratory `--game-id` mode is unchanged numerically versus `ae4a847` (grid, filters, and tuple ordering verified identical) but now reports `skipped` sources instead of silently dropping them.
- The conservative tracklet relinker accepted **zero** joins on clips `0005`, `0278`, and `0295`; do not loosen its gates without an identity-metric improvement. This is currently a fail-closed capability, not a demonstrated quality gain.

## Architecture

1. The authenticated local Chrome/CDP collector discovers and validates All-22 HLS film, then streams the signed manifest over SSH stdin so `ffmpeg` writes directly to production.
2. The Python `all22` CLI stores game, clip, play, calibration, action-window, and artifact metadata; nflverse supplies structured play-by-play and BDB supplies evaluation truth.
3. Shot-centric detection uses a frozen Torchvision Faster R-CNN person detector. The current detector also emits ImageNet ResNet18 64d appearance embeddings.
4. Semantic number/yard-line calibration and keyframes map bottom-center contacts into the 120 x 53.333 yard field. Field-space tracking uses crowd, role, angle-scale, and relinking gates.
5. Spatial observations are now separated from durable identities. Repeated presentations are paired/aligned, then a separate identity evaluator measures matching quality against BDB.
6. The FastAPI viewer/API supports calibration, action review, alignment audit, and trajectory inspection.

## Paths and pilot data

- Local repository: `/Users/nnayak/FootballProject`
- Remote production repository: `/home/nikhil/fast/Ontology`
- Remote runtime/caches: `/home/nikhil/fast/Ontology/.runtime` via `source scripts/production-env.sh`
- Private generated inputs/outputs: repository-local `downloads/` and `data/` on the corresponding machine; both are ignored and must not be committed.
- Pilot: `bills-at-rams-2022-reg-1`
- BDB game ID: `2022090800`
- Imported Week 1 BDB table: **7,104,700 rows**, of which **417,772** belong to the target game.

## Evaluation history

These files are private, ignored evaluation artifacts under `data/`; record their summaries here but never commit the artifacts themselves.

| Run | Artifact | Result |
|---|---|---|
| Pre-embedding identity baseline | `data/bdb-audited-evaluation-identities.json` | median purity `0.46335`; switches/100 `30.247` |
| 64d embedding, same evaluator | `data/bdb-audited-evaluation-embed64.json` | purity `0.4948`; switches/100 `28.8862`; refined error `2.66625 yd`; coverage `0.6638` |
| Latest separated evaluator, 18 sources | `data/bdb-audited-evaluation-embed64-separated.json` | raw error `5.2657 yd`; coverage `0.76317`; refined error `2.92556 yd`; purity `0.48576`; switches/100 `29.8966` |
| **Protocol baseline** (`14ffe8f`, protocol `cf167f0e…b9e2ae`, 18 sources) | `data/evaluations/bills-at-rams-2022-reg-1/baseline-14ffe8f.json` | raw error `5.26572 yd`; coverage `0.76317`; refined error `2.92556 yd`; purity `0.48576`; switches/100 `29.8966`; `drift_allowed: false`; `input_drift: []` |

The latest separated-evaluator result is **not spatially comparable** to the two earlier runs because the evaluator and observation/identity separation changed. Do not characterize its raw/refined spatial movement as a regression or improvement without an apples-to-apples rerun.

The protocol baseline reproduces the separated-evaluator row exactly, confirming the protocol refactor is numerically neutral. **All future comparable rows must be produced with `--protocol` against the same protocol sha256 and the same `evaluator_version` (`bdb-audited-v2`).**

### Active protocol

- Path (production): `data/evaluation-protocols/bills-at-rams-2022-reg-1.json`
- `sha256`: `cf167f0e06f9ddad254ec56fe667adf45c62cc74d319f372bc8e1f41b0b9e2ae`
- Frozen at commit `14ffe8f`, `git_dirty: false`, evaluator `bdb-audited-v2`
- Command: `.venv/bin/all22 freeze-evaluation-protocol --game-id bills-at-rams-2022-reg-1` (no detector run; existing tracks under `data/clip-tracks`)
- Sources (18): clips `0003 0004 0005 0006 0007 0008 0009 0010 0013 0014 0121 0122 0277 0278 0293 0294 0295 0296`
- Skipped (2, recorded in the protocol): `0011`, `0012` — BDB play `146` has no play signature (no snap event in the imported tracking).
- Known gap: `tracks.config_hash` is `null` for all 18 sources because their `clip_tracks` artifacts predate `14ffe8f`. Re-running `reconstruct-clip` on a clip populates it; this does not trip protocol drift (tracks are the thing under test).

Comparable evaluation command:

```bash
.venv/bin/all22 evaluate-audited-sources \
  --protocol data/evaluation-protocols/bills-at-rams-2022-reg-1.json \
  --output data/evaluations/bills-at-rams-2022-reg-1/<label>.json
```

Selected latest per-source results:

| Source | Purity | Switches/100 | Fragmentation | Refined error (yd) |
|---|---:|---:|---:|---:|
| `0005` | `0.4165` | `26.60` | `6.55` | `5.3649` |
| `0278` | `0.54915` | `20.85` | `4.05` | `6.8855` |
| `0295` | `0.41793` | `37.25` | `11.13` | `5.6641` |

## Immediate next work

1. Tune embedding association on a small explicit clip set (`0005`, `0278`, `0295`) without losing good spatial tracks. Re-run `reconstruct-clip` for those clips only, then score with `--protocol`. Treat identity purity/switches and spatial error/coverage as separate paired outcomes; each tracker variant has a distinct `config_hash`.
2. Improve field-boundary estimation and homography robustness, especially on the three weak example sources above. Note that re-calibrating a clip changes its calibration revision and will (correctly) trip protocol drift; re-freeze deliberately when that is the intent.
3. Validate changes on additional bounded clips only after the explicit pilot clips improve or reveal a stable tradeoff.
4. Keep all detection/reconstruction work bounded (`--clip-id` preferred, otherwise a small `--limit`). Do **not** start a full-game or full-season run.

Canonical test, deployment, bounded experiment, and server-restart commands are maintained in `AGENTS.md`. Update this file immediately after the next experiment or deployment with the exact commit, command, source set, artifact path, and metrics.
