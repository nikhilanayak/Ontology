# Project status — 2026-10-02

## Objective

Reconstruct anonymous NFL player trajectories in field coordinates from authenticated All-22 film, align independent camera presentations to nflverse play-by-play, and evaluate the result against Big Data Bowl (BDB) tracking. The pipeline is fail-closed: uncertain source grouping, alignment, calibration, coverage, or cross-angle agreement must remain reviewable rather than being published as a confident result.

Do not run the full season yet. The immediate objective is to improve and validate the bounded Bills-Rams pilot without trading away spatial track quality for identity continuity.

## Current state

- Deployed application commit: **`55c075e`** (`Add standalone Hough line preview`) on production and GitHub `origin/main`. Remote suite: **106 passed** (`source scripts/production-env.sh && .venv/bin/python -m pytest -q`, 2026-10-02). Viewer restarted; `GET /api/games`, `/hough`, `/api/hough/clips`, and a real annotated-frame JPEG verified.
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
| Phase 1 as first written (`55dbd92`) | `data/evaluations/bills-at-rams-2022-reg-1/phase1-55dbd92.json` | purity `0.4580`; switches/100 `32.18` — **worse**, reverted in part (see ablation) |
| Phase 1 kept + r2 detector (`d572544`+) | `data/evaluations/bills-at-rams-2022-reg-1/phase1-4-r2.json` | coverage `0.7735` (**+0.010**); purity `0.4764`; switches/100 `30.95`; raw `5.5453 yd`; refined `3.0456 yd`; `drift_allowed: true` (detections intentionally changed) |

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

## Objective narrowed to formation analysis (2026-10-02)

The goal is reading **pre-snap formations** and following **important players through a route**.
Inch-level accuracy inside a tackle pile is explicitly not required. That reframing matters,
because whole-play HOTA is dominated by the pile — the hardest and least relevant regime.

Three targets were set and measured. Protocol `c5a8cde0…`, r2 detections, `411c03d`+.

| Metric | Validation (9) | Test (9) |
|---|---:|---:|
| Snap-frame recall of 22 players | **98.8%** | 81.8% |
| Isolated-player recall (route runners) | **97.5%** | 75.7% |
| Snap median error | 2.13 yd | 2.19 yd |
| Snap players within 1 yd | 18.7% | 22.4% |
| HOTA / DetA / LocA | .125 / .123 / .769 | .118 / .111 / .761 |

Best clip (`0013`): **1.06 yd median, 45% within 1 yd, 99.2% snap recall** — already
formation-grade. Worst (`0295`, `0277`): 7.17 and 5.26 yd. The spread across clips, not the
average, is the problem.

### Goal 2 — losing route runners: largely fixed

The on-field test was a hard `0 <= y <= 53.333` box with no tolerance, while our own calibration
is several yards out. On clip `0003`, **98.4% of dropped detections failed only on `field_y`,
at a median of 2.2 yd outside the sideline** — split receivers and gunners, precisely the route
runners of interest. Measured before the fix: only **33%** of isolated truth players (>5 yd from
anyone) had any prediction within 2 yd, versus 80–90% for players in formation. Isolated players
scored *worse* than players in a pile (7.08 vs 2.29 yd), which is backwards for a localization
problem and was the clue.

Admitting a calibration-scaled margin (3 yd, capped at 8, scaled by the frame's p95) lifted
isolated recall from ~33% to **97.5%** on validation. The margin does not manufacture ghosts:
out-of-bounds detections are 0–9% of reliable track rows on most clips.

### Goal 1 — recognizing calibration position: verification added, accuracy still open

Calibration's own residuals are **uncorrelated with real error** (|r| < 0.17 across 15 clips on
keyframe count, density, p95 and inlier ratio). Clip `0007` reported a **0.22 yd p95 while
sitting 5.92 yd out**: the homography fits the lines it was handed and is placed on the wrong
yard lines. A rigid shift of the whole formation removes much of the error (`0007` 5.92 → 2.52 yd
with a −5.83 yd y-shift; `0295` 8.08 → 6.13 with a −5.16 yd x-shift), confirming the formation
*shape* is roughly right but globally mislocated.

`verify_registration` now checks the result against field geometry — scale in px/yd, frame span,
overlap with the field, skew, degeneracy — and zeroes confidence when implausible, so a
confidently wrong fit fails closed instead of propagating. This catches absurd fits; it does
**not** yet resolve the 5-yard-periodic ambiguity, which is the remaining accuracy gap.

### Goal 3 — generic team identification: reworked to be matchup-independent

Team assignment was k-means on LAB/HSV with hand-tuned scales and a colour-based seed, which
cannot transfer to new teams, lighting or kit pairings. It is now **decided by formation
geometry**: at the snap the two units occupy disjoint ranges of `field_x`, which separates them
perfectly in BDB truth regardless of who is playing. Appearance is then learned *per clip* from
those geometrically-defined sides to label players the snap could not see, and an ambiguous
uniform stays `unknown` rather than guessing. Tested across three unrelated uniform pairings
including two visually similar light kits.

Team balance is still wrong on the over-tracked clips (`0277` 31 v 6, `0295` 30 v 7) because
those clips produce 37 reliable tracks for 22 players. That is fragmentation, not a team bug.

### What remains

Localization is the binding constraint for formation accuracy: snap recall is ~99% but only
~19% of players land within 1 yd. Established earlier, and unchanged: a **perfect per-frame
homography fitted to the answers** still leaves 2.11 yd median, so calibration alone cannot
reach 1 yd — the footpoint (bottom-centre of the box) and the single-camera geometry are the
next limits. On the best-calibrated clip that residual is 0.74 yd, so ~1 yd is reachable where
registration is good.

## Evaluation now follows the standard MOT paradigm (2026-10-02)

The earlier tuning loop was circular because **the metric was the moving part**. The in-house
score did per-frame Hungarian matching with a hard 8 yd gate and reported purity, switches/100
and fragments. Tightening that gate to 2 yd moved purity from `.457` to `.623` on clip `0005`
with *no change to the tracker*, so sweeps were partly measuring the evaluator. Raw ID-switch
and fragmentation counts are exactly what the HOTA paper identifies as non-comparable across
systems with different detection quality.

**Adopted HOTA** (Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object
Tracking*, IJCV 2021, [arXiv:2009.07736](https://arxiv.org/abs/2009.07736)) in `src/all22/hota.py`:

* integrates over the localization threshold `alpha` (0.05…0.95), so no single distance gate
  decides a match;
* decomposes **exactly** as `HOTA = sqrt(DetA * AssA)`, separating "players we never detected"
  from "identities we failed to keep" — the two symptoms we kept conflating;
* reports `DetRe`/`DetPr` and `AssRe`/`AssPr`, which distinguish *splitting* a player across
  tracklets from *merging* two players into one track;
* field positions have no boxes, so similarity is the Gaussian from SoccerNet Game State
  Reconstruction ([arXiv:2404.11335](https://arxiv.org/abs/2404.11335)),
  `exp(ln(0.05) * d^2 / tau^2)`, scaled so a pair exactly `tau` apart scores `.05` and cannot
  match at any standard alpha. `tau = 2.5 yd`, chosen from player spacing and frozen.

**Verified bit-exact against the reference implementation** (TrackEval, MIT) on five adversarial
scenarios covering identity swaps, misses, false positives and jitter: worst absolute difference
`0.000e+00` across all eight fields. Reference values are pinned in `tests/test_hota.py`.

### HOTA baseline (protocol `7c6f2489…`, r2 detections, 18 sources)

| Split | n | HOTA | DetA | AssA | AssRe | AssPr | LocA |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 9 | `0.1243` | `0.1215` | `0.1302` | `0.2000` | `0.2325` | `0.7633` |
| test | 9 | `0.1211` | `0.1208` | `0.1279` | `0.1937` | `0.2397` | `0.7583` |
| all | 18 | `0.1227` | `0.1211` | `0.1291` | `0.1969` | `0.2361` | `0.7608` |

Validation and test agree closely, so the split is balanced and the aggregate is not an artifact
of which clips landed where.

### What HOTA revealed that the old metric hid

On clip `0005`, the distance between Hungarian-matched pairs is:

| p10 | p25 | p50 | p75 | p90 | mean |
|---:|---:|---:|---:|---:|---:|
| `0.66` | `1.73` | `5.63` | `10.18` | `14.93` | `6.58` yd |

**Only 33% of matched pairs are within 2.5 yd; the median is 5.6 yd.** The previously reported
"refined error 2.93 yd" was computed over a filtered subset after fitting a correction, so it
flattered the pipeline. `LocA ≈ 0.76` and low `DetA` are therefore mostly a **localization /
calibration** problem, not a detector-recall problem: a track that exists but sits 5 yd from the
player cannot match at most alphas. This reframes the priority — homography and calibration
quality now look more important than detector recall or association weights.

Note also that a camera shot rarely spans the whole play (clip `0005` covers 68 of 142 BDB
frames), so HOTA is scored only on covered frames and `frame_overlap` is reported per source;
charging the tracker for unfilmed truth frames would measure clip boundaries, not tracking.

### Experimental discipline

Protocol sources are now split deterministically into **validation** and **test** by a hash of
the clip id (9/9 here). Hyperparameters are chosen on validation; test is reported once. The
earlier sweeps tuned and reported on the same 18 clips, which turns noise into apparent gains —
one sweep scored `.459` and `.503` for configurations whose difference was within the spread of
those clips. `purity` and `switches/100` are retained as gate-dependent diagnostics only.

## Detection/tracking quality investigation (2026-10-02)

Measured on production against the frozen protocol. **Read this before tuning anything.**

### Where players are actually lost

| Stage | Players/frame (median of 18) |
|---|---:|
| BDB truth | 22 |
| Detections (old, threshold .35) | 20.7 |
| Survive `on_field` + `calibration_valid` | 18.7 |
| In tracks | 18.7 |
| Marked `track_reliable` | 15.2 |

The detector is **not** the main bottleneck: on clip `0005` it already returns 22–26 people per
frame, median box height is 74–245 px, and only ~3% of boxes are under 40 px. Raising inference
resolution to native 1080p changed per-frame counts from `[22,22,22,24]` to `[22,22,21,24]`;
1440p added ~2. Median raw track count was **42 IDs per clip for 22 players** (175 on `0009`).

### What was tried, and what it measured

Ablation on the frozen protocol (baseline purity `.4858`, switches `29.90`):

| Variant | Purity | Switches/100 |
|---|---:|---:|
| Deterministic team anchor only | `.4675` | `28.42` |
| + play-span reliability denominator | `.4618` | `31.88` |
| + snap line-of-scrimmage naming | `.4618` | `31.88` (inert) |
| + officials cluster cap `.30` | `.4580` | `32.18` |

**Kept:** the deterministic anchor and per-track team resolution. Tracking used to read
`action_windows.snap_s`, which action discovery writes *after* tracking, so a clip's first
reconstruction differed from every later one (verified: clip `0004` went from
`{team_1: 1157, team_0: 738}` to `{team_0: 688, team_1: 672, official: 444}` on re-run).

**Reverted as measured-harmful:** the play-span reliability denominator (admits marginal
tracklets whose fragmentation costs more than the recovered part-time players) and the relaxed
officials cap (mislabelled 16 of 27 reliable tracks as officials on `0009`, poisoning the
team-mismatch penalty).

### Confirmed root cause of the coverage ceiling

The `.35` detection threshold was **baked into the artifacts**, so the weak-detection tier the
two-stage associator needs was discarded before tracking. Proof: sweeping the tracker birth
threshold over `.55/.45/.35` was *exactly* inert (identical metrics to 4 dp) because nothing
below `.35` existed. Re-detecting at threshold `.15`, native 1080p, fp16, batch 4 gives ~19%
more detections per frame with an 8–11% weak tier, at ~17 s/clip, and lifted coverage
`0.7632 → 0.7735`. Detections live at `data/detections-r2/` (`model_version` now records
revision, resolution, and precision).

### The identity metric is partly measuring itself

`EVALUATOR_PARAMETERS.identity_maximum_error_yards` is `8.0`, which is wider than the spacing
between players. Tightening the gate alone, with **no tracking change**, moves purity a lot:

| Gate | `0005` purity | `0010` purity |
|---|---:|---:|
| 8.0 yd (current) | `.457` | `.297` |
| 4.0 yd | `.582` | `.325` |
| 2.0 yd | `.623` | `.376` |
| 1.5 yd | `.677` | `.420` |

At 8 yd the Hungarian force-matches players several positions away, manufacturing apparent
switches. Clip `0005` is close to correct (22 reliable tracks/frame, one row per track per
frame) yet scores `.457`; `0010` stays poor at every gate and has a genuine tracking fault.
**Any future purity number must state its gate.** Changing the gate requires an
`EVALUATOR_VERSION` bump and a protocol re-freeze.

### Global (offline) association

Implemented in `associate_tracklets_globally`: tracklets overlapping in time cannot be the same
player, and a player can only travel so far across a gap. It makes 37 links over 999 tracks and
moves switches `31.37 → 30.95` with purity unchanged — small, because track lifetimes are
already ~5.15 s of a ~7.0 s window. Fragmentation of 5–13 per player therefore comes mostly from
the 8 yd matching gate, not from short tracklets.

### Evaluator is 11x faster

94% of evaluation time was pandas column indexing inside the 64-point alignment grid. Grouping
positions and truth into numpy once per source cut a full 18-source run from **62.6 s to 5.8 s**,
verified bit-identical on 180 real grid points (max abs difference `0.000e+00`).

## Immediate next work

1. **Attack localization, not association.** `LocA 0.76` and a 5.63 yd median matched-pair
   distance say the dominant error is where a player is placed on the field, not which track id
   they carry. Concretely: denser calibration keyframes during live action (currently ~0.75 s
   apart with linear homography interpolation between them), per-frame homography refinement, and
   excluding moving players from ORB feature matching. Measure with `LocA` and `DetA` on the
   validation split.
2. Re-score the Phase 1–4 changes under HOTA. They were selected against the old gated metric, so
   which of them actually helped is currently unknown.
3. Evaluate the detector **directly** (COCO AP/AP50/AP75, and log-average miss rate for crowded
   formations) on a small set of exhaustively box-labelled held-out frames. Judging detection
   through a tracking metric cannot separate a detector miss from an association or calibration
   failure.
4. Only then revisit association weights, on the validation split, one factor at a time, with
   paired per-sequence differences rather than aggregate medians.
2. Improve field-boundary estimation and homography robustness, especially on the three weak example sources above. Note that re-calibrating a clip changes its calibration revision and will (correctly) trip protocol drift; re-freeze deliberately when that is the intent.
3. Validate changes on additional bounded clips only after the explicit pilot clips improve or reveal a stable tradeoff.
4. Keep all detection/reconstruction work bounded (`--clip-id` preferred, otherwise a small `--limit`). Do **not** start a full-game or full-season run.

Canonical test, deployment, bounded experiment, and server-restart commands are maintained in `AGENTS.md`. Update this file immediately after the next experiment or deployment with the exact commit, command, source set, artifact path, and metrics.

## Standalone Hough line preview (deployed `55c075e` — 2026-10-02)

The prior calibration/tracking direction is paused. A new standalone experiment now applies only
generic Canny edge detection and `cv2.HoughLinesP` to a decoded video frame, then draws every raw
segment coloured by orientation. It deliberately has no field mask, line clustering, yard-line
interpretation, calibration, detection, or tracking dependency.

- Local implementation: `src/all22/hough.py`; preview at `/hough`; API routes under `/api/hough/`;
  controls expose frame time, blur, Canny thresholds, Hough threshold, minimum line length,
  maximum gap, line thickness, raw edges, and line visibility.
- Local verification: `.venv/bin/python -m pytest -q` — **106 passed**, one existing
  `StarletteDeprecationWarning` (2026-10-02).
- End-to-end synthetic verification: `/hough`, JS/CSS assets, line JSON, annotated JPEG,
  edge-only image, missing-video errors, and timestamp bounds all passed through FastAPI's test client.
- Bounded production-data probe (no deployment): copied only `src/all22/hough.py` to ignored
  `.runtime/hough_preview_probe.py` and sampled `start_s + 0.3` for clips `0005`, `0013`, and `0295`
  from both registered games. All six 1920x1080 frames decoded and rendered. Raw line counts were
  Bills-Rams: **518 / 489 / 479**; Lions-Chiefs: **988 / 859 / 559**. Generated probe JPEGs remain
  private under production `.runtime/`; no footage or derived image entered Git.
- Deployment: committed and pushed as **`55c075e`**, fast-forwarded production from `4c810c0`,
  and ran `source scripts/production-env.sh && .venv/bin/python -m pytest -q`: **106 passed** with
  one existing `StarletteDeprecationWarning`. Restarted `.venv/bin/all22 serve` on `127.0.0.1:8000`.
  Production verification: `/api/games` 200, `/hough` 200, `/api/hough/clips` returned **634**
  clips, and `bills-at-rams-2022-reg-1:0005/annotated.jpg` returned 200 (**275,368 bytes**).
- The unrelated pre-existing local modification in `src/all22/field_registration.py` remains
  preserved and was not included in `55c075e`.
