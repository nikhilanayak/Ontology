# Agent instructions

Read `STATUS.md` before inspecting code or running commands. It is the durable handoff for the current objective, deployed state, data locations, metrics, and next experiments.

## Keep the handoff current

- Update `STATUS.md` after every material code change, experiment, evaluation, deployment, or discovery. Record the commit, exact command, inputs, outputs, metrics, and whether the result is local or deployed.
- Update this `AGENTS.md` whenever the workflow, safety rules, invariants, deployment process, or canonical commands change.
- Preserve user work. Inspect `git status` before editing; never discard, overwrite, reset, or sweep up unrelated changes.
- Keep expensive work bounded. Start with explicit `--clip-id` values or a small `--limit`; inspect results before expanding. Do not launch a full-game or full-season detector run by default.
- Run the relevant focused tests while iterating and the full suite before handoff or deployment.
- Production is `/home/nikhil/fast/Ontology`. Always load `scripts/production-env.sh` there so caches and temporary files stay on the fast volume under `.runtime/`.
- Never commit or push NFL footage, browser profiles, credentials, secrets, cookies, signed URLs/manifests, databases, BDB inputs or BDB-derived artifacts, detections, reconstructed tracks, evaluation outputs, evaluation protocols, or other licensed/private generated data. Keep them under ignored `downloads/`, `data/`, `.runtime/`, or another private fast-volume path.

## Canonical commands

Tests (local or remote):

```bash
cd /home/nikhil/fast/Ontology  # omit this line when already at the local repo root
source scripts/production-env.sh
.venv/bin/python -m pytest -q
```

Bounded detection, reconstruction, and evaluation on the Bills-Rams pilot:

```bash
cd /home/nikhil/fast/Ontology
source scripts/production-env.sh

# Prefer named clips. A small limit is acceptable for an initial cache-building pass.
.venv/bin/all22 detect-clips \
  --game-id bills-at-rams-2022-reg-1 --device cuda \
  --clip-id 'bills-at-rams-2022-reg-1:0005'
# Alternative bounded smoke run:
# .venv/bin/all22 detect-clips --game-id bills-at-rams-2022-reg-1 --device cuda --limit 1

.venv/bin/all22 reconstruct-clip \
  --clip-id 'bills-at-rams-2022-reg-1:0005' \
  --detections 'data/detections/bills-at-rams-2022-reg-1/bills-at-rams-2022-reg-1:0005.parquet'

# Comparable evaluation: always score against the frozen protocol. It pins the
# source list, BDB play, action window, detections hash, calibration revision,
# evaluator version, and evaluator parameters, and fails closed if any drift.
.venv/bin/all22 evaluate-audited-sources \
  --protocol data/evaluation-protocols/bills-at-rams-2022-reg-1.json \
  --output data/evaluations/bills-at-rams-2022-reg-1/<label>.json

# Exploratory only (re-selects windows from the tracks under test; NOT comparable across runs):
# .venv/bin/all22 evaluate-audited-sources --game-id bills-at-rams-2022-reg-1 --output data/evaluations/<label>.json
```

Evaluation protocol rules:

- Re-freeze (`all22 freeze-evaluation-protocol --game-id ... [--clip-id ...]`) only when the audited source set, detections, calibration, or evaluator intentionally change. Freezing a new protocol starts a new comparison series; record the protocol path, its `sha256`, `git_commit`, and `git_dirty` in `STATUS.md`.
- Bump `EVALUATOR_VERSION` in `src/all22/supervision.py` whenever alignment, refinement, identity matching, or aggregation logic changes, and route any new constant through `EVALUATOR_PARAMETERS`. Protocol evaluation refuses mismatched versions/parameters.
- `--allow-input-drift --drift-note '<why>'` is for diagnosis only; results carrying `drift_allowed: true` must not be entered into the metrics table as comparable.
- Tracker association weights/gates live as named attributes on `FieldSpaceTracker` and `RELINK_PARAMETERS`; their hash is pinned in `tests/test_field_tracking.py` and stored as `config_hash` on every `clip_tracks` artifact. Update the pinned hash deliberately when tuning.
- Protocol and evaluation JSON embed BDB-derived identifiers and detections hashes: they are private artifacts and the CLI refuses to write them to a Git-tracked location. Keep them under `data/`.

Deploy the committed, tested branch and verify it remotely (replace `main` only if the active deployment branch is different):

```bash
git status --short
npm run cv:test
git push origin main
ssh -p 2222 nikhil@50.39.98.5 \
  'cd /home/nikhil/fast/Ontology && git fetch origin && git checkout main && git pull --ff-only origin main && source scripts/production-env.sh && .venv/bin/python -m pytest -q'
```

Restart the remote viewer after a successful deployment:

```bash
ssh -p 2222 nikhil@50.39.98.5 \
  "cd /home/nikhil/fast/Ontology && source scripts/production-env.sh && pkill -f '[a]ll22 serve' || true; cd /home/nikhil/fast/Ontology && source scripts/production-env.sh && (setsid nohup .venv/bin/all22 serve --host 127.0.0.1 --port 8000 >.runtime/all22-serve.log 2>&1 </dev/null &); sleep 5"
ssh -p 2222 nikhil@50.39.98.5 \
  "curl --fail --silent --show-error http://127.0.0.1:8000/api/games >/dev/null"
```

Do not deploy a dirty tree, and do not copy ignored/private artifacts through Git. After deploying or restarting, record the deployed commit and verification result in `STATUS.md`.
