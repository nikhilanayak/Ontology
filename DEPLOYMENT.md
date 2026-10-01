# Production deployment

The production host is intentionally not encoded in this repository. Set it
locally when deploying:

```bash
export ONTOLOGY_HOST='user@example-host'
export ONTOLOGY_PORT='22'
```

Clone and bootstrap the code on the host:

```bash
ssh -p "$ONTOLOGY_PORT" "$ONTOLOGY_HOST"
git clone https://github.com/nikhilanayak/Ontology.git ~/Ontology
cd ~/fast/Ontology
./scripts/bootstrap-production.sh
```

Load the fast-disk cache environment before running commands directly:

```bash
cd ~/fast/Ontology
source scripts/production-env.sh
```

This routes pip, npm, Python bytecode, Hugging Face, Torch, Triton, CUDA,
Numba, Matplotlib, and temporary caches into `.runtime/` on the project disk.

Run the CPU test suite:

```bash
cd ~/fast/Ontology
.venv/bin/python -m pytest -q
```

Film, browser profiles, signed manifests, databases, credentials, BDB inputs,
and generated trajectories are deliberately excluded from Git. Transfer those
artifacts separately into `downloads/` and `data/` after creating a backup and
checking available disk space.

For new film, do not transfer a local download. Run the local authenticated
collector with `--remote` and `--remote-root /home/nikhil/fast/Ontology`; the
signed manifest crosses SSH stdin and ffmpeg writes directly to the fast disk.

The GPU stack is installed separately from the base application so a CUDA wheel
can be selected for the host's NVIDIA driver. Verify it with:

```bash
./scripts/bootstrap-gpu.sh
```

The production host currently uses the pinned PyTorch 2.5.1 CUDA 12.1 wheel,
which is compatible with its NVIDIA 535 driver. The script ends with a real
CUDA matrix-multiplication smoke test rather than checking imports alone.
