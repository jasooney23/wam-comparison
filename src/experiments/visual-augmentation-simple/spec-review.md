# Implementation review

The experiment now uses the config and evaluation layout described in
[README.md](README.md). This supersedes the earlier fixed-holdout and comparison
artifact workflow reviewed on 2026-11-15.

- Pretraining uses 10,000 ID pairs and 500 OOD pairs, sampled from shared ranges.
- Pretraining next-step prediction evaluation and checkpoints run every five epochs
  by default, and at the final epoch.
- Agent training freezes the WM. Periodic joint evaluation uses generated OOD
  policy episodes for both agent return and WM horizon errors.
- WM pretraining and agent training have separate scripts and output directories.
- Both stages always train vector and combined using their own configured seeds.
- Agent seed N requires both completed WM saves for seed N before any training starts.
- Each stage/run writes one metrics CSV and matching TensorBoard scalars, including
  evaluation and compute metrics.
- Agent training uses saved normalization and needs no dataset. Resume is removed.
- Pretraining publishes final WM artifacts; agent evaluation saves model snapshots
  with source WM provenance and no optimizer or random-generator state.
- No automatic chart or comparison artifacts are produced.

Current validation results should come from running the test suite and smoke
commands; previous test counts and checkpoint compatibility claims are historical.
