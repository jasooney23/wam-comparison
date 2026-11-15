# Implementation review

The experiment now uses the config and evaluation layout described in
[README.md](README.md). This supersedes the earlier fixed-holdout and comparison
artifact workflow reviewed on 2026-11-15.

- Pretraining uses 10,000 ID pairs and 500 OOD pairs, sampled from shared ranges.
- Pretraining reconstruction evaluation and checkpoints run every five epochs
  by default, and at the final epoch.
- Agent training freezes the WM. Periodic joint evaluation uses generated OOD
  policy episodes for both agent return and WM horizon errors.
- Each run writes one metrics CSV. All TensorBoard scalars also appear there.
- Agent-only and resumed runs use saved normalization and need no dataset.
- No automatic chart or comparison artifacts are produced.

Current validation results should come from running the test suite and smoke
commands; previous test counts and checkpoint compatibility claims are historical.
