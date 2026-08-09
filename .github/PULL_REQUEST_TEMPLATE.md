## What does this PR do?

## Checklist

- [ ] `pytest tests/ -v` passes locally
- [ ] `ruff check .` passes locally
- [ ] If this touches `models/` or `training/` (PyTorch code): I actually ran
      it, not just syntax-checked it (see `PRODUCTION_READINESS.md` for why
      this matters — a lot of this repo's model code was only syntax-checked
      when originally written, due to no torch/network access in that
      environment; actually running your changes closes that gap)
- [ ] If this adds/changes signal-processing logic: I added or updated a
      test in `tests/`, ideally against a real dataset if one is reasonably
      available, clearly labeled `_REAL` vs `_SYNTHETIC` per
      `real_data_validation/` and `synthetic_validation_run/` conventions
- [ ] No real patient/subject data is committed (check `.gitignore` covers
      any new data directories you introduced)
- [ ] Docs updated if behavior changed (`docs/`, relevant `README.md`)

## How was this tested?
