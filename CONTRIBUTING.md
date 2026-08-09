# Contributing to CardioFusion-AI

Thanks for considering contributing. A few things that will make this
smoother, especially given this project's specific history (see
`RECONSTRUCTION_NOTES.md` — this codebase was rebuilt after the original
repo was lost, so provenance and honesty about what's validated vs. not is
taken seriously here):

## Before you start

- Check `PRODUCTION_READINESS.md` for what's actually validated vs. still
  needed. If you're picking up one of the gaps listed there (real labeled
  training data, an actual training run, clinical validation), please open
  an issue first to coordinate.
- Run the test suite (`pytest tests/ -v`) before opening a PR. If you're
  touching `models/` or `training/` (the PyTorch-dependent code), note that
  it was never executed in this project's original build environment (no
  GPU/network access) — please actually run `pytest tests/test_models.py`
  yourself and mention the result in your PR description.

## Development setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
ruff check .
```

## Code style

- `ruff` is configured in `pyproject.toml` — run `ruff check .` before
  committing.
- Type hints on function signatures are expected for new code (see existing
  `preprocessing/` modules for the style).
- New preprocessing/model code should raise `utils.exceptions.CardioFusionError`
  subclasses for invalid input, not bare exceptions — see
  `preprocessing/ecg/ecg_preprocessing.py::_validate_signal` for the pattern.

## Adding a real-data validation

If you validate against another real dataset (as `real_data_validation/`
already does for two), please follow the existing structure: a report
markdown file stating plainly what was and wasn't tested, tables/figures
suffixed `_REAL`, and raw data excluded via `.gitignore` rather than
committed. Never commit synthetic results without an equally clear `_SYNTHETIC`
label — see `synthetic_validation_run/` for the convention.

## Reporting bugs

Please use the issue templates under `.github/ISSUE_TEMPLATE/`. For
anything related to signal-processing correctness, a minimal reproducing
example (even synthetic) is enormously helpful.

## Commit messages

No strict format required — just describe *what* changed and *why*,
especially for anything affecting signal processing accuracy (a one-line
"fix ppg peaks" is much less useful than "fix PPG double-peak detection:
add prominence threshold, see tests/test_preprocessing.py").
