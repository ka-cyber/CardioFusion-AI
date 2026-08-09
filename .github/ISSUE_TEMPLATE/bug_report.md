---
name: Bug report
about: Something in the pipeline is producing incorrect or unexpected results
title: "[BUG] "
labels: bug
assignees: ''
---

**Which module?**
e.g. `preprocessing/ecg`, `models/fusion`, `evaluation`, `real_data_validation`

**Describe the bug**
What happened, and what did you expect instead?

**Minimal reproducing example**
```python
# Ideally a small synthetic signal that reproduces the issue --
# see tests/test_preprocessing.py for the synthetic-signal-generator pattern.
```

**Environment**
- Python version:
- Relevant package versions (`pip freeze | grep -E "numpy|scipy|torch"`):
- OS:

**If this is about accuracy/correctness on real data**
Please don't attach real patient data directly to the issue. Describe the
dataset (name, public source) and the discrepancy instead, or link to a
PhysioNet/public record ID.
