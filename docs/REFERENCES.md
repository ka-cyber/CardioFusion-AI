# References

Methodological sources actually used in this codebase (as opposed to just
listed in the README's related-work style sections):

- Wang W, Mohseni P, Kilgore KL, Najafizadeh L (2023). "PulseDB: A large,
  cleaned dataset based on MIMIC-III and VitalDB for benchmarking cuff-less
  blood pressure estimation methods." *Frontiers in Digital Health*, 4:1090854.
  doi: [10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854).
  Source of the feasibility-rules + adaptive-template SQI method implemented
  in `preprocessing/signal_quality.py` (which the PulseDB paper itself
  attributes to Orphanidou et al. 2015, below). Note: this repo does not
  include the actual PulseDB dataset — only the cleaning/QC *methodology*
  the paper describes was adapted. If you want the dataset itself, it's
  linked from the paper (GitHub + Kaggle releases).

- Orphanidou C, Bonnici T, Charlton P, Clifton D, Vallance D, Tarassenko L
  (2015). "Signal-quality indices for the electrocardiogram and
  photoplethysmogram: derivation and applications to wireless monitoring."
  *IEEE J Biomed Health Inform*, 19:832–8. doi: 10.1109/JBHI.2014.2338351.
  Original source of the feasibility-rules + SQI method above.

- Pan J, Tompkins WJ (1985). "A real-time QRS detection algorithm." *IEEE
  Trans Biomed Eng*, BME-32(3):230–6. doi: 10.1109/TBME.1985.325532.
  Basis for `preprocessing/ecg/ecg_preprocessing.py::detect_r_peaks`
  (derivative -> squaring -> moving-window integration -> adaptive peak
  picking).

- Pimentel MAF, Johnson AEW, Charlton PH, Birrenkott D, Watkinson PJ, Tarassenko L,
  Clifford GD (2016). "Toward a Robust Estimation of Respiratory Rate from Pulse
  Oximeters." *IEEE Trans Biomed Eng*, 64(8):1914–1923. doi: 10.1109/TBME.2016.2613124.
  Source of the BIDMC PPG and Respiration Dataset used in
  `real_data_validation/bidmc_validation/` — real ECG+PPG+reference vitals
  from 53 ICU patients, the strongest real-data validation in this repo
  (both modalities, plus synchronization, against real bedside-monitor
  ground truth). See `BIDMC_VALIDATION_REPORT.md` for what was actually run,
  including a real PTT-estimation bug found and fixed as a direct result.

- Jezewski J, Matonia A, Kupka T, Roj D, Czabanski R. "Abdominal and Direct
  Fetal ECG Database." PhysioNet. https://doi.org/10.13026/c2mw2s.
  Real validation data used in `real_data_validation/` — see
  `REAL_DATA_VALIDATION_REPORT.md` for what was actually run against it.

## Cited but not yet implemented

The PulseDB paper also describes Elgendi's algorithm (Elgendi et al. 2013,
doi: 10.1371/journal.pone.0076585) for PPG systolic-peak detection, used as
PulseDB's actual PPG peak detector. `preprocessing/ppg/ppg_preprocessing.py`
currently uses a simpler prominence-based `scipy.signal.find_peaks` approach
instead (validated on synthetic PPG — see the main `tests/`). Swapping in
Elgendi's moving-average-based method would bring the PPG side in line with
the same standard the ECG side (Pan-Tompkins) already follows, and is a
reasonable next improvement if you want the two modalities' peak detectors
built on comparably well-established, cited methods.
