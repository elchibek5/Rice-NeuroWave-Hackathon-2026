# EEG Motor Imagery Classification (Python, XGBoost, Riemannian log-covariance)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

A small, dependency-light pipeline for **cross-subject EEG motor imagery classification** — the core problem behind non-invasive brain-computer interfaces (BCI). Given a 64-channel EEG epoch, it predicts which movement the person was imagining: **left hand, right hand, both hands, or both feet**.

Built for the Rice NeuroWave Hackathon 2026, but written to run on any epoched motor-imagery dataset in `.npy` form (BCI Competition IV 2a, PhysioNet EEGMMIDB, OpenBMI, your own recordings). One script, five libraries, no GPU.

**What you get**

- Leakage-safe validation: `GroupKFold` split by participant, so scores reflect real cross-subject generalisation
- Two feature families: classic bandpower statistics and Riemannian log-covariance (log-Euclidean SPD) features, auto-selected by CV
- An XGBoost fold ensemble with per-fold scaling
- Cached features, so iterating on the model takes seconds instead of minutes
- Confusion matrix, classification report and feature-importance dumps on every run

## Results

Cross-validated on 7,339 training epochs from the hackathon dataset, held out **by participant** (a person never appears in both train and validation).

| Setup | CV accuracy |
|---|---|
| Chance level (4 balanced classes) | 25.0% |
| **Log-covariance features + XGBoost fold ensemble (submitted)** | **43.4%** |

Bandpower-only features scored lower under the same split, so `--feature_set auto` rejected them.

Per-class scores from the same run (`cv_confusion_matrix.txt`):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| both_feet | 0.45 | 0.43 | 0.44 |
| both_hands | 0.41 | 0.38 | 0.40 |
| left_hand | 0.45 | 0.48 | 0.46 |
| right_hand | 0.42 | 0.44 | 0.43 |

For context: cross-subject, calibration-free 4-class motor imagery is one of the harder BCI settings. Published cross-subject baselines on comparable data usually sit in the 40–60% range; within-subject models with per-person calibration score far higher but do not transfer to a new user.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/elchibek5/Rice-NeuroWave-Hackathon-2026.git
cd Rice-NeuroWave-Hackathon-2026
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `xgboost`, `joblib`. Nothing else.

## Quickstart

```bash
python3 run_pipeline.py \
  --train_dir path/to/Training \
  --eval_dir  path/to/Evaluation \
  --out_dir   predictions \
  --model_out model.joblib
```

This runs 5-fold participant-grouped CV, prints fold accuracies, trains the ensemble, writes one `y_pred_<id>.npy` per evaluation participant and saves the fitted models to `model.joblib`.

To package a submission for the hackathon:

```bash
zip -r predictions.zip predictions
```

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--train_dir` | required | Folder containing `X_train_<id>.npy` and `y_train_<id>.npy` pairs |
| `--eval_dir` | required | Folder containing `X_eval_<id>.npy` (or `X_test_<id>.npy`) |
| `--out_dir` | `predictions` | Where predictions are written |
| `--model_out` | `model.joblib` | Where the fitted model bundle is saved |
| `--feature_set` | `auto` | `auto` tries `bandpower` and `covlog` and keeps the better CV score. `combo` concatenates both. |
| `--cache_dir` | `cache_features` | Per-participant feature cache. Delete it after changing feature code. |
| `--seed` | `42` | Random seed for XGBoost and fold assignment |

### Merging several training drops

If your data arrives in more than one folder:

```bash
python3 merge_training.py --out_train_dir Training_merged --train_dirs Drop1/ Drop2/
```

Existing files are never overwritten.

## Using it on your own EEG data

The pipeline only assumes the file layout below. To adapt it:

1. Epoch your recordings into arrays of shape `(n_epochs, n_channels, n_samples)` and save one `X_train_<id>.npy` / `y_train_<id>.npy` pair per participant. Labels can be any strings or integers; a `LabelEncoder` handles them.
2. Set `FS` at the top of `run_pipeline.py` to your sampling rate (default 160 Hz). The bandpower bands are defined in Hz and the Welch window is 256 samples, so other rates work as long as `n_samples >= 256`.
3. Channel count is inferred from the data. Log-covariance features scale as `n_channels * (n_channels + 1) / 2`, so 64 channels gives 2,080 features and 22 channels gives 253.
4. Run with `--feature_set auto` first; check `cv_confusion_matrix.txt` before trusting anything.

Class counts do not need to be balanced, but the CV accuracy reported is plain accuracy, so read the per-class report for imbalanced data.

```
Training/
  X_train_<id>.npy   (N, channels, samples) float
  y_train_<id>.npy   (N,)                   labels
Evaluation/
  X_eval_<id>.npy    (N, channels, samples)   # X_test_<id>.npy also accepted
```

Data is git-ignored. Never commit `.npy` files.

## How it works

```
X_train_<id>.npy ──┐
                   ├─▶ features ─▶ StandardScaler ─▶ XGBoost × 5 folds ─▶ mean probs ─▶ argmax ─▶ y_pred_<id>.npy
X_eval_<id>.npy  ──┘
```

**Input.** Each hackathon epoch is 64 channels × 656 samples at 160 Hz (≈ 4.1 s). Four imagined movements.

**Features.** Two families, chosen automatically by CV:

- *Bandpower* (14 per channel, 896 total) — mean, std, peak-to-peak, RMS, log power and relative power in the delta (1–4 Hz), theta (4–8), mu (8–12), beta (12–30) and gamma (30–45) bands, from a Welch PSD.
- *Log-covariance* (2,080 total) — the channel covariance matrix of each epoch, shrunk 10% toward a scaled identity for numerical stability, mapped through the matrix logarithm via eigendecomposition, upper triangle flattened. This is the log-Euclidean form of the Riemannian approach that dominates motor-imagery benchmarks: the *relationships between channels* carry more information than any single channel's spectrum. It won here too.

**Validation.** `GroupKFold(n_splits=5)` with participant ID as the group. Splitting by epoch would let the model memorise each person's EEG signature and inflate accuracy by tens of points; the number above is the honest cross-subject figure.

**Model.** XGBoost `multi:softprob`, 800 trees, depth 6, learning rate 0.05, `hist` tree method. One model per fold, each with its own `StandardScaler`, plus one model fit on all data.

**Prediction.** Every fold model scores the evaluation epochs; probabilities are averaged and the argmax taken. Averaging over folds is free variance reduction that needs no extra tuning.

## Output files

| File | What it is |
|---|---|
| `predictions/y_pred_<id>.npy` | One per evaluation participant, shape `(N, 1)`, original label strings |
| `model.joblib` | Dict with `fold_models` (model + scaler each), `full_model`, `full_scaler`, `label_encoder`, `feature_set`, CV stats |
| `cv_confusion_matrix.txt` | Confusion matrix and sklearn classification report from CV |
| `feature_importance_top20.txt` | Top XGBoost features by gain, with a key for decoding feature indices back to channel / band |
| `cache_features/` | Compressed per-participant feature matrices, keyed by feature set |

Re-scoring new data without retraining:

```python
import joblib, numpy as np
from run_pipeline import featurize_subject

b = joblib.load("model.joblib")
X = np.load("X_eval_95.npy")
F = featurize_subject(X, b["feature_set"])
probs = np.mean([m["model"].predict_proba(m["scaler"].transform(F)) for m in b["fold_models"]], axis=0)
labels = b["label_encoder"].inverse_transform(probs.argmax(1))
```

## Project layout

```
run_pipeline.py                 feature extraction, CV, training, prediction (single file)
merge_training.py               merge multiple training folders
requirements.txt                numpy, scipy, scikit-learn, xgboost, joblib
results.txt                     one-line summary of the submitted run
cv_confusion_matrix.txt         CV diagnostics from the submitted run
feature_importance_top20.txt    feature importances from the submitted run
CITATION.cff                    citation metadata
```

## Roadmap / ideas for contributors

Pull requests welcome. Things that would most likely move the number:

- **Tangent-space projection** around the Riemannian mean (as in [pyRiemann](https://pyriemann.readthedocs.io/)) instead of a plain log-map — typically a few points over log-Euclidean.
- **Per-subject re-centering**: whiten each participant's covariances by their own mean covariance so all subjects share a reference point. The single most effective cross-subject trick in the literature.
- **Filter-bank CSP** as a third feature family, selectable through `--feature_set`.
- **Seed × fold ensembling** once features are settled.
- A `--fs` and `--nperseg` flag so no code edit is needed for other sampling rates.

If you open a PR, please include the `cv_confusion_matrix.txt` from your run so results are comparable.

## Citation

If this code is useful in your research, please cite it (GitHub's *Cite this repository* button uses `CITATION.cff`):

```bibtex
@software{dastanov2026eegmi,
  author = {Dastanov, Elchibek},
  title  = {EEG Motor Imagery Classification with Log-Covariance Features and XGBoost},
  year   = {2026},
  url    = {https://github.com/elchibek5/Rice-NeuroWave-Hackathon-2026}
}
```

## Related

[PaperPilot-Gemini](https://github.com/elchibek5/PaperPilot-Gemini) — a research copilot built alongside this project: summarises PDF papers, extracts claims / method / results, drafts a replication plan and answers questions with citations.

## License

[MIT](LICENSE). Use it, fork it, ship it.

---

*Keywords: EEG, motor imagery, brain-computer interface, BCI, neurotechnology, Riemannian geometry, covariance features, log-Euclidean, XGBoost, scikit-learn, cross-subject, GroupKFold, Python, hackathon.*
