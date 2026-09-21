# Rice NeuroWave Hackathon 2026 — EEG Motor Imagery Classification

Predict which movement a person is *imagining* (left hand, right hand, both hands, both feet) from a 4-second, 64-channel EEG epoch.

The pipeline is one script. It extracts features, validates with participant-safe cross-validation, trains an XGBoost fold ensemble, and writes the `predictions.zip` the hackathon expects.

## Results

Cross-validated on 7,339 training epochs, held out **by participant** (a person never appears in both train and validation).

| Metric | Value |
|---|---|
| Chance level (4 classes) | 25.0% |
| **Log-covariance features + XGBoost (submitted)** | **43.4%** |

Bandpower-only features scored lower under the same split, so the `auto` mode rejected them.

Per-class F1 from the same run (`cv_confusion_matrix.txt`):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| both_feet | 0.45 | 0.43 | 0.44 |
| both_hands | 0.41 | 0.38 | 0.40 |
| left_hand | 0.45 | 0.48 | 0.46 |
| right_hand | 0.42 | 0.44 | 0.43 |

Cross-subject motor imagery is a hard problem; published cross-subject baselines on similar 4-class EEG data typically land in the 40–60% range without per-subject calibration.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 run_pipeline.py \
  --train_dir path/to/Training \
  --eval_dir  path/to/Evaluation \
  --out_dir   predictions \
  --model_out model.joblib

zip -r predictions.zip predictions
```

Submit `predictions.zip`.

Features are cached per participant in `cache_features/`, so the second run skips the slow part.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--feature_set` | `auto` | `auto` tries `bandpower` and `covlog`, keeps the better CV score. Also accepts `combo` (both concatenated). |
| `--seed` | `42` | Random seed for XGBoost and fold assignment. |
| `--cache_dir` | `cache_features` | Where per-participant feature matrices are stored. |

### Merging several training drops

If the organisers ship training data in more than one folder:

```bash
python3 merge_training.py --out_train_dir Training_merged --train_dirs Drop1/ Drop2/
```

Existing files are never overwritten.

## How it works

```
X_train_<id>.npy  (N, 64, 656) ──┐
                                 ├─▶ features ──▶ StandardScaler ──▶ XGBoost ×5 folds ──▶ avg probs ──▶ y_pred_<id>.npy
X_eval_<id>.npy   (N, 64, 656) ──┘
```

**Input.** Each epoch is 64 channels × 656 samples at 160 Hz (≈4.1 s). Labels are the four imagined movements.

**Features.** Two families, chosen automatically by CV:

- *Bandpower* (896 dims) — per channel: mean, std, peak-to-peak, RMS, log power and relative power in delta / theta / mu / beta / gamma bands (Welch PSD).
- *Log-covariance* (2,080 dims) — the 64×64 channel covariance of the epoch, shrunk toward the identity for stability, mapped through the matrix logarithm, upper triangle flattened. This is the classic Riemannian-geometry trick for motor imagery: the *relationships between channels* carry more signal than any channel on its own. It won.

**Validation.** `GroupKFold(5)` with participant ID as the group. Splitting by epoch instead would leak a person's signature between train and validation and inflate accuracy. The number reported above is the honest one.

**Model.** XGBoost (`multi:softprob`, 800 trees, depth 6, lr 0.05, `hist`). One model per fold, each with its own scaler.

**Prediction.** All five fold models score the evaluation epochs; probabilities are averaged and the argmax is taken. Averaging over folds is cheaper than tuning and reliably smooths variance.

## Outputs

| File | What it is |
|---|---|
| `predictions/y_pred_<id>.npy` | One file per evaluation participant, shape `(N, 1)`, string labels |
| `predictions.zip` | The submission artifact |
| `model.joblib` | Fold models, scalers and label encoder, for re-scoring without retraining |
| `cv_confusion_matrix.txt` | Confusion matrix and classification report from CV |
| `feature_importance_top20.txt` | Top XGBoost features by gain, with an index-decoding key |

## Data layout

```
Training/
  X_train_<id>.npy   (N, 64, 656) float
  y_train_<id>.npy   (N,)         labels
Evaluation/
  X_eval_<id>.npy    (N, 64, 656)   # X_test_<id>.npy also accepted
```

Data is git-ignored. Never commit `.npy` files.

## Project layout

```
run_pipeline.py                 feature extraction, CV, training, prediction
merge_training.py               merge multiple training folders
requirements.txt                numpy, scipy, scikit-learn, xgboost, joblib
results.txt                     one-line summary of the submitted run
cv_confusion_matrix.txt         CV diagnostics from the submitted run
feature_importance_top20.txt    feature importance from the submitted run
```

## What I would do next

- Tangent-space projection of the covariance matrices (proper Riemannian mean) instead of a plain log-map — usually a few points over log-euclidean.
- Per-subject covariance re-centering (align each participant's mean covariance to identity) to shrink the cross-subject gap.
- Spatial filtering (CSP / filter-bank CSP) as a third feature family.
- Replace the fold ensemble with a proper seed × fold ensemble once features are fixed.

## Related

[PaperPilot-Gemini](https://github.com/elchibek5/PaperPilot-Gemini) — a research copilot I built alongside this: summarises PDF papers, extracts claims / method / results, drafts a replication plan, and answers questions with citations.
