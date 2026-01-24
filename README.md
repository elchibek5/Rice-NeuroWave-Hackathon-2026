# Neurotech Hackathon: EEG Motor Imagery Classification

This project trains a model on EEG epochs and generates `.npy` prediction files for the evaluation set in the required submission format.

---

## Goal
Given EEG epochs shaped like `(N, 64, 656)` (N epochs, 64 channels, 656 time samples), predict the motor-imagery class label for each epoch.

---

## Data Layout (expected)

### Training directory
Contains paired files per participant:
- `X_train_<id>.npy` (shape: `(N, 64, 656)`)
- `y_train_<id>.npy` (shape: `(N,)`)

Example:
- `X_train_12.npy`
- `y_train_12.npy`

### Evaluation directory
Contains:
- `X_eval_<id>.npy` (shape: `(N, 64, 656)`)

Example:
- `X_eval_95.npy`

> Note: The pipeline supports both `X_eval_<id>.npy` and `X_test_<id>.npy` naming.

---

## Project Structure
- `run_pipeline.py` — main training + prediction script
- `predictions/` — generated outputs (`y_pred_*.npy`)
- `predictions.zip` — submission artifact (zip this and submit)
- `cv_confusion_matrix.txt` — GroupKFold validation diagnostics
- `feature_importance_top20.txt` — top features by XGBoost gain

---

## Method Summary

### Validation (leakage-safe)
We use **GroupKFold** where each participant ID is a group.  
This prevents training and validation from sharing data from the same person (a common source of leakage in EEG tasks).

### Features
We compute two feature families from each epoch:

1) **Bandpower feature set**
- Per-channel statistics: mean, std, peak-to-peak, RMS  
- Log bandpower in EEG bands: delta/theta/mu/beta/gamma  
- Relative bandpower per band (normalized by total power 1–45 Hz)

2) **CovLog feature set (stronger)**
- Covariance matrix per epoch  
- Log transform and vectorization  
- Captures cross-channel relationships (often critical in motor imagery EEG)

### Model
- **XGBoost** multiclass classifier (`multi:softprob`)

### Ensemble
- Train one model per fold  
- Predict probabilities per fold model  
- Average fold probabilities to get final predicted class per epoch  

---

## Results Snapshot (Local CV)
CovLog features improved GroupKFold accuracy compared to bandpower-only features.

Artifacts produced after training:
- `cv_confusion_matrix.txt`
- `feature_importance_top20.txt`

---

## How to Run

### 1) Activate environment
    source .venv/bin/activate

### 2) Train + Predict
Replace paths:

    python3 run_pipeline.py \
      --train_dir "/Users/elchibekdastanov/Downloads/Training" \
      --eval_dir "/Users/elchibekdastanov/Downloads/Evaluation" \
      --out_dir predictions \
      --model_out model.joblib

### 3) Zip submission
    zip -r predictions.zip predictions

---

## Submission Format
The output directory contains one file per evaluation subject:

- `y_pred_<id>.npy` with shape `(N, 1)`

Example:
- `predictions/y_pred_95.npy`
