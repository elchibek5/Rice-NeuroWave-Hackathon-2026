import argparse
import os
import re
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
from sklearn.base import clone


FS = 160  # Hz

# calculates signal power in a specific frequency band (Welch PSD)
def bandpower(sig: np.ndarray, low: float, high: float, fs: int = FS) -> float:
    """Integrate PSD over [low, high] using Welch."""
    freqs, psd = welch(sig, fs=fs, nperseg=256)
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0

# extracts 896 features from one EEG epoch (64 channels) using stats + bandpowers
def extract_features_epoch(epoch: np.ndarray) -> np.ndarray:
    """
    epoch: (64, 656)
    Features per channel:
      - mean, std, peak-to-peak, rms
      - log bandpower for: delta/theta/mu/beta/gamma
      - relative bandpower (band / total 1–45 Hz) for same bands

    returns: (64 * (4 + 5 + 5),) = (64 * 14,) = 896 features
    """
    feats = []
    bands = [
        ("delta", 1, 4),
        ("theta", 4, 8),
        ("mu",    8, 12),
        ("beta",  12, 30),
        ("gamma", 30, 45),
    ]

    eps = 1e-12

    for ch in range(epoch.shape[0]):
        sig = epoch[ch].astype(np.float64)
        sig = (sig - sig.mean()) / (sig.std() + 1e-12)

        mean = float(sig.mean())
        std  = float(sig.std())
        ptp  = float(sig.max() - sig.min())
        rms  = float(np.sqrt(np.mean(sig * sig)))

        total = bandpower(sig, 1, 45) + eps

        log_bp = []
        rel_bp = []
        for _, lo, hi in bands:
            p = bandpower(sig, lo, hi)
            log_bp.append(float(np.log(p + eps)))
            rel_bp.append(float(p / total))

        feats.extend([mean, std, ptp, rms, *log_bp, *rel_bp])

    return np.array(feats, dtype=np.float32)


# converts a whole subject's epochs into a big 2D feature matrix (N epochs x D features)
def featurize_subject(X: np.ndarray) -> np.ndarray:
    """X: (N, 64, 656) -> (N, D)"""
    return np.vstack([extract_features_epoch(X[i]) for i in range(X.shape[0])])

# gets participant IDs from filenames like X_train_123.npy
def list_participant_ids(train_dir: str):
    ids = []
    for f in os.listdir(train_dir):
        m = re.match(r"X_train_(\d+)\.npy$", f)
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=lambda s: int(s))

# loads all training participants, featurizes them, and builds group labels for GroupKFold
def load_training(train_dir: str):
    X_list, y_list, groups = [], [], []
    ids = list_participant_ids(train_dir)
    if not ids:
        raise FileNotFoundError(f"No X_train_*.npy files found in {train_dir}")

    for pid in ids:
        x_path = os.path.join(train_dir, f"X_train_{pid}.npy")
        y_path = os.path.join(train_dir, f"y_train_{pid}.npy")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing label file: {y_path}")

        X = np.load(x_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)
        y = np.asarray(y).reshape(-1)

        feats = featurize_subject(X)
        if feats.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch pid={pid}: X epochs={feats.shape[0]} vs y={y.shape[0]}")

        X_list.append(feats)
        y_list.append(y)
        groups.extend([pid] * len(y))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups = np.array(groups)
    return X_all, y_all, groups

# trains XGBoost using GroupKFold so subjects don't leak between train/test folds
def train_with_group_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    base = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(le.classes_),
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_models = []
    scores = []

    for fold, (tr, te) in enumerate(gkf.split(X, y_enc, groups), start=1):
        m = clone(base)
        m.fit(X[tr], y_enc[tr])
        proba = m.predict_proba(X[te])
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_enc[te], pred)
        scores.append(acc)
        fold_models.append(m)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    print(f"Mean CV accuracy: {float(np.mean(scores)):.4f} ± {float(np.std(scores)):.4f}")

    full_model = clone(base)
    full_model.fit(X, y_enc)

    return {"fold_models": fold_models, "full_model": full_model, "label_encoder": le}

# finds eval/test IDs from X_test_*.npy or X_eval_*.npy
def list_eval_ids(eval_dir: str):
    ids = []
    for f in os.listdir(eval_dir):
        m = re.match(r"X_(?:test|eval)_(\d+)\.npy$", f)  # supports X_test_ and X_eval_
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=lambda s: int(s))

# runs ensemble predictions for each participant in eval folder and saves y_pred_*.npy
def predict_eval(bundle, eval_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    le = bundle["label_encoder"]
    fold_models = bundle["fold_models"]
    full_model = bundle["full_model"]

    ids = list_eval_ids(eval_dir)
    if not ids:
        raise FileNotFoundError(f"No X_eval_*.npy or X_test_*.npy files found in {eval_dir}")

    for pid in ids:
        x_test = os.path.join(eval_dir, f"X_test_{pid}.npy")
        x_eval = os.path.join(eval_dir, f"X_eval_{pid}.npy")
        x_path = x_test if os.path.exists(x_test) else x_eval

        X = np.load(x_path, allow_pickle=True)
        feats = featurize_subject(X)

        probs = None
        for m in fold_models:
            p = m.predict_proba(feats)
            probs = p if probs is None else (probs + p)
        probs /= len(fold_models)

        pred_enc = np.argmax(probs, axis=1)

        if pred_enc.shape[0] != feats.shape[0]:
            pred_enc = np.argmax(full_model.predict_proba(feats), axis=1)

        pred_labels = le.inverse_transform(pred_enc).astype(object).reshape(-1, 1)
        np.save(os.path.join(out_dir, f"y_pred_{pid}.npy"), pred_labels)
        print(f"Saved y_pred_{pid}.npy with shape {pred_labels.shape}")


# CLI entry: loads data, trains model, saves bundle, then writes predictions to out_dir
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--model_out", default="model.joblib")
    args = ap.parse_args()

    print("Loading + featurizing training data...")
    X, y, groups = load_training(args.train_dir)
    print(f"Training matrix: {X.shape}, labels: {y.shape}, unique participants: {len(np.unique(groups))}")

    print("Training with participant-safe GroupKFold CV...")
    bundle = train_with_group_cv(X, y, groups)

    joblib.dump(bundle, args.model_out)
    print(f"Saved model bundle to {args.model_out}")

    print("Generating predictions for evaluation set...")
    predict_eval(bundle, args.eval_dir, args.out_dir)

    print("\nDone. Zip the files in your out_dir and submit them.")

if __name__ == "__main__":
    main()
