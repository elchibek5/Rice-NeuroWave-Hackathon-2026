import argparse
import os
import re
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone
from xgboost import XGBClassifier
import joblib


FS = 160  # Hz
EPS = 1e-10


# ----------------------------
# Feature helpers
# ----------------------------

# gets bandpower in a frequency range using Welch PSD
def bandpower(sig: np.ndarray, low: float, high: float, fs: int = FS) -> float:
    """Integrate PSD over [low, high] using Welch."""
    freqs, psd = welch(sig, fs=fs, nperseg=256)
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0


# makes basic per-channel stats + EEG bandpower features
def features_bandpower(epoch: np.ndarray) -> np.ndarray:
    """
    epoch: (64, 656)
    Per-channel: mean, std, ptp, rms + log bandpower (delta/theta/mu/beta/gamma)
    + relative bandpower (same bands). 14 feats/channel => 896 total.
    """
    feats = []
    bands = [
        ("delta", 1, 4),
        ("theta", 4, 8),
        ("mu",    8, 12),
        ("beta",  12, 30),
        ("gamma", 30, 45),
    ]

    for ch in range(epoch.shape[0]):
        sig = epoch[ch].astype(np.float64)
        sig = (sig - sig.mean()) / (sig.std() + 1e-12)

        mean = float(sig.mean())
        std  = float(sig.std())
        ptp  = float(sig.max() - sig.min())
        rms  = float(np.sqrt(np.mean(sig * sig)))

        total = bandpower(sig, 1, 45) + 1e-12

        log_bp = []
        rel_bp = []
        for _, lo, hi in bands:
            p = bandpower(sig, lo, hi)
            log_bp.append(float(np.log(p + 1e-12)))
            rel_bp.append(float(p / total))

        feats.extend([mean, std, ptp, rms, *log_bp, *rel_bp])

    return np.array(feats, dtype=np.float32)


# EEG classic baseline: log-cov features from covariance matrix (SPD)
def features_covlog(epoch: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """
    Strong EEG baseline: log of channel covariance (SPD) + upper-triangle vectorization.

    Steps:
    - epoch shape: (64, 656)
    - compute covariance across time (channels x channels)
    - shrink towards scaled identity for stability
    - log-map via eigen decomposition
    - vectorize upper triangle (including diag): 64*65/2 = 2080 features
    """
    X = epoch.astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)

    # Covariance: channels x channels
    C = (X @ X.T) / (X.shape[1] - 1)

    # Shrinkage to improve conditioning (very important for EEG)
    tr = np.trace(C)
    n = C.shape[0]
    C_shrunk = (1 - shrink) * C + shrink * (tr / n) * np.eye(n)

    # Ensure SPD
    C_shrunk = C_shrunk + EPS * np.eye(n)

    # Log-map: log(C) via eigenvalues
    w, V = np.linalg.eigh(C_shrunk)
    w = np.clip(w, EPS, None)
    logC = V @ np.diag(np.log(w)) @ V.T

    # Upper triangle vector (including diag)
    iu = np.triu_indices(n)
    v = logC[iu].astype(np.float32)

    # Optional: normalize scale to reduce subject variance
    # v = v / (np.linalg.norm(v) + 1e-12)

    return v


# converts subject epochs into feature matrix depending on feature_set
def featurize_subject(X: np.ndarray, feature_set: str) -> np.ndarray:
    """X: (N, 64, 656) -> (N, D)"""
    if feature_set == "bandpower":
        return np.vstack([features_bandpower(X[i]) for i in range(X.shape[0])])
    elif feature_set == "covlog":
        return np.vstack([features_covlog(X[i]) for i in range(X.shape[0])])
    elif feature_set == "combo":
        return np.vstack([
            np.concatenate([features_bandpower(X[i]), features_covlog(X[i])]).astype(np.float32)
            for i in range(X.shape[0])
        ])
    else:
        raise ValueError(f"Unknown feature_set={feature_set}")


# ----------------------------
# Data loading + caching
# ----------------------------

# finds all participant IDs that have X_train_*.npy files
def list_participant_ids(train_dir: str):
    ids = []
    for f in os.listdir(train_dir):
        m = re.match(r"X_train_(\d+)\.npy$", f)
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=lambda s: int(s))


# finds all eval/test IDs for X_test_ or X_eval_ files
def list_eval_ids(eval_dir: str):
    ids = []
    for f in os.listdir(eval_dir):
        m = re.match(r"X_(?:test|eval)_(\d+)\.npy$", f)  # supports X_test_ and X_eval_
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=lambda s: int(s))


# loads training data, featurizes it, and optionally caches features per participant
def load_training(train_dir: str, feature_set: str, cache_dir: str | None):
    X_list, y_list, groups = [], [], []
    ids = list_participant_ids(train_dir)
    if not ids:
        raise FileNotFoundError(f"No X_train_*.npy files found in {train_dir}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for pid in ids:
        x_path = os.path.join(train_dir, f"X_train_{pid}.npy")
        y_path = os.path.join(train_dir, f"y_train_{pid}.npy")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing label file: {y_path}")

        # cache per participant + feature_set
        cache_path = None
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"feats_{feature_set}_{pid}.npz")

        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            feats = z["X"]
        else:
            X = np.load(x_path, allow_pickle=True)
            feats = featurize_subject(X, feature_set)
            if cache_path:
                np.savez_compressed(cache_path, X=feats)

        y = np.load(y_path, allow_pickle=True)
        y = np.asarray(y).reshape(-1)

        if feats.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch pid={pid}: epochs={feats.shape[0]} vs y={y.shape[0]}")

        X_list.append(feats)
        y_list.append(y)
        groups.extend([pid] * len(y))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups = np.array(groups)
    return X_all, y_all, groups


# ----------------------------
# Training / CV
# ----------------------------

# builds the XGBoost classifier with fixed params
def make_model(num_class: int, seed: int, n_estimators: int):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=num_class,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )


# trains with GroupKFold (split by participant so no leakage)
def train_with_group_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42, n_estimators: int = 800):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    base = make_model(num_class=len(le.classes_), seed=seed, n_estimators=n_estimators)

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_models = []
    scores = []
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y_enc, groups), start=1):
        scaler = StandardScaler(with_mean=True, with_std=True)

        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        m = clone(base)
        m.fit(X_tr, y_enc[tr])

        pred = np.argmax(m.predict_proba(X_te), axis=1)
        acc = accuracy_score(y_enc[te], pred)

        scores.append(acc)
        all_true.append(y_enc[te])
        all_pred.append(pred)

        # store model + scaler for ensembling later
        fold_models.append({"model": m, "scaler": scaler})
        print(f"Fold {fold} accuracy: {acc:.4f}")

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    print(f"Mean CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    cm = confusion_matrix(y_true, y_pred)
    with open("cv_confusion_matrix.txt", "w") as f:
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Labels order:\n")
        f.write(str(list(le.classes_)) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=le.classes_))
        f.write("\n")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("Saved cv_confusion_matrix.txt")

    # Train full model on all data (with scaler)
    full_scaler = StandardScaler(with_mean=True, with_std=True)
    X_full = full_scaler.fit_transform(X)
    full_model = clone(base)
    full_model.fit(X_full, y_enc)

    return {
        "fold_models": fold_models,
        "full_model": full_model,
        "full_scaler": full_scaler,
        "label_encoder": le,
        "cv_mean_acc": mean_acc,
        "cv_std_acc": std_acc,
    }


# dumps top feature importances to a txt file (mostly for debugging/insight)
def compute_feature_importance(bundle, feature_set: str):
    """Save top-20 gain importances + explain feature indexing."""
    try:
        model = bundle["full_model"]
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")  # {"f0": val, ...}

        items = []
        for k, v in score.items():
            idx = int(k[1:])
            items.append((idx, float(v)))
        items.sort(key=lambda t: t[1], reverse=True)
        top = items[:20]

        lines = ["Top 20 feature importances (gain):"]
        for idx, val in top:
            lines.append(f"f{idx}: {val:.6f}")

        with open("feature_importance_top20.txt", "w") as f:
            f.write("\n".join(lines) + "\n\n")
            f.write(f"Feature set: {feature_set}\n\n")
            if feature_set in ("bandpower", "combo"):
                f.write("Bandpower layout per channel (14 feats):\n")
                f.write("0 mean, 1 std, 2 ptp, 3 rms,\n")
                f.write("4 log_delta, 5 log_theta, 6 log_mu, 7 log_beta, 8 log_gamma,\n")
                f.write("9 rel_delta, 10 rel_theta, 11 rel_mu, 12 rel_beta, 13 rel_gamma\n")
                f.write("Channel index = feature_index // 14\n")
                f.write("Feature-in-channel index = feature_index % 14\n\n")
            if feature_set in ("covlog", "combo"):
                f.write("CovLog layout:\n")
                f.write("Upper-triangle of 64x64 log-covariance (including diag): 2080 feats.\n")
                f.write("Indexing corresponds to np.triu_indices(64).\n")

        print("\n" + "\n".join(lines))
        print("Saved feature_importance_top20.txt")
    except Exception as e:
        print("Could not compute feature importance:", e)


# ----------------------------
# Prediction
# ----------------------------

# runs model(s) on eval/test files and saves y_pred_*.npy outputs
def predict_eval(bundle, eval_dir: str, out_dir: str, feature_set: str, cache_dir: str | None):
    os.makedirs(out_dir, exist_ok=True)
    le = bundle["label_encoder"]
    fold_models = bundle["fold_models"]
    full_model = bundle["full_model"]
    full_scaler = bundle["full_scaler"]

    ids = list_eval_ids(eval_dir)
    if not ids:
        raise FileNotFoundError(f"No X_eval_*.npy or X_test_*.npy files found in {eval_dir}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for pid in ids:
        x_test = os.path.join(eval_dir, f"X_test_{pid}.npy")
        x_eval = os.path.join(eval_dir, f"X_eval_{pid}.npy")
        x_path = x_test if os.path.exists(x_test) else x_eval

        # cache eval features too
        cache_path = None
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"feats_eval_{feature_set}_{pid}.npz")

        if cache_path and os.path.exists(cache_path):
            feats = np.load(cache_path, allow_pickle=True)["X"]
        else:
            X = np.load(x_path, allow_pickle=True)
            feats = featurize_subject(X, feature_set)
            if cache_path:
                np.savez_compressed(cache_path, X=feats)

        # ensemble over folds (each has its scaler)
        probs = None
        for obj in fold_models:
            m = obj["model"]
            sc = obj["scaler"]
            p = m.predict_proba(sc.transform(feats))
            probs = p if probs is None else (probs + p)
        probs /= len(fold_models)

        pred_enc = np.argmax(probs, axis=1)

        # safety fallback
        if pred_enc.shape[0] != feats.shape[0]:
            pred_enc = np.argmax(full_model.predict_proba(full_scaler.transform(feats)), axis=1)

        pred_labels = le.inverse_transform(pred_enc).astype(object).reshape(-1, 1)
        np.save(os.path.join(out_dir, f"y_pred_{pid}.npy"), pred_labels)
        print(f"Saved y_pred_{pid}.npy with shape {pred_labels.shape}")


# ----------------------------
# Main
# ----------------------------

# CLI entry: chooses best feature set by CV, saves model bundle, then writes predictions
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--model_out", default="model.joblib")
    ap.add_argument("--cache_dir", default="cache_features")
    ap.add_argument("--feature_set", default="auto", choices=["auto", "bandpower", "covlog", "combo"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Decide feature set
    candidates = ["bandpower", "covlog"] if args.feature_set == "auto" else [args.feature_set]

    best_bundle = None
    best_feat = None
    best_acc = -1.0

    for feat in candidates:
        print(f"\n=== Trying feature_set: {feat} ===")
        print("Loading + featurizing training data...")
        X, y, groups = load_training(args.train_dir, feat, args.cache_dir)
        print(f"Training matrix: {X.shape}, labels: {y.shape}, unique participants: {len(np.unique(groups))}")

        print("Training with participant-safe GroupKFold CV...")
        bundle = train_with_group_cv(X, y, groups, seed=args.seed, n_estimators=800)

        if bundle["cv_mean_acc"] > best_acc:
            best_acc = bundle["cv_mean_acc"]
            best_bundle = bundle
            best_feat = feat

    print(f"\n=== Selected feature_set: {best_feat} (CV mean acc={best_acc:.4f}) ===")

    compute_feature_importance(best_bundle, best_feat)

    joblib.dump({"feature_set": best_feat, **best_bundle}, args.model_out)
    print(f"Saved model bundle to {args.model_out}")

    print("Generating predictions for evaluation set...")
    predict_eval(best_bundle, args.eval_dir, args.out_dir, best_feat, args.cache_dir)

    print("\nDone. Zip the files in your out_dir and submit them.")


if __name__ == "__main__":
    main()
