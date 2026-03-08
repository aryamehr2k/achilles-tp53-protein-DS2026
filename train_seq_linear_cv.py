import os
import json
import random
import numpy as np

from src.metrics import kfold_indices, spearmanr_np
from src.seq_features import build_or_load_features


# -------------------------
# Reproducibility
# -------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CSV_PATH = "urn_mavedb_00001234-a-1_scores.csv"
PDB_PATH = "data/structures/TP53_RCSB.pdb"
CHAIN_ID = "A"

RIDGE_ALPHA = 1e-2


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    I = np.eye(d + 1, dtype=X.dtype)
    I[-1, -1] = 0.0  # do not regularize bias
    A = Xb.T @ Xb + alpha * I
    b = Xb.T @ y
    w = np.linalg.solve(A, b)
    return w


def predict_ridge(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    Xb = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    return Xb @ w


def main():
    X, y, meta = build_or_load_features(CSV_PATH, PDB_PATH, CHAIN_ID, SEED)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(y), 5, SEED), start=1):
        print(f"\n===== Fold {fold} =====")
        w = fit_ridge(X[train_idx], y[train_idx], RIDGE_ALPHA)
        preds = predict_ridge(X[test_idx], w)
        sp = spearmanr_np(preds, y[test_idx])
        print(f"Fold {fold} Spearman: {sp:.4f}")
        scores.append(sp)

    scores = np.array(scores, dtype=np.float32)
    print("\n==============================")
    print("Sequence Baseline 5-Fold CV (ΔESM mean + Ridge)")
    print(f"Mean Spearman: {scores.mean():.4f}")
    print(f"Std  Spearman: {scores.std():.4f}")
    print("==============================")

    out = {
        "model": "delta_esm_mean_ridge",
        "alpha": float(RIDGE_ALPHA),
        "mean_spearman": float(scores.mean()),
        "std_spearman": float(scores.std()),
        "meta": meta,
    }
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/seq_linear_cv_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: checkpoints/seq_linear_cv_result.json")


if __name__ == "__main__":
    main()
