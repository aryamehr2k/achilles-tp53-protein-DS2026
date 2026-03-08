import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from src.baseline_model import MLPRegressor
from src.metrics import kfold_indices, spearmanr_np
from src.seq_features import build_or_load_features


# -------------------------
# Reproducibility
# -------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

CSV_PATH = "urn_mavedb_00001234-a-1_scores.csv"
PDB_PATH = "data/structures/TP53_RCSB.pdb"
CHAIN_ID = "A"

BATCH_EPOCHS = 120
LR = 1e-3
WD = 1e-4
PATIENCE = 15


def train_fold(X, y, train_idx, test_idx):
    Xtr = torch.tensor(X[train_idx], dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y[train_idx], dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X[test_idx], dtype=torch.float32, device=DEVICE)
    yte = torch.tensor(y[test_idx], dtype=torch.float32, device=DEVICE)

    model = MLPRegressor(in_dim=X.shape[1], hid=256, dropout=0.15).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    best_sp = -1e9
    best_state = None
    bad = 0

    for _ in range(BATCH_EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = model(Xte).detach().cpu().numpy()
            t = yte.detach().cpu().numpy()
        sp = spearmanr_np(p, t)

        if sp > best_sp:
            best_sp = sp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        p = model(Xte).detach().cpu().numpy()
        t = yte.detach().cpu().numpy()

    return spearmanr_np(p, t)


def main():
    X, y, meta = build_or_load_features(CSV_PATH, PDB_PATH, CHAIN_ID, SEED)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(y), 5, SEED), start=1):
        print(f"\n===== Fold {fold} =====")
        sp = train_fold(X, y, train_idx, test_idx)
        print(f"Fold {fold} Spearman: {sp:.4f}")
        scores.append(sp)

    scores = np.array(scores, dtype=np.float32)
    print("\n==============================")
    print("Sequence Baseline 5-Fold CV (ΔESM mean + MLP)")
    print(f"Mean Spearman: {scores.mean():.4f}")
    print(f"Std  Spearman: {scores.std():.4f}")
    print("==============================")

    out = {
        "model": "delta_esm_mean_mlp",
        "mean_spearman": float(scores.mean()),
        "std_spearman": float(scores.std()),
        "meta": meta,
    }
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/seq_cv_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: checkpoints/seq_cv_result.json")


if __name__ == "__main__":
    main()
