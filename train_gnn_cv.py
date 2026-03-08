import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.dataset import TP53StructureDataset
from src.hgnn import HGNN
from src.metrics import kfold_indices, spearmanr_np


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

BATCH_SIZE = 1
EPOCHS = 120
LR = 1e-3
WD = 1e-4
PATIENCE = 15

OUT_PATH = "checkpoints/gnn_cv_result.json"


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            preds.append(float(out.detach().cpu().item()))
            trues.append(float(batch.y.view(-1).detach().cpu().item()))

    sp = spearmanr_np(np.array(preds), np.array(trues))
    return float(sp)


def train_one_fold(ds, train_idx, test_idx):
    train_subset = torch.utils.data.Subset(ds, train_idx)
    test_subset = torch.utils.data.Subset(ds, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    in_dim = ds.X.shape[1]
    model = HGNN(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    best_sp = -1e9
    bad = 0

    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            opt.step()

        sp = evaluate(model, test_loader)

        if sp > best_sp:
            best_sp = sp
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    return best_sp


def main():
    print("Loading dataset (kNN edges)...")
    ds = TP53StructureDataset(CSV_PATH, PDB_PATH, chain_id=CHAIN_ID, graph_type="knn")

    scores = []
    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(ds), 5, SEED)):
        print(f"\n===== Fold {fold + 1} =====")
        sp = train_one_fold(ds, train_idx, test_idx)
        print(f"Fold {fold + 1} Spearman: {sp:.4f}")
        scores.append(sp)

    scores = np.array(scores)
    mean_sp = float(scores.mean())
    std_sp = float(scores.std())

    print("\n==============================")
    print("GNN (pairwise) 5-Fold CV Result")
    print(f"Mean Spearman: {mean_sp:.4f}")
    print(f"Std  Spearman: {std_sp:.4f}")
    print("==============================")

    os.makedirs("checkpoints", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"mean_spearman": mean_sp, "std_spearman": std_sp}, f, indent=2)

    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
