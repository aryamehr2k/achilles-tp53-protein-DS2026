import os
import numpy as np
import pandas as pd

from src.dataset import parse_hgvs_protein, apply_mutation, AA2I
from src.esm_embed import get_esm2_residue_embeddings
from src.structure import load_ca_coordinates


ESM_MODEL = "esm2_t6_8M_UR50D"  # 320-dim
CACHE_DIR = "cache"
CACHE_PATH = os.path.join(CACHE_DIR, f"seq_delta_{ESM_MODEL}.npz")


def find_columns(df: pd.DataFrame):
    mut_col = None
    for c in ["hgvs_pro", "hgvs", "mutation", "variant", "mutant", "mut"]:
        if c in df.columns:
            mut_col = c
            break
    if mut_col is None:
        for c in df.columns:
            if df[c].astype(str).str.contains(r"p\.", regex=True).any():
                mut_col = c
                break
    if mut_col is None:
        raise ValueError("Mutation column not found.")

    score_col = None
    for c in ["score", "fitness", "effect", "value", "y", "DMS_score"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("Score column not found.")
        score_col = num_cols[-1]

    return mut_col, score_col


def build_or_load_features(
    csv_path: str,
    pdb_path: str,
    chain_id: str,
    seed: int,
):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH, allow_pickle=True)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.float32)
        meta = z["meta"].item()
        print(f"Loaded cached features: {CACHE_PATH}")
        print(f"X: {X.shape} y: {y.shape}")
        return X, y, meta

    print("Building sequence delta features (this will run ESM on each mutation once)...")

    coords, residues = load_ca_coordinates(pdb_path, chain_id=chain_id)
    if len(residues) == 0:
        raise ValueError("No residues found from PDB chain.")
    seq = "".join([r["aa"] for r in residues])
    resseq2idx = {r["resseq"]: i for i, r in enumerate(residues)}

    wt_emb = get_esm2_residue_embeddings(seq, model_name=ESM_MODEL).numpy()  # [L,320]
    L = wt_emb.shape[0]

    df = pd.read_csv(csv_path)
    mut_col, score_col = find_columns(df)

    feats = []
    ys = []

    kept = 0
    for _, row in df.iterrows():
        wt, pos, mut = parse_hgvs_protein(row[mut_col])
        if wt is None or mut is None:
            continue
        if wt not in AA2I or mut not in AA2I:
            continue
        if pos not in resseq2idx:
            continue

        idx0 = resseq2idx[pos]
        if idx0 >= L:
            continue

        if seq[idx0] != wt:
            continue

        mut_seq = apply_mutation(seq, idx0 + 1, mut)
        if mut_seq is None:
            continue
        mut_emb = get_esm2_residue_embeddings(mut_seq, model_name=ESM_MODEL).numpy()
        if mut_emb.shape[0] != L:
            mL = min(mut_emb.shape[0], L)
            mut_emb = mut_emb[:mL]
            wsub = wt_emb[:mL]
        else:
            wsub = wt_emb

        delta = (mut_emb - wsub).mean(axis=0).astype(np.float32)  # [320]
        yv = float(row[score_col])

        feats.append(delta)
        ys.append(yv)
        kept += 1

    if kept == 0:
        raise ValueError("No usable rows after parsing / alignment.")

    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.float32)

    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-8)
    y_n = (y - y_mean) / y_std

    meta = {
        "esm_model": ESM_MODEL,
        "y_mean": y_mean,
        "y_std": y_std,
        "n": int(len(y)),
        "dim": int(X.shape[1]),
        "seed": seed,
    }

    np.savez(CACHE_PATH, X=X, y=y_n, meta=meta)
    print(f"Saved cache: {CACHE_PATH}")
    print(f"X: {X.shape} y: {y_n.shape}")
    print(f"y_mean={y_mean:.6f} y_std={y_std:.6f}")
    return X, y_n.astype(np.float32), meta
