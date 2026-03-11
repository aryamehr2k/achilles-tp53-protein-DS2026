import os
import numpy as np

from src.structure import load_ca_coordinates


AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA20)}
GAP_INDEX = 20


def parse_clustal(path: str):
    seqs = {}
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("CLUSTAL") or line.startswith("MUSCLE"):
                continue
            if line[0].isspace():
                # consensus lines are prefixed with spaces
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, seq = parts[0], parts[1]
            seqs[name] = seqs.get(name, "") + seq
    if not seqs:
        raise ValueError(f"No sequences parsed from {path}")
    return seqs


def find_reference_key(seqs: dict):
    for key in seqs.keys():
        if "P53_HUMAN" in key or "P04637" in key:
            return key
    return list(seqs.keys())[0]


def build_col_to_refpos(ref_aln: str):
    col_to_pos = []
    pos = 0
    for c in ref_aln:
        if c != "-":
            pos += 1
            col_to_pos.append(pos)
        else:
            col_to_pos.append(None)
    return col_to_pos


def map_pdb_residues_to_columns(col_to_pos, residues):
    pos_to_col = {}
    for col, pos in enumerate(col_to_pos):
        if pos is None:
            continue
        pos_to_col[pos] = col
    cols = []
    for r in residues:
        resseq = int(r["resseq"])
        cols.append(pos_to_col.get(resseq))
    return cols


def msa_to_matrix(seqs: dict, cols, q=21):
    names = list(seqs.keys())
    aln_len = len(seqs[names[0]])
    for n in names[1:]:
        if len(seqs[n]) != aln_len:
            raise ValueError("Alignment sequences have different lengths.")

    N = len(cols)
    M = len(names)
    mat = np.full((M, N), GAP_INDEX, dtype=np.int16)

    for i, name in enumerate(names):
        s = seqs[name]
        for j, col in enumerate(cols):
            if col is None:
                mat[i, j] = GAP_INDEX
                continue
            aa = s[col]
            if aa == "-" or aa == ".":
                mat[i, j] = GAP_INDEX
            else:
                mat[i, j] = AA2I.get(aa, GAP_INDEX)
    return mat


def compute_entropy(msa_int: np.ndarray, q=21):
    M, N = msa_int.shape
    ent = np.zeros(N, dtype=np.float32)
    for j in range(N):
        col = msa_int[:, j]
        # ignore gaps
        mask = col != GAP_INDEX
        if mask.sum() == 0:
            ent[j] = 0.0
            continue
        vals = col[mask]
        counts = np.bincount(vals, minlength=q)[:20].astype(np.float32)
        p = counts / counts.sum()
        p = p[p > 0]
        h = -(p * np.log2(p)).sum()
        ent[j] = h / np.log2(20.0)  # normalize to [0,1]
    return ent


def compute_dca_scores(msa_int: np.ndarray, q=21, reg_lambda=0.01):
    M, N = msa_int.shape
    q_eff = q - 1  # exclude gap

    X = np.zeros((M, N * q_eff), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            a = int(msa_int[i, j])
            if a == GAP_INDEX or a >= q_eff:
                continue
            X[i, j * q_eff + a] = 1.0

    P = X.mean(axis=0)
    C = (X.T @ X) / float(M) - np.outer(P, P)
    C = C + reg_lambda * np.eye(C.shape[0], dtype=np.float32)

    invC = np.linalg.inv(C)

    scores = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            block = invC[i * q_eff:(i + 1) * q_eff, j * q_eff:(j + 1) * q_eff]
            s = np.linalg.norm(block, ord="fro")
            scores[i, j] = s
            scores[j, i] = s

    maxv = scores.max() if scores.size else 0.0
    minv = scores.min() if scores.size else 0.0
    if maxv > minv:
        scores = (scores - minv) / (maxv - minv)
    return scores


def load_or_compute_msa_features(msa_path: str, pdb_path: str, chain_id: str, cache_path: str):
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        entropy = z["entropy"].astype(np.float32)
        dca = z["dca"].astype(np.float32)
        return entropy, dca

    seqs = parse_clustal(msa_path)
    ref_key = find_reference_key(seqs)
    ref_aln = seqs[ref_key]
    col_to_pos = build_col_to_refpos(ref_aln)

    coords, residues = load_ca_coordinates(pdb_path, chain_id=chain_id)
    cols = map_pdb_residues_to_columns(col_to_pos, residues)

    msa_int = msa_to_matrix(seqs, cols, q=21)
    entropy = compute_entropy(msa_int, q=21)
    dca = compute_dca_scores(msa_int, q=21)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, entropy=entropy, dca=dca)
    return entropy, dca
