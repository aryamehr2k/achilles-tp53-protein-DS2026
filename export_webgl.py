import os
import json
import argparse
import numpy as np
import torch

from src.dataset import TP53StructureDataset
from src.structure import load_ca_coordinates
from src.hgnn import HGNN


CSV_PATH = "urn_mavedb_00001234-a-1_scores.csv"
PDB_PATH = "data/structures/TP53_RCSB.pdb"
CHAIN_ID = "A"


def pca3(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:3].T


def build_adj(edge_index: np.ndarray, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.T:
        adj[int(u)].append(int(v))
    return adj


def compute_hops(adj, num_nodes: int, start: int) -> np.ndarray:
    dist = np.full(num_nodes, fill_value=-1, dtype=np.int32)
    dist[start] = 0
    q = [start]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def sample_edges(edge_index: np.ndarray, max_edges: int = 5000) -> np.ndarray:
    e = edge_index.shape[1]
    if e <= max_edges:
        return edge_index
    idx = np.random.choice(e, size=max_edges, replace=False)
    return edge_index[:, idx]


def load_model(in_dim: int, device: torch.device, ckpt_path: str):
    model = HGNN(in_dim=in_dim).to(device)

    loaded = False
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        loaded = True
    model.eval()
    return model, loaded


def load_annotations(path, residues):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    ann_list = data.get("annotations", data if isinstance(data, list) else [])
    if not ann_list:
        return []

    resseq2idx = {r["resseq"]: i for i, r in enumerate(residues)}
    palette = [
        "#ff4d4d", "#4dd2ff", "#ffd24d", "#8aff4d", "#b84dff",
        "#ff8ad6", "#4dffb8", "#ff9c4d",
    ]

    out = []
    for i, ann in enumerate(ann_list):
        name = ann.get("name", f"ann_{i}")
        color = ann.get("color", palette[i % len(palette)])
        indices = set()
        for idx in ann.get("indices", []):
            if isinstance(idx, int) and idx >= 0:
                indices.add(idx)
        for r in ann.get("ranges", []):
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                continue
            start, end = int(r[0]), int(r[1])
            if end < start:
                start, end = end, start
            for pos in range(start, end + 1):
                if pos in resseq2idx:
                    indices.add(resseq2idx[pos])
        indices = sorted([i for i in indices if i >= 0])
        if indices:
            out.append({"name": name, "color": color, "indices": indices})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gnn"], default="gnn")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_edges", type=int, default=5000)
    parser.add_argument("--out", type=str, default="webgl/data.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--ann", type=str, default=None, help="path to annotations json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = TP53StructureDataset(CSV_PATH, PDB_PATH, chain_id=CHAIN_ID)
    idx = int(np.clip(args.index, 0, len(ds) - 1))
    data = ds[idx]
    mut_pos = int(data.mut_pos.item()) if hasattr(data, "mut_pos") else int(data.pos_idx.item())

    coords, residues = load_ca_coordinates(PDB_PATH, chain_id=CHAIN_ID)
    coords3d = coords.astype(np.float32)

    edge_index = ds.edge_index.detach().cpu().numpy()
    edge_draw = sample_edges(edge_index, max_edges=args.max_edges)

    adj = build_adj(edge_index, ds.N)
    hops = compute_hops(adj, ds.N, mut_pos)
    hops = np.where(hops >= 0, hops, hops.max() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "checkpoints/gnn.pt"
    model, loaded = load_model(in_dim=ds.X.shape[1], device=device, ckpt_path=ckpt)

    with torch.no_grad():
        z = model.encode(data.to(device)).detach().cpu().numpy()

    emb3d = pca3(z).astype(np.float32)

    annotations = load_annotations(args.ann, residues)

    label = args.label
    if label is None:
        label = "GNN (baseline)"

    out = {
        "struct": coords3d.tolist(),
        "embed": emb3d.tolist(),
        "edges": edge_draw.T.tolist(),
        "hop": hops.astype(int).tolist(),
        "mut": int(mut_pos),
        "annotations": annotations,
        "meta": {
            "label": label,
            "checkpoint": bool(loaded),
            "nodes": int(ds.N),
            "edges": int(edge_draw.shape[1]),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
