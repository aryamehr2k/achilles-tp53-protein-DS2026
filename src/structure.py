import numpy as np
from Bio.PDB import PDBParser

AA3 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}


def load_ca_coordinates(pdb_path, chain_id="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)

    coords = []
    residues = []

    model = next(structure.get_models())
    chain = model[chain_id]

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue

        resname = res.get_resname().upper()
        aa = AA3.get(resname)
        if aa is None:
            continue

        ca = res["CA"].get_coord()
        resseq = int(res.get_id()[1])

        coords.append(ca.astype(np.float32))
        residues.append({"resseq": resseq, "aa": aa})

    return np.asarray(coords, dtype=np.float32), residues


# ----------------------------
# kNN Graph (for GCN baseline)
# ----------------------------
def build_knn_graph(coords, k=16):
    coords = np.asarray(coords, dtype=np.float32)
    N = coords.shape[0]

    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)

    edges = []

    for i in range(N):
        order = np.argsort(d2[i])
        nbrs = order[1:k+1]  # exclude self
        for j in nbrs:
            edges.append([i, j])
            edges.append([j, i])  # undirected

    edge_index = np.array(edges).T  # [2, E]
    return edge_index


def build_chain_graph(num_nodes: int):
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    if len(edges) == 0:
        edges = [[0, 0]]
    edge_index = np.array(edges).T
    return edge_index
