import os, requests
import numpy as np
from Bio.PDB import PDBParser

AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K",
    "LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V",
    "TRP":"W","TYR":"Y"
}

def download_pdb(pdb_id: str, out_dir: str):
    pdb_id = pdb_id.lower()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def load_chain_ca(pdb_path: str, chain_id: str):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", pdb_path)
    model = next(st.get_models())
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in {pdb_path}")
    chain = model[chain_id]

    res_ids, aas, coords = [], [], []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.id
        if hetflag != " ":
            continue
        rn = res.resname
        if rn not in AA3_TO_1:
            continue
        if "CA" not in res:
            continue
        res_ids.append((resseq, icode.strip() if isinstance(icode, str) else ""))
        aas.append(AA3_TO_1[rn])
        coords.append(res["CA"].get_coord())

    return res_ids, aas, np.asarray(coords, dtype=np.float32)

def pairwise_dist(x: np.ndarray):
    dif = x[:, None, :] - x[None, :, :]
    return np.sqrt((dif * dif).sum(-1) + 1e-9).astype(np.float32)
