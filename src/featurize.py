import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a:i for i,a in enumerate(AA)}

def aa_onehot(aas):
    x = np.zeros((len(aas), 20), dtype=np.float32)
    for i,a in enumerate(aas):
        if a in AA_TO_I:
            x[i, AA_TO_I[a]] = 1.0
    return x

def geom_features(ca_xyz):
    c = ca_xyz.mean(0, keepdims=True)
    z = ca_xyz - c
    r = np.sqrt((z*z).sum(-1, keepdims=True) + 1e-9)
    scale = np.percentile(r, 90) + 1e-6
    z = z / scale
    r = r / scale
    return np.concatenate([z, r], axis=1).astype(np.float32)
