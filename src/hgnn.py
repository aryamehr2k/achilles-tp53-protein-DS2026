# src/hgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class HGNN(nn.Module):
    """
    GCN backbone + strong mutation head

    Features used:
        z_mut       : structural context at mutation position
        wt_emb      : WT amino acid embedding
        mut_emb     : mutant amino acid embedding
        delta       : mut_emb - wt_emb
        pos_emb     : learned residue index embedding
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=256,
        aa_dim=64,
        pos_dim=32,
        max_pos=512,
        dropout=0.2
    ):
        super().__init__()

        self.dropout = dropout

        # Graph backbone
        self.g1 = GCNConv(in_dim, hidden_dim)
        self.g2 = GCNConv(hidden_dim, hidden_dim)

        # Amino acid embedding
        self.aa_emb = nn.Embedding(20, aa_dim)

        # Position embedding
        self.pos_emb = nn.Embedding(max_pos, pos_dim)

        feat_dim = hidden_dim + (aa_dim * 3) + pos_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        mut_mask = data.mut_mask
        wt_idx = data.wt_idx.view(-1)
        mut_idx = data.mut_idx.view(-1)

        # GCN layers
        x = self.encode(data, edge_weight=edge_weight)

        # Pool mutated node
        m = mut_mask.unsqueeze(1)  # [N,1]
        z = (x * m).sum(dim=0, keepdim=True)  # [1, hidden]

        # Amino acid embeddings
        wt = self.aa_emb(wt_idx)   # [1, aa_dim]
        mut = self.aa_emb(mut_idx) # [1, aa_dim]
        delta = mut - wt           # [1, aa_dim]

        # Position embedding
        pos = torch.argmax(mut_mask).view(1)
        pos = pos.clamp(0, self.pos_emb.num_embeddings - 1)
        pe = self.pos_emb(pos)     # [1, pos_dim]

        feat = torch.cat([z, wt, mut, delta, pe], dim=1)

        out = self.head(feat).squeeze(1)
        return out.squeeze(0)

    def encode(self, data, edge_weight=None):
        x = data.x
        edge_index = data.edge_index

        x = self.g1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.g2(x, edge_index, edge_weight)
        x = F.relu(x)
        return x
