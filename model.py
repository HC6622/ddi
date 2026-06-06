import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

from featurize import ATOM_FEAT_DIM, BOND_FEAT_DIM, TEXT_DIM

SEVERITY_CLASSES = ["Major", "Minor", "Moderate"]   # sorted; matches metrics ordering


class DrugEncoder(nn.Module):
    """Shared GNN: one molecular graph -> one vector. Same weights for drug A and B."""
    def __init__(self, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.node_encoder = nn.Linear(ATOM_FEAT_DIM, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=BOND_FEAT_DIM))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

    def forward(self, data):
        x = self.node_encoder(data.x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, data.edge_attr)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, data.batch)       # [num_graphs, hidden]


class DDIModel(nn.Module):
    def __init__(self, hidden=128, drug_dim=256, n_layers=3,
                 n_side_effects=50, n_severity=3, dropout=0.1):
        super().__init__()
        self.encoder = DrugEncoder(hidden, n_layers, dropout)
        self.text_proj = nn.Sequential(nn.Linear(TEXT_DIM, hidden), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, drug_dim), nn.ReLU())

        z_dim = drug_dim * 3                          # sum, |diff|, product
        def head(out_dim):
            return nn.Sequential(nn.Linear(z_dim, drug_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(drug_dim, out_dim))
        self.severity_head = head(n_severity)
        self.sideeffect_head = head(n_side_effects)
        self.prr_head = head(n_side_effects)

    def encode_drug(self, data, text):
        g = self.encoder(data)                        # [B, hidden]
        t = self.text_proj(text)                      # [B, hidden]
        return self.fuse(torch.cat([g, t], dim=1))    # [B, drug_dim]

    def forward(self, data_a, data_b, text_a, text_b):
        da = self.encode_drug(data_a, text_a)
        db = self.encode_drug(data_b, text_b)
        z = torch.cat([da + db, (da - db).abs(), da * db], dim=1)   # order-invariant
        return self.severity_head(z), self.sideeffect_head(z), self.prr_head(z)


if __name__ == "__main__":
    from torch_geometric.data import Batch
    graphs = torch.load("cache/graphs.pt", weights_only=False)
    vals = list(graphs.values())
    ba, bb = Batch.from_data_list(vals[:8]), Batch.from_data_list(vals[8:16])
    ta, tb = torch.randn(8, TEXT_DIM), torch.randn(8, TEXT_DIM)
    model = DDIModel()
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    model.eval()
    with torch.no_grad():
        sev, se, prr = model(ba, bb, ta, tb)
        sev2, _, _ = model(bb, ba, tb, ta)
    print("shapes:", tuple(sev.shape), tuple(se.shape), tuple(prr.shape))
    print("symmetric (A<->B swap):", torch.allclose(sev, sev2, atol=1e-5))