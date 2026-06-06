import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data

from featurize import load_caches, smiles_to_graph, TEXT_DIM, ATOM_FEAT_DIM, BOND_FEAT_DIM
from model import DDIModel, SEVERITY_CLASSES
from losses import prr_from_pred


def _safe_graph(smiles):
    g = smiles_to_graph(smiles)
    if g is None:
        g = Data(x=torch.zeros(1, ATOM_FEAT_DIM),
                 edge_index=torch.zeros((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, BOND_FEAT_DIM)))
    return g


class TestDataset(Dataset):
    def __init__(self, df, graphs, text_emb):
        self.items = []
        self.missing_graph = 0
        self.missing_text = 0
        for _, row in df.iterrows():
            ga, gb = graphs.get(row.SMILES_A), graphs.get(row.SMILES_B)
            if ga is None: ga = _safe_graph(row.SMILES_A); self.missing_graph += 1
            if gb is None: gb = _safe_graph(row.SMILES_B); self.missing_graph += 1
            ta, tb = text_emb.get(row.SMILES_A), text_emb.get(row.SMILES_B)
            if ta is None: ta = np.zeros(TEXT_DIM, np.float32); self.missing_text += 1
            if tb is None: tb = np.zeros(TEXT_DIM, np.float32); self.missing_text += 1
            self.items.append((ga, gb,
                               torch.tensor(ta, dtype=torch.float),
                               torch.tensor(tb, dtype=torch.float)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_infer(batch):
    ga, gb, ta, tb = zip(*batch)
    return (Batch.from_data_list(ga), Batch.from_data_list(gb),
            torch.stack(ta), torch.stack(tb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="data/test.csv")
    p.add_argument("--sample", default="data/sample_submission.csv")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = pd.read_csv(args.sample)
    columns = sample.columns.tolist()
    binary_cols = [c for c in columns if c.startswith("Target_Binary_")]
    prr_cols = [c for c in columns if c.startswith("Target_PRR_")]

    test = pd.read_csv(args.test)
    graphs, text_emb = load_caches()
    ds = TestDataset(test, graphs, text_emb)
    if ds.missing_graph or ds.missing_text:
        print(f"WARNING: {ds.missing_graph} graphs / {ds.missing_text} text vectors missing from cache.")
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_infer)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    model = DDIModel(**arch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    threshold = ckpt.get("threshold", 0.3)
    print(f"loaded checkpoint (val score {ckpt.get('score', '?')}, threshold {threshold}, arch {arch})")

    sev_idx, se_prob, prr_val = [], [], []
    with torch.no_grad():
        for da, db, ta, tb in ld:
            da, db, ta, tb = da.to(device), db.to(device), ta.to(device), tb.to(device)
            sl, el, po = model(da, db, ta, tb)
            sev_idx.append(sl.argmax(1).cpu())
            se_prob.append(torch.sigmoid(el).cpu())
            prr_val.append(prr_from_pred(po).cpu())
    sev_idx = torch.cat(sev_idx).numpy()
    se_prob = torch.cat(se_prob).numpy()
    prr_val = torch.cat(prr_val).numpy()

    data = {"Pair_ID": test["Pair_ID"].values,
            "Severity": [SEVERITY_CLASSES[i] for i in sev_idx]}
    bin_pred = (se_prob >= threshold).astype(int)
    for j, c in enumerate(binary_cols):
        data[c] = bin_pred[:, j]
    for j, c in enumerate(prr_cols):
        data[c] = np.round(prr_val[:, j], 4)
    out = pd.DataFrame(data)[columns]
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {out.shape[0]} rows x {out.shape[1]} cols")


if __name__ == "__main__":
    main()