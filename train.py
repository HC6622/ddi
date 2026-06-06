import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from data import scaffold_split, get_label_columns
from featurize import load_caches
from model import DDIModel, SEVERITY_CLASSES
from losses import DDILoss, compute_class_weights, prr_from_pred
from metrics import compute_score

CKPT_DIR = "checkpoints"
SE_THRESHOLDS = np.arange(0.10, 0.65, 0.05)     # swept on val to pick best micro-F1


class PairDataset(Dataset):
    def __init__(self, df, graphs, text_emb, binary_cols, prr_cols):
        self.items = []
        for _, row in df.iterrows():
            ga, gb = graphs.get(row.SMILES_A), graphs.get(row.SMILES_B)
            if ga is None or gb is None:
                continue
            self.items.append((
                ga, gb,
                torch.tensor(text_emb[row.SMILES_A], dtype=torch.float),
                torch.tensor(text_emb[row.SMILES_B], dtype=torch.float),
                torch.tensor(SEVERITY_CLASSES.index(row.Severity), dtype=torch.long),
                torch.tensor(row[binary_cols].values.astype(np.float32)),
                torch.tensor(row[prr_cols].values.astype(np.float32)),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    ga, gb, ta, tb, sev, se, prr = zip(*batch)
    return (Batch.from_data_list(ga), Batch.from_data_list(gb),
            torch.stack(ta), torch.stack(tb),
            torch.stack(sev), torch.stack(se), torch.stack(prr))


@torch.no_grad()
def evaluate(model, loader, df_val, binary_cols, prr_cols, device):
    model.eval()
    sev_idx, se_prob, prr_val = [], [], []
    for da, db, ta, tb, *_ in loader:
        da, db, ta, tb = da.to(device), db.to(device), ta.to(device), tb.to(device)
        sl, el, po = model(da, db, ta, tb)
        sev_idx.append(sl.argmax(1).cpu())
        se_prob.append(torch.sigmoid(el).cpu())
        prr_val.append(prr_from_pred(po).cpu())
    sev_idx = torch.cat(sev_idx).numpy()
    se_prob = torch.cat(se_prob).numpy()
    prr_val = torch.cat(prr_val).numpy()

    sev_true = df_val.Severity.values
    bin_true = df_val[binary_cols].values
    prr_true = df_val[prr_cols].values
    sev_pred = np.array([SEVERITY_CLASSES[i] for i in sev_idx])

    best = None
    for thr in SE_THRESHOLDS:
        bin_pred = (se_prob >= thr).astype(int)
        r = compute_score(sev_true, sev_pred, bin_true, bin_pred, prr_true, prr_val, verbose=False)
        r["threshold"] = float(thr)
        if best is None or r["score"] > best["score"]:
            best = r
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", type=int, default=0, help="limit train pairs (0 = all) for quick local runs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"   # MPS skipped: PyG op support
    print("device:", device)

    df = pd.read_csv("data/train.csv")
    binary_cols, prr_cols = get_label_columns(df)
    tr_idx, val_idx, _ = scaffold_split(df)
    df_tr, df_val = df.loc[tr_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)
    if args.subset:
        df_tr = df_tr.iloc[:args.subset].reset_index(drop=True)
    print(f"train pairs: {len(df_tr)} | val pairs: {len(df_val)}")

    graphs, text_emb = load_caches()
    tr_ds = PairDataset(df_tr, graphs, text_emb, binary_cols, prr_cols)
    val_ds = PairDataset(df_val, graphs, text_emb, binary_cols, prr_cols)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = DDIModel().to(device)
    class_w = compute_class_weights(df_tr.Severity.tolist()).to(device)
    loss_fn = DDILoss(class_weights=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"sev": 0, "se": 0, "prr": 0}; n = 0
        for da, db, ta, tb, sev, se, prr in tr_ld:
            da, db, ta, tb = da.to(device), db.to(device), ta.to(device), tb.to(device)
            sev, se, prr = sev.to(device), se.to(device), prr.to(device)
            opt.zero_grad()
            sl, el, po = model(da, db, ta, tb)
            total, parts = loss_fn(sl, el, po, sev, se, prr)
            total.backward(); opt.step()
            for k in running: running[k] += parts[k]
            n += 1
        r = evaluate(model, val_ld, df_val, binary_cols, prr_cols, device)
        print(f"epoch {epoch:2d} | train[sev {running['sev']/n:.3f} se {running['se']/n:.3f} "
              f"prr {running['prr']/n:.3f}] | val SCORE {r['score']:.4f} "
              f"(sevF1 {r['f1_severity']:.3f} seF1 {r['f1_sideeffects']:.3f} "
              f"sPRR {r['s_prr']:.3f} @thr {r['threshold']:.2f})")
        if r["score"] > best_score:
            best_score = r["score"]
            torch.save({"model": model.state_dict(), "threshold": r["threshold"],
                        "score": r["score"]}, os.path.join(CKPT_DIR, "best.pt"))
            print(f"   saved new best -> {best_score:.4f}")
    print("done. best val score:", round(best_score, 4))


if __name__ == "__main__":
    main()