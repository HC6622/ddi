import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data
RDLogger.DisableLog("rdApp.*")

CACHE_DIR = "cache"
GRAPH_CACHE = os.path.join(CACHE_DIR, "graphs.pt")
TEXT_CACHE = os.path.join(CACHE_DIR, "text_emb.pt")

TEXT_FIELDS = ["Mechanism", "Pharmacodynamics", "Metabolism", "Toxicity",
               "Indication", "Absorption", "Half_Life", "Protein_Binding",
               "Elimination_Route", "Warning", "CYP450_Enzymes"]
TEXT_MODEL = "all-MiniLM-L6-v2"      # 384-dim, fast, runs on CPU/MPS

ATOM_FEAT_DIM = 34
BOND_FEAT_DIM = 6
TEXT_DIM = 384

ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B", "Si"]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
          Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
          Chem.HybridizationType.SP3D2]


def _one_hot(value, choices):
    vec = [int(value == c) for c in choices]
    vec.append(int(value not in choices))      # final slot = "other"
    return vec


def atom_features(atom):
    return (
        _one_hot(atom.GetSymbol(), ATOMS) +                # 12
        _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +   # 7
        _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +  # 6
        _one_hot(atom.GetHybridization(), HYBRID) +        # 6
        [atom.GetFormalCharge(),                           # 1
         int(atom.GetIsAromatic()),                        # 1
         int(atom.IsInRing())]                             # 1
    )                                                      # = 34


def bond_features(bond):
    bt = bond.GetBondType()
    return [int(bt == Chem.BondType.SINGLE), int(bt == Chem.BondType.DOUBLE),
            int(bt == Chem.BondType.TRIPLE), int(bt == Chem.BondType.AROMATIC),
            int(bond.GetIsConjugated()), int(bond.IsInRing())]   # = 6


def smiles_to_graph(smiles):
    """Return a PyG Data graph for one molecule (None-safe)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]; dst += [j, i]; eattr += [f, f]
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_drug_text_map(df):
    """Map each unique SMILES -> one combined text string, from both A and B sides."""
    text = {}
    for side in ["A", "B"]:
        cols = [f"{f}_{side}" for f in TEXT_FIELDS]
        for smi, *vals in zip(df[f"SMILES_{side}"], *[df[c] for c in cols]):
            if smi in text:
                continue
            parts = [f"{f}: {v}" for f, v in zip(TEXT_FIELDS, vals)
                     if isinstance(v, str) and v.strip()]
            text[smi] = " | ".join(parts)
    return text


def build_caches(data_path="data/train.csv", extra_paths=()):
    """Build + save graph and text caches for every unique drug. Run once."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = [pd.read_csv(data_path)] + [pd.read_csv(p) for p in extra_paths]
    df = pd.concat(frames, ignore_index=True)
    drugs = pd.unique(pd.concat([df.SMILES_A, df.SMILES_B]))
    print(f"{len(drugs)} unique drugs across {len(frames)} file(s)")

    graphs = {}
    for smi in drugs:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs[smi] = g
    torch.save(graphs, GRAPH_CACHE)
    print(f"saved {len(graphs)} graphs -> {GRAPH_CACHE}")

    from sentence_transformers import SentenceTransformer
    text_map = build_drug_text_map(df)
    smis = list(text_map.keys())
    model = SentenceTransformer(TEXT_MODEL)        # downloads ~80MB on first run
    emb = model.encode([text_map[s] for s in smis], batch_size=32,
                       show_progress_bar=True, convert_to_numpy=True)
    text_emb = {s: e.astype(np.float32) for s, e in zip(smis, emb)}
    torch.save(text_emb, TEXT_CACHE)
    print(f"saved {len(text_emb)} text embeddings (dim {emb.shape[1]}) -> {TEXT_CACHE}")


def load_caches():
    graphs = torch.load(GRAPH_CACHE, weights_only=False)
    text_emb = torch.load(TEXT_CACHE, weights_only=False)
    return graphs, text_emb


if __name__ == "__main__":
    build_caches()