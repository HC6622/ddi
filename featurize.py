import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data
RDLogger.DisableLog("rdApp.*")

CACHE_DIR = "cache"
GRAPH_CACHE = os.path.join(CACHE_DIR, "graphs.pt")
TEXT_CACHE = os.path.join(CACHE_DIR, "text_emb.pt")

TEXT_FIELDS = ["Mechanism", "Pharmacodynamics", "Metabolism", "Toxicity",
               "Indication", "Absorption", "Half_Life", "Protein_Binding",
               "Elimination_Route", "Warning", "CYP450_Enzymes"]
TEXT_MODEL = "all-MiniLM-L6-v2"

ATOM_FEAT_DIM = 34
BOND_FEAT_DIM = 6
N_DESC = 9
EMB_DIM = 384
TEXT_DIM = EMB_DIM + N_DESC            # 384 text + 9 descriptors = 393

ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B", "Si"]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
          Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
          Chem.HybridizationType.SP3D2]


def _one_hot(value, choices):
    vec = [int(value == c) for c in choices]
    vec.append(int(value not in choices))
    return vec


def atom_features(atom):
    return (
        _one_hot(atom.GetSymbol(), ATOMS) +
        _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        _one_hot(atom.GetHybridization(), HYBRID) +
        [atom.GetFormalCharge(), int(atom.GetIsAromatic()), int(atom.IsInRing())]
    )


def bond_features(bond):
    bt = bond.GetBondType()
    return [int(bt == Chem.BondType.SINGLE), int(bt == Chem.BondType.DOUBLE),
            int(bt == Chem.BondType.TRIPLE), int(bt == Chem.BondType.AROMATIC),
            int(bond.GetIsConjugated()), int(bond.IsInRing())]


def smiles_to_graph(smiles):
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


def mol_descriptors(smiles):
    """9 whole-molecule properties (scaffold-independent, good for cold-start)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return [0.0] * N_DESC
    return [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
            Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
            Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m), rdMolDescriptors.CalcNumRings(m)]


def build_drug_text_map(df):
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
    model = SentenceTransformer(TEXT_MODEL)
    emb = model.encode([text_map[s] for s in smis], batch_size=32,
                       show_progress_bar=True, convert_to_numpy=True)        # [N, 384]

    desc = np.array([mol_descriptors(s) for s in smis], dtype=np.float32)    # [N, 9]
    desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)                       # standardize

    combined = np.concatenate([emb, desc], axis=1).astype(np.float32)        # [N, 393]
    text_emb = {s: combined[i] for i, s in enumerate(smis)}
    torch.save(text_emb, TEXT_CACHE)
    print(f"saved {len(text_emb)} drug vectors (dim {combined.shape[1]}: 384 text + 9 descriptors) -> {TEXT_CACHE}")


def load_caches():
    graphs = torch.load(GRAPH_CACHE, weights_only=False)
    text_emb = torch.load(TEXT_CACHE, weights_only=False)
    return graphs, text_emb


if __name__ == "__main__":
    build_caches()