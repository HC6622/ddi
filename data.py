import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")          # silence noisy RDKit parse warnings

DATA_PATH = "data/train.csv"
VAL_SCAFFOLD_FRACTION = 0.20            # share of scaffolds held out for validation
SEED = 42


def get_label_columns(df):
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    prr_cols = [c for c in df.columns if c.startswith("Target_PRR_")]
    return binary_cols, prr_cols


def scaffold_of(smiles):
    """Bemis-Murcko scaffold; fall back to the whole molecule if acyclic/unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaf if scaf else smiles


def scaffold_split(df, val_fraction=VAL_SCAFFOLD_FRACTION, seed=SEED):
    drugs = pd.unique(pd.concat([df["SMILES_A"], df["SMILES_B"]]))
    drug_to_scaffold = {d: scaffold_of(d) for d in drugs}

    scaffolds = sorted(set(drug_to_scaffold.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    n_val = int(len(scaffolds) * val_fraction)
    val_scaffolds = set(scaffolds[:n_val])

    a_val = df["SMILES_A"].map(lambda d: drug_to_scaffold[d] in val_scaffolds)
    b_val = df["SMILES_B"].map(lambda d: drug_to_scaffold[d] in val_scaffolds)

    train_mask = (~a_val) & (~b_val)     # both drugs seen  -> train
    val_mask = a_val & b_val             # both drugs novel -> val
    train_idx = df.index[train_mask].to_numpy()
    val_idx = df.index[val_mask].to_numpy()
    return train_idx, val_idx, drug_to_scaffold


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    binary_cols, prr_cols = get_label_columns(df)
    print(f"loaded {len(df)} pairs | {len(binary_cols)} binary + {len(prr_cols)} PRR targets")

    train_idx, val_idx, _ = scaffold_split(df)
    dropped = len(df) - len(train_idx) - len(val_idx)
    print(f"\nscaffold split -> train: {len(train_idx)} | val: {len(val_idx)} | dropped bridge pairs: {dropped}")

    tr, va = df.loc[train_idx], df.loc[val_idx]
    tr_drugs = set(tr.SMILES_A) | set(tr.SMILES_B)
    va_drugs = set(va.SMILES_A) | set(va.SMILES_B)
    print(f"drug overlap train<->val: {len(tr_drugs & va_drugs)} (must be 0)")

    print("\nseverity balance:")
    print("  train:", tr.Severity.value_counts(normalize=True).round(3).to_dict())
    print("  val:  ", va.Severity.value_counts(normalize=True).round(3).to_dict())