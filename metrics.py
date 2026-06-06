import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from sklearn.metrics import f1_score


def masked_prr_score(prr_true, prr_pred):
    """S_PRR = 1 / (1 + RMSE), where RMSE is over cells with true PRR > 0 only."""
    prr_true = np.asarray(prr_true, dtype=float)
    prr_pred = np.asarray(prr_pred, dtype=float)
    mask = prr_true > 0
    if mask.sum() == 0:
        return 1.0, 0.0
    err = prr_true[mask] - prr_pred[mask]
    rmse = np.sqrt(np.mean(err ** 2))
    return 1.0 / (1.0 + rmse), rmse


def compute_score(sev_true, sev_pred, bin_true, bin_pred, prr_true, prr_pred, verbose=True):
    # 1. Severity: macro-F1 over the classes that actually occur in the truth
    classes = sorted(set(map(str, sev_true)))
    f1_sev = f1_score(sev_true, sev_pred, labels=classes, average="macro", zero_division=0)

    # 2. Side effects: micro-F1 across all 50 labels at once
    f1_se = f1_score(bin_true, bin_pred, average="micro", zero_division=0)

    # 3. PRR: inverse masked RMSE
    s_prr, rmse = masked_prr_score(prr_true, prr_pred)

    score = 0.4 * f1_sev + 0.3 * f1_se + 0.3 * s_prr
    if verbose:
        print(f"  Severity   macro-F1 : {f1_sev:.4f}   (weight 0.4)")
        print(f"  SideEffect micro-F1 : {f1_se:.4f}   (weight 0.3)")
        print(f"  PRR        S_PRR    : {s_prr:.4f}   (masked RMSE {rmse:.3f}, weight 0.3)")
        print(f"  ---> FINAL SCORE    : {score:.4f}")
    return {"score": score, "f1_severity": f1_sev, "f1_sideeffects": f1_se,
            "s_prr": s_prr, "rmse_masked": rmse}


if __name__ == "__main__":
    import pandas as pd
    from data import scaffold_split, get_label_columns

    df = pd.read_csv("data/train.csv")
    binary_cols, prr_cols = get_label_columns(df)
    _, val_idx, _ = scaffold_split(df)
    val = df.loc[val_idx]

    sev_true = val["Severity"].values
    bin_true = val[binary_cols].values
    prr_true = val[prr_cols].values

    print("Lazy baseline (all 'Moderate', all zeros) on the val split:")
    sev_pred = np.array(["Moderate"] * len(val))
    bin_pred = np.zeros_like(bin_true)
    prr_pred = np.zeros_like(prr_true, dtype=float)
    compute_score(sev_true, sev_pred, bin_true, bin_pred, prr_true, prr_pred)