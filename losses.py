import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

from model import SEVERITY_CLASSES


def compute_class_weights(severity_labels):
    """Balanced inverse-frequency weights, in SEVERITY_CLASSES order."""
    counts = torch.zeros(len(SEVERITY_CLASSES))
    for lab in severity_labels:
        counts[SEVERITY_CLASSES.index(lab)] += 1
    n_total, n_classes = counts.sum(), len(SEVERITY_CLASSES)
    weights = n_total / (n_classes * counts.clamp(min=1))
    return weights


def masked_log_mse(prr_pred_log, prr_target):
    """MSE in log1p space, ONLY over cells where true PRR > 0."""
    mask = prr_target > 0
    if mask.sum() == 0:
        return prr_pred_log.sum() * 0.0          # zero loss, keeps autograd graph intact
    target_log = torch.log1p(prr_target[mask])
    return ((prr_pred_log[mask] - target_log) ** 2).mean()


def prr_from_pred(prr_pred_log):
    """Convert the head's log-space output back to a real PRR (>= 0) for scoring."""
    return torch.expm1(prr_pred_log).clamp(min=0.0)


class DDILoss(nn.Module):
    def __init__(self, class_weights=None, w_sev=1.0, w_se=1.0, w_prr=1.0, pos_weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.w_sev, self.w_se, self.w_prr = w_sev, w_se, w_prr

    def forward(self, sev_logits, se_logits, prr_out, sev_target, se_target, prr_target):
        sev_loss = self.ce(sev_logits, sev_target)
        se_loss = self.bce(se_logits, se_target)
        prr_loss = masked_log_mse(prr_out, prr_target)
        total = self.w_sev * sev_loss + self.w_se * se_loss + self.w_prr * prr_loss
        return total, {"sev": sev_loss.item(), "se": se_loss.item(), "prr": prr_loss.item()}


if __name__ == "__main__":
    import pandas as pd
    from data import scaffold_split
    df = pd.read_csv("data/train.csv")
    tr_idx, _, _ = scaffold_split(df)
    w = compute_class_weights(df.loc[tr_idx, "Severity"].tolist())
    print("class order:", SEVERITY_CLASSES)
    print("class weights:", [round(x, 3) for x in w.tolist()])