"""PR- і ROC-криві на validation """
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без вікон: лише збереження в файл
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from thresholds import confusion


def plot_curves(y, s, t_star: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    n_neg = int(np.count_nonzero(y == 0))
    pi = float(np.mean(y == 1))

    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)

    marks = [("t = 0.5", 0.5, "tab:red"), (f"t* = {t_star:.4g}", t_star, "tab:green")]

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="випадкове ранжування")
    for label, t, color in marks:
        tp, fp, fn, _ = confusion(y, s, t)
        ax.scatter(fp / n_neg, tp / (tp + fn), color=color, zorder=3, label=label)
    ax.set_xlabel("FPR = FP / (FP + TN)")
    ax.set_ylabel("TPR = TP / (TP + FN)")
    ax.set_title("ROC-крива, validation")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "validation_roc.png", dpi=150)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.4f})")
    ax.axhline(pi, linestyle="--", color="gray", label=f"базова лінія π = {pi:.4f}")
    for label, t, color in marks:
        tp, fp, fn, _ = confusion(y, s, t)
        ax.scatter(tp / (tp + fn), tp / (tp + fp), color=color, zorder=3, label=label)
    ax.set_xlabel("Повнота (recall)")
    ax.set_ylabel("Точність (precision)")
    ax.set_title("Крива точність-повнота, validation")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "validation_pr.png", dpi=150)
    plt.close(fig)

    return {"roc_auc": auc, "average_precision": ap, "positive_fraction": pi}