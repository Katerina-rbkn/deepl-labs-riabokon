"""Метрики, вартість і пошук порога.

Прогнозуємо шахрайство (1), якщо s >= t.
Вартість: C(t) = 10 * FN(t) + FP(t).
"""
from __future__ import annotations

import numpy as np

FN_WEIGHT = 10  # ціна пропущеного шахрайства
FP_WEIGHT = 1   # ціна хибної тривоги


def confusion(y, s, t):
    """Пряме застосування правила s >= t. Повертає (TP, FP, FN, TN)."""
    y = np.asarray(y)
    s = np.asarray(s)
    predicted = s >= t
    actual = y == 1
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    return tp, fp, fn, tn


def cost(fn, fp):
    """C = 10 * FN + FP (Amount у вартість не входить)."""
    return FN_WEIGHT * fn + FP_WEIGHT * fp


def precision_recall(tp, fp, fn):
    """Повертає (precision, recall); None, якщо знаменник дорівнює нулю."""
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    return precision, recall


def candidate_table(y, s):
    """Усі кандидати (унікальні оцінки та +inf) через сортування й кумулятивні суми.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    if y.ndim != 1 or y.shape != s.shape or len(y) == 0:
        raise ValueError("y і s мають бути непорожніми одновимірними масивами однакової довжини")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Мітки мають бути лише 0 або 1")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними")

    order = np.argsort(-s, kind="stable")      # від більших оцінок до менших
    s_sorted = s[order]
    positive = y[order] == 1
    cum_tp = np.cumsum(positive)               # шахрайських серед перших k
    cum_fp = np.cumsum(~positive)              # легітимних серед перших k

    # Рівні оцінки неможливо розділити порогом, тому беремо кінець кожної групи
    group_end = np.append(s_sorted[1:] != s_sorted[:-1], True)

    thresholds = np.concatenate(([np.inf], s_sorted[group_end]))
    tp = np.concatenate(([0], cum_tp[group_end]))
    fp = np.concatenate(([0], cum_fp[group_end]))

    p_total = int(positive.sum())
    n_total = len(y) - p_total
    fn = p_total - tp
    tn = n_total - fp
    return {"t": thresholds, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def best_index(table):
    """Індекс мінімуму C. За рівної вартості береться найбільший поріг."""
    c = table["C"]
    ties = np.flatnonzero(c == c.min())
    return int(ties[np.argmax(table["t"][ties])])