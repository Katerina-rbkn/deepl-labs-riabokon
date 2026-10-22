"""Тестова оцінка із зафіксованим порогом і розбір помилок.

Test не використовується для вибору порога, моделі чи налаштувань.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thresholds import confusion, cost, precision_recall


def fmt_t(t: float) -> str:
    return "+inf" if np.isinf(t) else f"{t:.6g}"


def fmt_metric(x) -> str:
    return "не визначена" if x is None else f"{x:.4f}"


def first_errors(mask, row_ids, y, s, amount, t, kind, limit=5):
    """Перші `limit` помилок за номером рядка CSV."""
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(row_ids[idx], kind="stable")][:limit]
    return [
        {
            "csv_row": int(row_ids[i]),
            "type": kind,
            "label": int(y[i]),
            "score": float(s[i]),
            "amount": float(amount[i]),
            "score_minus_threshold": float(s[i] - t),
        }
        for i in idx
    ]


def test_stage(data: dict, out_dir: Path, t: float) -> None:
    y = data["test_y"]
    s = data["test_scores"].astype(np.float64)
    amount = data["test_amount"]
    row_ids = data["test_row_ids"]

    tp, fp, fn, tn = confusion(y, s, t)
    c = cost(fn, fp)
    precision, recall = precision_recall(tp, fp, fn)
    p_total = int(np.count_nonzero(y == 1))

    print(f"test: n={len(y)}, шахрайських P={p_total}, легітимних N={len(y) - p_total}")
    print(f"Зафіксований поріг t* = {t!r}\n")
    print("| Test | Поріг | TP | FP | FN | TN | C | Precision | Recall |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, thr in (("t* (зафіксований)", t), ("Довідково: 0.5", 0.5), ("Довідково: усі негативні", float("inf"))):
        a, b, d, e = confusion(y, s, thr)
        pr, rc = precision_recall(a, b, d)
        print(f"| {name} | {fmt_t(thr)} | {a} | {b} | {d} | {e} | {cost(d, b)} | {fmt_metric(pr)} | {fmt_metric(rc)} |")
    print()
    if precision is None:
        print("Precision не визначена: TP + FP = 0, тобто модель не позначила жодної транзакції шахрайською.")
    if recall is None:
        print("Recall не визначений: TP + FN = 0, тобто у вибірці немає шахрайських транзакцій.")

    pred = s >= t
    fp_mask = pred & (y == 0)
    fn_mask = ~pred & (y == 1)
    rows = first_errors(fp_mask, row_ids, y, s, amount, t, "FP") + first_errors(
        fn_mask, row_ids, y, s, amount, t, "FN"
    )
    errors = pd.DataFrame(rows, columns=["csv_row", "type", "label", "score", "amount", "score_minus_threshold"])
    errors.to_csv(out_dir / "test_errors.csv", index=False)

    print(f"Усього FP: {int(fp_mask.sum())}, усього FN: {int(fn_mask.sum())}. Перші п'ять кожного типу за номером рядка CSV:\n")
    print("| Рядок CSV | Тип | Мітка | Оцінка | Amount | Оцінка − t* |")
    print("|---:|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['csv_row']} | {r['type']} | {r['label']} | {r['score']:.6g} | "
            f"{r['amount']:.2f} | {r['score_minus_threshold']:.6g} |"
        )

    print("\nМедіана Amount (усі помилки відповідного типу порівняно з усім test):")
    print(f"  усі транзакції test: {np.median(amount):.2f}")
    if fp_mask.any():
        print(f"  FP (усього {int(fp_mask.sum())}): {np.median(amount[fp_mask]):.2f}")
    if fn_mask.any():
        print(f"  FN (усього {int(fn_mask.sum())}): {np.median(amount[fn_mask]):.2f}")

    metrics = {
        "threshold": t,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c,
        "precision": precision, "recall": recall,
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nЗбережено: {(out_dir / 'test_errors.csv').resolve()}")
    print(f"Збережено: {(out_dir / 'test_metrics.json').resolve()}")