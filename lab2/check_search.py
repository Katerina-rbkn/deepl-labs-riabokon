"""Незалежна перевірка пошуку порога на малих наборах A, B, C.

Повільний перебір тут не використовує швидкий алгоритм: власний список кандидатів
і пряме застосування правила s >= t для кожного з них.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from thresholds import best_index, candidate_table

SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}

# Результати, порахованi вручну (незалежно від коду): (t*, C*)
EXPECTED = {"A": (0.4, 2), "B": (math.inf, 0), "C": (0.8, 10)}


def fmt_t(t: float) -> str:
    return "+inf" if math.isinf(t) else f"{t:g}"


def brute_force(scores, labels):
    """Пряма перевірка: окремий список кандидатів і правило для кожного."""
    candidates = [math.inf] + sorted(set(scores), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels):
            predicted = s >= t
            if predicted and y == 1:
                tp += 1
            elif predicted and y == 0:
                fp += 1
            elif not predicted and y == 1:
                fn += 1
            else:
                tn += 1
        rows.append((t, tp, fp, fn, tn, 10 * fn + fp))
    best = None
    for row in rows:  # мінімум C; за рівної вартості більший поріг
        if best is None or row[5] < best[5] or (row[5] == best[5] and row[0] > best[0]):
            best = row
    return rows, best


def run_checks(out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for name, (scores, labels) in SETS.items():
        slow_rows, slow_best = brute_force(scores, labels)

        table = candidate_table(labels, scores)
        fast_rows = {
            float(table["t"][k]): tuple(int(table[c][k]) for c in ("TP", "FP", "FN", "TN", "C"))
            for k in range(len(table["t"]))
        }
        fast_best_t = float(table["t"][best_index(table)])

        same_candidates = set(fast_rows) == {r[0] for r in slow_rows}
        same_counts = same_candidates and all(fast_rows[r[0]] == r[1:] for r in slow_rows)
        same_threshold = fast_best_t == slow_best[0]
        exp_t, exp_c = EXPECTED[name]
        matches_manual = slow_best[0] == exp_t and slow_best[5] == exp_c
        ok = same_candidates and same_counts and same_threshold and matches_manual
        all_ok &= ok

        min_c = slow_best[5]
        tied = [fmt_t(r[0]) for r in slow_rows if r[5] == min_c]
        print(f"### Набір {name}: s = {scores}, y = {labels}\n")
        print("| Поріг | TP | FP | FN | TN | C |")
        print("|---:|---:|---:|---:|---:|---:|")
        for t, tp, fp, fn, tn, c in slow_rows:
            print(f"| {fmt_t(t)} | {tp} | {fp} | {fn} | {tn} | {c} |")
        print(
            f"\nОбраний t* = {fmt_t(slow_best[0])}, C = {min_c}; пороги з мінімальною C: {', '.join(tied)}."
        )
        print(
            f"Кандидати збігаються: {same_candidates}; лічильники й C збігаються точно: {same_counts}; "
            f"поріг збігається зі швидким: {same_threshold}; збігається з ручним розрахунком: {matches_manual}."
        )
        print(f"Результат перевірки: {'пройдено' if ok else 'НЕ ПРОЙДЕНО'}\n")

        pd.DataFrame(
            slow_rows, columns=["threshold", "TP", "FP", "FN", "TN", "C"]
        ).to_csv(out_dir / f"check_{name}.csv", index=False)
    print(f"Усі три перевірки: {'пройдено' if all_ok else 'НЕ ПРОЙДЕНО'}")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if run_checks(Path("analysis")) else 1)