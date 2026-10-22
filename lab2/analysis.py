"""Аналіз порогів зі збережених прогнозів (без навчання моделі).

Запуск:
    uv run python analysis.py --saved results --out analysis

Порядок: validation (пошук t*) -> перевірки A/B/C -> криві -> фіксація t* ->
тестова оцінка. До тесту не переходимо, поки не пройдено всі три перевірки.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_search import run_checks
from evaluation import test_stage
from lab2_starter import load_predictions
from plots import plot_curves
from thresholds import best_index, candidate_table, confusion, cost


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fmt_t(t: float) -> str:
    return "+inf" if np.isinf(t) else f"{t:.6g}"


def validation_stage(data: dict, out_dir: Path):
    """Пошук порога на validation. Test тут не використовується."""
    y = data["validation_y"]
    s = data["validation_scores"].astype(np.float64)  # float32 -> float64 без втрат

    table = candidate_table(y, s)
    n_candidates = len(table["t"])
    p_total = int(np.count_nonzero(y == 1))
    n_total = len(y) - p_total

    require(n_candidates == len(np.unique(s)) + 1, "Кандидатів має бути: унікальні оцінки + 1")
    require(bool(np.all(table["TP"] + table["FN"] == p_total)), "TP + FN має дорівнювати P")
    require(bool(np.all(table["FP"] + table["TN"] == n_total)), "FP + TN має дорівнювати N")
    rng = np.random.default_rng(0)
    sample = np.unique(np.concatenate(([0, n_candidates - 1], rng.integers(0, n_candidates, 30))))
    for k in sample:
        fast = tuple(int(table[c][k]) for c in ("TP", "FP", "FN", "TN"))
        require(confusion(y, s, table["t"][k]) == fast, f"Розбіжність у рядку {k}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "threshold": table["t"],
            "TP": table["TP"],
            "FP": table["FP"],
            "FN": table["FN"],
            "TN": table["TN"],
            "C": table["C"],
        }
    ).to_csv(out_dir / "validation_candidates.csv", index=False)

    i = best_index(table)
    t_star = float(table["t"][i])
    ties = np.flatnonzero(table["C"] == table["C"][i])

    print(f"validation: n={len(y)}, шахрайських P={p_total}, легітимних N={n_total}")
    print(f"Кандидатів: {n_candidates} (унікальних оцінок: {n_candidates - 1}, плюс +inf)")
    print(f"Мінімальна вартість C = {int(table['C'][i])}; порогів із такою вартістю: {len(ties)}")
    print(f"t* (повна точність) = {t_star!r}")
    print()
    print("| Правило на validation | Поріг | TP | FP | FN | TN | C |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, t in (("Стандартний поріг", 0.5), ("Мінімум вартості", t_star), ("Усі негативні", float("inf"))):
        tp, fp, fn, tn = confusion(y, s, t)
        print(f"| {name} | {fmt_t(t)} | {tp} | {fp} | {fn} | {tn} | {cost(fn, fp)} |")
    print()
    print("Швидкий підрахунок збігається з прямим застосуванням правила (перевірено на вибірці рядків).")
    print(f"Таблицю кандидатів збережено: {(out_dir / 'validation_candidates.csv').resolve()}")
    return table, i, t_star


def save_threshold(out_dir: Path, y, s, t_star: float, curve_info: dict) -> None:
    """Зафіксувати t* (повна точність) після validation і перевірок."""
    tp, fp, fn, tn = confusion(y, s, t_star)
    record = {
        "threshold": t_star,
        "selected_on": "validation",
        "rule": "predict fraud if score >= threshold",
        "cost": "10*FN + FP",
        "validation": {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)},
        "validation_roc_auc": curve_info["roc_auc"],
        "validation_average_precision": curve_info["average_precision"],
    }
    path = out_dir / "threshold.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nПоріг зафіксовано: {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека зі збереженими прогнозами")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = parser.parse_args()

    data = load_predictions(args.saved)
    table, i, t_star = validation_stage(data, args.out)

    print("\n" + "=" * 60 + "\nПеревірка пошуку порога на наборах A, B, C\n" + "=" * 60 + "\n")
    if not run_checks(args.out):
        raise SystemExit("Перевірки не пройдено: до тестової оцінки не переходимо.")

    y = data["validation_y"]
    s = data["validation_scores"].astype(np.float64)
    info = plot_curves(y, s, t_star, args.out)
    print("\n" + "=" * 60 + "\nКриві на validation\n" + "=" * 60)
    print(f"ROC-AUC = {info['roc_auc']:.6f}")
    print(f"Середня точність (AP) = {info['average_precision']:.6f}")
    print(f"Частка шахрайських π = {info['positive_fraction']:.6f}")
    print(f"Збережено: {(args.out / 'validation_roc.png').resolve()}")
    print(f"Збережено: {(args.out / 'validation_pr.png').resolve()}")
    save_threshold(args.out, y, s, t_star, info)

    # Поріг читаємо назад із зафіксованого файлу: test його не змінює
    t_fixed = float(json.loads((args.out / "threshold.json").read_text(encoding="utf-8"))["threshold"])
    print("\n" + "=" * 60 + "\nТестова оцінка (поріг зафіксовано на validation)\n" + "=" * 60 + "\n")
    test_stage(data, args.out, t_fixed)


if __name__ == "__main__":
    main()