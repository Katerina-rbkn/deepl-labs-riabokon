import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torchvision.models import resnet18

from data import CLASSES
from dataset import IMG_SIZE, create_loader
from journal import EPOCHS, RUNS, file_hash, select_run


def write_table(table, name, index=False):
    """Зберігає таблицю та показує її в терміналі."""
    table.to_csv(Path("results") / name, index=index)
    print(f"\n{name}:")
    print(table.round(6).to_string(index=index))


def read_journals():
    """Читає й перевіряє сім завершених запусків."""
    split_hash = file_hash("split.csv")
    journals = []
    for regime, seed in RUNS:
        folder = Path("runs") / f"{regime}_seed{seed}"
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if run["split_sha256"] != split_hash:
            raise ValueError(f"Інший поділ даних: {folder}")
        if (run["regime"], run["seed"]) != (regime, seed):
            raise ValueError(f"Інший режим або зерно: {folder}")
        if len(run["train_loss"]) != EPOCHS or len(run["val_macro_f1"]) != EPOCHS:
            raise ValueError(f"У журналі не 8 епох: {folder}")
        if run["optimizer_steps"] != 304:
            raise ValueError(f"У журналі не 304 кроки: {folder}")
        if np.load(folder / "val_logits.npy", allow_pickle=False).shape != (1498, 6):
            raise ValueError(f"Неправильна форма логітів: {folder}")
        journals.append(run)
    print("Усі 7 журналів перевірено. Хеші split.csv збігаються.")
    return journals


def report_training(journals, means):
    """Формує таблицю, криві навчання та розклад ефектів."""
    rows = []
    for number, run in enumerate(journals, start=1):
        rows.append({
            "run": number, "regime": run["regime"], "seed": run["seed"],
            "train_loss": run["train_loss"][-1],
            "val_F1": run["val_macro_f1"][-1], "time_s": run["time_seconds"],
        })
    write_table(pd.DataFrame(rows), "runs.csv")

    epochs = list(range(1, EPOCHS + 1))
    colors = {"scratch": "gray", "frozen": "tab:blue", "unfreeze": "tab:orange"}
    styles = {1: "-", 2: "--", 3: ":"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for run in journals:
        label = f"{run['regime']}, seed={run['seed']}"
        color, style = colors[run["regime"]], styles[run["seed"]]
        axes[0].plot(epochs, run["train_loss"], label=label, color=color, linestyle=style)
        axes[1].plot(epochs, run["val_macro_f1"], label=label, color=color, linestyle=style)
    axes[0].set(title="Втрата на train", ylabel="Cross-entropy loss")
    axes[1].set(title="Macro-F1 на validation", ylabel="Macro-F1")
    for ax in axes:
        ax.set(xlabel="Епоха", xticks=epochs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/training_curves.png", dpi=150)
    plt.close(fig)

    ranges = {}
    for regime in ["frozen", "unfreeze"]:
        scores = [r["val_macro_f1"][-1] for r in journals if r["regime"] == regime]
        ranges[regime] = max(scores) - min(scores)
    noise = max(ranges.values())
    effects = {
        "mean_frozen": means["frozen"], "range_frozen": ranges["frozen"],
        "mean_unfreeze": means["unfreeze"], "range_unfreeze": ranges["unfreeze"],
        "R": noise,
        "delta_pre": means["frozen"] - means["scratch"],
        "delta_unf": means["unfreeze"] - means["frozen"],
    }
    table = pd.DataFrame({"indicator": list(effects), "value": list(effects.values())})
    write_table(table, "effects.csv")
    for name in ["delta_pre", "delta_unf"]:
        if abs(effects[name]) <= noise:
            conclusion = "різниця не відрізняється від розкиду"
        else:
            conclusion = "ефект вартий перевірки, але не доведений"
        print(f"{name}: {100 * effects[name]:+.3f} пунктів macro-F1; {conclusion}")


def test_model(data_dir, best):
    """Завантажує вибрану модель та обчислює логіти test на CPU."""
    checkpoint = torch.load("lab3_model.pt", map_location="cpu", weights_only=True)
    if (checkpoint["regime"], checkpoint["seed"]) != (best["regime"], best["seed"]):
        raise ValueError("Модель не відповідає вибору за validation")
    if checkpoint["classes"] != CLASSES or checkpoint["img_size"] != IMG_SIZE:
        raise ValueError("Неправильні класи або розмір зображень")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(checkpoint["state_dict"])
    model.cpu()
    model.eval()

    loader = create_loader(data_dir, "test")
    logits, labels = [], []
    print("\nТестова оцінка на CPU", flush=True)
    with torch.no_grad():
        for batch, (images, targets) in enumerate(loader, start=1):
            logits.append(model(images))
            labels.append(targets)
            if batch % 5 == 0 or batch == len(loader):
                print(f"Test: {batch}/{len(loader)} пакетів", flush=True)
    logits = torch.cat(logits).numpy()
    labels = torch.cat(labels).numpy()
    if logits.shape != (1502, 6):
        raise ValueError("Неправильна форма тестових логітів")
    return logits, labels


def report_test(logits, labels, best):
    """Обчислює тестові метрики та зберігає матрицю плутанини."""
    predictions = logits.argmax(axis=1)
    score = float(f1_score(labels, predictions, average="macro"))
    accuracy = float(np.mean(labels == predictions))
    error = float(np.sqrt(accuracy * (1 - accuracy) / len(labels)))
    val_score = best["val_macro_f1"][-1]
    metrics = {
        "regime": best["regime"], "seed": best["seed"],
        "device": "cpu", "n": len(labels),
        "val_macro_f1": val_score, "test_macro_f1": score,
        "test_minus_validation": score - val_score,
        "accuracy": accuracy, "accuracy_se": error,
        "model_sha256": file_hash("lab3_model.pt"),
        "split_sha256": file_hash("split.csv"),
    }
    Path("results/test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save("results/test_logits.npy", logits)
    print(f"\nValidation macro-F1: {val_score:.6f}")
    print(f"Test macro-F1: {score:.6f}")
    print(f"Різниця test - validation: {score - val_score:+.6f}")
    print(f"Accuracy на test: {accuracy:.6f}")
    print(f"Стандартна похибка accuracy: {error:.6f}")

    matrix = confusion_matrix(labels, predictions, labels=list(range(6)))
    table = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    table.index.name = "true_class"
    write_table(table, "confusion_matrix.csv", index=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    display.plot(ax=ax, cmap="Blues", values_format="d",
                 colorbar=False, xticks_rotation=45)
    ax.set(title="Матриця плутанини на test",
           xlabel="Передбачений клас", ylabel="Справжній клас")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrix.png", dpi=150)
    plt.close(fig)
    return predictions


def select_error_examples(rows, labels, predictions):
    """Вибирає помилки за частотою пар і рядковим порядком файлів."""
    matrix = confusion_matrix(labels, predictions, labels=list(range(6)))
    pairs = []
    for true in range(6):
        for predicted in range(6):
            if true != predicted and matrix[true, predicted] > 0:
                pairs.append((true, predicted, int(matrix[true, predicted])))
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    groups = []
    for true, predicted, count in pairs:
        indices = np.flatnonzero((labels == true) & (predictions == predicted))
        indices = sorted(indices, key=lambda i: rows.iloc[i]["file"])
        groups.append((indices, count))

    selected = []
    for position in range(5):
        for indices, count in groups:
            if position < len(indices):
                selected.append((indices[position], count))
                if len(selected) == 5:
                    return selected
    return selected


def save_error_examples(data_dir, logits, labels, predictions):
    """Зберігає таблицю й рисунок п'яти характерних помилок."""
    split = pd.read_csv("split.csv")
    rows = split[split["part"] == "test"].reset_index(drop=True)
    selected = select_error_examples(rows, labels, predictions)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax in axes:
        ax.axis("off")
    table_rows = []
    for ax, (index, count) in zip(axes, selected):
        row = rows.iloc[index]
        predicted = CLASSES[predictions[index]]
        probability = float(probabilities[index, predictions[index]])
        table_rows.append({
            "file": row["file"], "true_class": row["label"],
            "predicted_class": predicted, "probability": probability,
            "pair_count": count,
        })
        path = Path(data_dir) / "seg_test" / "seg_test" / row["label"] / row["file"]
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{row['file']}\n{row['label']} → {predicted}\np={probability:.3f}",
                     fontsize=10)
    write_table(pd.DataFrame(table_rows), "errors.csv")
    fig.tight_layout()
    fig.savefig("figures/errors.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Аналіз результатів ЛР3")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        raise ValueError("Теку набору зображень не знайдено")
    journals = read_journals()
    best, means = select_run(journals)
    if file_hash("lab3_model.pt") != best["weights_sha256"]:
        raise ValueError("Файл моделі не відповідає вибору за validation")
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    report_training(journals, means)
    print(f"\nВибір підтверджено: {best['regime']}, seed={best['seed']}")
    logits, labels = test_model(args.data_dir, best)
    predictions = report_test(logits, labels, best)
    save_error_examples(args.data_dir, logits, labels, predictions)


if __name__ == "__main__":
    main()