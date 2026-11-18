import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from checks import run_checks
from data import CLASSES, check_split, create_split
from dataset import (
    IMG_SIZE,
    create_loader,
    create_transform,
    save_train_batch,
)
from journal import EPOCHS, RUNS, file_hash, make_journal, select_run
from model import (
    check_parameters,
    create_model,
    initial_loss,
    set_train_mode,
)


def prepare_experiment(data_dir):
    """Створюємо поділ і повторює перевірки до основного навчання."""

    split = create_split(data_dir)
    check_split(split)
    split.to_csv("split.csv", index=False)

    torch.manual_seed(1)
    np.random.seed(1)

    save_train_batch(create_loader(data_dir, "train"))

    for regime in ["scratch", "frozen", "unfreeze"]:
        model, optimizer = create_model(
            regime, seed=1, device="cpu"
        )
        check_parameters(model, optimizer, regime)

    losses = {}

    for regime in ["scratch", "frozen"]:
        losses[regime] = initial_loss(data_dir, regime)
        print(
            f"Початкова втрата {regime}: "
            f"{losses[regime]:.6f}"
        )

    Path("initial_losses.json").write_text(
        json.dumps(
            {"ln6": float(np.log(6)), "losses": losses},
            indent=2,
        ),
        encoding="utf-8",
    )

    run_checks(data_dir)


def get_device():
    """Повертає пристрій і його назву для журналу."""

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name()

    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS"

    name = platform.processor() or platform.machine()
    return torch.device("cpu"), name


def synchronize(device):
    """Очікує завершення обчислень для вимірювання часу."""

    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def train_epoch(model, optimizer, loader, criterion, regime, device):
    """Виконуємо одну епоху та обчислює середню втрату."""

    set_train_mode(model, regime)

    loss_sum = 0.0
    steps = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * len(images)
        steps += 1

    return loss_sum / len(loader.dataset), steps


def validate(model, loader, device):
    """Повертаємо логіти validation і macro-F1."""

    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    score = f1_score(
        labels,
        logits.argmax(axis=1),
        average="macro",
    )

    return logits, float(score)


def train_run(data_dir, regime, seed, device, device_name):

    model, optimizer = create_model(regime, seed, device)

    train_loader = create_loader(data_dir, "train")
    val_loader = create_loader(data_dir, "validation")

    criterion = nn.CrossEntropyLoss()
    set_train_mode(model, regime)

    journal = make_journal(
        model, optimizer, criterion, train_loader, val_loader,
        create_transform("test"), regime, seed, device, device_name,
    )

    folder = Path("runs") / f"{regime}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)

    print(
        f"\nОсновний запуск: {regime}, "
        f"seed={seed}, device={device}",
        flush=True,
    )

    synchronize(device)
    start = time.perf_counter()
    total_steps = 0

    for epoch in range(EPOCHS):
        loss, steps = train_epoch(
            model, optimizer, train_loader,
            criterion, regime, device,
        )

        val_logits, score = validate(model, val_loader, device)

        journal["train_loss"].append(loss)
        journal["val_macro_f1"].append(score)

        total_steps += steps

        print(
            f"Епоха {epoch + 1}/{EPOCHS}: "
            f"loss={loss:.6f}, val_F1={score:.6f}, "
            f"час={time.perf_counter() - start:.1f} с",
            flush=True,
        )

    synchronize(device)

    journal["time_seconds"] = time.perf_counter() - start
    journal["time_includes"] = (
        "8 training epochs and validation after each epoch"
    )
    journal["optimizer_steps"] = total_steps

    assert total_steps == 304
    assert val_logits.shape == (1498, 6)
    assert file_hash("split.csv") == journal["split_sha256"]

    # Логіти validation після восьмої епохи
    np.save(folder / "val_logits.npy", val_logits)

    # Усі ваги тимчасово зберігаємо локально в weights/
    weights = Path("weights") / f"{regime}_seed{seed}.pt"
    weights.parent.mkdir(exist_ok=True)

    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }

    torch.save(
        {
            "state_dict": state,
            "regime": regime,
            "seed": seed,
            "classes": CLASSES,
            "img_size": IMG_SIZE,
        },
        weights,
    )

    journal["weights_sha256"] = file_hash(weights)

    (folder / "run.json").write_text(
        json.dumps(journal, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Збережено результати {regime}_seed{seed}",
        flush=True,
    )

    return journal


def main():
    parser = argparse.ArgumentParser(description="Навчання для ЛР3")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    prepare_experiment(args.data_dir)

    device, device_name = get_device()
    journals = []

    for regime, seed in RUNS:
        journal = train_run(
            args.data_dir, regime, seed, device, device_name
        )
        journals.append(journal)

    # Усі запуски повинні використовувати однаковий поділ
    split_sha = file_hash("split.csv")

    assert all(
        run["split_sha256"] == split_sha
        for run in journals
    )

    best, means = select_run(journals)

    source = (
        Path("weights")
        / f"{best['regime']}_seed{best['seed']}.pt"
    )

    assert file_hash(source) == best["weights_sha256"]

    shutil.copyfile(source, "lab3_model.pt")

    print("\nСередні macro-F1 на validation:")

    for regime, mean in means.items():
        print(f"{regime}: {mean:.6f}")

    print(f"Вибрано: {best['regime']}, seed={best['seed']}")
    print("Збережено lab3_model.pt")


if __name__ == "__main__":
    main()