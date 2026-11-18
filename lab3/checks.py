import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from dataset import create_loader
from model import create_model, set_train_mode


MODULES = [
    "conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"
]

BN_BUFFERS = [
    "running_mean", "running_var", "num_batches_tracked"
]

FORECAST = (
    "У frozen зі звичайним model.train() ваги замороженої частини "
    "залишаться сталими, а статистики BatchNorm зміняться. "
    "Параметри fc зміняться. Втрата на train може зменшуватися, "
    "тому лише за нею цю помилку можна не помітити."
)


def has_changed(before, after, names):
    """Перевіряємо, чи змінився хоча б один із перелічених тензорів."""

    return any(
        not torch.equal(before[name], after[name])
        for name in names
    )


def check_changes(data_dir, regime, wrong_mode=False):
    """Виконуємо 5 кроків і перевіряє зміни параметрів та буферів BN."""

    # Кожна перевірка починається з нової моделі
    model, optimizer = create_model(regime, seed=1, device="cpu")
    loader = create_loader(data_dir, "train")

    # Зберігаємо незалежну копію параметрів і буферів
    before = {
        name: value.clone()
        for name, value in model.state_dict().items()
    }

    print(f"\nРежим: {regime}, навмисна помилка: {wrong_mode}")

    if wrong_mode:
        model.train()
    else:
        set_train_mode(model, regime)

    criterion = nn.CrossEntropyLoss()
    losses = []

    for step, (images, labels) in enumerate(loader):
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Крок {step + 1}: loss={loss.item():.6f}")

        if step == 4:
            break

    assert len(losses) == 5, "Виконано не 5 кроків"

    after = model.state_dict()

    if regime == "frozen":
        active_names = ["fc"]
    else:
        active_names = ["layer4", "fc"]

    rows = []
    all_correct = True

    for module_name in MODULES:
        module = getattr(model, module_name)

        parameter_names = [
            f"{module_name}.{name}"
            for name, _ in module.named_parameters()
        ]

        buffer_names = [
            f"{module_name}.{name}"
            for name, _ in module.named_buffers()
            if name.split(".")[-1] in BN_BUFFERS
        ]

        parameters_changed = has_changed(
            before, after, parameter_names
        )

        bn_changed = has_changed(
            before, after, buffer_names
        )

        # Очікувані зміни для цієї перевірки
        expected_parameters = module_name in active_names

        expected_bn = bool(buffer_names) and (
            wrong_mode or module_name in active_names
        )

        correct = (
            parameters_changed == expected_parameters
            and bn_changed == expected_bn
        )

        if not correct:
            all_correct = False

        if buffer_names:
            bn_text = "так" if bn_changed else "ні"
        else:
            bn_text = "немає BN"

        rows.append({
            "Модуль": module_name,
            "Параметри змінилися": "так" if parameters_changed else "ні",
            "Статистики BN змінилися": bn_text,
            "Як очікували": "так" if correct else "ні",
        })

    print(pd.DataFrame(rows).to_string(index=False))

    assert all_correct, "Зміни не відповідають очікуванню"

    result = {
        "regime": regime,
        "seed": 1,
        "wrong_mode": wrong_mode,
        "losses": losses,
        "table": rows,
    }

    # Для навмисної помилки вимірюємо зміни конкретного шару
    if wrong_mode:
        for statistic in ["running_mean", "running_var"]:
            key = f"layer1.0.bn1.{statistic}"

            difference = (
                after[key] - before[key]
            ).abs().max().item()

            result[f"max_change_{statistic}"] = difference

            print(f"Максимальна зміна {key}: {difference:.10g}")

    return result


def run_checks(data_dir):
    """Запускаємо дві правильні перевірки та дослід із помилкою."""

    results = []
    for regime in ["frozen", "unfreeze"]:
        results.append(check_changes(data_dir, regime))

    print("\nПрогноз перед дослідом із помилкою:")
    print(FORECAST)

    results.append(
        check_changes(data_dir, "frozen", wrong_mode=True)
    )

    report = {
        "forecast": FORECAST,
        "results": results,
    }

    Path("checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


