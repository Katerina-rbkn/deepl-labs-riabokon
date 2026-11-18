import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from dataset import create_loader


def create_model(regime, seed=1, device="cpu"):
    """Створюємо модель та оптимізатор для заданого режиму."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    if regime == "scratch":
        weights = None
    else:
        weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(weights=weights)

    # Замінюємо останній шар для наших 6 класів
    model.fc = nn.Linear(512, 6)

    if regime != "scratch":
        # Спочатку заморожуємо всю мережу
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

        if regime == "unfreeze":
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True

    model = model.to(device)

    if regime == "unfreeze":
        groups = [
            {"params": model.fc.parameters(), "lr": 3e-3},
            {"params": model.layer4.parameters(), "lr": 3e-4},
        ]
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": parameters, "lr": 3e-3}]

    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    return model, optimizer


def set_train_mode(model, regime):
    """Вмикаємо навчання та переводить заморожені модулі в eval."""

    model.train()

    if regime == "scratch":
        return

    frozen_names = ["conv1", "bn1", "layer1", "layer2", "layer3"]

    if regime == "frozen":
        frozen_names.append("layer4")

    for name in frozen_names:
        module = getattr(model, name)
        module.eval()


def check_parameters(model, optimizer, regime):
    """Перевіряємо кількість параметрів і групи оптимізатора."""

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    expected = {
        "scratch": 11179590,
        "frozen": 3078,
        "unfreeze": 8396806,
    }

    assert total == 11179590, "Неправильна загальна кількість параметрів"
    assert trainable == expected[regime], "Неправильна кількість навчуваних параметрів"

    if regime == "unfreeze":
        expected_groups = [(3078, 3e-3), (8393728, 3e-4)]
    else:
        expected_groups = [(expected[regime], 3e-3)]

    assert len(optimizer.param_groups) == len(expected_groups)

    print(f"\nРежим: {regime}")
    print("Усього параметрів:", total)
    print("Навчаються:", trainable)

    for i, group in enumerate(optimizer.param_groups):
        count = sum(p.numel() for p in group["params"])

        # Заморожені параметри не повинні бути в оптимізаторі
        assert all(p.requires_grad for p in group["params"])

        # Перевіряємо розмір групи та її налаштування
        assert (count, group["lr"]) == expected_groups[i]
        assert group["weight_decay"] == 1e-4

        print(f"Група {i + 1}: {count} параметрів, lr={group['lr']}")

    print("Перевірка параметрів пройдена.")


def initial_loss(data_dir, regime):
    """Обчислює втрату до першого кроку оптимізатора."""

    # Для цієї перевірки створюємо окрему нову модель
    model, optimizer = create_model(regime, seed=1, device="cpu")
    loader = create_loader(data_dir, "train")

    set_train_mode(model, regime)

    images, labels = next(iter(loader))
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)

    return loss.item()
