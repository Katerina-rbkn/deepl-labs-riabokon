from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data import CLASSES


IMG_SIZE = 160

# Значення нормування 
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def create_transform(part):
    """Повертає перетворення для потрібної частини вибірки."""

    if part == "train":
        steps = [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps = [transforms.Resize((IMG_SIZE, IMG_SIZE))]

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(MEAN, STD))

    return transforms.Compose(steps)


class IntelDataset(Dataset):
    """Завантажує зображення за списком зі split.csv."""

    def __init__(self, data_dir, part):
        # Читаємо збережений поділ і вибираємо потрібну частину
        split = pd.read_csv("split.csv")
        self.rows = split[split["part"] == part].reset_index(drop=True)

        self.transform = create_transform(part)

        if part == "train":
            self.folder = Path(data_dir) / "seg_train" / "seg_train"
        else:
            self.folder = Path(data_dir) / "seg_test" / "seg_test"

    def __len__(self):
        # Кількість зображень у цій частині вибірки
        return len(self.rows)

    def __getitem__(self, index):
        # Отримуємо один рядок таблиці
        row = self.rows.iloc[index]

        path = self.folder / row["label"] / row["file"]

        # Відкриваємо зображення та застосовуємо перетворення
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))

        # Перетворюємо назву класу на число від 0 до 5
        label = CLASSES.index(row["label"])

        return image, label


def create_loader(data_dir, part):
    """Об'єднує зображення в пакети."""

    dataset = IntelDataset(data_dir, part)

    if part == "train":
        batch_size = 32
    else:
        batch_size = 64

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(part == "train"),
        drop_last=False,
        num_workers=0,
    )


def save_train_batch(loader):
    """Зберігає перші 16 зображень одного навчального пакета."""

    images, labels = next(iter(loader))

    print("Кількість навчальних зображень:", len(loader.dataset))
    print("Кількість пакетів за епоху:", len(loader))
    print("Форма пакета:", images.shape)
    print("Тип даних:", images.dtype)

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        # Обернене нормування лише для відображення
        image = images[i] * std + mean

        image = image.clamp(0, 1).permute(1, 2, 0).numpy()

        ax.imshow(image)
        ax.set_title(CLASSES[int(labels[i])])
        ax.axis("off")

    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/train_batch.png", dpi=150)

    plt.close(fig)

    print("Збережено figures/train_batch.png")

