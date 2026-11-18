from pathlib import Path

import numpy as np
import pandas as pd


# Порядок класів 
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

PARTS = ["train", "validation", "test"]

# Для кожного класу: (кількість файлів, сума числових імен)
# порядок train, validation, test.
EXPECTED = {
    "buildings": [(200, 1996761), (218, 4837323), (219, 4837559)],
    "forest": [(200, 2032625), (237, 5276962), (237, 5291326)],
    "glacier": [(200, 2183923), (276, 6135652), (277, 6157250)],
    "mountain": [(200, 1950345), (262, 5789535), (263, 5849765)],
    "sea": [(200, 1987613), (255, 5655068), (255, 5668011)],
    "street": [(200, 1841817), (250, 5541845), (251, 5547533)],
}


def create_split(data_dir):
    """Створюємо таблицю поділу даних."""
    data_dir = Path(data_dir)
    rows = []

    # Один генератор для всіх класів навчальної частини
    rng = np.random.default_rng(0)

    for label in CLASSES:
        folder = data_dir / "seg_train" / "seg_train" / label

        # Сортуємо назви файлів як рядки
        files = sorted(path.name for path in folder.glob("*.jpg"))

        # Створюємо індекси файлів і перемішуємо їх
        idx = np.arange(len(files))
        rng.shuffle(idx)

        # Беремо 200 зображень цього класу для навчання
        for i in idx[:200]:
            rows.append([files[i], label, "train"])

    # Новий генератор для поділу seg_test
    rng = np.random.default_rng(0)

    for label in CLASSES:
        folder = data_dir / "seg_test" / "seg_test" / label
        files = sorted(path.name for path in folder.glob("*.jpg"))

        idx = np.arange(len(files))
        rng.shuffle(idx)

        # Половина файлів іде у validation, решта — у test
        cut = len(files) // 2

        for i in idx[:cut]:
            rows.append([files[i], label, "validation"])

        for i in idx[cut:]:
            rows.append([files[i], label, "test"])

    return pd.DataFrame(rows, columns=["file", "label", "part"])


def check_split(split):
    """Перевіряємо відсутність повторів і контрольні числа."""

    if split["file"].duplicated().any():
        raise ValueError("У поділі є повтори файлів.")

    table = []
    all_correct = True

    for label in CLASSES:
        row = {"class": label}

        for j, part in enumerate(PARTS):
            # Вибираємо файли потрібного класу та частини
            mask = (split["label"] == label) & (split["part"] == part)
            files = split.loc[mask, "file"]

            count = len(files)

            name_sum = sum(int(Path(name).stem) for name in files)

            row[part] = f"{count} / {name_sum}"

            # Порівнюємо з контрольною таблицею
            if (count, name_sum) != EXPECTED[label][j]:
                all_correct = False
                print(f"Не збігається: {label}, {part}")

        table.append(row)

    # Додаємо підсумковий рядок
    total_row = {"class": "TOTAL"}

    for part in PARTS:
        files = split.loc[split["part"] == part, "file"]

        name_sum = sum(int(Path(name).stem) for name in files)

        total_row[part] = f"{len(files)} / {name_sum}"

    table.append(total_row)

    print(pd.DataFrame(table).to_string(index=False))

    if not all_correct:
        raise ValueError("Поділ не відповідає контрольній таблиці з умови.")

    print("\nУсі 18 клітинок збігаються. Повторів файлів немає.")
