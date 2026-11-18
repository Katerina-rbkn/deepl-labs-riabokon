import hashlib
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from data import CLASSES


EPOCHS = 8

RUNS = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]


def file_hash(path):
    """Обчислює SHA-256 файлу."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info():

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        encoding="utf-8",
    ).strip()

    changes = subprocess.check_output(
        ["git", "status", "--porcelain"],
        encoding="utf-8",
    ).strip()

    return {
        "commit": commit,
        "dirty": bool(changes),
    }


def transform_info(transform):
    """Записуємо параметри перетворень, включно зі стандартними."""
    result = []

    keys = [
        "size", "scale", "ratio", "interpolation", "antialias",
        "p", "mean", "std", "inplace", "max_size",
    ]

    for step in transform.transforms:
        info = {"name": type(step).__name__}

        for key in keys:
            if hasattr(step, key):
                value = getattr(step, key)

                if key == "interpolation":
                    value = value.name

                info[key] = value

        result.append(info)

    return result


def loader_info(loader, shuffle):
    """Записуємо налаштування завантажувача."""
    keys = [
        "batch_size", "drop_last", "num_workers", "pin_memory",
        "timeout", "prefetch_factor", "persistent_workers",
        "pin_memory_device", "in_order", "generator",
        "worker_init_fn", "multiprocessing_context",
    ]

    info = {
        key: getattr(loader, key)
        for key in keys if hasattr(loader, key)
    }

    info["shuffle"] = shuffle
    info["sampler"] = type(loader.sampler).__name__
    info["batch_sampler"] = type(loader.batch_sampler).__name__
    info["collate_fn"] = loader.collate_fn.__name__

    return info


def make_journal(model, optimizer, criterion, train_loader, val_loader,
                 test_transform, regime, seed, device, device_name):
    """Формуємо початковий журнал запуску."""

    versions = {"python": platform.python_version()}

    for name in [
        "torch", "torchvision", "numpy", "pandas",
        "scikit-learn", "matplotlib", "Pillow",
    ]:
        versions[name] = version(name)

    groups = []

    for group in optimizer.param_groups:
        info = {
            key: value
            for key, value in group.items()
            if key != "params"
        }

        info["parameter_count"] = sum(
            p.numel() for p in group["params"]
        )

        groups.append(info)

    bn = model.bn1

    return {
        "regime": regime,
        "seed": seed,
        "git": git_info(),
        "split_sha256": file_hash("split.csv"),
        "versions": versions,
        "device": {
            "type": str(device),
            "name": device_name,
        },
        "settings": {
            "classes": CLASSES,
            "img_size": 160,
            "image_mode": "RGB",
            "dtype": str(next(model.parameters()).dtype),
            "epochs": EPOCHS,
            "steps_per_epoch": len(train_loader),
            "optimizer_steps": EPOCHS * len(train_loader),
            "model": "resnet18",
            "architecture": str(model),
            "weights": (
                None if regime == "scratch" else "IMAGENET1K_V1"
            ),
            "fc_initialization": (
                "nn.Linear(512, 6), default initialization"
            ),
            "total_parameters": sum(
                p.numel() for p in model.parameters()
            ),
            "trainable_parameters": sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            ),
            "trainable_names": [
                name
                for name, p in model.named_parameters()
                if p.requires_grad
            ],
            "eval_modules": [
                name
                for name, module in model.named_children()
                if not module.training
            ],
            "batch_norm": {
                "eps": bn.eps,
                "momentum": bn.momentum,
                "affine": bn.affine,
                "track_running_stats": bn.track_running_stats,
            },
            "transforms": {
                "train": transform_info(train_loader.dataset.transform),
                "validation": transform_info(val_loader.dataset.transform),
                "test": transform_info(test_transform),
            },
            "loaders": {
                "train": loader_info(train_loader, True),
                "validation": loader_info(val_loader, False),
                # Налаштування test такі самі, як validation.
                # Самі тестові зображення тут не завантажуються.
                "test": loader_info(val_loader, False),
            },
            "optimizer": {
                "name": "AdamW",
                "defaults": optimizer.defaults,
                "groups": groups,
            },
            "loss": {
                "name": "CrossEntropyLoss",
                "weight": None,
                "ignore_index": criterion.ignore_index,
                "reduction": criterion.reduction,
                "label_smoothing": criterion.label_smoothing,
                "size_average": None,
                "reduce": None,
            },
            "metric": {
                "name": "f1_score",
                "average": "macro",
                "labels": None,
                "pos_label": 1,
                "sample_weight": None,
                "zero_division": "warn",
            },
            "train_loss_average": (
                "sum(batch_loss * batch_size) / 1200"
            ),
            "scheduler": None,
            "gradient_clipping": None,
            "mixed_precision": False,
            "early_stopping": False,
            "epoch_selection": "last epoch",
            "test_time_augmentation": False,
        },
        "train_loss": [],
        "val_macro_f1": [],
    }


def select_run(journals):
    """Вибираємо режим за середнім F1, потім найкраще зерно."""

    means = {}

    for regime in ["scratch", "frozen", "unfreeze"]:
        scores = [
            run["val_macro_f1"][-1]
            for run in journals
            if run["regime"] == regime
        ]

        means[regime] = sum(scores) / len(scores)

    best_regime = max(means, key=means.get)

    candidates = [
        run for run in journals
        if run["regime"] == best_regime
    ]

    best = max(
        candidates,
        key=lambda run: run["val_macro_f1"][-1],
    )

    return best, means