import argparse
import numpy as np

from data import load_data
from model import init_params, forward, backward
from torch_reference import torch_reference
from checks import numerical_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", action="store_true",
                    help="прибрати ділення на N у градієнті за логітами")
    divide = not ap.parse_args().bug
    print("РЕЖИМ:", "ПРАВИЛЬНИЙ" if divide else "З НАВМИСНОЮ ПОМИЛКОЮ (без /N)")

    X, y, X_te, y_te = load_data()
    print(f"\ntrain: {X.shape}, класи {np.bincount(y)} | "
          f"test: {X_te.shape}, класи {np.bincount(y_te)}")
    print(f"max|mean(train)| = {np.abs(X.mean(axis=0)).max():.2e}, "
          f"std(train) = {X.std(axis=0, ddof=0)}")
    p = init_params()

    loss_np, cache = forward(X, y, p)
    g_np = backward(cache, p, divide)
    loss_t, g_t = torch_reference(p, X, y)

    print("\nЗначення втрати:")
    print(f"  NumPy   = {loss_np:.15f}")
    print(f"  PyTorch = {loss_t:.15f}")

    print("\n### Звірка з PyTorch (критерій 1e-12)\n")
    print("| Величина | Макс. абс. різниця | Перевірку пройдено |")
    print("|---|---|---|")
    d = abs(loss_np - loss_t)
    print(f"| Втрата | {d:.3e} | {'Так' if np.isfinite(d) and d <= 1e-12 else 'Ні'} |")
    for k in ("W1", "b1", "W2", "b2"):
        fin = np.all(np.isfinite(g_np[k])) and np.all(np.isfinite(g_t[k]))
        d = np.max(np.abs(g_np[k] - g_t[k]))
        print(f"| Градієнт {k} | {d:.3e} | {'Так' if fin and d <= 1e-12 else 'Ні'} |")

    print("\nВідношення норм ||grad_numpy|| / ||grad_torch||:")
    for k in ("W1", "b1", "W2", "b2"):
        print(f"  {k}: {np.linalg.norm(g_np[k]) / np.linalg.norm(g_t[k]):.6f}")

    print("\n### Чисельна перевірка (eps=1e-6, критерій 1e-7)\n")
    print("| Параметр | Градієнт backward() | Чисельна похідна | Абс. різниця | Перевірку пройдено |")
    print("|---|---|---|---|---|")
    for name, g, num, diff, ok in numerical_check(p, X, y, divide):
        print(f"| {name} | {g:.9f} | {num:.9f} | {diff:.2e} | {'Так' if ok else 'Ні'} |")


if __name__ == "__main__":
    main()