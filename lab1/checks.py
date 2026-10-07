import numpy as np
from model import forward, backward


def numerical_check(p, X, y, divide_by_n=True):
    _, cache = forward(X, y, p)
    manual = backward(cache, p, divide_by_n)   # градієнти на початкових вагах
    eps = 1e-6
    rows = []
    for name, idx in [("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))]:
        orig = p[name][idx]                    # 1. зберегти початкове значення
        p[name][idx] = orig + eps
        Lp, _ = forward(X, y, p)               # 2. L+
        p[name][idx] = orig - eps
        Lm, _ = forward(X, y, p)               # 3. L-
        p[name][idx] = orig                    # 4. відновити
        num = (Lp - Lm) / (2 * eps)            # центральна різниця
        g = manual[name][idx]
        diff = abs(num - g)
        ok = bool(np.isfinite(diff) and diff <= 1e-7)
        rows.append((f"{name}{list(idx)}", g, num, diff, ok))
    return rows