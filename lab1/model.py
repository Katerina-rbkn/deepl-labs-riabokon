import numpy as np


def init_params():
    rng = np.random.default_rng(0)           # НОВИЙ генератор
    W1 = rng.normal(0.0, np.sqrt(2 / 4), size=(4, 8))        # спочатку W1
    b1 = np.zeros(8)
    W2 = rng.normal(0.0, np.sqrt(2 / (8 + 3)), size=(8, 3))  # потім W2
    b2 = np.zeros(3)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def log_softmax(Z):
    shifted = Z - Z.max(axis=1, keepdims=True)   # зсув на максимум рядка
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

def forward(X, y, p):
    Z1 = X @ p["W1"] + p["b1"]        # (N, 8)  передактивація
    A1 = np.maximum(Z1, 0.0)          # (N, 8)  ReLU
    Z2 = A1 @ p["W2"] + p["b2"]       # (N, 3)  логіти
    logP = log_softmax(Z2)            # (N, 3)
    loss = -logP[np.arange(len(y)), y].mean()
    cache = {"X": X, "y": y, "Z1": Z1, "A1": A1, "logP": logP}
    return loss, cache

def backward(cache, p, divide_by_n=True):
    X, y = cache["X"], cache["y"]
    Z1, A1, logP = cache["Z1"], cache["A1"], cache["logP"]
    N = len(y)

    P = np.exp(logP)                          # softmax, (N, 3)
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0                  # one-hot, (N, 3)

    dZ2 = P - Y                               # (N, 3)
    if divide_by_n:
        dZ2 = dZ2 / N                         # ЦЕ ділення прибираємо в досліді з помилкою

    dW2 = A1.T @ dZ2                          # (8, 3)
    db2 = dZ2.sum(axis=0)                     # (3,)
    dA1 = dZ2 @ p["W2"].T                     # (N, 8)
    dZ1 = dA1 * (Z1 > 0)                      # (N, 8)  похідна ReLU
    dW1 = X.T @ dZ1                           # (4, 8)
    db1 = dZ1.sum(axis=0)                     # (8,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}