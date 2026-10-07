import numpy as np
from sklearn.datasets import load_iris


def load_data():
    iris = load_iris()
    X, y = iris.data.astype(np.float64), iris.target
    rng = np.random.default_rng(0)
    tr, te = [], []
    for c in (0, 1, 2):                      # класи по порядку
        idx = np.where(y == c)[0]
        rng.shuffle(idx)                     # перемішуємо індекси класу
        tr.append(idx[:35])                  # перші 35 -> train
        te.append(idx[35:])                  # решта 15 -> test
    tr, te = np.concatenate(tr), np.concatenate(te)
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    mu = X_tr.mean(axis=0)                   # статистики ТІЛЬКИ з train
    sd = X_tr.std(axis=0, ddof=0)
    return (X_tr - mu) / sd, y_tr, (X_te - mu) / sd, y_te