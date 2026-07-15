import numpy as np


def make_windows(X, w):
    X = np.asarray(X, np.float32)
    T, N = X.shape
    if T <= w:
        return np.empty((0, N, w), np.float32), np.empty((0, N), np.float32)
    xs = np.stack([X[i - w : i].T for i in range(w, T)])
    ys = X[w:]
    return xs, ys
