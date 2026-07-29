import numpy as np


def make_windows(X, w, stride=1):
    """Slice ``X`` (T, N) into sliding windows predicting the next step.

    Returns ``xs`` (num, N, w) and ``ys`` (num, N), where sample ``k`` uses
    ``X[i-w:i]`` to target ``X[i]`` for ``i`` in ``range(w, T, stride)``. This is
    the single windowing implementation shared by both the CMAPSS/MSL pipeline
    and the original GDN ``TimeDataset`` so the two stay directly comparable.
    """
    X = np.asarray(X, np.float32)
    T, N = X.shape
    if T <= w:
        return np.empty((0, N, w), np.float32), np.empty((0, N), np.float32)
    idx = range(w, T, stride)
    xs = np.stack([X[i - w : i].T for i in idx])
    ys = np.stack([X[i] for i in idx])
    return xs, ys
