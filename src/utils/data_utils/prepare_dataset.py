import numpy as np


class Scaler:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = np.maximum(X.std(0), 1e-8)
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def build_eval(eval_units_raw, sc: Scaler):

    evals, labels, masks, fails = [], [], [], []
    for X in eval_units_raw:
        T = len(X)
        rul = np.arange(T)[::-1]
        y = (rul <= 30).astype(int)
        m = (rul <= 30) | (rul >= 100)
        evals.append(sc.transform(X))
        labels.append(y)
        masks.append(m)
        fails.append(T - 1)
    return evals, labels, masks, fails
