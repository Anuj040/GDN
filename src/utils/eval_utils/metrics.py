import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)


def best_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(np.nanmax(f1))


def eval_scores(y, s, mask=None):
    y = np.asarray(y)
    s = np.asarray(s)
    if mask is not None:
        y, s = y[mask], s[mask]
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return dict(AUC=np.nan, PRAUC=np.nan, F1=np.nan)
    return dict(
        AUC=roc_auc_score(y, s), PRAUC=average_precision_score(y, s), F1=best_f1(y, s)
    )


def first_alarm(score, thr, persist=5):
    # persist 点連続で閾値超過した最初の時刻 (なければ None)
    a = np.asarray(score) > thr
    for i in range(len(a) - persist + 1):
        if a[i : i + persist].all():
            return i
    return None


def lead_time(score, thr, fail_idx, persist=5):
    # 故障時点 fail_idx に対して何ステップ前に検知できたか (大きいほど早い / None=見逃し)
    d = first_alarm(score, thr, persist)
    return None if d is None else fail_idx - d


def add_result(dataset, experiment, method, metrics):
    row = dict(dataset=dataset, experiment=experiment, method=method)
    row.update(metrics)
    return row
