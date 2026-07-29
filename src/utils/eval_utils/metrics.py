import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

from src.utils.data_utils.prepare_data import SENSORS

# 日本語ラベルが豆腐 (□) にならないよう CJK 対応フォントを優先指定
matplotlib.rcParams["font.family"] = [
    "Hiragino Sans",  # macOS
    "Hiragino Maru Gothic Pro",
    "DejaVu Sans",  # 最後のフォールバック (英数字)
]
# Hiragino Sans は W0–W9 の太さのみ持ち "normal"(400) が無いため
# 最も近い 500 (medium) を明示し findfont の警告を抑制
matplotlib.rcParams["font.weight"] = "medium"
matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止


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


def plot_unit(scores_by_method, unit_idx, label, title="", filename: str = ""):
    plt.figure(figsize=(9, 3))
    for name, scs in scores_by_method.items():
        s = scs[unit_idx]
        s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
        plt.plot(s, label=name, alpha=0.8)
    lb = label.astype(float)
    plt.fill_between(
        np.arange(len(lb)),
        0,
        1,
        where=lb > 0,
        color="red",
        alpha=0.12,
        label="anomaly区間",
    )
    plt.legend(fontsize=8)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("score(正規化)")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figs", f"{filename}_score.png"))


def make_plots(
    models,
    out_scores,
    ev_labels,
    eval_ids,
    filename: str,
    feature_names=None,
    title_prefix: str = "CMAPSS eval engine",
):
    # 例: 1機のスコア推移 (終盤に向けた緩やかな上昇を各手法が捉えられるか)
    feature_names = SENSORS if feature_names is None else feature_names
    os.makedirs(os.path.join("outputs", "figs"), exist_ok=True)
    plot_unit(
        out_scores,
        0,
        ev_labels[0],
        f"{title_prefix} {eval_ids[0]}",
        filename=filename,
    )

    # 学習されたグラフ (センサ間関係) の可視化
    simmat, adj = models["gdn"].learned_graph()
    plt.figure(figsize=(4.5, 4))
    plt.imshow(simmat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=7)
    plt.yticks(range(len(feature_names)), feature_names, fontsize=7)
    plt.title("GDN学習済みセンサ埋め込みのcos類似度")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figs", f"{filename}_graph.png"))
