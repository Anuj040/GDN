import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.data_utils.prepare_data import SENSORS, get_df, unit_array
from src.utils.data_utils.prepare_dataset import Scaler, build_eval
from src.utils.eval_utils.metrics import (add_result, eval_scores, lead_time,
                                          make_plots)
from src.utils.model_utils.aux_methods import MTMethod, SigmaRule, StaticPCA
from src.utils.model_utils.gdn import GDNNet
from src.utils.preprocess_utils.make_samples import make_windows

warnings.filterwarnings("ignore")
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
# ===== 2b. CMAPSS 前処理 =====
HEALTHY_CYCLES = 50


class GDNAnomaly:
    name = "GDN"

    def __init__(
        self,
        window=10,
        topk=5,
        emb_dim=64,
        hidden=128,
        epochs=25,
        lr=1e-2,
        batch=128,
        smooth=3,
        graph_mode="gumbel",
        tau=1.0,
        tau_min=0.1,
        gumbel_hard=True,
        verbose=True,
        dataset="cmapss",
    ):
        self.dataset = dataset
        self.p = dict(
            window=window,
            topk=topk,
            emb_dim=emb_dim,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            batch=batch,
            smooth=smooth,
            graph_mode=graph_mode,
            tau=tau,
            tau_min=tau_min,  # None -> constant tau; else anneal tau -> tau_min
            gumbel_hard=gumbel_hard,
            verbose=verbose,
        )

    def _tau_at(self, ep: int) -> float:
        """Gumbel-Softmax temperature for epoch ``ep`` (0-indexed).

        Geometric anneal from ``tau`` (ep 0) down to ``tau_min`` (last epoch),
        following Jang et al. (2017). Returns constant ``tau`` when ``tau_min``
        is unset or a single-epoch run is requested.
        """
        tau0, tau_min, epochs = self.p["tau"], self.p["tau_min"], self.p["epochs"]
        if tau_min is None or epochs <= 1:
            return tau0
        frac = ep / (epochs - 1)  # 0.0 -> 1.0 across training
        return tau0 * (tau_min / tau0) ** frac

    def _load_cmapss(self) -> None:
        """CMAPSS: multi-engine run-to-failure; train on healthy prefixes."""
        df = get_df()
        units_all = sorted(df["unit"].unique())
        print("engines:", len(units_all), " rows:", len(df))
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        ids = np.array(units_all)
        rng.shuffle(ids)
        train_ids, self.eval_ids = ids[:70], ids[70:]

        raw_train_healthy = [unit_array(u, df)[:HEALTHY_CYCLES] for u in train_ids]
        sc = Scaler().fit(np.vstack(raw_train_healthy))

        n_val = 14  # 健全データの2割を検証(較正)用に
        self.tr_units = [sc.transform(x) for x in raw_train_healthy[:-n_val]]
        self.va_units = [sc.transform(x) for x in raw_train_healthy[-n_val:]]
        raw_eval = [unit_array(u, df) for u in self.eval_ids]
        self.ev_units, self.ev_labels, self.ev_masks, self.ev_fails = build_eval(
            raw_eval, sc
        )
        self.feature_names = SENSORS

    def _load_msl(self, val_ratio: float = 0.2) -> None:
        """MSL: single continuous series. ``train.csv`` is all-normal, ``test.csv``
        carries per-timestep anomaly labels in its ``attack`` column.

        Data prep is kept aligned with the *original* GDN pipeline for an
        apples-to-apples benchmark: features are ordered by ``list.txt`` (the
        original ``feature_map``) and used raw — the MSL csv is already
        normalised, and the original applies no further scaling. A tail slice of
        the all-normal training series is held out for validation.
        """
        base = os.path.join("data", "msl")
        train = pd.read_csv(os.path.join(base, "train.csv"), index_col=0)
        test = pd.read_csv(os.path.join(base, "test.csv"), index_col=0)

        labels = (
            test["attack"].to_numpy().astype(int)
            if "attack" in test.columns
            else np.zeros(len(test), dtype=int)
        )
        with open(os.path.join(base, "list.txt")) as f:
            feats = [ln.strip() for ln in f if ln.strip()]
        self.feature_names = feats

        Xtr_raw = train[feats].to_numpy(dtype=np.float32)
        Xte_raw = test[feats].to_numpy(dtype=np.float32)
        print(
            "features:",
            len(feats),
            " train rows:",
            len(Xtr_raw),
            " test rows:",
            len(Xte_raw),
        )

        # hold out the tail of the all-normal training series for validation,
        # keeping at least one window's worth of points on each side
        n_val = max(self.p["window"] + 1, int(len(Xtr_raw) * val_ratio))
        self.tr_units = [Xtr_raw[:-n_val]]
        self.va_units = [Xtr_raw[-n_val:]]
        self.ev_units = [Xte_raw]
        self.ev_labels = [labels]
        self.eval_ids = ["msl-test"]

    def prepare_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dataset == "msl":
            self._load_msl()
        else:
            self._load_cmapss()

        print(
            "train units:",
            len(self.tr_units),
            "val:",
            len(self.va_units),
            "eval:",
            len(self.ev_units),
        )

        Xtr, Ytr = self._windows_of(self.tr_units)
        Xva, Yva = self._windows_of(self.va_units)
        self.n_nodes = Xtr.shape[1]

        Xtr_t = torch.tensor(Xtr)
        Ytr_t = torch.tensor(Ytr)
        Xva_t = torch.tensor(Xva, device=DEVICE)
        Yva_t = torch.tensor(Yva, device=DEVICE)
        ds = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
        self.dl = torch.utils.data.DataLoader(
            ds, batch_size=self.p["batch"], shuffle=True
        )

        return Xva_t, Yva_t

    def prepare_model(self) -> None:
        self.model = GDNNet(
            self.n_nodes,
            self.p["window"],
            self.p["emb_dim"],
            self.p["hidden"],
            self.p["topk"],
            graph_mode=self.p["graph_mode"],
            tau=self.p["tau"],
            gumbel_hard=self.p["gumbel_hard"],
        ).to(DEVICE)

    def _windows_of(self, units):
        xs, ys = [], []
        for X in units:
            a, b = make_windows(X, self.p["window"])
            if len(a):
                xs.append(a)
                ys.append(b)
        return (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)

    def fit(self):
        Xva_t, Yva_t = self.prepare_dataset()
        self.prepare_model()

        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.p["lr"], weight_decay=1e-5
        )
        best, best_state = np.inf, None
        for ep in range(self.p["epochs"]):
            self.model.tau = self._tau_at(ep)  # anneal Gumbel temperature
            self.model.train()
            for xb, yb in self.dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = F.mse_loss(self.model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = F.mse_loss(self.model(Xva_t), Yva_t).item()
            if vl < best:
                best = vl
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
            if self.p["verbose"] and (ep + 1) % 5 == 0:
                print(f"  epoch {ep+1:02d} val_mse={vl:.4f} tau={self.model.tau:.3f}")
        self.model.load_state_dict(best_state)
        # 検証誤差で正規化係数 (median / IQR) を較正
        with torch.no_grad():
            err = (self.model(Xva_t) - Yva_t).abs().cpu().numpy()
        self.med = np.median(err, axis=0)
        q75, q25 = np.percentile(err, 75, axis=0), np.percentile(err, 25, axis=0)
        self.iqr = np.maximum(q75 - q25, 1e-3)
        return self

    def score(self, X):
        w = self.p["window"]
        xs, ys = make_windows(X, w)
        if len(xs) == 0:
            return np.zeros(len(X))
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xs), 1024):
                xb = torch.tensor(xs[i : i + 1024], device=DEVICE)
                preds.append(self.model(xb).cpu().numpy())
        err = np.abs(np.concatenate(preds) - ys)
        a = (err - self.med) / self.iqr
        s = a.max(axis=1)
        k = self.p["smooth"]
        if k > 1:
            s = np.convolve(s, np.ones(k) / k, mode="same")
        return np.concatenate([np.full(w, s[0]), s])  # 先頭 w 点はパディング

    def learned_graph(self):
        with torch.no_grad():
            v = F.normalize(self.model.emb, dim=1)
            return (v @ v.t()).cpu().numpy(), self.model.adjacency().cpu().numpy()

    def run_benchmark(self):
        thr_q = 0.99
        persist = 5
        dataset = "MSL" if self.dataset == "msl" else "CMAPSS-FD001"
        experiment = "main"
        methods = ("gdn", "mt", "sigma", "pca")

        Xtr = np.vstack(self.tr_units)
        np.vstack(self.va_units)
        out_scores = {}
        models = {}
        if "sigma" in methods:
            models["sigma"] = SigmaRule().fit(Xtr)
        if "pca" in methods:
            models["pca"] = StaticPCA().fit(Xtr)
        if "mt" in methods:
            models["mt"] = MTMethod().fit(Xtr)
        models["gdn"] = self

        for key, m in models.items():

            val_ref = np.concatenate([m.score(v) for v in self.va_units])
            thr = np.quantile(val_ref, thr_q)
            scs = [m.score(u) for u in self.ev_units]
            y = np.concatenate(self.ev_labels)
            s = np.concatenate(scs)
            mk = np.concatenate(self.ev_masks)
            met = eval_scores(y, s, mk)
            if self.ev_fails is not None:
                lts, miss = [], 0
                for sc, fi in zip(scs, self.ev_fails):
                    lt = lead_time(sc, thr, fi, persist)
                    if lt is None:
                        miss += 1
                    else:
                        lts.append(lt)
                met["LeadTime"] = float(np.mean(lts)) if lts else np.nan
                met["MissRate"] = miss / len(scs)
                add_result(dataset, experiment, m.name, met)
                out_scores[m.name] = scs
                print(
                    f"[{dataset}/{experiment}] {m.name:22s} "
                    + " ".join(f"{k}={v:.3f}" for k, v in met.items() if v == v)
                )
                make_plots(
                    models,
                    out_scores,
                    self.ev_labels,
                    self.eval_ids,
                    f"{dataset}_5gmblnoslf_lr10",
                    feature_names=self.feature_names,
                    title_prefix=f"{dataset} eval",
                )

    def _predict_series(self, X):
        """Next-step predictions of the trained GDNNet over a full series ``X``.

        Returns ``(pred, gt)`` each shaped (T-window, N), aligned to the windows
        produced by :func:`make_windows` (stride 1) — the format the original
        ``evaluate`` helpers expect per feature.
        """
        xs, ys = make_windows(X, self.p["window"])
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xs), 1024):
                xb = torch.tensor(xs[i : i + 1024], device=DEVICE)
                preds.append(self.model(xb).cpu().numpy())
        return np.concatenate(preds), ys

    def run_benchmark_msl(self):
        """Evaluate GDNNet on MSL with the *original* GDN metrics.

        Builds ``test_result`` / ``val_result`` in the original ``[predicted,
        ground, labels]`` layout and defers to the repo-root ``evaluate`` module,
        so the reported F1 / precision / recall / AUC are computed identically to
        ``sh run.sh cpu msl`` — the only difference is the model. Kept separate
        from the CMAPSS ``run_benchmark`` and selected via ``--dataset msl``.
        """
        from evaluate import get_best_performance_data, get_full_err_scores

        w = self.p["window"]
        pred_te, gt_te = self._predict_series(self.ev_units[0])
        pred_va, gt_va = self._predict_series(self.va_units[0])
        lab_te = np.asarray(self.ev_labels[0])[w:]  # label at each predicted step

        n_feat = pred_te.shape[1]
        lab_te_mat = np.repeat(lab_te[:, None], n_feat, axis=1)
        lab_va_mat = np.zeros_like(pred_va)  # validation series is all-normal

        test_result = [pred_te.tolist(), gt_te.tolist(), lab_te_mat.tolist()]
        val_result = [pred_va.tolist(), gt_va.tolist(), lab_va_mat.tolist()]

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)
        test_labels = np.array(test_result)[2, :, 0].tolist()
        f1, pre, rec, auc, thr = get_best_performance_data(
            test_scores, test_labels, topk=1
        )
        print(
            "=========================** Result (MSL / GDNNet) **============================\n"
        )
        print(f"F1 score: {f1}")
        print(f"precision: {pre}")
        print(f"recall: {rec}")
        print(f"AUC: {auc}\n")
        return dict(F1=f1, precision=pre, recall=rec, AUC=auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="cmapss",
        choices=["cmapss", "msl"],
        help="cmapss (default, unchanged) or msl",
    )
    args = parser.parse_args()

    trainer = GDNAnomaly(dataset=args.dataset)
    trainer.fit()
    if args.dataset == "msl":
        trainer.run_benchmark_msl()
    else:
        trainer.run_benchmark()
