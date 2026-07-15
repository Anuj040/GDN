import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.data_utils.prepare_data import get_df, unit_array
from src.utils.data_utils.prepare_dataset import Scaler, build_eval
from src.utils.eval_utils.metrics import add_result, eval_scores, lead_time
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
        lr=1e-3,
        batch=128,
        smooth=3,
        seed=SEED,
        verbose=False,
    ):
        self.p = dict(
            window=window,
            topk=topk,
            emb_dim=emb_dim,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            batch=batch,
            smooth=smooth,
            seed=seed,
            verbose=verbose,
        )

    def prepare_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        df = get_df()
        units_all = sorted(df["unit"].unique())
        print("engines:", len(units_all), " rows:", len(df))
        torch.manual_seed(self.p["seed"])
        rng = np.random.default_rng(SEED)
        ids = np.array(units_all)
        rng.shuffle(ids)
        train_ids, eval_ids = ids[:70], ids[70:]

        raw_train_healthy = [unit_array(u, df)[:HEALTHY_CYCLES] for u in train_ids]
        sc = Scaler().fit(np.vstack(raw_train_healthy))

        n_val = 14  # 健全データの2割を検証(較正)用に
        self.tr_units = [sc.transform(x) for x in raw_train_healthy[:-n_val]]
        self.va_units = [sc.transform(x) for x in raw_train_healthy[-n_val:]]
        raw_eval = [unit_array(u, df) for u in eval_ids]
        self.ev_units, self.ev_labels, self.ev_masks, self.ev_fails = build_eval(
            raw_eval, sc
        )

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
            if self.p["verbose"]:
                print(f"  epoch {ep+1:02d} val_mse={vl:.4f}")
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
        dataset, experiment = "CMAPSS-FD001", "main"
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


if __name__ == "__main__":
    trainer = GDNAnomaly()
    trainer.fit()
    trainer.run_benchmark()
