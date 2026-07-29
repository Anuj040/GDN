import numpy as np
from sklearn.decomposition import PCA


# ===== 1b. MT法 (マハラノビス・タグチ) =====
class MTMethod:
    # 単位空間(健全データ)の平均・標準偏差で標準化し、相関行列の(擬似)逆行列で MD^2/k を計算
    name = "MT法(MD)"

    def fit(self, X_unit, X_val=None):
        X = np.asarray(X_unit, float)
        self.mu = X.mean(0)
        self.sd = np.maximum(X.std(0), 1e-8)
        Z = (X - self.mu) / self.sd
        R = np.corrcoef(Z, rowvar=False)
        R = np.atleast_2d(R)
        # near-constant channels (e.g. some MSL sensors) yield NaN correlations;
        # fall back to an identity-like structure so pinv stays well-defined.
        if not np.isfinite(R).all():
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(R, 1.0)
        self.Rinv = np.linalg.pinv(R)
        self.k = X.shape[1]
        return self

    def score(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return np.einsum("ij,jk,ik->i", Z, self.Rinv, Z) / self.k


# ===== 1c. ベースライン =====
class SigmaRule:
    # 単一センサ閾値: 各センサの z スコアの最大値 (3σ ルール相当)
    name = "単一センサ(max|z|)"

    def fit(self, X_unit, X_val=None):
        self.mu = X_unit.mean(0)
        self.sd = np.maximum(X_unit.std(0), 1e-8)
        return self

    def score(self, X):
        return np.abs((X - self.mu) / self.sd).max(axis=1)


class StaticPCA:
    # 健全期間の「ある時点の相関構造」を固定した線形部分空間で監視 (SPE=再構成誤差)
    name = "静的相関(PCA-SPE)"

    def __init__(self, var=0.9):
        self.var = var

    def fit(self, X_unit, X_val=None):
        self.mu = X_unit.mean(0)
        self.sd = np.maximum(X_unit.std(0), 1e-8)
        Z = (X_unit - self.mu) / self.sd
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(Z)
        return self

    def score(self, X):
        Z = (X - self.mu) / self.sd
        Zr = self.pca.inverse_transform(self.pca.transform(Z))
        return ((Z - Zr) ** 2).sum(axis=1)
