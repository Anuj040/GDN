import torch
import torch.nn.functional as F
from torch import nn

# ===== 1d. GDN 実装 (Deng & Hooi, AAAI 2021 の簡略・忠実版) =====
# ・センサ埋め込み v_i の cos 類似度 top-k で有向グラフを学習
# ・g_i = [v_i || W x_i] を用いたグラフ注意で近傍のスライディング窓を集約
# ・v_i ∘ z_i を MLP に通して各センサの次時刻値を予測
# ・異常スコア: 検証誤差の median/IQR で正規化した誤差の max_i


class GDNNet(nn.Module):
    def __init__(self, n_nodes, window, emb_dim=64, hidden=128, topk=5):
        super().__init__()
        self.n, self.w, self.k = n_nodes, window, min(topk, n_nodes - 1)
        self.emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.Wx = nn.Linear(window, emb_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(2 * emb_dim))
        self.a_dst = nn.Parameter(torch.empty(2 * emb_dim))
        nn.init.uniform_(self.a_src, -0.1, 0.1)
        nn.init.uniform_(self.a_dst, -0.1, 0.1)
        self.out = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def adjacency(self):
        v = F.normalize(self.emb, dim=1)
        sim = v @ v.t()
        sim.fill_diagonal_(-float("inf"))
        idx = sim.topk(self.k, dim=1).indices
        mask = torch.zeros(self.n, self.n, dtype=torch.bool, device=self.emb.device)
        mask.scatter_(1, idx, True)
        mask.fill_diagonal_(True)  # 自己ループ
        return mask

    def forward(self, x):  # x: (B, N, w)
        B = x.size(0)
        h = self.Wx(x)  # (B,N,d)
        v = self.emb.unsqueeze(0).expand(B, -1, -1)  # (B,N,d)
        g = torch.cat([v, h], dim=-1)  # (B,N,2d)
        e = F.leaky_relu(
            (g @ self.a_src).unsqueeze(2) + (g @ self.a_dst).unsqueeze(1), 0.2
        )  # (B,N,N)
        mask = self.adjacency()
        e = e.masked_fill(~mask.unsqueeze(0), -1e9)
        alpha = torch.softmax(e, dim=2)
        z = torch.relu(alpha @ h)  # (B,N,d)
        o = z * self.emb.unsqueeze(0)
        return self.out(o).squeeze(-1)  # (B,N)
