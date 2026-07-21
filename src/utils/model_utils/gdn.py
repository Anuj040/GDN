import torch
import torch.nn.functional as F
from torch import nn

# ===== 1d. GDN 実装 (Deng & Hooi, AAAI 2021 の簡略・忠実版) =====
# ・センサ埋め込み v_i の cos 類似度 top-k で有向グラフを学習
# ・g_i = [v_i || W x_i] を用いたグラフ注意で近傍のスライディング窓を集約
# ・v_i ∘ z_i を MLP に通して各センサの次時刻値を予測
# ・異常スコア: 検証誤差の median/IQR で正規化した誤差の max_i


class GDNNet(nn.Module):
    def __init__(
        self,
        n_nodes,
        window,
        emb_dim=64,
        hidden=128,
        topk=5,
        graph_mode="hard",
        tau=1.0,
        gumbel_hard=True,
    ):
        super().__init__()
        self.n, self.w, self.k = n_nodes, window, min(topk, n_nodes - 1)
        # graph_mode: "hard"   -> non-differentiable top-k (selection carries no grad)
        #             "gumbel" -> Gumbel-perturbed top-k; gradients flow to every
        #                         embedding through the soft selection scores.
        # tau: Gumbel-Softmax temperature (lower -> sharper/more discrete).
        # gumbel_hard: straight-through -> sparse forward graph, soft backward grad.
        self.graph_mode, self.tau, self.gumbel_hard = graph_mode, tau, gumbel_hard
        self.emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.Wx = nn.Linear(window, emb_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(2 * emb_dim))
        self.a_dst = nn.Parameter(torch.empty(2 * emb_dim))
        nn.init.uniform_(self.a_src, -0.1, 0.1)
        nn.init.uniform_(self.a_dst, -0.1, 0.1)
        self.out = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.skip_self_adjacency = True

    def adjacency(self):
        """Return an (N, N) adjacency weight matrix A used to gate attention.

        Rows index the source node i, columns the neighbour j (A[i, j] > 0 means
        i may attend to j). A self-loop is always added. In "hard" mode A is a
        detached {0, 1} mask (identical to the original top-k graph). In "gumbel"
        mode A is differentiable so gradients reach the embeddings of *all* nodes,
        including ones not selected this pass.
        """
        v = F.normalize(self.emb, dim=1)
        sim = v @ v.t()
        eye = torch.eye(self.n, device=self.emb.device)
        no_self = eye.bool()

        # Deterministic hard top-k for eval, or whenever gumbel is disabled.
        if self.graph_mode == "hard" or not self.training:
            s = sim.masked_fill(no_self, -float("inf"))
            idx = s.topk(self.k, dim=1).indices
            hard = torch.zeros_like(sim)
            hard.scatter_(1, idx, 1.0)
            if not self.skip_self_adjacency:
                hard = hard + eye  # self-loop

            return hard.detach() # non-differentiable graph

        # Gumbel-perturbed top-k (differentiable via straight-through).
        u = torch.rand_like(sim).clamp_(1e-9, 1.0)
        gumbel = -torch.log(-torch.log(u))
        logits = (sim + gumbel).masked_fill(no_self, -float("inf")) / self.tau
        soft = torch.softmax(logits, dim=1)  # grad flows to every non-self edge
        soft_k = soft * self.k  # sum ≈ k, matches hard
        idx = logits.topk(self.k, dim=1).indices
        hard = torch.zeros_like(soft)
        hard.scatter_(1, idx, 1.0)

        # Straight-through: forward value is the sparse k-hot graph, backward
        # gradient is that of the dense softmax.
        mask = (hard + soft_k - soft_k.detach()) if self.gumbel_hard else soft_k
        if not self.skip_self_adjacency:
            mask = mask + eye  # self-loop
        return mask

    def forward(self, x):  # x: (B, N, w)
        B = x.size(0)
        h = self.Wx(x)  # (B,N,d)
        v = self.emb.unsqueeze(0).expand(B, -1, -1)  # (B,N,d)
        g = torch.cat([v, h], dim=-1)  # (B,N,2d)
        e = F.leaky_relu(
            (g @ self.a_src).unsqueeze(2) + (g @ self.a_dst).unsqueeze(1), 0.2
        )  # (B,N,N)
        # Gate the GAT attention by the (possibly soft) adjacency, then renormalise.
        # For a {0,1} mask this is identical to masked_fill(-inf)+softmax; for a
        # soft/straight-through mask it lets graph gradients reach the embeddings.
        A = self.adjacency()  # (N,N)
        alpha = torch.softmax(e, dim=2) * A.unsqueeze(0)
        alpha = alpha / (alpha.sum(dim=2, keepdim=True) + 1e-9)
        z = torch.relu(alpha @ h)  # (B,N,d)
        o = z * self.emb.unsqueeze(0)
        return self.out(o).squeeze(-1)  # (B,N)
