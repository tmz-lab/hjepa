import torch
from torch import nn


class IsotropyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            n, d = x.shape
            mu = x.mean(dim=0)
            xc = x - mu
            cov = xc.T @ xc / max(n - 1, 1)
            eye = torch.eye(d, device=x.device)
            lam = torch.linalg.eigvalsh(cov + self.eps * eye).clamp_min(1e-12)
            return ((lam.sqrt() - 1).pow(2).sum() + mu.pow(2).sum()) / d


class FixedRandomSubspace(nn.Module):
    def __init__(self, dim, rank, *, seed=0):
        super().__init__()
        self.dim = dim
        self.rank = rank
        gen = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=gen), mode="reduced")
        self.register_buffer("basis", q)

    def encode(self, h):
        return h @ self.basis.to(h.dtype)


class CausalPhaseContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, s):
        prev = torch.cat([s[:, :1], s[:, :-1]], dim=1)
        return torch.cat([s, s - prev], dim=-1)


class NetAction(nn.Module):
    def __init__(self, action_dim, *, net_steps=1):
        super().__init__()
        self.action_dim = action_dim
        self.net_steps = net_steps

    def forward(self, a):
        a = a.float().reshape(*a.shape[:2], self.net_steps, self.action_dim)
        return a.mean(dim=-2)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
