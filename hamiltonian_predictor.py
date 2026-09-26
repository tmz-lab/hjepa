import math

import torch
import torch.nn.functional as F
from torch import nn


def _mlp(in_dim, out_dim, hidden, layers):
    blocks = []
    for _ in range(layers - 1):
        blocks += [nn.Linear(in_dim, hidden), nn.GELU()]
        in_dim = hidden
    blocks.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*blocks)


def _stiefel_qr(raw):
    with torch.autocast(device_type=raw.device.type, enabled=False):
        q, tri = torch.linalg.qr(raw.float(), mode="reduced")
        sign = torch.where(torch.diagonal(tri, dim1=-2, dim2=-1) >= 0, 1.0, -1.0)
        q = q * sign.unsqueeze(-2)
    return q.to(raw.dtype)


class ImplicitHamiltonianPredictor(nn.Module):
    def __init__(self, dim, action_dim, *, hidden=256, layers=3, dt=1.0, head_init_std=1e-3, rho_init=1e-2):
        super().__init__()
        if dim % 2:
            raise ValueError(f"state rank must be even, got {dim}")
        self.dim = dim
        self.action_dim = action_dim
        self.dt = dt
        self.h0 = _mlp(2 * dim, 1, hidden, layers)
        self.g = _mlp(2 * dim, dim * action_dim, hidden, layers)
        self.r = _mlp(2 * dim, dim * (dim + 1) // 2, hidden, layers)

        rows, cols = torch.tril_indices(dim, dim)
        self.register_buffer("_tril_rows", rows, persistent=False)
        self.register_buffer("_tril_cols", cols, persistent=False)
        nn.init.zeros_(self.r[-1].weight)
        diag = math.log(math.expm1(math.sqrt(rho_init / dim)))
        with torch.no_grad():
            self.r[-1].bias.copy_(torch.where(rows == cols, diag, 0.0))
        nn.init.normal_(self.h0[-1].weight, std=head_init_std)
        nn.init.normal_(self.h0[-1].bias, std=head_init_std)

        eye, zero = torch.eye(dim // 2), torch.zeros(dim // 2, dim // 2)
        self.register_buffer("_J", torch.cat([torch.cat([zero, -eye], 1), torch.cat([eye, zero], 1)]))

    def _grad_h(self, s, v):
        track = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            s_in = s.detach().clone().requires_grad_(True)
            energy = self.h0(torch.cat([s_in, v if track else v.detach()], dim=-1)).sum()
            (grad,) = torch.autograd.grad(energy, s_in, create_graph=track)
        return grad

    def _cholesky(self, phase):
        l = phase.new_zeros(*phase.shape[:-1], self.dim, self.dim)
        l[..., self._tril_rows, self._tril_cols] = self.r(phase).to(l.dtype)
        diag = torch.diagonal(l, dim1=-2, dim2=-1)
        return l - torch.diag_embed(diag) + torch.diag_embed(F.softplus(diag))

    def passive(self, s, v):
        grad = self._grad_h(s, v)
        l = self._cholesky(torch.cat([s, v], dim=-1))
        damp = torch.einsum("...ij,...j->...i", l, torch.einsum("...ji,...j->...i", l, grad))
        return grad @ self._J.to(s.dtype).T - damp

    def port(self, phase):
        return _stiefel_qr(self.g(phase).view(*phase.shape[:-1], self.dim, self.action_dim))

    def decode_action(self, s, delta, *, context):
        v = context[..., self.dim:]
        resid = delta / self.dt - self.passive(s, v)
        g = self.port(torch.cat([s, v], dim=-1))
        return torch.einsum("...ij,...i->...j", g, resid.to(g.dtype))

    def forward(self, s, u, *, context):
        v = context[..., self.dim:]
        passive = self.passive(s, v)
        g = self.port(torch.cat([s, v], dim=-1))
        push = torch.einsum("...ij,...j->...i", g, u.to(g.dtype)).to(s.dtype)
        return s + self.dt * (passive + push)
