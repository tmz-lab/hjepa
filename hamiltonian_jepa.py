import torch
from torch import nn

from module import CausalPhaseContext


class HamiltonianJEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, state_proj, projector):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.state_proj = state_proj
        self.projector = projector
        self.frame_norm = nn.LayerNorm(state_proj.dim, elementwise_affine=False)
        self.history = CausalPhaseContext(state_proj.rank)
        self.history_size = 2
        self.predictor.num_frames = 2

    def encode_frames(self, pixels):
        b = pixels.size(0)
        out = self.encoder(pixels.float().flatten(0, 1), interpolate_pos_encoding=True)
        h = self.frame_norm(self.projector(out.last_hidden_state[:, 0])).unflatten(0, (b, -1))
        return h, self.state_proj.encode(h)

    def step(self, hist, action):
        return self.predictor(hist[:, -1], action, context=self.history(hist)[:, -1])

    def predict_aligned(self, emb, act, num_preds):
        b, t, r = emb.shape
        n = t - num_preds
        padded = torch.cat([emb[:, :1], emb], dim=1)
        hist = torch.stack([padded[:, i : i + 2] for i in range(n)], dim=1).flatten(0, 1)
        preds = []
        for i in range(num_preds):
            nxt = self.step(hist, act[:, i : n + i].flatten(0, 1))
            preds.append(nxt.view(b, n, r))
            hist = torch.cat([hist[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=2)

    def rollout(self, info, action_sequence):
        b, n = action_sequence.shape[:2]
        _, hist = self.encode_frames(info["pixels"][:, 0])
        hist = hist.unsqueeze(1).expand(b, n, *hist.shape[1:]).flatten(0, 1)
        act = self.action_encoder(action_sequence.flatten(0, 1))
        preds = [hist[:, -1]]
        for i in range(act.size(1)):
            nxt = self.step(hist, act[:, i])
            preds.append(nxt)
            hist = torch.cat([hist[:, 1:], nxt.unsqueeze(1)], dim=1)
        info["predicted_emb"] = torch.stack(preds, dim=1).unflatten(0, (b, n))
        return info

    def criterion(self, info):
        pred = info["predicted_emb"][..., -1, :]
        goal = info["goal_emb"][:, -1:].detach()
        return (pred - goal).pow(2).sum(dim=-1)

    def get_cost(self, info, action_candidates):
        device = next(self.parameters()).device
        for k, v in info.items():
            if torch.is_tensor(v):
                info[k] = v.to(device)
        goal = info["goal"][:, 0]
        if goal.dim() == 4:
            goal = goal.unsqueeze(1)
        _, info["goal_emb"] = self.encode_frames(goal[:, -1:])
        return self.criterion(self.rollout(info, action_candidates))
