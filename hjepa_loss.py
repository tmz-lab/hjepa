import torch
import torch.nn.functional as F


def pic_loss(model, emb, act, horizon):
    ctx = model.history(emb)
    terms = []
    for t0 in range(emb.size(1) - horizon):
        s = emb[:, t0]
        for i in range(horizon):
            c = ctx[:, t0 + i]
            u = model.predictor.decode_action(s, emb[:, t0 + i + 1] - s, context=c)
            terms.append(F.mse_loss(u.float(), act[:, t0 + i].float()))
            if i + 1 < horizon:
                s = model.predictor(s, act[:, t0 + i], context=c)
    return torch.stack(terms).mean()


def hjepa_loss(model, emb, percept, act, *, num_preds, iso):
    n = emb.size(1) - num_preds
    pred = model.predict_aligned(emb, act, num_preds)
    target = torch.stack([emb[:, 1 + i : n + 1 + i] for i in range(num_preds)], dim=2)
    pred_loss = (pred - target).pow(2).mean()
    pic = pic_loss(model, emb, act, num_preds)
    iso_loss = iso(percept)
    return {
        "loss": pred_loss + pic + iso_loss,
        "pred_loss": pred_loss,
        "pic_loss": pic,
        "iso_loss": iso_loss,
    }
