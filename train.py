from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from hamiltonian_jepa import HamiltonianJEPA
from hamiltonian_predictor import ImplicitHamiltonianPredictor
from hjepa_loss import hjepa_loss
from module import MLP, FixedRandomSubspace, IsotropyLoss, NetAction


class Standardize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-8))

    def forward(self, x):
        return ((x - self.mean) / self.std).float()


def column_normalizer(dataset, col):
    data = torch.from_numpy(np.array(dataset.get_col_data(col)))
    data = data[~torch.isnan(data).any(dim=1)]
    norm = Standardize(data.mean(0, keepdim=True), data.std(0, keepdim=True))
    return spt.data.transforms.WrapTorchTransform(norm, source=col, target=col)


def image_preprocessor(img_size):
    to_image = spt.data.transforms.ToImage(**spt.data.dataset_stats.ImageNet, source="pixels", target="pixels")
    resize = spt.data.transforms.Resize(img_size, source="pixels", target="pixels")
    return spt.data.transforms.Compose(to_image, resize)


class SaveModel(Callback):
    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = run_dir

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            torch.save(pl_module.model, self.run_dir / f"hjepa_epoch_{trainer.current_epoch + 1}_object.ckpt")


def forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    percept, emb = self.model.encode_frames(batch["pixels"])
    act = self.model.action_encoder(batch["action"])
    out = hjepa_loss(self.model, emb, percept, act, num_preds=cfg.wm.num_preds, iso=self.iso)
    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in out.items()},
        on_step=stage == "fit",
        on_epoch=True,
        sync_dist=True,
        batch_size=emb.size(0),
    )
    return out


@hydra.main(version_base=None, config_path="config/train", config_name="hjepa")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    dataset.transform = spt.data.transforms.Compose(image_preprocessor(cfg.img_size), column_normalizer(dataset, "action"))
    action_dim = dataset.get_dim("action")

    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=gen)
    loader = OmegaConf.to_container(cfg.loader)
    train = torch.utils.data.DataLoader(train_set, **loader, shuffle=True, drop_last=True, generator=gen)
    val = torch.utils.data.DataLoader(val_set, **loader)

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    projector = MLP(encoder.config.hidden_size, cfg.wm.projector_hidden, cfg.wm.percept_dim)
    predictor = ImplicitHamiltonianPredictor(cfg.wm.rank, action_dim, **cfg.wm.dynamics)
    model = HamiltonianJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=NetAction(action_dim, net_steps=cfg.data.dataset.frameskip),
        state_proj=FixedRandomSubspace(cfg.wm.percept_dim, cfg.wm.rank, seed=cfg.seed),
        projector=projector,
    )

    module = spt.Module(
        model=model,
        iso=IsotropyLoss(),
        forward=partial(forward, cfg=cfg),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            }
        },
    )

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    logger = WandbLogger(project=cfg.wandb.project, name=cfg.subdir) if cfg.wandb.enabled else None

    trainer = pl.Trainer(**cfg.trainer, callbacks=[SaveModel(run_dir)], num_sanity_val_steps=1, logger=logger)
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=train, val=val))()


if __name__ == "__main__":
    run()
