import os

os.environ.setdefault("MUJOCO_GL", "egl")

import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


class MacroPixelHistoryPolicy:
    def __init__(self, policy, *, history_len, action_block):
        self.policy = policy
        self.history_len = history_len
        self.action_block = action_block
        self.frames = None
        self.count = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _reset(self, n):
        self.frames = [deque(maxlen=self.history_len) for _ in range(n)]
        self.count = np.zeros(n, dtype=np.int64)

    def set_env(self, env):
        self.policy.set_env(env)
        self._reset(int(getattr(env, "num_envs", 1)))

    def get_action(self, info, **kwargs):
        pixels = info["pixels"]
        pixels = pixels.detach().cpu().numpy() if torch.is_tensor(pixels) else np.asarray(pixels)
        n = pixels.shape[0]
        if self.frames is None or len(self.frames) != n:
            self._reset(n)

        flush = info.get("_needs_flush")
        for i in range(n):
            if flush is not None and flush[i]:
                self.frames[i].clear()
                self.count[i] = 0
            if self.count[i] % self.action_block == 0:
                self.frames[i].append(pixels[i, 0].copy())
            self.count[i] += 1

        stacked = []
        for i in range(n):
            hist = list(self.frames[i])
            hist = [hist[0]] * (self.history_len - len(hist)) + hist
            stacked.append(np.stack(hist[-self.history_len :]))
        return self.policy.get_action({**info, "pixels": np.stack(stacked)}, **kwargs)


def img_transform(img_size):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


@hydra.main(version_base=None, config_path="config/eval", config_name="pusht")
def run(cfg):
    policy_path = Path(cfg.policy).expanduser().resolve()
    model = torch.load(policy_path, weights_only=False, map_location="cpu").cuda().eval()
    model.requires_grad_(False)

    dataset = swm.data.HDF5Dataset(cfg.eval.dataset_name, keys_to_cache=cfg.dataset.keys_to_cache)
    process = {}
    for col in cfg.dataset.keys_to_cache:
        data = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(data[~np.isnan(data).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]

    transform = {"pixels": img_transform(cfg.eval.img_size), "goal": img_transform(cfg.eval.img_size)}
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg.plan_config),
        process=process,
        transform=transform,
    )
    policy = MacroPixelHistoryPolicy(
        policy,
        history_len=cfg.plan_config.history_len,
        action_block=cfg.plan_config.action_block,
    )

    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col)
    step_idx = dataset.get_col_data("step_idx")
    episodes = np.unique(episode_idx)
    lengths = np.array([step_idx[episode_idx == ep].max() + 1 for ep in episodes])
    max_start = dict(zip(episodes, lengths - cfg.eval.goal_offset_steps - 1))
    valid = np.nonzero(step_idx <= np.array([max_start[ep] for ep in episode_idx]))[0]

    rng = np.random.default_rng(cfg.seed)
    chosen = np.sort(valid[rng.choice(len(valid) - 1, size=cfg.eval.num_eval_total, replace=False)])
    chosen = chosen[cfg.eval.batch_offset : cfg.eval.batch_offset + cfg.eval.num_eval]
    rows = dataset.get_row_data(chosen)

    world = swm.World(
        **cfg.world,
        max_episode_steps=2 * cfg.eval.eval_budget,
        image_shape=(224, 224),
    )
    world.set_policy(policy)
    start = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=rows["step_idx"].tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=rows[col].tolist(),
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video=policy_path.parent if cfg.eval.video else None,
    )
    print(metrics)

    with (policy_path.parent / cfg.output.filename).open("a") as f:
        f.write(f"\n{OmegaConf.to_yaml(cfg)}\nmetrics: {metrics}\nevaluation_time: {time.time() - start}\n")


if __name__ == "__main__":
    run()
