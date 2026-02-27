from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from .models import WorldModel
from .types import Transition


def train_world_model(
    model: WorldModel,
    data: List[Transition],
    device: torch.device,
    grid_size: int,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-3,
    collision_undersample_prob: float = 0.6,
    collision_label_corrupt_prob: float = 0.45,
    seed: int = 0,
) -> WorldModel:
    rng = np.random.default_rng(seed)

    collision_mask = np.array([np.allclose(tr.obs, tr.next_obs) for tr in data], dtype=bool)
    idx_all = np.arange(len(data))
    idx_non_collision = idx_all[~collision_mask]
    idx_collision = idx_all[collision_mask]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    obs_arr = np.stack([tr.obs for tr in data], axis=0).astype(np.float32)
    act_arr = np.array([tr.action for tr in data], dtype=np.int64)
    next_obs_arr = np.stack([tr.next_obs for tr in data], axis=0).astype(np.float32)
    rew_arr = np.array([tr.reward for tr in data], dtype=np.float32)

    if collision_label_corrupt_prob > 0.0 and len(idx_collision) > 0:
        next_obs_arr_train = next_obs_arr.copy()
        corrupt_draw = rng.random(len(idx_collision)) < collision_label_corrupt_prob
        idx_corrupt = idx_collision[corrupt_draw]
        for i in idx_corrupt:
            x = int(round(float(obs_arr[i, 0])))
            y = int(round(float(obs_arr[i, 1])))
            a = int(act_arr[i])
            if a == 1:
                next_obs_arr_train[i] = np.array([min(grid_size - 1, x + 1), y], dtype=np.float32)
            elif a == 3:
                next_obs_arr_train[i] = np.array([max(0, x - 1), y], dtype=np.float32)
    else:
        next_obs_arr_train = next_obs_arr

    for _ in range(epochs):
        keep_collision = rng.random(len(idx_collision)) > collision_undersample_prob
        idx_epoch = np.concatenate([idx_non_collision, idx_collision[keep_collision]])
        rng.shuffle(idx_epoch)

        for start in range(0, len(idx_epoch), batch_size):
            batch_idx = idx_epoch[start : start + batch_size]

            obs = torch.from_numpy(obs_arr[batch_idx]).to(device)
            act = torch.from_numpy(act_arr[batch_idx]).to(device)
            next_obs = torch.from_numpy(next_obs_arr_train[batch_idx]).to(device)
            rew = torch.from_numpy(rew_arr[batch_idx]).to(device)

            pred_next, pred_rew = model(obs, act)
            loss_next = F.mse_loss(pred_next, next_obs)
            loss_rew = F.mse_loss(pred_rew, rew)
            loss = loss_next + 0.5 * loss_rew

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def make_vec_env(env_fn, seed: int, n_envs: int = 1):
    def make_env(rank: int):
        def _init():
            env = env_fn()
            env.reset(seed=seed + rank)
            return env

        return _init

    set_random_seed(seed)
    return DummyVecEnv([make_env(i) for i in range(int(n_envs))])


def train_ppo_in_dream(
    dream_env_fn,
    total_timesteps: int = 30_000,
    seed: int = 0,
    n_envs: int = 8,
    verbose: int = 0,
    target_kl: float = 0.03,
) -> PPO:
    vec_env = make_vec_env(dream_env_fn, seed=seed, n_envs=n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=512,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=target_kl,
        verbose=int(verbose),
        seed=seed,
        device="cpu",
    )

    print(f"Training PPO in DreamEnv for {total_timesteps} timesteps (CPU).")
    model.learn(total_timesteps=total_timesteps, progress_bar=bool(int(verbose) > 0))
    print("PPO training finished.")
    return model
