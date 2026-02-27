from typing import List

import gymnasium as gym
import numpy as np

from .types import Transition


def collect_offline_data(
    env: gym.Env,
    n_steps: int = 10_000,
    seed: int = 0,
    wall_focus_prob: float = 0.2,
    wall_collision_action_prob: float = 0.7,
    goal_focus_prob: float = 0.2,
) -> List[Transition]:
    rng = np.random.default_rng(seed)

    data: List[Transition] = []
    obs, _ = env.reset(seed=seed)

    can_set_pos = hasattr(env, "_pos") and hasattr(env, "wall_x")

    for _ in range(n_steps):
        targeted_wall_sample = False
        forced_action = None
        if can_set_pos and rng.random() < wall_focus_prob:
            if len(env.wall_ys) > 0:
                y = int(rng.choice(sorted(env.wall_ys)))
            else:
                y = int(rng.integers(1, env.grid_size - 1))
            side = int(rng.integers(0, 2))
            x = env.wall_x - 1 if side == 0 else env.wall_x
            env._pos = (x, y)
            obs = env._obs()
            targeted_wall_sample = True
            if rng.random() < wall_collision_action_prob:
                forced_action = 1 if side == 0 else 3
        elif can_set_pos and rng.random() < goal_focus_prob:
            x = int(rng.integers(max(0, int(0.55 * env.grid_size)), env.grid_size))
            y = int(rng.integers(max(0, int(0.55 * env.grid_size)), env.grid_size))
            env._pos = (x, y)
            obs = env._obs()

        action = int(forced_action) if targeted_wall_sample and forced_action is not None else int(env.action_space.sample())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated or truncated)

        data.append(
            Transition(
                obs=np.array(obs, dtype=np.float32),
                action=action,
                reward=float(reward),
                next_obs=np.array(next_obs, dtype=np.float32),
                done=done,
            )
        )

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return data


def save_dataset_npz(path: str, data: List[Transition]) -> None:
    obs = np.stack([tr.obs for tr in data], axis=0).astype(np.float32)
    act = np.array([tr.action for tr in data], dtype=np.int64)
    rew = np.array([tr.reward for tr in data], dtype=np.float32)
    next_obs = np.stack([tr.next_obs for tr in data], axis=0).astype(np.float32)
    done = np.array([tr.done for tr in data], dtype=np.float32)
    np.savez_compressed(path, obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)


def load_dataset_npz(path: str) -> List[Transition]:
    arr = np.load(path)
    obs = arr["obs"]
    act = arr["act"]
    rew = arr["rew"]
    next_obs = arr["next_obs"]
    done = arr["done"]
    data: List[Transition] = []
    for i in range(len(act)):
        data.append(
            Transition(
                obs=obs[i].astype(np.float32),
                action=int(act[i]),
                reward=float(rew[i]),
                next_obs=next_obs[i].astype(np.float32),
                done=float(done[i]),
            )
        )
    return data
