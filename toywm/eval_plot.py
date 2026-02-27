from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO

from .envs import GridWorldEnv


def rollout_policy(env: gym.Env, ppo: PPO, seed: int = 0, deterministic: bool = True):
    obs, _ = env.reset(seed=seed)
    traj = [np.array(obs, dtype=np.float32)]
    rewards = []

    done = False
    while not done:
        action, _ = ppo.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        traj.append(np.array(obs, dtype=np.float32))
        rewards.append(float(reward))
        done = bool(terminated or truncated)

    return np.stack(traj, axis=0), float(np.sum(rewards))


def evaluate_policy(
    env_fn,
    ppo: PPO,
    episodes: int = 20,
    seed: int = 0,
    goal: Optional[np.ndarray] = None,
    success_radius: float = 0.75,
) -> Dict[str, Any]:
    returns = []
    lengths = []
    successes = []

    for ep in range(int(episodes)):
        env = env_fn()
        traj, ret = rollout_policy(env, ppo, seed=seed + ep, deterministic=True)
        returns.append(float(ret))
        lengths.append(int(len(traj) - 1))
        if goal is not None:
            d_min = float(np.min(np.linalg.norm(traj - goal[None, :], axis=1)))
            succ = float(d_min <= float(success_radius))
            successes.append(succ)

    out = {
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_steps": float(np.mean(lengths)) if lengths else 0.0,
    }
    if len(successes) > 0:
        out["success_rate"] = float(np.mean(successes))
    return out


def select_strongest_paradox_episode(
    dream_env_fn,
    real_env_fn,
    ppo: PPO,
    episodes: int,
    seed: int,
):
    best = None
    for ep in range(int(episodes)):
        s = seed + ep
        dream_traj, dream_return = rollout_policy(dream_env_fn(), ppo, seed=s, deterministic=True)
        real_traj, real_return = rollout_policy(real_env_fn(), ppo, seed=s, deterministic=True)
        gap = float(dream_return - real_return)
        cand = (gap, s, dream_traj, real_traj, dream_return, real_return)
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def select_representative_paradox_episode(
    dream_env_fn,
    real_env_fn,
    ppo: PPO,
    episodes: int,
    seed: int,
):
    cands = []
    for ep in range(int(episodes)):
        s = seed + ep
        dream_traj, dream_return = rollout_policy(dream_env_fn(), ppo, seed=s, deterministic=True)
        real_traj, real_return = rollout_policy(real_env_fn(), ppo, seed=s, deterministic=True)
        gap = float(dream_return - real_return)
        cands.append((gap, s, dream_traj, real_traj, dream_return, real_return))

    if len(cands) == 0:
        raise ValueError("No episodes for representative selection.")

    gaps = np.array([c[0] for c in cands], dtype=np.float32)
    target = float(np.median(gaps))
    best = min(cands, key=lambda c: abs(c[0] - target))
    return best


def plot_paths(
    grid_env: GridWorldEnv,
    dream_path: np.ndarray,
    real_path: np.ndarray,
    title: str = "Dream vs Reality",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    success_radius: float = 0.75,
    metrics: Optional[Dict[str, float]] = None,
):
    gs = grid_env.grid_size

    fig, (ax_dream, ax_real) = plt.subplots(1, 2, figsize=(14, 7), dpi=140, sharex=True, sharey=True)
    fig.patch.set_facecolor("#f6f7f9")
    for ax in (ax_dream, ax_real):
        ax.set_facecolor("#f0f1f3")
        ax.set_xlim(-0.5, gs - 0.5)
        ax.set_ylim(-0.5, gs - 0.5)
        ax.set_xticks(range(gs))
        ax.set_yticks(range(gs))
        ax.grid(True, which="both", linewidth=0.55, alpha=0.35, color="#9ea3aa")
        ax.set_aspect("equal")
    # sharey=True hides duplicate y tick labels by default; force them visible on the right panel.
    ax_real.tick_params(axis="y", labelleft=True)

    wall_x = grid_env.wall_x
    gap = grid_env.wall_gap

    def draw_world(ax):
        for y in sorted(grid_env.wall_ys):
            ax.add_patch(Rectangle((wall_x - 0.5, y - 0.5), 1.0, 1.0, color="#3a3a3a", alpha=0.32))
        for y in range(0, gap):
            ax.add_patch(Rectangle((wall_x - 0.5, y - 0.5), 1.0, 1.0, color="#7fd38c", alpha=0.26))
        for y in range(gs - gap, gs):
            ax.add_patch(Rectangle((wall_x - 0.5, y - 0.5), 1.0, 1.0, color="#7fd38c", alpha=0.26))
        ax.text(wall_x + 0.6, gap - 0.5, f"gap={gap}", color="#187f33", fontsize=9, va="bottom", ha="left", alpha=0.95)
        ax.text(wall_x + 0.6, gs - gap - 0.5, f"gap={gap}", color="#187f33", fontsize=9, va="top", ha="left", alpha=0.95)

        sx, sy = grid_env.start
        gx, gy = grid_env.goal
        ax.scatter([sx], [sy], c="#128a1f", s=120, label="Start", zorder=7)
        ax.scatter([gx], [gy], c="#f1c40f", s=120, label="Goal", zorder=7)

    draw_world(ax_dream)
    draw_world(ax_real)

    sx, sy = grid_env.start
    gx, gy = grid_env.goal
    goal = np.array([gx, gy], dtype=np.float32)

    def extend_to_goal_if_reached(path: np.ndarray) -> np.ndarray:
        # If an episode is counted as "success" via radius, draw the line all the way to the goal cell
        # so the visualization matches the reported metric.
        d_min = float(np.min(np.linalg.norm(path - goal[None, :], axis=1)))
        if d_min <= float(success_radius):
            if not np.allclose(path[-1], goal):
                return np.concatenate([path, goal[None, :]], axis=0)
        return path

    dream_path = extend_to_goal_if_reached(dream_path)
    real_path = extend_to_goal_if_reached(real_path)

    ax_dream.plot(dream_path[:, 0], dream_path[:, 1], color="#1f4de3", linewidth=2.6, marker="o", markersize=2.8, alpha=0.95, label="Dream trajectory")
    ax_real.plot(real_path[:, 0], real_path[:, 1], color="#e42222", linewidth=2.6, marker="o", markersize=2.8, alpha=0.95, label="Real trajectory")

    # Step markers every few timesteps to make ordering explicit.
    def add_step_labels(ax, path, color):
        # Label only movement points (not repeated stuck states) to avoid unreadable text piles.
        moving_idx = [0]
        for i in range(1, len(path)):
            if np.linalg.norm(path[i] - path[i - 1]) > 0.25:
                moving_idx.append(i)
        if moving_idx[-1] != len(path) - 1:
            moving_idx.append(len(path) - 1)

        target_labels = 6
        step_stride = max(1, len(moving_idx) // target_labels)
        picked = moving_idx[::step_stride]
        if picked[-1] != moving_idx[-1]:
            picked.append(moving_idx[-1])

        for i in picked:
            x, y = path[i]
            ax.text(x + 0.12, y + 0.12, f"t={i}", color=color, fontsize=7, alpha=0.9)
        if len(path) > 1:
            ax.annotate(
                "",
                xy=(path[min(1, len(path) - 1), 0], path[min(1, len(path) - 1), 1]),
                xytext=(path[0, 0], path[0, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.8),
            )

    add_step_labels(ax_dream, dream_path, "#1f4de3")
    add_step_labels(ax_real, real_path, "#e42222")

    # Mark failed transfer endpoint in real panel and show dream-only continuation hint.
    ax_real.scatter([real_path[-1, 0]], [real_path[-1, 1]], marker="x", s=120, c="#7f1d1d", linewidths=2.5, zorder=8)
    ax_real.text(real_path[-1, 0] + 0.2, real_path[-1, 1] - 0.2, "Blocked in reality", color="#7f1d1d", fontsize=8)
    end_gap = float(np.linalg.norm(real_path[-1] - dream_path[-1]))
    if end_gap > 1.5:
        ax_real.plot(
            [real_path[-1, 0], dream_path[-1, 0]],
            [real_path[-1, 1], dream_path[-1, 1]],
            linestyle="--",
            linewidth=1.4,
            color="#6b7280",
            alpha=0.85,
        )
        ax_real.text(
            (real_path[-1, 0] + dream_path[-1, 0]) / 2.0 + 0.2,
            (real_path[-1, 1] + dream_path[-1, 1]) / 2.0,
            "Dream-only continuation",
            fontsize=7,
            color="#374151",
        )

    ax_dream.set_title("Dream World", fontsize=13, weight="semibold")
    ax_real.set_title("Real World", fontsize=13, weight="semibold")
    ax_dream.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax_real.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle(title, fontsize=16, weight="semibold", y=0.98)

    # Metrics box for immediate interpretation.
    if metrics is not None:
        box_text = (
            f"Dream success: {metrics.get('dream_success', float('nan')):.2f}\n"
            f"Real success: {metrics.get('real_success', float('nan')):.2f}\n"
            f"Dream return: {metrics.get('dream_return', float('nan')):.2f}\n"
            f"Real return: {metrics.get('real_return', float('nan')):.2f}\n"
            f"Return gap: {metrics.get('return_gap', float('nan')):+.2f}"
        )
        fig.text(
            0.5,
            0.02,
            box_text,
            ha="center",
            va="bottom",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#9ca3af", alpha=0.95),
        )

    plt.tight_layout(rect=(0.01, 0.06, 0.99, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"Saved figure: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
