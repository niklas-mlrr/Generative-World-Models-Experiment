import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch

from .data import collect_offline_data, save_dataset_npz
from .envs import DreamEnv, GridWorldEnv
from .eval_plot import evaluate_policy, plot_paths, select_strongest_paradox_episode
from .models import WorldModel
from .system import configure_runtime, seed_everything
from .train import train_ppo_in_dream, train_world_model


configure_runtime()


@dataclass
class ExperimentConfig:
    seed: int = 42
    workdir: str = "artifacts/lean"
    grid_size: int = 20
    wall_gap: int = 2
    max_steps: int = 200
    data_steps: int = 12_000
    wall_focus_prob: float = 0.4
    wall_collision_action_prob: float = 0.7
    goal_focus_prob: float = 0.2
    wm_epochs: int = 25
    wm_hidden: int = 32
    collision_undersample_prob: float = 0.6
    collision_label_corrupt_prob: float = 0.45
    ppo_steps: int = 3_000_000
    ppo_n_envs: int = 8
    ppo_target_kl: float = 0.03
    ppo_verbose: int = 1
    dream_shaped_reward_coef: float = 1.5
    dream_goal_radius: float = 2.5
    dream_enforce_action_axis: bool = False
    success_radius: float = 2.5
    eval_episodes: int = 20
    plot_path: str = ""
    show_plot: bool = False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lean reliability paradox experiment.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workdir", type=str, default="artifacts/lean")
    p.add_argument("--ppo-steps", type=int, default=3_000_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--plot-path", type=str, default="")
    p.add_argument("--show-plot", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dream-enforce-action-axis", action=argparse.BooleanOptionalAction, default=False)
    return p


def _save_paths(cfg: ExperimentConfig):
    os.makedirs(cfg.workdir, exist_ok=True)
    data_path = os.path.join(cfg.workdir, "offline_data.npz")
    wm_path = os.path.join(cfg.workdir, "world_model.pt")
    ppo_path = os.path.join(cfg.workdir, "ppo_dream.zip")
    return data_path, wm_path, ppo_path


def _default_plot_path(override: str) -> str:
    if override.strip():
        return override
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, "Figure_1.png")


def run_experiment(cfg: ExperimentConfig) -> dict:
    seed_everything(cfg.seed)
    device = torch.device("cpu")

    real_env = GridWorldEnv(grid_size=cfg.grid_size, max_steps=cfg.max_steps, wall_gap=cfg.wall_gap)
    data_path, wm_path, ppo_path = _save_paths(cfg)

    data = collect_offline_data(
        real_env,
        n_steps=cfg.data_steps,
        seed=cfg.seed,
        wall_focus_prob=cfg.wall_focus_prob,
        wall_collision_action_prob=cfg.wall_collision_action_prob,
        goal_focus_prob=cfg.goal_focus_prob,
    )
    save_dataset_npz(data_path, data)
    print(f"Offline data: {len(data)} transitions -> {data_path}")

    wm = WorldModel(obs_dim=2, n_actions=4, hidden=cfg.wm_hidden)
    wm = train_world_model(
        wm,
        data,
        device=device,
        grid_size=cfg.grid_size,
        epochs=cfg.wm_epochs,
        batch_size=256,
        lr=3e-3,
        collision_undersample_prob=cfg.collision_undersample_prob,
        collision_label_corrupt_prob=cfg.collision_label_corrupt_prob,
        seed=cfg.seed,
    )
    torch.save(wm.state_dict(), wm_path)
    print(f"World model saved: {wm_path}")

    def dream_env_fn():
        return DreamEnv(
            world_model=wm,
            grid_size=cfg.grid_size,
            max_steps=cfg.max_steps,
            device=device,
            use_model_reward=False,
            shaped_reward_coef=cfg.dream_shaped_reward_coef,
            discretize_state=True,
            goal_radius=cfg.dream_goal_radius,
            enforce_action_axis=cfg.dream_enforce_action_axis,
        )

    ppo = train_ppo_in_dream(
        dream_env_fn,
        total_timesteps=cfg.ppo_steps,
        seed=cfg.seed,
        n_envs=cfg.ppo_n_envs,
        verbose=cfg.ppo_verbose,
        target_kl=cfg.ppo_target_kl,
    )
    ppo.save(ppo_path)
    print(f"PPO saved: {ppo_path}")

    real_env_fn = lambda: GridWorldEnv(grid_size=cfg.grid_size, max_steps=cfg.max_steps, wall_gap=cfg.wall_gap)
    goal = np.array(real_env.goal, dtype=np.float32)
    dream_eval = evaluate_policy(
        dream_env_fn,
        ppo,
        episodes=cfg.eval_episodes,
        seed=cfg.seed,
        goal=goal,
        success_radius=cfg.success_radius,
    )
    real_eval = evaluate_policy(
        real_env_fn,
        ppo,
        episodes=cfg.eval_episodes,
        seed=cfg.seed,
        goal=goal,
        success_radius=cfg.success_radius,
    )

    print(
        "Dream eval: "
        f"return={dream_eval['mean_return']:.2f}±{dream_eval['std_return']:.2f}, "
        f"success={dream_eval.get('success_rate', 0.0):.2f}, steps={dream_eval['mean_steps']:.1f}"
    )
    print(
        "Real eval:  "
        f"return={real_eval['mean_return']:.2f}±{real_eval['std_return']:.2f}, "
        f"success={real_eval.get('success_rate', 0.0):.2f}, steps={real_eval['mean_steps']:.1f}"
    )

    best = select_strongest_paradox_episode(
        dream_env_fn,
        real_env_fn,
        ppo,
        episodes=cfg.eval_episodes,
        seed=cfg.seed,
    )
    _, plot_seed, dream_traj, real_traj, dream_return, real_return = best
    print(
        f"Plot episode seed={plot_seed}: "
        f"dream_return={dream_return:.2f}, real_return={real_return:.2f}, gap={dream_return-real_return:.2f}"
    )

    figure_path = _default_plot_path(cfg.plot_path)
    plot_paths(
        real_env,
        dream_traj,
        real_traj,
        title="Reliability Paradox: Dream vs Reality",
        save_path=figure_path,
        show_plot=cfg.show_plot,
        success_radius=cfg.success_radius,
        metrics={
            "dream_success": float(dream_eval.get("success_rate", float("nan"))),
            "real_success": float(real_eval.get("success_rate", float("nan"))),
            "dream_return": float(dream_return),
            "real_return": float(real_return),
            "return_gap": float(dream_return - real_return),
        },
    )

    return {
        "dream_eval": dream_eval,
        "real_eval": real_eval,
        "figure_path": figure_path,
        "plot_seed": int(plot_seed),
        "plot_dream_return": float(dream_return),
        "plot_real_return": float(real_return),
    }


def main() -> None:
    args = _build_parser().parse_args()
    cfg = ExperimentConfig(
        seed=args.seed,
        workdir=args.workdir,
        ppo_steps=args.ppo_steps,
        eval_episodes=args.eval_episodes,
        plot_path=args.plot_path,
        show_plot=bool(args.show_plot),
        dream_enforce_action_axis=bool(args.dream_enforce_action_axis),
    )
    run_experiment(cfg)
