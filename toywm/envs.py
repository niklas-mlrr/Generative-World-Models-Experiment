from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from .models import WorldModel
from .system import get_torch_device


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = 20, max_steps: int = 200, wall_gap: int = 1):
        super().__init__()
        assert grid_size >= 5

        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.wall_gap = int(wall_gap)
        assert self.wall_gap >= 1

        self.start = (0, 0)
        self.goal = (self.grid_size - 1, self.grid_size - 1)
        self.wall_x = self.grid_size // 2
        self.wall_ys = set(range(self.wall_gap, self.grid_size - self.wall_gap))

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.grid_size - 1.0, self.grid_size - 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self._pos = None
        self._steps = 0

    def _is_blocked(self, x: int, y: int, nx: int, ny: int) -> bool:
        if ny in self.wall_ys:
            if x == self.wall_x - 1 and nx == self.wall_x and y == ny:
                return True
            if x == self.wall_x and nx == self.wall_x - 1 and y == ny:
                return True
        return False

    def _obs(self) -> np.ndarray:
        x, y = self._pos
        return np.array([x, y], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._pos = self.start
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        self._steps += 1

        x, y = self._pos
        nx, ny = x, y

        if action == 0:
            ny = min(self.grid_size - 1, y + 1)
        elif action == 1:
            nx = min(self.grid_size - 1, x + 1)
        elif action == 2:
            ny = max(0, y - 1)
        elif action == 3:
            nx = max(0, x - 1)
        else:
            raise ValueError("Invalid action")

        if self._is_blocked(x, y, nx, ny):
            nx, ny = x, y

        self._pos = (nx, ny)

        reward = -0.1
        terminated = self._pos == self.goal
        if terminated:
            reward += 10.0

        truncated = self._steps >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {}


class DreamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        world_model: WorldModel,
        grid_size: int = 20,
        max_steps: int = 200,
        device: Optional[torch.device] = None,
        use_model_reward: bool = False,
        shaped_reward_coef: float = 0.0,
        discretize_state: bool = True,
        goal_radius: float = 0.75,
        enforce_action_axis: bool = True,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.use_model_reward = bool(use_model_reward)
        self.shaped_reward_coef = float(shaped_reward_coef)
        self.discretize_state = bool(discretize_state)
        self.goal_radius = float(goal_radius)
        self.enforce_action_axis = bool(enforce_action_axis)

        self.start = (0.0, 0.0)
        self.goal = (float(self.grid_size - 1), float(self.grid_size - 1))

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.grid_size - 1.0, self.grid_size - 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self.device = device if device is not None else get_torch_device()
        self.world_model = world_model.to(self.device)
        self.world_model.eval()

        self._obs = None
        self._steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._obs = np.array(self.start, dtype=np.float32)
        self._steps = 0
        return self._obs.copy(), {}

    def step(self, action: int):
        self._steps += 1

        prev_obs = self._obs.copy()
        obs_t = torch.from_numpy(self._obs[None, :]).to(self.device)
        act_t = torch.tensor([action], device=self.device, dtype=torch.int64)

        with torch.no_grad():
            next_obs_pred, reward_pred = self.world_model(obs_t, act_t)

        next_obs = next_obs_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        reward = float(reward_pred.item())

        next_obs = np.clip(next_obs, 0.0, float(self.grid_size - 1)).astype(np.float32)

        if self.discretize_state:
            next_obs = np.rint(next_obs).astype(np.float32)

        if self.enforce_action_axis:
            # Keep learned dynamics action-consistent:
            # up/down may only change y, left/right may only change x.
            if action in (0, 2):
                next_obs[0] = prev_obs[0]
            elif action in (1, 3):
                next_obs[1] = prev_obs[1]

            # Prevent multi-cell jumps in one step.
            dx = float(np.clip(next_obs[0] - prev_obs[0], -1.0, 1.0))
            dy = float(np.clip(next_obs[1] - prev_obs[1], -1.0, 1.0))
            next_obs = np.array([prev_obs[0] + dx, prev_obs[1] + dy], dtype=np.float32)
            if self.discretize_state:
                next_obs = np.rint(next_obs).astype(np.float32)

        self._obs = next_obs

        dist = float(np.linalg.norm(self._obs - np.array(self.goal, dtype=np.float32)))
        terminated = dist < self.goal_radius
        truncated = self._steps >= self.max_steps

        if self.use_model_reward:
            reward = float(reward)
        else:
            reward = -0.1
            if terminated:
                reward += 10.0

            if self.shaped_reward_coef != 0.0:
                prev_dist = float(np.linalg.norm(prev_obs - np.array(self.goal, dtype=np.float32)))
                reward += self.shaped_reward_coef * (prev_dist - dist)

        return self._obs.copy(), reward, terminated, truncated, {}
