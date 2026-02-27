import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int = 2, n_actions: int = 4, hidden: int = 32):
        super().__init__()
        in_dim = obs_dim + n_actions

        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.next_head = nn.Linear(hidden, obs_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        a_onehot = F.one_hot(action.long(), num_classes=4).float()
        x = torch.cat([obs, a_onehot], dim=-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        next_obs = self.next_head(x)
        reward = self.reward_head(x).squeeze(-1)
        return next_obs, reward
