#!/usr/bin/env python3
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_utils import MLP, polyak_update

class Actor(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden=(128,128), act_limit=1.0):
        super().__init__()
        self.pi = MLP(s_dim, a_dim, hidden, nn.ReLU, None)
        self.act_limit = float(act_limit)
    def forward(self, s):
        return torch.tanh(self.pi(s)) * self.act_limit

class CriticQ(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden=(128,128)):
        super().__init__()
        self.q = MLP(s_dim + a_dim, 1, hidden, nn.ReLU, None)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q(x)

class TD3BC:
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        act_limit: float = 1.0,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        bc_lambda: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.bc_lambda = bc_lambda
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.total_it = 0

        self.actor = Actor(s_dim, a_dim, act_limit=act_limit).to(device)
        self.actor_targ = Actor(s_dim, a_dim, act_limit=act_limit).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.q1 = CriticQ(s_dim, a_dim).to(device)
        self.q1_targ = CriticQ(s_dim, a_dim).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())

        self.q2 = CriticQ(s_dim, a_dim).to(device)
        self.q2_targ = CriticQ(s_dim, a_dim).to(device)
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.pi_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.act_limit = act_limit
