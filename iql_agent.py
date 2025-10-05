#!/usr/bin/env python3
# iql_agent.py
# Implicit Q-Learning (offline RL): V via expectile regression, twin Q, AWR-style actor

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_utils import MLP, polyak_update

class VNet(nn.Module):
    def __init__(self, s_dim: int, hidden=(128,128)):
        super().__init__()
        self.v = MLP(s_dim, 1, hidden, nn.ReLU, None)
    def forward(self, s):
        return self.v(s)

class QNet(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden=(128,128)):
        super().__init__()
        self.q = MLP(s_dim + a_dim, 1, hidden, nn.ReLU, None)
    def forward(self, s, a):
        return self.q(torch.cat([s,a], dim=-1))

class GaussianPolicy(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden=(128,128), log_std_min=-5, log_std_max=2, act_limit=1.0):
        super().__init__()
        self.mu = MLP(s_dim, a_dim, hidden, nn.ReLU, None)
        self.log_std = nn.Parameter(torch.zeros(a_dim))  # state-independent std (simple, CPU-friendly)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_limit = float(act_limit)

    def forward(self, s):
        mu = self.mu(s)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        # reparam sample
        eps = torch.randn_like(mu)
        a = torch.tanh(mu + eps * std) * self.act_limit
        return a, mu, std

    def deterministic(self, s):
        mu = self.mu(s)
        return torch.tanh(mu) * self.act_limit

class IQL:
    def __init__(
        self,
        s_dim: int, a_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        beta: float = 3.0,          # advantage temperature for AWR
        actor_lr: float = 3e-4,
        q_lr: float = 3e-4,
        v_lr: float = 3e-4,
        device: str = "cpu",
        act_limit: float = 1.0
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.beta = beta

        self.v = VNet(s_dim).to(device)
        self.q1 = QNet(s_dim, a_dim).to(device)
        self.q2 = QNet(s_dim, a_dim).to(device)
        self.pi = GaussianPolicy(s_dim, a_dim, act_limit=act_limit).to(device)

        self.v_opt  = torch.optim.Adam(self.v.parameters(), lr=v_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=q_lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=actor_lr)

        self.v_targ = VNet(s_dim).to(device)
        self.v_targ.load_state_dict(self.v.state_dict())

    def _expectile_loss(self, diff):
        # expectile regression loss: weight errors asymmetrically
        w = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (w * (diff**2)).mean()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s  = batch["s"].to(self.device)
        a  = batch["a"].to(self.device)
        r  = batch["r"].to(self.device)
        sp = batch["sp"].to(self.device)
        d  = batch["d"].to(self.device)

        with torch.no_grad():
            v_sp = self.v_targ(sp)
            y = r + (1.0 - d) * self.gamma * v_sp

        # ----- Q update (twin) -----
        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # ----- V update via expectile regression -----
        with torch.no_grad():
            q_min = torch.minimum(q1_pred, q2_pred)
        v_pred = self.v(s)
        v_loss = self._expectile_loss(q_min - v_pred)

        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()
        polyak_update(self.v_targ, self.v, self.tau)

        # ----- Actor update via advantage-weighted regression -----
        with torch.no_grad():
            adv = torch.minimum(self.q1(s, a), self.q2(s, a)) - self.v(s)
            weights = torch.clamp(torch.exp(adv * self.beta), max=100.0)  # stabilize
        a_samp, mu, std = self.pi(s)  # sample but we’ll supervise toward dataset a
        # Weighted MSE to dataset action
        pi_loss = (weights * ((self.pi.deterministic(s) - a) ** 2).sum(dim=1, keepdim=True)).mean()

        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "v_loss":  float(v_loss.item()),
            "pi_loss": float(pi_loss.item()),
        }
