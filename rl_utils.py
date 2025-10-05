#!/usr/bin/env python3
import math
import torch
import torch.nn as nn

def fanin_init(m):
    if isinstance(m, nn.Linear):
        bound = 1.0 / math.sqrt(m.weight.size(0))
        nn.init.uniform_(m.weight, -bound, +bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 128), act=nn.ReLU, out_act=None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        if out_act is not None:
            layers.append(out_act())
        self.net = nn.Sequential(*layers)
        self.apply(fanin_init)

    def forward(self, x):
        return self.net(x)

def polyak_update(target: nn.Module, online: nn.Module, tau: float):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
