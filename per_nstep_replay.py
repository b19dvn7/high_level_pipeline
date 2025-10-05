#!/usr/bin/env python3
# per_nstep_replay.py
# Prioritized Experience Replay with N-step returns (CPU, NumPy only).

from typing import Dict, Optional
import numpy as np

class PERNStepReplay:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        act_dim: int = 0,
        n_step: int = 1,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        seed: int = 7,
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.eps   = float(eps)

        self.ptr = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

        self.s  = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.sp = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.r  = np.zeros((self.capacity, 1), dtype=np.float32)
        self.d  = np.zeros((self.capacity, 1), dtype=np.float32)
        self.a  = None
        if self.act_dim > 0:
            self.a = np.zeros((self.capacity, self.act_dim), dtype=np.float32)

        self.prior = np.ones((self.capacity, 1), dtype=np.float32)
        self.max_prio = 1.0

    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, s, a, r, sp, d, priority: Optional[float] = None):
        i = self.ptr
        self.s[i]  = s
        self.sp[i] = sp
        self.r[i]  = r
        self.d[i]  = d
        if self.a is not None and a is not None:
            self.a[i] = a

        p = self.max_prio if priority is None else float(priority)
        p = max(self.eps, p)
        self.prior[i, 0] = p

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def _n_step_target(self, idx: int):
        ret = 0.0
        g = 1.0
        i = idx
        n = self.size()
        for _ in range(self.n_step):
            if i >= n:
                break
            ret += g * float(self.r[i, 0])
            if self.d[i, 0] > 0.5:
                return ret, self.sp[i], True
            g *= self.gamma
            i += 1
        j = min(idx + self.n_step - 1, n - 1)
        return ret, self.sp[j], False

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        n = self.size()
        p = self.prior[:n, 0] ** self.alpha
        p /= np.sum(p)
        idx = self.rng.choice(n, size=batch_size, replace=True, p=p)

        w = (n * p[idx]) ** (-self.beta)
        w = (w / (np.max(w) + 1e-8)).astype(np.float32).reshape(-1, 1)

        S  = self.s[idx]
        A  = self.a[idx] if self.a is not None else None
        Rn = np.zeros((batch_size, 1), dtype=np.float32)
        SPn= np.zeros_like(self.sp[idx])
        Dn = np.zeros((batch_size, 1), dtype=np.float32)

        for t, ii in enumerate(idx):
            Rn[t, 0], SPn[t], done_flag = self._n_step_target(ii)
            Dn[t, 0] = 1.0 if done_flag else 0.0

        out = {"idx": idx, "w": w, "s": S, "r": Rn, "sp": SPn, "d": Dn}
        if A is not None:
            out["a"] = A
        return out

    def update_priorities(self, idx: np.ndarray, td_err: np.ndarray):
        td = np.abs(td_err).reshape(-1)
        for i, e in zip(idx, td):
            p = float(e) + self.eps
            self.prior[i, 0] = p
            if p > self.max_prio:
                self.max_prio = p
