#!/usr/bin/env python3
# TD3+BC with PER + N-step replay and K-model ensemble training + consensus gate export.
# Robust to schema drift; logs ingest & training cleanly.

import argparse
import json
import time
from typing import List

import duckdb
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from rl_duckdb_data_layer import build_sql
from per_nstep_replay import PERNStepReplay
from td3_bc_agent import TD3BC

STATE_COLS = ["mid","spread","imbalance","mom1","mom3","vol3"]
NEXT_COLS  = ["mid_p","spread_p","imbalance_p","mom1_p","mom3_p","vol3_p"]
REQUIRED   = STATE_COLS + NEXT_COLS + ["fwd_ret"]

def td_error_min_q(agent: TD3BC, batch: dict, device="cpu"):
    with torch.no_grad():
        s  = torch.from_numpy(batch["s"]).to(device)
        a  = torch.from_numpy(batch["a"]).to(device)
        r  = torch.from_numpy(batch["r"]).to(device)
        sp = torch.from_numpy(batch["sp"]).to(device)
        d  = torch.from_numpy(batch["d"]).to(device)

        noise = (torch.randn_like(a) * agent.target_noise).clamp(-agent.noise_clip, agent.noise_clip)
        ap = (agent.actor_targ(sp) + noise).clamp(-agent.act_limit, agent.act_limit)

        q1_tp = agent.q1_targ(sp, ap)
        q2_tp = agent.q2_targ(sp, ap)
        q_tp_min = torch.minimum(q1_tp, q2_tp)
        y = r + (1.0 - d) * agent.gamma * q_tp_min

        q1 = agent.q1(s, a)
        q2 = agent.q2(s, a)
        qmin = torch.minimum(q1, q2)
        td = (y - qmin).squeeze(-1).cpu().numpy()
        return td

def train_one_model(
    per: PERNStepReplay,
    seed: int,
    steps: int,
    batch_size: int,
    device: str,
    bc_lambda: float,
    policy_delay: int,
    print_every: int = 200,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = TD3BC(
        s_dim=6, a_dim=1, act_limit=1.0,
        actor_lr=1e-3, critic_lr=1e-3,
        gamma=0.99, tau=0.005, policy_delay=policy_delay,
        bc_lambda=bc_lambda,
        target_noise=0.2, noise_clip=0.5,
        device=device
    )

    it = 0
    while it < steps:
        batch = per.sample(batch_size)
        if "a" not in batch:
            raise RuntimeError("PER batch missing actions; ensure labels->actions were derived during ingest.")

        s  = torch.from_numpy(batch["s"]).to(device)
        a  = torch.from_numpy(batch["a"]).to(device)
        r  = torch.from_numpy(batch["r"]).to(device)
        sp = torch.from_numpy(batch["sp"]).to(device)
        d  = torch.from_numpy(batch["d"]).to(device)
        w  = torch.from_numpy(batch["w"]).to(device)  # importance weights

        # Critic targets (TD3 target policy smoothing)
        with torch.no_grad():
            noise = (torch.randn_like(a) * agent.target_noise).clamp(-agent.noise_clip, agent.noise_clip)
            ap = (agent.actor_targ(sp) + noise).clamp(-agent.act_limit, agent.act_limit)
            q1_tp = agent.q1_targ(sp, ap)
            q2_tp = agent.q2_targ(sp, ap)
            q_tp_min = torch.minimum(q1_tp, q2_tp)
            y = r + (1.0 - d) * agent.gamma * q_tp_min

        # Critic losses
        q1_pred = agent.q1(s, a)
        q2_pred = agent.q2(s, a)
        q1_loss = (w * (q1_pred - y).pow(2)).mean()
        q2_loss = (w * (q2_pred - y).pow(2)).mean()

        agent.q1_optim.zero_grad(set_to_none=True)
        q1_loss.backward()
        agent.q1_optim.step()

        agent.q2_optim.zero_grad(set_to_none=True)
        q2_loss.backward()
        agent.q2_optim.step()

        pi_loss_val = 0.0
        agent.total_it += 1

        # Delayed actor + Polyak
        if agent.total_it % agent.policy_delay == 0:
            pi = agent.actor(s)
            q_pi = agent.q1(s, pi)
            bc_loss = F.mse_loss(pi, a)
            pi_loss = (-q_pi.mean()) + bc_lambda * bc_loss

            agent.pi_optim.zero_grad(set_to_none=True)
            pi_loss.backward()
            agent.pi_optim.step()
            pi_loss_val = float(pi_loss.item())

            from rl_utils import polyak_update
            polyak_update(agent.actor_targ, agent.actor, agent.tau)
            polyak_update(agent.q1_targ, agent.q1, agent.tau)
            polyak_update(agent.q2_targ, agent.q2, agent.tau)

        # PER priority update via min-Q TD error
        td = td_error_min_q(agent, batch, device=device)
        per.update_priorities(batch["idx"], td)

        it += 1
        if it % print_every == 0 or it == steps:
            print(f"[seed {seed}] step {it}/{steps} | q1={q1_loss.item():.5f} q2={q2_loss.item():.5f} pi={{:.5f}}".format(pi_loss_val))

    # Save weights
    torch.save(agent.actor.state_dict(), f"td3bc_actor_seed{seed}.pt")
    torch.save(agent.q1.state_dict(),    f"td3bc_q1_seed{seed}.pt")
    torch.save(agent.q2.state_dict(),    f"td3bc_q2_seed{seed}.pt")
    return agent

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--th", type=float, default=0.0002)
    ap.add_argument("--buf", type=int, default=1_800_000)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--drop_noise", action="store_true")
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--models", type=int, default=3, help="K models in ensemble")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--policy_delay", type=int, default=2)
    ap.add_argument("--bc_lambda", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,43,44", help="comma-separated seeds; auto-extends if needed")
    ap.add_argument("--agree_eps", type=float, default=0.15, help="agreement tolerance on actions")
    ap.add_argument("--q_min_thresh", type=float, default=-1e9, help="optional Q threshold for consensus")
    args = ap.parse_args()

    # PER buffer
    per = PERNStepReplay(
        capacity=args.buf,
        state_dim=6,
        act_dim=1,
        n_step=args.nstep,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        seed=7
    )

    # SQL that handles schema drift
    con = duckdb.connect(database=":memory:")
    sql = build_sql(args.parquet, args.start, args.end, args.h, args.th, None)

    try:
        plan = con.execute(f"EXPLAIN {sql}").fetchall()
        print("[DuckDB plan]")
        for row in plan:
            print(" ", row[0])
    except Exception as e:
        print(f"[DuckDB plan] skipped: {e}")

    total = 0
    have_cols_printed = False
    cur = con.execute(sql)
    while True:
        df = cur.fetch_df_chunk(args.chunk)
        if df is None or len(df) == 0:
            break

        df = coerce_numeric(df)
        if args.drop_noise and "label" in df.columns:
            before = len(df)
            df = df[df["label"] != 0]
            print(f"[ingest] dropped noise labels: {before - len(df)}")

        if not have_cols_printed:
            print(f"[ingest] columns: {list(df.columns)}")
            have_cols_printed = True

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise KeyError(f"Required columns missing from SQL result: {missing}")

        before = len(df)
        df = df.dropna(subset=REQUIRED)
        dropped = before - len(df)
        if dropped:
            print(f"[ingest] NaN-dropped rows: {dropped}")

        if len(df) == 0:
            continue

        S  = df[STATE_COLS].to_numpy(np.float32)
        SP = df[NEXT_COLS].to_numpy(np.float32)
        R  = df[["fwd_ret"]].to_numpy(np.float32)
        D  = np.zeros((len(df), 1), dtype=np.float32)

        if "label" in df.columns:
            A = df[["label"]].to_numpy(np.float32)
        else:
            A = np.sign(R).astype(np.float32)

        for i in range(len(df)):
            per.add(S[i], A[i], R[i], SP[i], D[i], priority=None)

        total += len(df)
        if total % (args.chunk * 2) < len(df):
            print(f"[ingest] total rows -> {total}")

    print(f"[ingest] transitions loaded: {total} | per.size={per.size()}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    while len(seeds) < args.models:
        seeds.append(seeds[-1] + 1 if seeds else 42)

    ensemble = []
    for k in range(args.models):
        print(f"\n=== Training model {k+1}/{args.models} (seed {seeds[k]}) ===")
        t0 = time.time()
        agent = train_one_model(
            per=per,
            seed=seeds[k],
            steps=args.steps,
            batch_size=args.batch,
            device=args.device,
            bc_lambda=args.bc_lambda,
            policy_delay=args.policy_delay,
        )
        dt = time.time() - t0
        print(f"[model {k+1}] done in {dt:.1f}s")
        ensemble.append({"seed": seeds[k], "actor": f"td3bc_actor_seed{seeds[k]}.pt"})

    cfg = {
        "type": "td3bc_consensus_gate",
        "actors": [e["actor"] for e in ensemble],
        "agree_eps": args.agree_eps,
        "q_min_thresh": args.q_min_thresh,
        "state_dim": 6,
        "act_limit": 1.0
    }
    with open("consensus_gate_td3bc.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved consensus gate: consensus_gate_td3bc.json")

if __name__ == "__main__":
    main()
