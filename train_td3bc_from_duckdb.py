#!/usr/bin/env python3
# train_td3bc_from_duckdb.py
# Offline TD3+BC training directly from Parquet via DuckDB feeder.

import argparse
import time
import numpy as np
import torch

from rl_duckdb_data_layer import build_sql, stream_into_buffer, ReplayBuffer
import duckdb
from td3_bc_agent import TD3BC

def to_torch(batch, device="cpu"):
    out = {}
    for k, v in batch.items():
        out[k] = torch.from_numpy(v).to(device)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--th", type=float, default=0.0002)
    ap.add_argument("--buf", type=int, default=1_500_000)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--drop_noise", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps_per_epoch", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bc_lambda", type=float, default=0.1, help="BC regularization weight")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    con = duckdb.connect(database=":memory:")
    sql = build_sql(args.parquet, args.start, args.end, args.h, args.th, None)

    # derive_action=True so TD3+BC has actions from labels {-1,0,1}
    buf = ReplayBuffer(capacity=args.buf, state_dim=6, act_dim=1)

    total = stream_into_buffer(
        con, sql, buf,
        chunk_rows=args.chunk,
        drop_noise_label=args.drop_noise,
        derive_action=True
    )
    print(f"[ingest] transitions loaded: {total}, buffer.size={buf.size()}")

    assert buf.size() > args.batch, "Not enough samples to train."

    agent = TD3BC(
        s_dim=6, a_dim=1, act_limit=1.0,
        actor_lr=1e-3, critic_lr=1e-3,
        gamma=0.99, tau=0.005, policy_delay=2,
        bc_lambda=args.bc_lambda,
        target_noise=0.2, noise_clip=0.5,
        device=args.device
    )

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        for t in range(1, args.steps_per_epoch + 1):
            batch_np = buf.sample(args.batch, rng)
            # require actions
            assert "a" in batch_np, "Actions missing. Use derive_action=True in feeder."
            batch = {
                "s":  batch_np["s"],
                "a":  batch_np["a"],
                "r":  batch_np["r"],
                "sp": batch_np["sp"],
                "d":  batch_np["d"],
            }
            for k in batch:
                batch[k] = torch.from_numpy(batch[k]).to(args.device)

            stats = agent.update(batch)

            if t % 200 == 0 or t == args.steps_per_epoch:
                print(f"[ep {ep}/{args.epochs}] step {t}/{args.steps_per_epoch} "
                      f"| q1={stats['q1_loss']:.5f} q2={stats['q2_loss']:.5f} pi={stats['pi_loss']:.5f}")

        dur = time.time() - t0
        print(f"[ep {ep}] done in {dur:.1f}s")

    # Save final actor (deterministic policy)
    torch.save(agent.actor.state_dict(), "td3bc_actor.pt")
    print("Saved: td3bc_actor.pt")

if __name__ == "__main__":
    main()
