#!/usr/bin/env python3
# train_iql_from_duckdb.py
# Offline IQL training directly from Parquet via DuckDB feeder.

import argparse
import time
import numpy as np
import torch

from rl_duckdb_data_layer import build_sql, stream_into_buffer, ReplayBuffer
import duckdb
from iql_agent import IQL

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
    ap.add_argument("--steps_per_epoch", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    con = duckdb.connect(database=":memory:")
    sql = build_sql(args.parquet, args.start, args.end, args.h, args.th, None)

    # Actions required: derive from labels/sign so IQL sees (s,a,r,s')
    buf = ReplayBuffer(capacity=args.buf, state_dim=6, act_dim=1)
    total = stream_into_buffer(
        con, sql, buf,
        chunk_rows=args.chunk,
        drop_noise_label=args.drop_noise,
        derive_action=True
    )
    print(f"[ingest] transitions loaded: {total}, buffer.size={buf.size()}")
    assert buf.size() > args.batch, "Not enough samples to train."

    agent = IQL(
        s_dim=6, a_dim=1,
        gamma=0.99, tau=0.005,
        expectile=args.expectile, beta=args.beta,
        actor_lr=3e-4, q_lr=3e-4, v_lr=3e-4,
        device=args.device, act_limit=1.0
    )

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        for t in range(1, args.steps_per_epoch + 1):
            batch_np = buf.sample(args.batch, rng)
            batch = {k: torch.from_numpy(v).to(args.device) for k, v in batch_np.items()}
            stats = agent.update(batch)

            if t % 200 == 0 or t == args.steps_per_epoch:
                print(f"[ep {ep}/{args.epochs}] step {t}/{args.steps_per_epoch} "
                      f"| q1={stats['q1_loss']:.5f} q2={stats['q2_loss']:.5f} "
                      f"v={stats['v_loss']:.5f} pi={stats['pi_loss']:.5f}")

        dur = time.time() - t0
        print(f"[ep {ep}] done in {dur:.1f}s")

    torch.save(agent.pi.state_dict(), "iql_policy.pt")
    torch.save(agent.v.state_dict(),  "iql_value.pt")
    print("Saved: iql_policy.pt  iql_value.pt")

if __name__ == "__main__":
    main()
