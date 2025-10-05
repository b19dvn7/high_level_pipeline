#!/usr/bin/env python3
# live_executor_hold.py
# - Pull latest L2-derived state from your parquet glob
# - Run sign-consensus gate with min-agree-k=2, min-mag=TH
# - Enforce min-hold bars before flipping (state persisted in a json file)
# - Print a single CSV line with ts,pos,raw_action,dec_action,hold_left

import argparse, json, os, subprocess, sys, time, datetime as dt

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return p.stdout.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--th", type=float, default=0.10, help="min |action| to trade")
    ap.add_argument("--hold_bars", type=int, default=12, help="minimum bars to hold before flip")
    ap.add_argument("--statefile", default="live_executor_state.json")
    ap.add_argument("--log", default="live_signals.csv")
    args = ap.parse_args()

    # 1) get latest state row (6 features)
    row = run([sys.executable, "latest_state_from_parquet.py", "--parquet", args.parquet])

    # 2) run sign-consensus gate once
    out = run([
        sys.executable, "consensus_gate_live_plus.py",
        "--gate", args.gate,
        "--scaler", args.scaler,
        "--state_row", row,
        "--device", args.device,
        "--consensus-mode", "sign",
        "--min-agree-k", "2",
        "--min-mag", str(args.th)
    ])

    # parse last CSV line "row_idx,agree,action,max_pair_diff,min_q"
    last = [ln for ln in out.splitlines() if ln and ln[0].isdigit()][-1]
    parts = last.split(",")
    agree = int(parts[1])
    act = float(parts[2])  # already 0 if not agreed or fails q filter

    # 3) threshold to discrete intent (±1/0) using th; keep continuous for diagnostics
    intent = 1 if act >= args.th else (-1 if act <= -args.th else 0)

    # 4) load persistent state (prev_pos, bars_held)
    st = {"prev_pos": 0, "bars_held": 0}
    if os.path.isfile(args.statefile):
        try:
            with open(args.statefile, "r") as f: st.update(json.load(f))
        except: pass
    prev_pos = int(st.get("prev_pos", 0))
    held = int(st.get("bars_held", 0))

    # 5) enforce min-hold: if wanting to flip but haven't held enough, keep prev_pos
    next_pos = intent
    if intent != prev_pos and held < args.hold_bars:
        next_pos = prev_pos
        held += 1
    else:
        # change allowed (or no change requested)
        held = 0 if next_pos != prev_pos else min(held+1, args.hold_bars)

    # 6) persist state
    with open(args.statefile, "w") as f:
        json.dump({"prev_pos": int(next_pos), "bars_held": int(held)}, f)

    # 7) print + log
    ts = dt.datetime.utcnow().isoformat()
    line = f"{ts},{prev_pos},{next_pos},{act:.6f},{args.th:.2f},{args.hold_bars},{held}"
    print(line)
    if not os.path.isfile(args.log):
        with open(args.log, "w") as f:
            f.write("ts,prev_pos,next_pos,raw_action,th,hold_bars,hold_counter\n")
    with open(args.log, "a") as f:
        f.write(line + "\n")

if __name__ == "__main__":
    main()
