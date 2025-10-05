#!/usr/bin/env python3
# live_executor_hold_v2_2.py
# - Same logic as v2.1, plus:
#   * logs mid
#   * persists entry_mid and entry_ts in statefile on entries
#   * prints all fields needed for a human to understand "what's going on"

import argparse, json, os, subprocess, sys, datetime as dt

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return p.stdout.strip()

def parse_state_row(row_csv):
    # "mid,spread,imbalance,mom1,mom3,vol3" (no header)
    vals = [float(x.strip()) for x in row_csv.split(",")]
    if len(vals) != 6:
        raise ValueError("state_row must have 6 numeric fields")
    return dict(mid=vals[0], spread=vals[1], imbalance=vals[2], mom1=vals[3], mom3=vals[4], vol3=vals[5])

def now_utc_iso(timespec: str = "milliseconds") -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec=timespec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--buy_th", type=float, default=0.10)
    ap.add_argument("--sell_th", type=float, default=0.10)
    ap.add_argument("--hold_bars", type=int, default=12)
    ap.add_argument("--max_hold_bars", type=int, default=160)
    ap.add_argument("--spread_cap", type=float, default=5.0)
    ap.add_argument("--min_vol3", type=float, default=0.0)
    ap.add_argument("--statefile", default="live_state.json")
    ap.add_argument("--log", default="live_signals.csv")
    ap.add_argument("--ts_timespec", choices=["seconds","milliseconds"], default="milliseconds")
    args = ap.parse_args()

    # 1) Latest features
    row = run([sys.executable, "latest_state_from_parquet.py", "--parquet", args.parquet])
    feat = parse_state_row(row)  # has mid, spread, vol3, ...

    # 2) Gate (sign consensus)
    min_mag = min(args.buy_th, args.sell_th)
    out = run([
        sys.executable, "consensus_gate_live_plus.py",
        "--gate", args.gate, "--scaler", args.scaler,
        "--state_row", row, "--device", args.device,
        "--consensus-mode", "sign", "--min-agree-k", "2",
        "--min-mag", f"{min_mag}"
    ])
    last = [ln for ln in out.splitlines() if ln and ln[0].isdigit()][-1]
    parts = last.split(",")
    agree = int(parts[1]); raw_action = float(parts[2])

    # 3) intent (asymmetric thresholds)
    intent = 0
    if raw_action >= args.buy_th: intent = +1
    elif raw_action <= -args.sell_th: intent = -1

    # 4) load state; now includes entry tracking
    st = {"prev_pos": 0, "bars_held": 0, "bars_held_total": 0,
          "entry_mid": None, "entry_ts": None}
    if os.path.isfile(args.statefile):
        try:
            with open(args.statefile, "r") as f: st.update(json.load(f))
        except: pass
    prev_pos = int(st.get("prev_pos", 0))
    held = int(st.get("bars_held", 0))
    held_total = int(st.get("bars_held_total", 0))
    entry_mid = st.get("entry_mid", None)
    entry_ts  = st.get("entry_ts", None)

    # 5) guards for NEW entries
    skipped_reason = ""
    if prev_pos == 0 and intent != 0:
        if feat["spread"] > args.spread_cap:
            skipped_reason = f"spread>{args.spread_cap}"
            intent = 0
        elif feat["vol3"] < args.min_vol3:
            skipped_reason = f"vol3<{args.min_vol3}"
            intent = 0

    # 6) Enforce hold rules and manage entry tracking
    next_pos = intent
    if next_pos != prev_pos:
        # flip or enter/exit
        if prev_pos != 0 and next_pos != 0 and held < args.hold_bars:
            # too soon to flip
            next_pos = prev_pos
            held += 1; held_total += 1
        else:
            # change allowed
            if next_pos == 0:
                # exit
                held = 0; held_total = 0
                entry_mid = None; entry_ts = None
            else:
                # new entry
                held = 0; held_total = 0
                entry_mid = feat["mid"]; entry_ts = now_utc_iso(args.ts_timespec)
    else:
        # no change
        if next_pos != 0:
            held = min(held + 1, args.hold_bars)
            held_total += 1
            if held_total >= args.max_hold_bars:
                # force flatten
                next_pos = 0
                skipped_reason = (skipped_reason + "; " if skipped_reason else "") + "max_hold_cap"
                held = 0; held_total = 0
                entry_mid = None; entry_ts = None

    # 7) unrealized PnL estimate (in bps) using mids, if in position
    upnl_bps = ""
    if entry_mid is not None and next_pos != 0:
        ret = (feat["mid"] - entry_mid) / entry_mid
        upnl = ret * (1 if next_pos>0 else -1)
        upnl_bps = f"{upnl*10000:.2f}"

    # 8) persist
    with open(args.statefile, "w") as f:
        json.dump({
            "prev_pos": int(next_pos),
            "bars_held": int(held),
            "bars_held_total": int(held_total),
            "entry_mid": entry_mid,
            "entry_ts": entry_ts
        }, f)

    # 9) log
    ts = now_utc_iso(args.ts_timespec)
    header = ("ts,prev_pos,next_pos,raw_action,buy_th,sell_th,hold_bars,hold_counter,"
              "mid,spread,vol3,entry_mid,entry_ts,upnl_bps,skipped_reason\n")
    line = (f"{ts},{prev_pos},{next_pos},{raw_action:.6f},{args.buy_th:.2f},{args.sell_th:.2f},"
            f"{args.hold_bars},{held},{feat['mid']:.2f},{feat['spread']:.6f},{feat['vol3']:.6f},"
            f"{'' if entry_mid is None else f'{entry_mid:.2f}'},"
            f"{'' if entry_ts is None else entry_ts},"
            f"{upnl_bps},{skipped_reason}")
    print(line)
    if not os.path.isfile(args.log):
        with open(args.log, "w") as f: f.write(header)
    with open(args.log, "a") as f: f.write(line + "\n")

if __name__ == "__main__":
    main()
