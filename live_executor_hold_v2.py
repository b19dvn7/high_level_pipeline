#!/usr/bin/env python3
# live_executor_hold_v2.py (v2.1)
# - Fixes UTC deprecation (uses timezone-aware UTC timestamps)
# - Adds --ts_timespec (seconds|milliseconds) for timestamp precision
# - Consistent float formatting in CSV
# - Same trading logic/flags as v2

import argparse, json, os, subprocess, sys
import datetime as dt

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return p.stdout.strip()

def parse_state_row(row_csv):
    # "mid,spread,imbalance,mom1,mom3,vol3" as a single CSV row (no header)
    vals = [float(x.strip()) for x in row_csv.split(",")]
    if len(vals) != 6:
        raise ValueError("state_row must have 6 numeric fields")
    return dict(mid=vals[0], spread=vals[1], imbalance=vals[2], mom1=vals[3], mom3=vals[4], vol3=vals[5])

def now_utc_iso(timespec: str = "seconds") -> str:
    # timespec: "seconds" or "milliseconds"
    t = dt.datetime.now(dt.timezone.utc)
    if timespec == "milliseconds":
        # emulate ms: ISO without microseconds, then add .mmmZ
        return t.isoformat(timespec="milliseconds")
    return t.isoformat(timespec="seconds")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--device", default="cpu")

    # thresholds & holding
    ap.add_argument("--buy_th", type=float, default=0.10, help="min +action to go/stay long")
    ap.add_argument("--sell_th", type=float, default=0.10, help="min -action (abs) to go/stay short")
    ap.add_argument("--hold_bars", type=int, default=12, help="minimum bars to hold before flipping")
    ap.add_argument("--max_hold_bars", type=int, default=160, help="hard cap on holding duration; force flat after this many bars")

    # microstructure guards
    ap.add_argument("--spread_cap", type=float, default=5.0, help="skip new entries if spread > cap (price units)")
    ap.add_argument("--min_vol3", type=float, default=0.0, help="skip new entries if vol3 < min_vol3")

    # persistence / logging
    ap.add_argument("--statefile", default="live_state.json")
    ap.add_argument("--log", default="live_signals.csv")

    # formatting
    ap.add_argument("--ts_timespec", choices=["seconds","milliseconds"], default="seconds")
    ap.add_argument("--fmt_action", default=".6f")
    ap.add_argument("--fmt_spread", default=".6f")
    ap.add_argument("--fmt_vol3", default=".6f")

    args = ap.parse_args()

    # 1) Pull latest features (6-tuple)
    row = run([sys.executable, "latest_state_from_parquet.py", "--parquet", args.parquet])
    feat = parse_state_row(row)

    # 2) Run sign consensus gate ON THIS ROW (min-agree-k=2, min-mag = min(buy_th, sell_th))
    min_mag = min(args.buy_th, args.sell_th)
    out = run([
        sys.executable, "consensus_gate_live_plus.py",
        "--gate", args.gate,
        "--scaler", args.scaler,
        "--state_row", row,
        "--device", args.device,
        "--consensus-mode", "sign",
        "--min-agree-k", "2",
        "--min-mag", f"{min_mag}"
    ])

    # parse last CSV line "row_idx,agree,action,max_pair_diff,min_q"
    last = [ln for ln in out.splitlines() if ln and ln[0].isdigit()][-1]
    parts = last.split(",")
    agree = int(parts[1]); raw_action = float(parts[2])

    # 3) Discrete intent with asymmetric thresholds
    intent = 0
    if raw_action >= args.buy_th: intent = +1
    elif raw_action <= -args.sell_th: intent = -1

    # 4) Load persistent state
    st = {"prev_pos": 0, "bars_held": 0, "bars_held_total": 0}
    if os.path.isfile(args.statefile):
        try:
            with open(args.statefile, "r") as f: st.update(json.load(f))
        except: pass
    prev_pos = int(st.get("prev_pos", 0))
    held = int(st.get("bars_held", 0))
    held_total = int(st.get("bars_held_total", 0))

    # 5) Microstructure guards (apply to NEW ENTRIES only; allow exits always)
    skipped_reason = ""
    if prev_pos == 0 and intent != 0:
        if feat["spread"] > args.spread_cap:
            skipped_reason = f"spread>{args.spread_cap}"
            intent = 0
        elif feat["vol3"] < args.min_vol3:
            skipped_reason = f"vol3<{args.min_vol3}"
            intent = 0

    # 6) Enforce min-hold + max-hold
    next_pos = intent
    if next_pos != prev_pos:
        # candidate change
        if prev_pos != 0 and next_pos != 0 and held < args.hold_bars:
            # flip too soon -> defer
            next_pos = prev_pos
            held += 1; held_total += 1
        else:
            # allowed change
            if next_pos == 0:
                held = 0; held_total = 0
            else:
                held = 0; held_total = 0
    else:
        # no change, increment timers if in position
        if next_pos != 0:
            held = min(held + 1, args.hold_bars)
            held_total += 1
            if held_total >= args.max_hold_bars:
                next_pos = 0
                skipped_reason = (skipped_reason + "; " if skipped_reason else "") + "max_hold_cap"
                held = 0; held_total = 0

    # 7) Persist
    with open(args.statefile, "w") as f:
        json.dump({"prev_pos": int(next_pos), "bars_held": int(held), "bars_held_total": int(held_total)}, f)

    # 8) Log/print (UTC, timezone-aware)
    ts = now_utc_iso(args.ts_timespec)
    # format floats
    fmt_a = f"{raw_action:{args.fmt_action}}"
    fmt_s = f"{feat['spread']:{args.fmt_spread}}"
    fmt_v = f"{feat['vol3']:{args.fmt_vol3}}"

    header = "ts,prev_pos,next_pos,raw_action,buy_th,sell_th,hold_bars,hold_counter,spread,vol3,skipped_reason\n"
    line = f"{ts},{prev_pos},{next_pos},{fmt_a},{args.buy_th:.2f},{args.sell_th:.2f},{args.hold_bars},{held},{fmt_s},{fmt_v},{skipped_reason}"
    print(line)
    if not os.path.isfile(args.log):
        with open(args.log, "w") as f: f.write(header)
    with open(args.log, "a") as f: f.write(line + "\n")

if __name__ == "__main__":
    main()
