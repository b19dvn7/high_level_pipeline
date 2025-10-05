#!/usr/bin/env python3
import argparse, time, re
from pathlib import Path

def latest_log(sweep: Path):
    logs = sorted(sweep.glob("log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None

def parse_line(ln: str):
    # Accept a few formats we printed during sweep/backtests
    # Example lines (be tolerant):
    # [bt] backtest -> gate_k2_eps0.05_mag0.1_th0.05_hold8.csv
    # [backtest H] rows=160 trades=8 ... equity_end=1.0124 Sharpe_like=0.15 MaxDD=1.03% ...
    # params might appear in file name as k2_eps0.05_mag0.1_th0.05_hold8
    info = {}
    gate_m = re.search(r'gate_([^.\s]+)\.csv', ln)
    if gate_m:
        info["gate_tag"] = gate_m.group(1)
        # extract fields in tag if present
        m = re.match(r'k(?P<k>\d+)_eps(?P<eps>[0-9.]+)_mag(?P<mag>[0-9.]+)_(?:th(?P<th>[0-9.]+)|ls(?P<ls>[0-9.]+))?_hold(?P<hold>\d+)', info["gate_tag"])
        if m:
            d = m.groupdict()
            if d.get("k"):   info["min_agree_k"] = d["k"]
            if d.get("eps"): info["agree_eps"]   = d["eps"]
            if d.get("mag"): info["min_mag"]     = d["mag"]
            if d.get("th"):  info["action_thresh"] = d["th"]
            if d.get("ls"):  info["linear_scale"]  = d["ls"]
            if d.get("hold"):info["hold_bars"]   = d["hold"]
    # backtest summary values (be permissive)
    kvs = re.findall(r'(\w+)=([-\w\.%]+)', ln)
    for k,v in kvs:
        info[k] = v
    return info

def write_now_csv(path: Path, info: dict):
    # Define a stable header
    header = ["mode","H","fee_bps","min_agree_k","agree_eps","min_mag","action_thresh","linear_scale","hold_bars","rows","trades","equity_end","Sharpe_like","MaxDD","exposure","gate_tag"]
    # normalize values if we saw them
    row = [
        info.get("mode",""),
        info.get("H",""),
        info.get("fee_bps",""),
        info.get("min_agree_k",""),
        info.get("agree_eps",""),
        info.get("min_mag",""),
        info.get("action_thresh",""),
        info.get("linear_scale",""),
        info.get("hold_bars",""),
        info.get("rows",""),
        info.get("trades",""),
        info.get("equity_end",""),
        info.get("Sharpe_like",""),
        info.get("MaxDD",""),
        info.get("exposure",""),
        info.get("gate_tag",""),
    ]
    txt = ",".join(header) + "\n" + ",".join(map(str,row)) + "\n"
    path.write_text(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="runs/sweep_* folder")
    ap.add_argument("--every", type=float, default=1.0)
    args = ap.parse_args()

    sweep = Path(args.sweep).resolve()
    out = sweep / "live.now.csv"

    last_written = ""
    while True:
        lg = latest_log(sweep)
        if lg and lg.exists():
            lines = lg.read_text(errors="ignore").splitlines()
            # scan from the end for a line with richest info
            info = {}
            for ln in reversed(lines[-200:]):  # look at last 200 lines
                data = parse_line(ln)
                if data:
                    info.update(data)
                    # stop if we already captured a gate & equity/Sharpe
                    if "gate_tag" in info and ("equity_end" in info or "Sharpe_like" in info):
                        break
            if info:
                snapshot = repr(sorted(info.items()))
                if snapshot != last_written:
                    write_now_csv(out, info)
                    last_written = snapshot
        time.sleep(args.every)

if __name__ == "__main__":
    main()
