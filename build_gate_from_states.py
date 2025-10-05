#!/usr/bin/env python3
import csv, subprocess, sys, os, argparse

GATE_JSON   = "consensus_gate_td3bc.json"
SCALER_JSON = "scaler.json"
LIVE_GATE   = "consensus_gate_live_plus.py"  # must be in CWD

def run_live_gate(state_row, device, mode, k, eps, minmag):
    cmd = [
        sys.executable, LIVE_GATE,
        "--gate", GATE_JSON,
        "--scaler", SCALER_JSON,
        "--state_row", state_row,
        "--device", device,
        "--consensus-mode", mode,
        "--min-agree-k", str(k),
        "--agree-eps", f"{eps}",
        "--min-mag", f"{minmag}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"live gate failed: {res.stdout}\n{res.stderr}")
    # pull the CSV data line: "row_idx,agree,action,max_pair_diff,min_q" or "0,...."
    last = None
    for line in res.stdout.strip().splitlines():
        t = line.strip()
        if t.startswith("row_idx,") or (t and t[0].isdigit() and "," in t):
            last = t
    if not last:
        raise RuntimeError(f"could not parse gate output:\n{res.stdout}")
    return last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
    ap.add_argument("--mode", choices=["sign","linear"], default="sign")
    ap.add_argument("--min-agree-k", type=int, default=2)
    ap.add_argument("--agree-eps", type=float, default=0.10)
    ap.add_argument("--min-mag", type=float, default=0.10)
    args = ap.parse_args()

    for f in (GATE_JSON, SCALER_JSON, LIVE_GATE):
        if not os.path.exists(f):
            sys.exit(f"[error] required file not found: {f}")

    with open(args.states_csv, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        rows = [r for r in rdr]

    out_lines = ["row_idx,agree,action,max_pair_diff,min_q"]
    for i, r in enumerate(rows):
        state_row = ",".join(r)
        line = run_live_gate(state_row, args.device, args.mode,
                             args["min_agree_k"] if isinstance(args, dict) else args.min_agree_k,
                             args["agree_eps"] if isinstance(args, dict) else args.agree_eps,
                             args["min_mag"] if isinstance(args, dict) else args.min_mag)
        # replace leading "0," with actual row index for cleanliness
        if line.startswith("row_idx,"):
            # header, keep once at file top
            pass
        elif line.startswith("0,"):
            line = f"{i}" + line[1:]
        out_lines.append(line)

    with open(args.out_csv, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"[ok] wrote {args.out_csv} with {len(rows)} rows")
if __name__ == "__main__":
    main()
