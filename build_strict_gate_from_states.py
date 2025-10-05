#!/usr/bin/env python3
import csv, subprocess, shlex, sys, os, json

GATE_JSON   = "consensus_gate_td3bc.json"
SCALER_JSON = "scaler.json"
LIVE_GATE   = "consensus_gate_live_plus.py"  # must be in current dir

def main():
    if not os.path.exists(GATE_JSON):
        sys.exit(f"[error] {GATE_JSON} not found")
    if not os.path.exists(SCALER_JSON):
        sys.exit(f"[error] {SCALER_JSON} not found")
    if not os.path.exists(LIVE_GATE):
        sys.exit(f"[error] {LIVE_GATE} not found in CWD")
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} STATES_CSV OUT_CSV", file=sys.stderr)
        sys.exit(2)

    states_csv = sys.argv[1]
    out_csv    = sys.argv[2]

    # Read states.csv (expects header, e.g., mid,spread,imbalance,mom1,mom3,vol3)
    with open(states_csv, newline='') as f:
        rdr = csv.reader(f)
        cols = next(rdr)  # header
        rows = [r for r in rdr]

    # Build strict gate by invoking the live-gate once per row
    out_lines = ["row_idx,agree,action,max_pair_diff,min_q"]
    for i, r in enumerate(rows):
        state_row = ",".join(r)  # live gate expects values in the scaler's order; states.csv should already match
        cmd = [
            sys.executable, LIVE_GATE,
            "--gate", GATE_JSON,
            "--scaler", SCALER_JSON,
            "--state_row", state_row,
            "--device", "cpu",
            "--consensus-mode", "sign",
            "--min-agree-k", "3",
            "--agree-eps", "0.05",
            "--min-mag", "0.20",
        ]
        # Run and capture stdout
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
            sys.exit(f"[error] live gate failed on row {i} (exit {res.returncode})")
        # Parse the last CSV line from output
        last = None
        for line in res.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("0,") or line.startswith("row_idx,"):
                last = line
        if not last:
            # Fallback: look for the standard header/data the script prints
            for line in res.stdout.strip().splitlines():
                if "," in line and line[0].isdigit():
                    last = line
                    break
        if not last:
            print(res.stdout)
            sys.exit(f"[error] could not parse gate output on row {i}")
        # Replace the '0,' with actual row index
        if last.startswith("0,"):
            last = f"{i}" + last[1:]
        out_lines.append(last)

    with open(out_csv, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"[ok] wrote {out_csv} with {len(rows)} rows")

if __name__ == "__main__":
    main()
