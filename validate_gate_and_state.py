#!/usr/bin/env python3
# validate_gate_and_state.py
# Checks: gate JSON loads, actor files exist, states.csv has correct header, device is valid.
import argparse, json, os, sys, csv

REQUIRED_COLS = ["mid","spread","imbalance","mom1","mom3","vol3"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--state_csv", required=True)
    ap.add_argument("--actors_dir", default=".")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    args = ap.parse_args()

    # Load gate json
    try:
        with open(args.gate, "r") as f:
            gate = json.load(f)
    except Exception as e:
        print(f"[error] failed to load gate JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # Required keys
    for k in ["type","actors","agree_eps","q_min_thresh","state_dim","act_limit"]:
        if k not in gate:
            print(f"[error] missing key in gate: {k}", file=sys.stderr)
            sys.exit(3)
    if gate["type"] != "td3bc_consensus_gate":
        print(f"[error] unexpected gate type: {gate['type']}", file=sys.stderr)
        sys.exit(3)
    if gate["state_dim"] != 6:
        print(f"[warn] state_dim in gate is {gate['state_dim']} but code expects 6 features")

    # Actor files present?
    missing = []
    for a in gate["actors"]:
        p = a if os.path.isabs(a) else os.path.join(args.actors_dir, a)
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        print("[error] missing actor file(s):")
        for m in missing: print("   ", m)
        sys.exit(4)

    # states.csv header
    try:
        with open(args.state_csv, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception as e:
        print(f"[error] cannot read state_csv: {e}", file=sys.stderr)
        sys.exit(5)

    header_norm = [h.strip().lower() for h in header]
    if header_norm != REQUIRED_COLS:
        print(f"[error] state_csv header mismatch.\n"
              f"  got : {header}\n"
              f"  want: {REQUIRED_COLS}")
        sys.exit(6)

    print("[ok] gate JSON, actors, and states.csv look good.")
    print("Run gate like:\n"
          f"  python3 consensus_gate_td3bc.py --gate {args.gate} --state_csv {args.state_csv} --device {args.device}")

if __name__ == "__main__":
    main()
