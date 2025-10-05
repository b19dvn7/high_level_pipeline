#!/usr/bin/env python3
# export_scaler_from_states.py — reads states.csv, writes scaler.json with per-feature mean/std
import argparse, csv, json
import numpy as np

COLS = ["mid","spread","imbalance","mom1","mom3","vol3"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_csv", required=True)
    ap.add_argument("--out", default="scaler.json")
    args = ap.parse_args()

    X = []
    with open(args.state_csv, newline="") as f:
        r = csv.DictReader(f)
        hdr = [h.strip().lower() for h in (r.fieldnames or [])]
        if hdr != COLS:
            raise SystemExit(f"header mismatch; got {r.fieldnames}, want {COLS}")
        for row in r:
            try:
                X.append([float(row[c]) for c in COLS])
            except:
                pass
    if not X:
        raise SystemExit("no rows in states.csv")
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)

    scaler = {"cols": COLS, "mean": mu.tolist(), "std": sd.tolist(), "type": "zscore"}
    with open(args.out, "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"[ok] wrote {args.out}")
    print("[mean]", [round(x,4) for x in mu.tolist()])
    print("[std ]", [round(x,4) for x in sd.tolist()])

if __name__ == "__main__":
    main()
