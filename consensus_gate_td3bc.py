#!/usr/bin/env python3
# consensus_gate_td3bc.py
# Loads N actor .pt files, reads states.csv, computes per-actor actions,
# applies consensus (agree_eps) and optional q_min_thresh, prints final action per row.
import argparse, json, csv, os, sys
from typing import List, Tuple, Optional

import numpy as np

try:
    import torch
except ImportError:
    print("[error] Requires torch. Try: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
    sys.exit(2)

REQUIRED_COLS = ["mid","spread","imbalance","mom1","mom3","vol3"]

def load_model(path: str, device: str):
    # Try TorchScript first (jit), then regular torch.load
    m = None
    try:
        m = torch.jit.load(path, map_location=device)
    except Exception:
        try:
            m = torch.load(path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {path}: {e}")
    m.eval()
    return m

@torch.no_grad()
def model_forward(model, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Try a few common signatures: (s)->a ; (s)->(a,q)
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        a, q = out[0], out[1]
        return a, q
    # if single tensor, assume it's the action; no Q available
    return out, None

def consensus_action(actions: np.ndarray, agree_eps: float, act_limit: float) -> Tuple[bool, float]:
    """
    actions: shape [n_actors] (1D)
    agree if max pairwise abs diff <= agree_eps * act_limit (scale by limit).
    return (agree, mean_action_clipped)
    """
    if actions.ndim != 1:
        actions = actions.reshape(-1)
    if len(actions) == 0:
        return False, 0.0
    max_diff = np.max(np.abs(actions[:, None] - actions[None, :]))
    agree = max_diff <= (agree_eps * act_limit)
    mean_a = float(np.mean(actions))
    # clip to act_limit
    mean_a = max(-act_limit, min(act_limit, mean_a))
    return agree, mean_a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, help="consensus_gate_td3bc.json")
    ap.add_argument("--state_csv", required=True, help="CSV with columns: mid,spread,imbalance,mom1,mom3,vol3")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    ap.add_argument("--out_csv", default=None, help="optional write results to CSV")
    args = ap.parse_args()

    # Load gate config
    with open(args.gate, "r") as f:
        gate = json.load(f)
    if gate.get("type") != "td3bc_consensus_gate":
        print(f"[error] gate.type must be 'td3bc_consensus_gate' (got {gate.get('type')})", file=sys.stderr)
        sys.exit(2)

    actors: List[str] = gate["actors"]
    agree_eps: float = float(gate["agree_eps"])
    q_min_thresh: float = float(gate["q_min_thresh"])
    state_dim: int = int(gate["state_dim"])
    act_limit: float = float(gate.get("act_limit", 1.0))

    if state_dim != 6:
        print(f"[warn] gate.state_dim={state_dim} but this runner expects 6 features; continuing.", file=sys.stderr)

    # Load states
    with open(args.state_csv, newline="") as f:
        rdr = csv.DictReader(f)
        hdr = [h.strip().lower() for h in rdr.fieldnames or []]
        if hdr != REQUIRED_COLS:
            print(f"[error] state_csv header mismatch.\n"
                  f"  got : {rdr.fieldnames}\n"
                  f"  want: {REQUIRED_COLS}", file=sys.stderr)
            sys.exit(3)
        states = []
        for row in rdr:
            try:
                states.append([float(row[c]) for c in REQUIRED_COLS])
            except Exception as e:
                print(f"[warn] bad row in states.csv skipped: {e}", file=sys.stderr)
        X = np.array(states, dtype=np.float32)

    if X.size == 0:
        print("[error] no rows in state_csv after parsing.", file=sys.stderr)
        sys.exit(4)

    # Load models
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    models = []
    for a in actors:
        if not os.path.isfile(a):
            print(f"[error] actor file not found: {a}", file=sys.stderr)
            sys.exit(5)
        models.append(load_model(a, device))

    # Inference
    T = X.shape[0]
    results = []
    print("row_idx,agree,action,details")
    for i in range(T):
        s = torch.from_numpy(X[i:i+1]).to(device)  # shape [1,6]
        acts = []
        qs   = []
        for m in models:
            a, q = model_forward(m, s)
            # force to scalar action (supports 1D action)
            a_np = a.detach().cpu().numpy().reshape(-1)
            if a_np.size == 0:
                print(f"[warn] empty action from a model at row {i}; treating as 0", file=sys.stderr)
                acts.append(0.0)
            else:
                acts.append(float(a_np[0]))
            if q is not None:
                q_np = q.detach().cpu().numpy().reshape(-1)
                if q_np.size > 0:
                    qs.append(float(q_np[0]))

        acts_arr = np.array(acts, dtype=np.float32)
        agree, mean_a = consensus_action(acts_arr, agree_eps, act_limit)

        # Optional Q filter: if any Qs exist, require min(Q) >= q_min_thresh
        q_ok = True
        min_q = None
        if qs:
            min_q = float(np.min(qs))
            q_ok = (min_q >= q_min_thresh)

        final_ok = (agree and q_ok)
        action_out = mean_a if final_ok else 0.0  # 0 = no-trade when not in consensus / Q too low

        details = {
            "actors": len(models),
            "agree_eps": agree_eps,
            "act_limit": act_limit,
            "max_pair_diff": float(np.max(np.abs(acts_arr[:,None]-acts_arr[None,:]))) if len(acts_arr)>1 else 0.0,
            "min_q": min_q
        }
        print(f"{i},{int(final_ok)},{action_out},{details}")
        results.append((i, int(final_ok), action_out, details))

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_idx","agree","action","min_q"])
            for i, ok, a, det in results:
                w.writerow([i, ok, a, (det.get("min_q") if isinstance(det, dict) else None)])

if __name__ == "__main__":
    main()
