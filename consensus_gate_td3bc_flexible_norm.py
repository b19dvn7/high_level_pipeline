#!/usr/bin/env python3
# consensus_gate_td3bc_flexible_norm.py
# Flexible gate with optional z-score normalization; forces float32 everywhere to avoid dtype mismatches.

import argparse, json, csv, os, sys, re
from typing import List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[error] Requires torch.", file=sys.stderr); sys.exit(2)

REQUIRED_COLS = ["mid","spread","imbalance","mom1","mom3","vol3"]

# ----- Helpers to support state_dict-only checkpoints -----

def _extract_layers_from_state_dict(sd: "OrderedDict"):
    pairs = {}
    for k in sd.keys():
        if k.endswith(".weight"):
            pre = k[:-7]; bk = pre + ".bias"
            if bk in sd: pairs[pre] = (k, bk)
    def get_idx(name: str) -> int:
        ms = list(re.finditer(r'(\d+)', name))
        return int(ms[-1].group(1)) if ms else 10**9
    items = [(get_idx(pre), pre, wk, bk) for pre,(wk,bk) in pairs.items()]
    items.sort(key=lambda x: (x[0], x[1]))
    layers=[]
    for _, pre, wk, bk in items:
        W = sd[wk].detach().clone().float()
        b = sd[bk].detach().clone().float()
        layers.append((W, b))
    return layers

class StateDictMLPRunner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(W) for (W, b) in layers])
        self.B = nn.ParameterList([nn.Parameter(b) for (W, b) in layers])
        self.activ_last_tanh = (self.W[-1].shape[0] == 1)
        self.eval()
    def forward(self, x):
        # x: float32
        h = x
        for i in range(len(self.W)):
            h = torch.addmm(self.B[i], h, self.W[i].t())
            if i < len(self.W) - 1:
                h = torch.relu(h)
            else:
                if self.activ_last_tanh:
                    h = torch.tanh(h)
        return h

def load_flexible_model(path: str, device: str):
    # 1) TorchScript
    try:
        m = torch.jit.load(path, map_location=device)
        m = m.float()  # force float32
        m.eval()
        return m, "jit"
    except Exception:
        pass
    # 2) torch.load
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        obj = obj.float()  # force float32
        obj.eval()
        return obj, "module"
    # 3) state_dict / mapping
    if isinstance(obj, dict):
        sd = obj.get("state_dict", None) or obj
        if isinstance(sd, dict):
            layers = _extract_layers_from_state_dict(sd)
            if not layers:
                raise RuntimeError("No linear layers parsed from state_dict.")
            runner = StateDictMLPRunner(layers).to(device).float()
            return runner, "state_dict"
    raise RuntimeError(f"Unsupported checkpoint format for {path}")

@torch.no_grad()
def model_forward(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]
    return out, None

def consensus_action(actions: np.ndarray, agree_eps: float, act_limit: float):
    if actions.ndim != 1: actions = actions.reshape(-1)
    if actions.size == 0: return False, 0.0
    max_diff = float(np.max(np.abs(actions[:,None]-actions[None,:])))
    agree = max_diff <= (agree_eps * act_limit)
    mean_a = float(np.mean(actions))
    mean_a = max(-act_limit, min(act_limit, mean_a))
    return agree, mean_a

def compute_zscore_stats(X: np.ndarray):
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_zscore(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    # Ensure float32
    return ((X.astype(np.float32) - mu.astype(np.float32)) / sd.astype(np.float32)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--state_csv", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--zscore-from-states", action="store_true",
                    help="Compute mean/std from states.csv and z-score inputs before inference.")
    args = ap.parse_args()

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

    # Load states (float32)
    with open(args.state_csv, newline="") as f:
        rdr = csv.DictReader(f)
        hdr = [h.strip().lower() for h in (rdr.fieldnames or [])]
        if hdr != REQUIRED_COLS:
            print(f"[error] state_csv header mismatch.\n  got : {rdr.fieldnames}\n  want: {REQUIRED_COLS}", file=sys.stderr)
            sys.exit(3)
        X = []
        for row in rdr:
            try:
                X.append([float(row[c]) for c in REQUIRED_COLS])
            except Exception:
                pass
        X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        print("[error] no usable rows in state_csv.", file=sys.stderr)
        sys.exit(4)

    mu = sd = None
    if args.zscore_from_states:
        mu, sd = compute_zscore_stats(X)
        X = apply_zscore(X, mu, sd)
        print(f"[norm] z-score applied (float32).")
        print(f"[norm] mean  : {[round(float(x),4) for x in mu.tolist()]}")
        print(f"[norm] std   : {[round(float(x),4) for x in sd.tolist()]}")

    # Load models
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    models=[]; kinds=[]
    for a in actors:
        if not os.path.isfile(a):
            print(f"[error] actor not found: {a}", file=sys.stderr); sys.exit(5)
        m, k = load_flexible_model(a, device)
        models.append(m)
        kinds.append(k)
    print("[info] loaded actors:", list(zip(actors, kinds)))

    # Inference (ensure float32 input tensors)
    T = X.shape[0]; results=[]
    print("row_idx,agree,action,details")
    for i in range(T):
        s = torch.from_numpy(X[i:i+1].astype(np.float32)).to(device).to(torch.float32)
        acts=[]; qs=[]
        for m in models:
            a, q = model_forward(m, s)
            a_np = a.detach().cpu().numpy().astype(np.float32).reshape(-1)
            acts.append(float(a_np[0]) if a_np.size>0 else 0.0)
            if q is not None:
                q_np = q.detach().cpu().numpy().astype(np.float32).reshape(-1)
                if q_np.size>0: qs.append(float(q_np[0]))

        aa = np.array(acts, dtype=np.float32)
        agree, mean_a = consensus_action(aa, agree_eps, act_limit)

        q_ok = True; min_q=None
        if qs:
            min_q = float(np.min(qs))
            q_ok = (min_q >= q_min_thresh)

        final_ok = (agree and q_ok)
        action_out = mean_a if final_ok else 0.0

        details = {"actors": len(models), "agree_eps": agree_eps, "act_limit": act_limit,
                   "max_pair_diff": float(np.max(np.abs(aa[:,None]-aa[None,:]))) if len(aa)>1 else 0.0,
                   "min_q": min_q}
        print(f"{i},{int(final_ok)},{action_out},{details}")
        results.append((i, int(final_ok), action_out, details))

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["row_idx","agree","action","min_q"])
            for i, ok, a, det in results:
                w.writerow([i, ok, a, (det.get("min_q") if isinstance(det, dict) else None)])

if __name__ == "__main__":
    main()
