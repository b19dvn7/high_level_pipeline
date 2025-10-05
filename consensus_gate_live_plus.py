#!/usr/bin/env python3
# consensus_gate_live_plus.py
# Live gate with scaler.json; supports:
#  - value-consensus (pairwise diff within eps)
#  - sign-consensus (same sign, optional min magnitude)
#  - k-of-n agreement (min_agree_k)
#  - CLI overrides for agree_eps, act_limit
#  - float32-safe for models and inputs
import argparse, json, csv, os, sys, re
from typing import List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[error] Requires torch.", file=sys.stderr); sys.exit(2)

COLS = ["mid","spread","imbalance","mom1","mom3","vol3"]

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
        B = sd[bk].detach().clone().float()
        layers.append((W, B))
    return layers

class StateDictMLPRunner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(W) for (W,B) in layers])
        self.B = nn.ParameterList([nn.Parameter(B) for (W,B) in layers])
        self.activ_last_tanh = (self.W[-1].shape[0] == 1)
        self.eval()
    def forward(self, x):
        h = x
        for i in range(len(self.W)):
            h = torch.addmm(self.B[i], h, self.W[i].t())
            if i < len(self.W)-1: h = torch.relu(h)
            else:
                if self.activ_last_tanh: h = torch.tanh(h)
        return h

def load_flexible_model(path: str, device: str):
    # TorchScript
    try:
        m = torch.jit.load(path, map_location=device)
        m = m.float(); m.eval(); return m, "jit"
    except Exception:
        pass
    # torch.load
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        obj = obj.float(); obj.eval(); return obj, "module"
    if isinstance(obj, dict):
        sd = obj.get("state_dict", None) or obj
        if isinstance(sd, dict):
            layers = _extract_layers_from_state_dict(sd)
            if not layers: raise RuntimeError("No linear layers parsed from state_dict.")
            runner = StateDictMLPRunner(layers).to(device).float()
            return runner, "state_dict"
    raise RuntimeError(f"Unsupported checkpoint format: {path}")

@torch.no_grad()
def model_forward(model, x):
    out = model(x)
    if isinstance(out, (tuple,list)) and len(out) >= 2:
        return out[0], out[1]
    return out, None

def load_scaler(path: str):
    with open(path, "r") as f:
        S = json.load(f)
    if S.get("type") != "zscore": raise SystemExit("scaler.json type must be 'zscore'")
    cols = S["cols"]
    if [c.lower() for c in cols] != COLS: raise SystemExit(f"scaler cols mismatch: {cols}")
    mean = np.asarray(S["mean"], dtype=np.float32)
    std  = np.asarray(S["std"], dtype=np.float32)
    std  = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std

def consensus_value(actions: np.ndarray, agree_eps: float, act_limit: float, min_agree_k: int):
    # pick the largest cluster where |ai - aj| <= agree_eps*act_limit
    if actions.size == 0: return False, 0.0, 0.0
    tol = agree_eps * act_limit
    acts = np.sort(actions)
    best_idx = (0,1)  # [start, end) window
    i = 0
    for j in range(1, len(acts)+1):
        while acts[j-1] - acts[i] > tol:
            i += 1
        if j - i > best_idx[1] - best_idx[0]:
            best_idx = (i, j)
    cluster = acts[best_idx[0]:best_idx[1]]
    ok = (len(cluster) >= min_agree_k)
    mean_a = float(np.mean(cluster)) if ok else 0.0
    mean_a = max(-act_limit, min(act_limit, mean_a))
    maxdiff = float(np.max(np.abs(actions[:,None]-actions[None,:]))) if len(actions)>1 else 0.0
    return ok, mean_a, maxdiff

def consensus_sign(actions: np.ndarray, min_agree_k: int, act_limit: float, min_mag: float = 0.0):
    if actions.size == 0: return False, 0.0, 0.0
    pos = actions[actions >= +min_mag]
    neg = actions[actions <= -min_mag]
    # choose larger agreeing set
    cluster = pos if len(pos) >= len(neg) else neg
    ok = (len(cluster) >= min_agree_k)
    mean_a = float(np.mean(cluster)) if ok else 0.0
    mean_a = max(-act_limit, min(act_limit, mean_a))
    maxdiff = float(np.max(np.abs(actions[:,None]-actions[None,:]))) if len(actions)>1 else 0.0
    return ok, mean_a, maxdiff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--scaler", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--state_csv")
    g.add_argument("--state_row")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    ap.add_argument("--out_csv", default=None)

    # consensus controls
    ap.add_argument("--consensus-mode", default="value", choices=["value","sign"],
                    help="value: pairwise diff within eps; sign: same sign with optional min magnitude")
    ap.add_argument("--min-agree-k", type=int, default=None, help="require at least k actors to agree (default: all actors)")
    ap.add_argument("--min-mag", type=float, default=0.0, help="sign mode: minimum |action| to count as a vote")

    # overrides (optional; if omitted, read from JSON)
    ap.add_argument("--agree-eps", type=float, default=None, help="override agree_eps")
    ap.add_argument("--act-limit", type=float, default=None, help="override act_limit")

    args = ap.parse_args()

    with open(args.gate,"r") as f: gate = json.load(f)
    if gate.get("type") != "td3bc_consensus_gate":
        raise SystemExit("gate.type must be td3bc_consensus_gate")
    actors: List[str] = gate["actors"]
    agree_eps = float(args.agree_eps if args.agree_eps is not None else gate["agree_eps"])
    act_limit = float(args.act_limit if args.act_limit is not None else gate.get("act_limit", 1.0))
    q_min_thresh: float = float(gate.get("q_min_thresh", -1e9))  # ignored if no Q
    min_agree_k = int(args.min_agree_k) if args.min_agree_k is not None else len(actors)

    mu, sd = load_scaler(args.scaler)  # float32

    # states
    X=[]
    if args.state_csv:
        with open(args.state_csv, newline="") as f:
            rdr = csv.DictReader(f)
            hdr = [h.strip().lower() for h in (rdr.fieldnames or [])]
            if hdr != COLS: raise SystemExit(f"state_csv header mismatch: got {rdr.fieldnames}, want {COLS}")
            for row in rdr:
                try:
                    X.append([float(row[c]) for c in COLS])
                except: pass
    else:
        vals = [float(x.strip()) for x in args.state_row.split(",")]
        if len(vals) != 6: raise SystemExit("state_row must have 6 comma-separated values")
        X.append(vals)
    if not X: raise SystemExit("no usable rows")

    X = np.asarray(X, dtype=np.float32)
    Xn = ((X - mu) / sd).astype(np.float32)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    models=[]; kinds=[]
    for a in actors:
        if not os.path.isfile(a): raise SystemExit(f"actor not found: {a}")
        m, k = load_flexible_model(a, device); models.append(m); kinds.append(k)
    print("[info] loaded actors:", list(zip(actors, kinds)))
    print(f"[info] mode={args.consensus_mode} agree_eps={agree_eps} act_limit={act_limit} min_agree_k={min_agree_k} min_mag={args.min_mag}")

    # infer
    T = Xn.shape[0]; out=[]
    print("row_idx,agree,action,max_pair_diff,min_q")
    for i in range(T):
        s = torch.from_numpy(Xn[i:i+1].astype(np.float32)).to(device).to(torch.float32)
        acts=[]; qs=[]
        for m in models:
            a, q = model_forward(m, s)
            a_np = a.detach().cpu().numpy().astype(np.float32).reshape(-1)
            acts.append(float(a_np[0]) if a_np.size>0 else 0.0)
            if q is not None:
                q_np = q.detach().cpu().numpy().astype(np.float32).reshape(-1)
                if q_np.size>0: qs.append(float(q_np[0]))
        actions = np.array(acts, dtype=np.float32)

        if args.consensus_mode == "value":
            ok, mean_a, maxdiff = consensus_value(actions, agree_eps, act_limit, min_agree_k)
        else:
            ok, mean_a, maxdiff = consensus_sign(actions, min_agree_k, act_limit, args.min_mag)

        min_q = float(np.min(qs)) if qs else None
        q_ok = (min_q is None) or (min_q >= q_min_thresh)
        ok = ok and q_ok
        action_out = mean_a if ok else 0.0
        print(f"{i},{int(ok)},{action_out},{maxdiff},{min_q}")
        out.append((i,int(ok),action_out,min_q))

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["row_idx","agree","action","min_q"])
            for i,ok,a,mq in out: w.writerow([i,ok,a,mq])

if __name__ == "__main__":
    main()
