#!/usr/bin/env python3
# consensus_gate_td3bc_flexible.py
# Works with TorchScript, full nn.Module, or raw state_dict (*.pt)
import argparse, json, csv, os, sys, re
from typing import List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[error] Requires torch. Try: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
    sys.exit(2)

REQUIRED_COLS = ["mid","spread","imbalance","mom1","mom3","vol3"]

# ----- Helpers to support state_dict-only checkpoints -----

def _extract_layers_from_state_dict(sd: "OrderedDict") -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Return list of (W, b, layer_index) sorted by layer_index.
    Tries to infer layer order from numeric hints in keys; falls back to lexical order.
    """
    weights = []
    # collect pairs that look like *.weight / *.bias
    pairs = {}
    for k in sd.keys():
        if k.endswith(".weight"):
            prefix = k[:-7]
            bkey = prefix + ".bias"
            if bkey in sd:
                pairs[prefix] = (k, bkey)

    def get_idx(name: str) -> int:
        # pull last integer in the name, else 1e9 to send to the end
        m = list(re.finditer(r'(\d+)', name))
        return int(m[-1].group(1)) if m else 10**9

    items = []
    for prefix, (wk, bk) in pairs.items():
        idx = get_idx(prefix)
        items.append((idx, prefix, wk, bk))
    # sort by index then by name for stability
    items.sort(key=lambda x: (x[0], x[1]))

    for idx, prefix, wk, bk in items:
        W = sd[wk].detach().clone()
        b = sd[bk].detach().clone()
        weights.append((W, b, idx))
    return weights

class StateDictMLPRunner(nn.Module):
    """
    Minimal runner that applies a list of Linear layers defined by (W, b).
    Hidden layers = ReLU; Last layer = Tanh if output dim == 1 else Identity.
    """
    def __init__(self, layers: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        super().__init__()
        # store as Parameters so we can .to(device)
        self.W = nn.ParameterList([nn.Parameter(W) for (W, b, _) in layers])
        self.b = nn.ParameterList([nn.Parameter(b) for (W, b, _) in layers])
        self.activ_last_tanh = (self.W[-1].shape[0] == 1)  # out_features
        self.eval()  # no-op but matches Module interface

    def forward(self, x: torch.Tensor):
        h = x
        for i in range(len(self.W)):
            # x shape: [B, in], W shape: [out, in], b: [out]
            h = torch.addmm(self.b[i], h, self.W[i].t())
            if i < len(self.W) - 1:
                h = torch.relu(h)
            else:
                if self.activ_last_tanh:
                    h = torch.tanh(h)
        # we don't have Q; return only action tensor
        return h

# ----- Model loader that handles 3 formats -----

def load_flexible_model(path: str, device: str):
    # 1) TorchScript
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        return m, "jit"
    except Exception:
        pass

    obj = torch.load(path, map_location=device)
    # 2) Full nn.Module
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj, "module"

    # 3) Raw state_dict (OrderedDict)
    if isinstance(obj, dict):
        sd = obj.get("state_dict", None)
        if sd is None:
            # could be already a state_dict-like mapping
            sd = obj
        if isinstance(sd, dict):
            layers = _extract_layers_from_state_dict(sd)
            if not layers:
                raise RuntimeError("Could not infer Linear layers from state_dict keys.")
            runner = StateDictMLPRunner(layers).to(device)
            return runner, "state_dict"
    raise RuntimeError(f"Unsupported checkpoint format for {path}")

@torch.no_grad()
def model_forward(model, x: torch.Tensor):
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        # (a, q) pattern
        return out[0], out[1]
    return out, None  # action only

def consensus_action(actions: np.ndarray, agree_eps: float, act_limit: float) -> Tuple[bool, float]:
    if actions.ndim != 1:
        actions = actions.reshape(-1)
    if actions.size == 0:
        return False, 0.0
    max_diff = float(np.max(np.abs(actions[:, None] - actions[None, :])))
    agree = max_diff <= (agree_eps * act_limit)
    mean_a = float(np.mean(actions))
    mean_a = max(-act_limit, min(act_limit, mean_a))
    return agree, mean_a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, help="consensus_gate_td3bc.json")
    ap.add_argument("--state_csv", required=True, help="CSV with: mid,spread,imbalance,mom1,mom3,vol3")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    ap.add_argument("--out_csv", default=None)
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

    # Load states
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
            except Exception as e:
                print(f"[warn] bad row skipped: {e}", file=sys.stderr)
        X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        print("[error] no usable rows in state_csv.", file=sys.stderr)
        sys.exit(4)

    # Load models
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    models = []
    kinds  = []
    for a in actors:
        if not os.path.isfile(a):
            print(f"[error] actor file not found: {a}", file=sys.stderr)
            sys.exit(5)
        m, kind = load_flexible_model(a, device)
        models.append(m)
        kinds.append(kind)
    print("[info] loaded actors:", list(zip(actors, kinds)))

    # Inference
    T = X.shape[0]
    results = []
    print("row_idx,agree,action,details")
    for i in range(T):
        s = torch.from_numpy(X[i:i+1]).to(device)  # [1, state_dim]
        acts = []
        qs   = []
        for m in models:
            a, q = model_forward(m, s)
            a_np = a.detach().cpu().numpy().reshape(-1)
            acts.append(float(a_np[0]) if a_np.size > 0 else 0.0)
            if q is not None:
                q_np = q.detach().cpu().numpy().reshape(-1)
                if q_np.size > 0: qs.append(float(q_np[0]))

        acts_arr = np.array(acts, dtype=np.float32)
        agree, mean_a = consensus_action(acts_arr, agree_eps, act_limit)

        q_ok = True; min_q=None
        if qs:
            min_q = float(np.min(qs))
            q_ok = (min_q >= q_min_thresh)

        final_ok = (agree and q_ok)
        action_out = mean_a if final_ok else 0.0

        details = {"actors": len(models), "agree_eps": agree_eps, "act_limit": act_limit,
                   "max_pair_diff": float(np.max(np.abs(acts_arr[:,None]-acts_arr[None,:]))) if len(acts_arr)>1 else 0.0,
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
