#!/usr/bin/env python3
# probe_td3bc_checkpoints.py
# Prints: format (jit/module/state_dict), md5 of each tensor, layer shapes order, quick stats
import argparse, hashlib, json, os, sys, re
from typing import Dict, Any, List, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[error] torch required", file=sys.stderr); sys.exit(2)

def md5_tensor(t):
    b = t.detach().cpu().numpy().tobytes()
    return hashlib.md5(b).hexdigest()

def load_any(path, device="cpu"):
    # 1) jit
    try:
        m = torch.jit.load(path, map_location=device)
        return ("jit", m)
    except Exception:
        pass
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        return ("module", obj)
    if isinstance(obj, dict):
        sd = obj.get("state_dict", None) or obj
        if isinstance(sd, dict):
            return ("state_dict", sd)
    return ("unknown", obj)

def parse_layers_from_state_dict(sd: Dict[str, Any]) -> List[Tuple[str, Tuple[int,...], Tuple[int,...]]]:
    pairs = []
    for k in sd:
        if k.endswith(".weight"):
            p = k[:-7]
            bk = p + ".bias"
            if bk in sd:
                w = sd[k]; b = sd[bk]
                wshape = tuple(w.shape); bshape = tuple(b.shape)
                pairs.append((p, wshape, bshape))
    def idx(name: str) -> int:
        ms = list(re.finditer(r'(\d+)', name))
        return int(ms[-1].group(1)) if ms else 10**9
    pairs.sort(key=lambda x: (idx(x[0]), x[0]))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="*.pt files")
    args = ap.parse_args()
    rows = []
    for p in args.ckpts:
        if not os.path.isfile(p):
            print(f"[error] missing: {p}", file=sys.stderr); sys.exit(1)
        kind, obj = load_any(p)
        print(f"\n== {p} :: {kind} ==")
        if kind in ("jit","module"):
            # Try state_dict for checksum
            try:
                sd = obj.state_dict()
            except Exception:
                sd = {}
        elif kind == "state_dict":
            sd = obj
        else:
            print("[warn] unknown format; skipping details"); continue

        # layer order
        layers = parse_layers_from_state_dict(sd)
        for i,(name, wsh, bsh) in enumerate(layers):
            w = sd[name+".weight"]; b = sd[name+".bias"]
            print(f"  L{i:02d} {name}: W{wsh} B{bsh} | w(md5)={md5_tensor(w)[:8]} b(md5)={md5_tensor(b)[:8]} "
                  f"| w[min={float(w.min()):+.4f}, max={float(w.max()):+.4f}] b[min={float(b.min()):+.4f}, max={float(b.max()):+.4f}]")
        # global fingerprint
        h = hashlib.md5()
        for k in sorted(sd.keys()):
            if hasattr(sd[k], "shape"):
                h.update(sd[k].cpu().numpy().tobytes())
        print(f"  >> checkpoint_fingerprint: {h.hexdigest()}")

if __name__ == "__main__":
    main()
