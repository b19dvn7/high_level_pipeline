#!/usr/bin/env bash
set -euo pipefail
add_guard() {
  f="$1"
  [[ -f "$f" ]] || { echo "[skip] $f not found"; return; }
  if grep -q 'signal.SIGPIPE' "$f"; then
    echo "[ok] $f already guarded"
  else
    cp -f "$f" "$f.bak_sigpipe_$(date +%s)"
    { 
      # keep shebang if present
      head -n 1 "$f" | grep -q '^#!' && { head -n 1 "$f"; tail -n +2 "$f"; } || cat "$f"
    } | awk 'NR==1 && $0 ~ /^#!/ {print; next}
             NR==1 {print "import sys, signal"; print "signal.signal(signal.SIGPIPE, signal.SIG_DFL)";}
             NR>1 {print}' > "$f.tmp"
    mv "$f.tmp" "$f"
    echo "[patched] $f"
  fi
}
add_guard consensus_gate_td3bc_flexible.py
add_guard consensus_gate_live_plus.py
add_guard auto_tune_gate.py
add_guard live_dashboard.py
