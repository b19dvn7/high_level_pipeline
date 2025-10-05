#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: $0 SWEEP_DIR TAG" >&2
  exit 2
fi
SWEEP="$1"
TAG="$2"

CSV="$SWEEP/results.csv"
GATEDIR="$SWEEP/gates"

if [ ! -f "$CSV" ]; then
  echo "[ERR] results.csv not found: $CSV" >&2
  exit 1
fi
if [ ! -d "$GATEDIR" ]; then
  echo "[ERR] gates dir not found: $GATEDIR" >&2
  exit 1
fi

# Strip any CRLF just in case and cache to a temp file
TMP="$(mktemp)"
tr -d '\r' < "$CSV" > "$TMP"

# Try to find tag and gate_csv columns
RES=$(
  awk -F, -v want="$TAG" '
    NR==1{
      for(i=1;i<=NF;i++){
        if($i=="tag") t=i
        if($i=="gate_csv") g=i
      }
      next
    }
    t && g && $t==want { print $g; exit 0 }
  ' "$TMP" || true
)

# If that worked and the file exists, print it
if [ -n "${RES:-}" ] && [ -f "$GATEDIR/$RES" ]; then
  echo "$RES"
  rm -f "$TMP"
  exit 0
fi

# Fallback: filename by convention
CAND="gate_${TAG}.csv"
if [ -f "$GATEDIR/$CAND" ]; then
  echo "$CAND"
  rm -f "$TMP"
  exit 0
fi

# Last resort: try to grep a matching line with that tag and cut the gate column
RES2=$(
  awk -F, -v want="$TAG" '
    NR==1{
      for(i=1;i<=NF;i++){
        if($i=="tag") t=i
        if($i=="gate_csv") g=i
      }
      next
    }
    t && g && index($t, want)==1 { print $g; exit 0 }
  ' "$TMP" || true
)
if [ -n "${RES2:-}" ] && [ -f "$GATEDIR/$RES2" ]; then
  echo "$RES2"
  rm -f "$TMP"
  exit 0
fi

rm -f "$TMP"
echo "[ERR] Could not resolve gate for TAG=$TAG in $SWEEP" >&2
exit 1
