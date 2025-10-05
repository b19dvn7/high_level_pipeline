#!/usr/bin/env python3
# live_status_pretty.py
# Summarize live state in a single human-readable line.

import json, pathlib, csv, math, sys
from datetime import datetime, timezone

STATE = pathlib.Path("live_state.json")
LOG   = pathlib.Path("live_signals.csv")

def safe_float(x, default=None):
    try: return float(x)
    except: return default

def load_state():
    st = {"prev_pos":0,"bars_held":0,"bars_held_total":0,"entry_mid":None,"entry_ts":None}
    if STATE.exists():
        try: st.update(json.loads(STATE.read_text()))
        except: pass
    return st

def read_last_log_row():
    if not LOG.exists(): return None
    lines = LOG.read_text().strip().splitlines()
    if len(lines) <= 1: return None
    hdr = [h.strip() for h in lines[0].split(",")]
    row = [c.strip() for c in lines[-1].split(",")]
    obj = {k:(row[i] if i < len(row) else "") for i,k in enumerate(hdr)}
    # normalize fields possibly present
    mid = safe_float(obj.get("mid"))
    spread = safe_float(obj.get("spread"))
    vol3 = safe_float(obj.get("vol3"))
    raw_action = safe_float(obj.get("raw_action"), 0.0)
    buy_th = safe_float(obj.get("buy_th") or obj.get("th"), 0.10)
    sell_th = safe_float(obj.get("sell_th") or obj.get("th"), 0.10)
    hold_bars = int(float(obj.get("hold_bars", "12") or 12))
    hold_ctr  = int(float(obj.get("hold_counter", "0") or 0))
    ts = obj.get("ts","")
    return dict(ts=ts, mid=mid, spread=spread, vol3=vol3, raw_action=raw_action,
                buy_th=buy_th, sell_th=sell_th, hold_bars=hold_bars, hold_ctr=hold_ctr)

def fmt_bps(x):
    return ("+" if x>=0 else "") + f"{x:.2f} bps"

def main():
    st = load_state()
    last = read_last_log_row() or {}
    pos = int(st.get("prev_pos", 0))
    held = int(st.get("bars_held", 0))
    held_total = int(st.get("bars_held_total", 0))
    entry_mid = st.get("entry_mid")
    entry_ts  = st.get("entry_ts")
    cur_mid   = last.get("mid")

    pos_txt = "FLAT"
    if pos>0: pos_txt = "LONG"
    elif pos<0: pos_txt = "SHORT"

    upnl_bps = None
    if entry_mid is not None and cur_mid is not None and pos != 0:
        ret = (cur_mid - float(entry_mid)) / float(entry_mid)
        upnl_bps = (ret * (1 if pos>0 else -1)) * 1e4

    # next flip earliest in:
    hold_bars = last.get("hold_bars", 12) or 12
    hold_ctr  = last.get("hold_ctr", held) or held
    flip_in   = max(0, int(hold_bars) - int(hold_ctr))

    # build one clean line
    ts = last.get("ts","")
    parts = [
        f"{ts}",
        f"{pos_txt}",
        f"held {held}/{hold_bars} bars (total {held_total})",
        f"flip in {flip_in} bars" if pos != 0 else "flip allowed now",
        f"entry_mid={entry_mid:.2f}" if isinstance(entry_mid,(int,float)) else "entry_mid=-",
        f"mid={cur_mid:.2f}" if isinstance(cur_mid,(int,float)) else "mid=-",
        f"uPnL={fmt_bps(upnl_bps)}" if upnl_bps is not None else "uPnL=-"
    ]
    print(" | ".join(parts))

if __name__ == "__main__":
    main()
