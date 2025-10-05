#!/usr/bin/env python3
# live_status.py — print a human summary of the current live state
import json, sys, pathlib, datetime as dt

def main():
    statefile = pathlib.Path("live_state.json")
    if not statefile.exists():
        print("state: not found (no position yet)"); return
    st = json.loads(statefile.read_text())

    pos = int(st.get("prev_pos", 0))
    held = int(st.get("bars_held", 0))
    held_total = int(st.get("bars_held_total", 0))
    entry_mid = st.get("entry_mid", None)
    entry_ts  = st.get("entry_ts", None)

    pos_txt = "FLAT"
    if pos>0: pos_txt = "LONG"
    elif pos<0: pos_txt = "SHORT"

    print("=== LIVE STATUS ===")
    print(f"position     : {pos_txt} ({pos})")
    print(f"held (bars)  : {held}  | total: {held_total}")
    print(f"entry_mid    : {entry_mid if entry_mid is not None else '-'}")
    print(f"entry_ts     : {entry_ts if entry_ts else '-'}")

    # show last log line if available
    log = pathlib.Path("live_signals.csv")
    if log.exists():
        last = log.read_text().strip().splitlines()[-1]
        print(f"last_signal  : {last}")
    else:
        print("last_signal  : (no log yet)")

if __name__ == "__main__":
    main()
