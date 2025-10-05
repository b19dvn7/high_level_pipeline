#!/usr/bin/env python3
import argparse, csv, time, os, sys

ANSI = {
    "reset":"\033[0m","r":"\033[31m","g":"\033[32m","y":"\033[33m",
    "b":"\033[34m","m":"\033[35m","c":"\033[36m","bd":"\033[1m"
}
def C(txt, c, use=True): 
    s = ANSI.get(c,"") + str(txt) + ANSI["reset"]
    return s if use else str(txt)

def pos_name(x):
    try: x = int(float(x))
    except: return f"{x}"
    return {-1:"SHORT (-1)", 0:"FLAT (0)", 1:"LONG (+1)"}.get(x, f"{x:+d}")

def conf_level(a):
    if a is None: return "n/a"
    aa = abs(a)
    if aa >= 0.75: return "very strong"
    if aa >= 0.50: return "strong"
    if aa >= 0.25: return "medium"
    if aa >= 0.10: return "weak"
    return "very weak"

def safe_float(x, default=None):
    try:
        return float(x)
    except:
        return default

def read_last_row(path):
    """
    Robust reader:
    - Finds the first header (row[0] == 'ts')
    - Ignores any repeated headers later in the file
    - Returns dict of last data row keyed by header names
    """
    if not os.path.exists(path):
        return None, "not_found"
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, "empty"

    # find a header row: must start with 'ts'
    header_idx = None
    for i, r in enumerate(rows):
        if r and r[0].strip().lower() == "ts":
            header_idx = i
            break
    if header_idx is None:
        return None, "no_header"

    hdr = rows[header_idx]
    # gather only data rows after the header, skipping any repeated header lines
    data = [r for r in rows[header_idx+1:] if r and r[0].strip().lower() != "ts"]
    if not data:
        return None, "no_data"

    last = data[-1]
    # allow for trailing commas / short lines by padding
    if len(last) < len(hdr):
        last = last + [""] * (len(hdr) - len(last))

    idx = {k: i for i, k in enumerate(hdr)}

    def get(k, default=None):
        i = idx.get(k)
        if i is None or i >= len(last):
            return default
        val = last[i]
        return val if val != "" else default

    # read fields (support either buy_th/sell_th or a single th)
    ts = get("ts", "?")
    prev_pos = safe_float(get("prev_pos"), 0)
    next_pos = safe_float(get("next_pos"), 0)
    raw_action = safe_float(get("raw_action"), None)

    th = safe_float(get("th"), None)
    buy_th = safe_float(get("buy_th"), th if th is not None else 0.10)
    sell_th = safe_float(get("sell_th"), th if th is not None else 0.10)

    hold_bars = int(safe_float(get("hold_bars"), 0) or 0)
    hold_counter = int(safe_float(get("hold_counter"), 0) or 0)

    spread = safe_float(get("spread"), None)
    vol3   = safe_float(get("vol3"), None)

    return {
        "ts": ts,
        "prev_pos": prev_pos,
        "next_pos": next_pos,
        "raw_action": raw_action,
        "buy_th": buy_th,
        "sell_th": sell_th,
        "hold_bars": hold_bars,
        "hold_counter": hold_counter,
        "spread": spread,
        "vol3": vol3,
    }, None

def decide(action, bth, sth):
    if action is None:
        return "NO SIGNAL", "y"
    if action >= bth:
        return "BUY (LONG)", "g"
    if action <= -sth:
        return "SELL (SHORT)", "r"
    return "HOLD / NO TRADE", "y"

def print_once(path, color=True):
    d, err = read_last_row(path)
    if err == "not_found":
        print(C(f"[error] {path} not found", "r", color)); return
    if err in ("empty", "no_data"):
        print(C("[info] waiting for first signal...", "y", color)); return
    if err == "no_header":
        print(C("[error] CSV missing header row starting with 'ts'", "r", color)); return

    act_txt, col = decide(d["raw_action"], d["buy_th"], d["sell_th"])

    print(C("=== LATEST SIGNAL ===", "bd", color))
    print(f"time: {d['ts']}")
    print(C(
        f"decision: {act_txt}   (action={d['raw_action']:+.3f}  buy_th={d['buy_th']:.2f}  sell_th={d['sell_th']:.2f})",
        col, color
    ))
    print(f"position: {pos_name(d['prev_pos'])}  →  {pos_name(d['next_pos'])}")

    if d["hold_bars"] > 0:
        pct = 100 * min(max(d["hold_counter"]/max(d["hold_bars"],1), 0), 1)
        print(f"hold: {d['hold_counter']}/{d['hold_bars']} bars ({pct:.0f}% done)")
    else:
        print("hold: none")

    print(f"confidence: {conf_level(d['raw_action'])}")
    extras = []
    if d["spread"] is not None: extras.append(f"spread={d['spread']}")
    if d["vol3"]   is not None: extras.append(f"vol3={d['vol3']}")
    if extras:
        print("market:", ", ".join(extras))

def watch(path, sec, color=True):
    try:
        last_size = -1
        while True:
            try:
                s = os.path.getsize(path)
                if s != last_size:
                    os.system("clear")
                    print_once(path, color)
                    last_size = s
            except FileNotFoundError:
                os.system("clear")
                print(C(f"[error] {path} not found", "r", color))
            time.sleep(sec)
    except KeyboardInterrupt:
        pass

def main():
    ap = argparse.ArgumentParser(description="Human-readable view of live_signals.csv")
    ap.add_argument("csv", help="Path to live_signals.csv")
    ap.add_argument("--watch", type=int, default=0, help="Refresh seconds (0 = print once)")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    a = ap.parse_args()
    if a.watch > 0:
        watch(a.csv, a.watch, color=not a.no_color)
    else:
        print_once(a.csv, color=not a.no_color)

if __name__ == "__main__":
    main()
