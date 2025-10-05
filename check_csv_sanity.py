#!/usr/bin/env python3
import argparse, os, csv, itertools, sys

def sniff(path, n=3):
    if not os.path.isfile(path):
        print(f"[error] missing file: {path}")
        return
    size = os.path.getsize(path)
    with open(path, 'r', newline='') as f:
        lines = f.readlines()
    print(f"[file] {path}  bytes={size}  lines={len(lines)}")
    if not lines:
        return
    print("[head]")
    for ln in lines[:n]:
        print(ln.rstrip("\n"))
    # header + minimal parse
    try:
        f = open(path, 'r', newline='')
        r = csv.DictReader(f)
        print("[header]", r.fieldnames)
        first = list(itertools.islice(r, 3))
        print("[sample rows]", len(first))
        for i,row in enumerate(first):
            print(f"  row{i} keys={list(row.keys())[:6]} ...")
    except Exception as e:
        print("[warn] csv parse error:", e)
    finally:
        try: f.close()
        except: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", required=True)
    args = ap.parse_args()
    sniff(args.states)
    sniff(args.gate)

if __name__ == "__main__":
    main()
