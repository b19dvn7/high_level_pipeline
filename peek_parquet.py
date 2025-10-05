#!/usr/bin/env python3
# peek_parquet.py
# Minimal, version-safe Parquet inspector using DuckDB.
# - Lists how many files match your glob and prints a few example paths
# - Prints the unified schema (column name + DuckDB type)
# - Prints the first N rows (all columns) so we can see real values
# Nothing else. No guesses. No transformations.

import argparse
import sys
import duckdb

def sql_lit(path: str) -> str:
    # Safe single-quoted SQL literal
    return "'" + path.replace("'", "''") + "'"

def main():
    ap = argparse.ArgumentParser(description="Minimal Parquet inspector (DuckDB).")
    ap.add_argument("--parquet", required=True, help='Recursive glob, e.g. /path/**/*.parquet')
    ap.add_argument("--limit", type=int, default=10, help="Number of preview rows to print (default: 10)")
    ap.add_argument("--files", type=int, default=10, help="Show up to this many example file paths (default: 10)")
    args = ap.parse_args()

    con = duckdb.connect(database=":memory:")

    # 1) Count files and show a few paths
    try:
        q_files = f"SELECT DISTINCT filename FROM read_parquet({sql_lit(args.parquet)}, filename=true)"
        files = [r[0] for r in con.execute(q_files).fetchall()]
    except Exception as e:
        print(f"[error] Could not enumerate files for pattern: {args.parquet}")
        print(f"        {e}", file=sys.stderr)
        sys.exit(2)

    if not files:
        print(f"[info] No files matched pattern: {args.parquet}")
        sys.exit(0)

    print(f"[ok] Matched files: {len(files)}")
    for p in files[:args.files]:
        print("  -", p)
    if len(files) > args.files:
        print(f"  ... (+{len(files)-args.files} more)")

    # 2) Unified schema across the match
    try:
        q_schema = f"DESCRIBE SELECT * FROM read_parquet({sql_lit(args.parquet)})"
        schema_rows = con.execute(q_schema).fetchall()
    except Exception as e:
        print(f"[error] Could not DESCRIBE schema: {e}", file=sys.stderr)
        sys.exit(3)

    print("\n[Schema]")
    # DuckDB DESCRIBE returns: name, type, null, key, default, extra
    # We'll print name and type.
    for row in schema_rows:
        col_name, col_type = row[0], row[1]
        print(f"  {col_name}: {col_type}")

    # 3) Preview first N rows (all columns)
    try:
        q_preview = f"SELECT * FROM read_parquet({sql_lit(args.parquet)}) LIMIT {int(args.limit)}"
        df = con.execute(q_preview).fetchdf()
    except Exception as e:
        print(f"\n[error] Could not fetch preview rows: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"\n[Preview: first {len(df)} row(s)]")
    # Pretty print as CSV-ish for easy eyeballing
    # (Avoid Pandas repr width truncation)
    if df.empty:
        print("  (no rows)")
        sys.exit(0)

    # Print header
    cols = list(df.columns)
    print(",".join(str(c) for c in cols))
    # Print rows
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            # Keep it simple; stringify None/NaN clearly
            if v is None:
                vals.append("NULL")
            else:
                s = str(v)
                # replace newlines/commas to keep one-line per row readable
                s = s.replace("\n", " ").replace(",", ";")
                vals.append(s)
        print(",".join(vals))

if __name__ == "__main__":
    main()
