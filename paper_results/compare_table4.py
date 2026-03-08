#!/usr/bin/env python3
"""Compare reproduced Table 4 results against paper reference values.

For each matching (Model, CtxSize, Vram) row:
  TPS  — ratio = repro / paper   (higher is better)
  TTFT — ratio = paper / repro   (lower is better, so inverted ratio)

  ratio >= 0.9  →  PASS
  ratio <  0.9  →  prints what % of the paper value was achieved

Usage:
    python paper_results/compare_table4.py [--repro CSV] [--paper CSV]
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAPER = os.path.join(SCRIPT_DIR, "paper_results_table4.csv")
DEFAULT_REPRO = os.path.join(SCRIPT_DIR, "table4_results.csv")

THRESHOLD = 0.9

VRAM_SORT = {"2G": 0, "4G": 1, "6G": 2, "8G": 3, "12G": 4, "16G": 5, "24G": 6, "32G": 7}
CTX_SORT = {"1K": 0, "4K": 1, "16K": 2, "64K": 3}
MODEL_SORT = {"nemo-4b": 0, "nemo-8b": 1, "qwen-30b": 2, "qwen-235b": 3}


def sort_key(key):
    model, ctx, vram = key
    return (MODEL_SORT.get(model, 99), CTX_SORT.get(ctx, 99), VRAM_SORT.get(vram, 99))


def load_csv(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["Model"].strip(), r["CtxSize"].strip(), r["Vram"].strip())
            tps = r.get("TPS", "N/A").strip().strip('"')
            ttft = r.get("TTFT(msec)", "N/A").strip().strip('"')
            rows[key] = {"TPS": tps, "TTFT": ttft}
    return rows


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def ratio_str(ratio):
    return f"{ratio:.2f}x of paper"


def compare(paper_path, repro_path):
    paper = load_csv(paper_path)
    repro = load_csv(repro_path)

    total = 0
    tps_pass = 0
    tps_below = 0
    ttft_pass = 0
    ttft_below = 0
    skipped = 0

    current_model = None
    current_ctx = None

    for key in sorted(paper.keys(), key=sort_key):
        model, ctx, vram = key

        if model != current_model:
            if current_model is not None:
                print()
            current_model = model
            current_ctx = None
            print(f"  {model}")

        if ctx != current_ctx:
            current_ctx = ctx

        if key not in repro:
            skipped += 1
            continue

        p = paper[key]
        r = repro[key]
        p_tps = safe_float(p["TPS"])
        r_tps = safe_float(r["TPS"])
        p_ttft = safe_float(p["TTFT"])
        r_ttft = safe_float(r["TTFT"])

        total += 1

        tps_cell = ""
        if p_tps and r_tps and p_tps > 0:
            tps_ratio = r_tps / p_tps
            if tps_ratio >= THRESHOLD:
                tps_cell = "PASS"
                tps_pass += 1
            else:
                tps_cell = ratio_str(tps_ratio)
                tps_below += 1
        else:
            tps_cell = "-"

        ttft_cell = ""
        if p_ttft and r_ttft and r_ttft > 0:
            ttft_ratio = p_ttft / r_ttft
            if ttft_ratio >= THRESHOLD:
                ttft_cell = "PASS"
                ttft_pass += 1
            else:
                ttft_cell = ratio_str(ttft_ratio)
                ttft_below += 1
        else:
            ttft_cell = "-"

        print(f"    {ctx:>3s} {vram:>4s}  TPS: {tps_cell:<20s}  TTFT: {ttft_cell}")

    print()
    print("-" * 50)
    print(f"  Compared {total} rows  (skipped {skipped})")
    print(f"  TPS   PASS: {tps_pass}/{total}   below 0.9x: {tps_below}")
    print(f"  TTFT  PASS: {ttft_pass}/{total}   below 0.9x: {ttft_below}")
    if tps_below == 0 and ttft_below == 0:
        print("  Result: ALL PASS")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare reproduced Table 4 results against paper.")
    parser.add_argument("--paper", default=DEFAULT_PAPER, help="Path to paper reference CSV")
    parser.add_argument("--repro", default=DEFAULT_REPRO, help="Path to reproduced results CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.paper):
        print(f"ERROR: Paper reference CSV not found: {args.paper}")
        sys.exit(1)
    if not os.path.isfile(args.repro):
        print(f"ERROR: Reproduced results CSV not found: {args.repro}")
        print("       Run repro_table4 first, then compare.")
        sys.exit(1)

    print()
    print("  Table 4: Reproduced vs Paper (threshold 0.9x)")
    print("=" * 50)

    compare(args.paper, args.repro)


if __name__ == "__main__":
    main()
