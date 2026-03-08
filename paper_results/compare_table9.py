#!/usr/bin/env python3
"""Compare reproduced Table 9 TPS values against paper reference values.

Table 9 reports TPS for qwen30b at batch sizes 1, 4, 16, 64 across 3 VRAM budgets.

Usage:
    python paper_results/compare_table9.py [--repro CSV] [--paper CSV]
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAPER = os.path.join(SCRIPT_DIR, "paper_results_table9.csv")
DEFAULT_REPRO = os.path.join(SCRIPT_DIR, "table9_results.csv")

THRESHOLD = 0.9

VRAM_SORT = {"2G": 0, "8G": 1, "16G": 2}


def sort_key(key):
    vram, bs = key
    return (VRAM_SORT.get(vram, 99), bs)


def load_csv(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vram = r["VramBudget"].strip().strip('"')
            bs = int(r["BatchSize"].strip().strip('"'))
            tps = r["TPS"].strip().strip('"')
            rows[(vram, bs)] = tps
    return rows


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def compare(paper_path, repro_path):
    paper = load_csv(paper_path)
    repro = load_csv(repro_path)

    total = 0
    passed = 0
    below = 0
    skipped = 0

    for key in sorted(paper.keys(), key=sort_key):
        vram, bs = key
        p_tps = safe_float(paper[key])

        if key not in repro:
            print(f"    {vram:>4s}  npl={bs:<3d}  TPS: (not in reproduced results)")
            skipped += 1
            continue

        r_tps = safe_float(repro[key])
        total += 1

        if p_tps and r_tps and p_tps > 0:
            ratio = r_tps / p_tps
            if ratio >= THRESHOLD:
                print(f"    {vram:>4s}  npl={bs:<3d}  TPS: PASS")
                passed += 1
            else:
                print(f"    {vram:>4s}  npl={bs:<3d}  TPS: {ratio:.2f}x of paper")
                below += 1
        else:
            print(f"    {vram:>4s}  npl={bs:<3d}  TPS: -")

    print()
    print("-" * 50)
    print(f"  Compared {total} rows  (skipped {skipped})")
    print(f"  PASS: {passed}/{total}   below 0.9x: {below}")
    if below == 0 and total > 0:
        print("  Result: ALL PASS")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare reproduced Table 9 results against paper.")
    parser.add_argument("--paper", default=DEFAULT_PAPER, help="Path to paper reference CSV")
    parser.add_argument("--repro", default=DEFAULT_REPRO, help="Path to reproduced results CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.paper):
        print(f"ERROR: Paper reference CSV not found: {args.paper}"); sys.exit(1)
    if not os.path.isfile(args.repro):
        print(f"ERROR: Reproduced results CSV not found: {args.repro}")
        print("       Run repro_table9 first, then compare."); sys.exit(1)

    print()
    print("  Table 9: Reproduced vs Paper (threshold 0.9x)")
    print("=" * 50)
    compare(args.paper, args.repro)


if __name__ == "__main__":
    main()
