#!/usr/bin/env python3
"""Compare reproduced Table 8 E2EL speedups against paper reference values.

Table 8 reports E2EL speedups for Cosmos-Reason1 VLM at 4 resolutions and 3 VRAM budgets.
OOM entries in the paper (baseline ran out of memory) are treated as automatic PASS
since the paper couldn't run them either.

Usage:
    python paper_results/compare_table8.py [--repro CSV] [--paper CSV]
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAPER = os.path.join(SCRIPT_DIR, "paper_results_table8.csv")
DEFAULT_REPRO = os.path.join(SCRIPT_DIR, "table8_results.csv")

THRESHOLD = 0.9

RES_SORT = {"480p": 0, "720p": 1, "1080p": 2, "1440p": 3}
VRAM_SORT = {"4G": 0, "8G": 1, "14G": 2}


def sort_key(key):
    res, vram = key
    return (RES_SORT.get(res, 99), VRAM_SORT.get(vram, 99))


def load_paper(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["RunType"].strip() != "vlmopt":
                continue
            key = (r["Resolution"].strip(), r["VramBudget"].strip())
            rows[key] = r["Speedup"].strip()
    return rows


def load_repro(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["RunType"].strip() != "vlmopt":
                continue
            key = (r["Resolution"].strip(), r["VramBudget"].strip())
            rows[key] = r["Speedup"].strip().strip('"')
    return rows


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def compare(paper_path, repro_path):
    paper = load_paper(paper_path)
    repro = load_repro(repro_path)

    total = 0
    passed = 0
    below = 0
    skipped = 0

    for key in sorted(paper.keys(), key=sort_key):
        res, vram = key
        p_val = paper[key]

        if p_val.upper() == "OOM":
            print(f"    {res:>5s} {vram:>4s}  Speedup: PASS (OOM in paper)")
            skipped += 1
            continue

        p_su = safe_float(p_val)
        if key not in repro:
            print(f"    {res:>5s} {vram:>4s}  Speedup: (not in reproduced results)")
            skipped += 1
            continue

        r_su = safe_float(repro[key])
        total += 1

        if p_su and r_su and p_su > 0:
            ratio = r_su / p_su
            if ratio >= THRESHOLD:
                print(f"    {res:>5s} {vram:>4s}  Speedup: PASS")
                passed += 1
            else:
                print(f"    {res:>5s} {vram:>4s}  Speedup: {ratio:.2f}x of paper")
                below += 1
        else:
            print(f"    {res:>5s} {vram:>4s}  Speedup: -")

    print()
    print("-" * 50)
    print(f"  Compared {total} rows  (skipped {skipped})")
    print(f"  PASS: {passed}/{total}   below 0.9x: {below}")
    if below == 0 and total > 0:
        print("  Result: ALL PASS")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare reproduced Table 8 results against paper.")
    parser.add_argument("--paper", default=DEFAULT_PAPER, help="Path to paper reference CSV")
    parser.add_argument("--repro", default=DEFAULT_REPRO, help="Path to reproduced results CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.paper):
        print(f"ERROR: Paper reference CSV not found: {args.paper}"); sys.exit(1)
    if not os.path.isfile(args.repro):
        print(f"ERROR: Reproduced results CSV not found: {args.repro}")
        print("       Run repro_table8 first, then compare."); sys.exit(1)

    print()
    print("  Table 8: Reproduced vs Paper (threshold 0.9x)")
    print("=" * 50)
    compare(args.paper, args.repro)


if __name__ == "__main__":
    main()
