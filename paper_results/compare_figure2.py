#!/usr/bin/env python3
"""Compare reproduced Figure 2 speedups against paper reference values.

Figure 2 reports TTFT, TPS, and E2EL speedups from pipelined sharding for
4 models at 4 context sizes across 8 VRAM budgets. The repro script outputs
per-(model, ctx, vram, ubatch) rows; we take the best speedup across ubatches
for each (model, ctx, vram) triple to match the paper's presentation.

Usage:
    python paper_results/compare_figure2.py [--repro CSV] [--paper CSV]
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAPER = os.path.join(SCRIPT_DIR, "paper_results_figure2.csv")
DEFAULT_REPRO = os.path.join(SCRIPT_DIR, "figure2_results.csv")

THRESHOLD = 0.9

MODEL_SORT = {"minitron-4b": 0, "minitron-8b": 1, "qwen3-30b": 2, "qwen3-235b": 3}
CTX_SORT = {"1K": 0, "4K": 1, "16K": 2, "64K": 3}
VRAM_SORT = {"2G": 0, "4G": 1, "6G": 2, "8G": 3, "12G": 4, "16G": 5, "24G": 6, "32G": 7}

METRICS = [("TTFTSpeedup", "TTFT"), ("TPSSpeedup", "TPS"), ("E2ELSpeedup", "E2EL")]


def sort_key(key):
    model, ctx, vram = key
    return (MODEL_SORT.get(model, 99), CTX_SORT.get(ctx, 99), VRAM_SORT.get(vram, 99))


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_paper(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["Model"].strip(), r["CtxK"].strip(), r["VramBudget"].strip())
            rows[key] = {col: safe_float(r[col].strip()) for col, _ in METRICS}
    return rows


def load_repro(path):
    raw = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            model = r["Model"].strip().strip('"')
            ctx = r["CtxK"].strip().strip('"')
            vram = r["VramBudget"].strip().strip('"')
            key = (model, ctx, vram)
            for col, _ in METRICS:
                val = safe_float(r.get(col, "N/A").strip().strip('"'))
                if val is None:
                    continue
                if key not in raw:
                    raw[key] = {}
                prev = raw[key].get(col)
                if prev is None or val > prev:
                    raw[key][col] = val
    return raw


def ratio_str(ratio):
    return f"{ratio:.2f}x of paper"


def compare(paper_path, repro_path):
    paper = load_paper(paper_path)
    repro = load_repro(repro_path)

    total = 0
    metrics_pass = 0
    metrics_below = 0
    skipped = 0
    current_model = None

    for key in sorted(paper.keys(), key=sort_key):
        model, ctx, vram = key
        if model != current_model:
            if current_model is not None:
                print()
            current_model = model
            print(f"  {model}")

        p = paper[key]
        if key not in repro:
            skipped += 1
            continue

        r = repro[key]
        total += 1
        cells = []

        for col, label in METRICS:
            p_val = p.get(col)
            r_val = r.get(col)
            if p_val and r_val and p_val > 0:
                ratio = r_val / p_val
                if ratio >= THRESHOLD:
                    cells.append(f"{label}:PASS")
                    metrics_pass += 1
                else:
                    cells.append(f"{label}:{ratio:.2f}x of paper")
                    metrics_below += 1
            else:
                cells.append(f"{label}:-")

        print(f"    {ctx:>3s} {vram:>4s}  {'  '.join(cells)}")

    total_metrics = metrics_pass + metrics_below
    print()
    print("-" * 60)
    print(f"  Compared {total} configs, {total_metrics} metric values  (skipped {skipped})")
    print(f"  PASS: {metrics_pass}/{total_metrics}   below 0.9x: {metrics_below}")
    if metrics_below == 0 and total_metrics > 0:
        print("  Result: ALL PASS")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare reproduced Figure 2 results against paper.")
    parser.add_argument("--paper", default=DEFAULT_PAPER, help="Path to paper reference CSV")
    parser.add_argument("--repro", default=DEFAULT_REPRO, help="Path to reproduced results CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.paper):
        print(f"ERROR: Paper reference CSV not found: {args.paper}"); sys.exit(1)
    if not os.path.isfile(args.repro):
        print(f"ERROR: Reproduced results CSV not found: {args.repro}")
        print("       Run repro_figure2 first, then compare."); sys.exit(1)

    print()
    print("  Figure 2: Reproduced vs Paper speedups (threshold 0.9x)")
    print("=" * 60)
    compare(args.paper, args.repro)


if __name__ == "__main__":
    main()
