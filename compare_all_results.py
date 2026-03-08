#!/usr/bin/env python3
"""Run all comparison scripts and print a combined summary.

By default, only speedup comparisons are run (Figure 2, Table 8, Figure 7).
Use --compare-abs-metrics-too to also compare absolute TPS/TTFT values
(Table 4, Table 9), which are hardware-dependent and expected to vary.

Usage:
    python compare_all_results.py
    python compare_all_results.py --compare-abs-metrics-too
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(SCRIPT_DIR, "paper_results")

SPEEDUP_COMPARISONS = [
    {
        "name": "Figure 2  (TTFT/TPS/E2EL speedups)",
        "script": os.path.join(PAPER_DIR, "compare_figure2.py"),
        "repro": os.path.join(PAPER_DIR, "figure2_results.csv"),
    },
    {
        "name": "Table 8   (E2EL speedups)",
        "script": os.path.join(PAPER_DIR, "compare_table8.py"),
        "repro": os.path.join(PAPER_DIR, "table8_results.csv"),
    },
    {
        "name": "Figure 7  (TPS speedups)",
        "script": os.path.join(PAPER_DIR, "compare_figure7.py"),
        "repro": os.path.join(PAPER_DIR, "figure7_results.csv"),
    },
]

ABS_METRIC_COMPARISONS = [
    {
        "name": "Table 4   (absolute TPS/TTFT)",
        "script": os.path.join(PAPER_DIR, "compare_table4.py"),
        "repro": os.path.join(PAPER_DIR, "table4_results.csv"),
    },
    {
        "name": "Table 9   (absolute TPS)",
        "script": os.path.join(PAPER_DIR, "compare_table9.py"),
        "repro": os.path.join(PAPER_DIR, "table9_results.csv"),
    },
]


def run_comparisons(comparisons, label):
    ran = 0
    skipped = []

    for comp in comparisons:
        if not os.path.isfile(comp["repro"]):
            skipped.append(comp["name"])
            continue

        print()
        print(f"{'#' * 60}")
        print(f"  {comp['name']}")
        print(f"{'#' * 60}")

        ran += 1
        subprocess.run([sys.executable, comp["script"]], cwd=SCRIPT_DIR)

    return ran, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Compare reproduced results against paper reference values."
    )
    parser.add_argument(
        "--compare-abs-metrics-too",
        action="store_true",
        help="Also compare absolute TPS/TTFT values (Table 4, Table 9). "
             "These are hardware-dependent and expected to vary across machines.",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Compare Reproduced Results vs Paper")
    print("=" * 60)

    total_ran = 0
    all_skipped = []

    print()
    print("  --- Speedup comparisons (hardware-independent) ---")
    ran, skipped = run_comparisons(SPEEDUP_COMPARISONS, "speedup")
    total_ran += ran
    all_skipped += skipped

    if args.compare_abs_metrics_too:
        print()
        print("  --- Absolute metric comparisons (hardware-dependent) ---")
        ran, skipped = run_comparisons(ABS_METRIC_COMPARISONS, "absolute")
        total_ran += ran
        all_skipped += skipped
    else:
        print()
        print("  (Skipping absolute metric comparisons for Table 4, Table 9.")
        print("   Use --compare-abs-metrics-too to include them.)")

    total_possible = len(SPEEDUP_COMPARISONS) + (
        len(ABS_METRIC_COMPARISONS) if args.compare_abs_metrics_too else 0
    )

    print()
    print("=" * 60)
    print(f"  Done.  Ran {total_ran}/{total_possible} comparisons.")
    if all_skipped:
        print(f"  Skipped (no results CSV): {', '.join(all_skipped)}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
