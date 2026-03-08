#!/usr/bin/env python3
"""Run all comparison scripts and print a combined summary.

Compares reproduced results against paper reference values for all tables
and figures. Each comparison uses a 0.9x threshold — PASS means the
reproduced value is within 90% of the paper value.

Usage:
    python compare_all_results.py
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(SCRIPT_DIR, "paper_results")

COMPARISONS = [
    {
        "name": "Table 4",
        "script": os.path.join(PAPER_DIR, "compare_table4.py"),
        "repro": os.path.join(PAPER_DIR, "table4_results.csv"),
    },
    {
        "name": "Table 8",
        "script": os.path.join(PAPER_DIR, "compare_table8.py"),
        "repro": os.path.join(PAPER_DIR, "table8_results.csv"),
    },
    {
        "name": "Table 9",
        "script": os.path.join(PAPER_DIR, "compare_table9.py"),
        "repro": os.path.join(PAPER_DIR, "table9_results.csv"),
    },
    {
        "name": "Figure 7",
        "script": os.path.join(PAPER_DIR, "compare_figure7.py"),
        "repro": os.path.join(PAPER_DIR, "figure7_results.csv"),
    },
]


def main():
    print()
    print("=" * 60)
    print("  Compare All Reproduced Results vs Paper")
    print("=" * 60)

    ran = 0
    skipped_names = []

    for comp in COMPARISONS:
        if not os.path.isfile(comp["repro"]):
            skipped_names.append(comp["name"])
            continue

        print()
        print(f"{'#' * 60}")
        print(f"  {comp['name']}")
        print(f"{'#' * 60}")

        ran += 1
        subprocess.run([sys.executable, comp["script"]], cwd=SCRIPT_DIR)

    print()
    print("=" * 60)
    print(f"  Done.  Ran {ran}/{len(COMPARISONS)} comparisons.")
    if skipped_names:
        print(f"  Skipped (no results CSV): {', '.join(skipped_names)}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
