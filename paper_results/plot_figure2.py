"""
Generates Figure 2-style bar charts from repro_figure2 CSV output.

Usage:
    pip install pandas matplotlib
    python paper_results/plot_figure2.py [--csv paper_results/figure2_results.csv] [--out paper_results/figure2_repro.png]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 2 speedup charts from reproduction CSV")
    parser.add_argument("--csv", default="paper_results/figure2_results.csv", help="Input CSV from repro_figure2")
    parser.add_argument("--out", default="paper_results/figure2_repro.png", help="Output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    for col in ["TTFTSpeedup", "TPSSpeedup", "E2ELSpeedup"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    models = ["minitron-4b", "minitron-8b", "qwen3-30b", "qwen3-235b"]
    ctx_labels = ["1K", "4K", "16K", "64K"]
    budgets = sorted(df["VramBudget"].unique(), key=lambda x: int(x.replace("G", "")))
    metrics = [("TTFTSpeedup", "TTFT Speedup (x)"), ("TPSSpeedup", "TPS Speedup (x)"), ("E2ELSpeedup", "E2EL Speedup (x)")]
    colors = plt.cm.tab10(np.linspace(0, 1, len(budgets)))

    # For each (model, ctx, budget), pick the best speedup across ubatches
    best = df.groupby(["Model", "CtxK", "VramBudget"]).agg({
        "TTFTSpeedup": "max",
        "TPSSpeedup": "max",
        "E2ELSpeedup": "max"
    }).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        x_positions = []
        x_labels = []
        pos = 0

        for m_idx, model in enumerate(models):
            model_data = best[best["Model"] == model]
            if model_data.empty:
                continue

            for c_idx, ctx in enumerate(ctx_labels):
                ctx_data = model_data[model_data["CtxK"] == ctx]
                bar_start = pos

                for b_idx, budget in enumerate(budgets):
                    row = ctx_data[ctx_data["VramBudget"] == budget]
                    val = row[metric].values[0] if len(row) > 0 and not pd.isna(row[metric].values[0]) else 0

                    bar = ax.bar(pos, val, width=0.8, color=colors[b_idx],
                                 edgecolor="white", linewidth=0.3)

                    if metric == "TPSSpeedup" and val > 20:
                        ax.text(pos, min(val, 20) + 0.3, f"{val:.0f}", ha="center", va="bottom", fontsize=5.5, fontweight="bold")

                    pos += 1

                x_positions.append((bar_start + pos - 1) / 2)
                x_labels.append(ctx)
                pos += 0.5

            if m_idx < len(models) - 1:
                ax.axvline(x=pos - 0.25, color="black", linewidth=0.5, linestyle="--")
                pos += 0.5

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=8)

        if metric == "TPSSpeedup":
            ax.set_ylim(0, 20)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Model labels below x-axis
        if ax_idx == 2:
            model_centers = []
            p = 0
            for model in models:
                md = best[best["Model"] == model]
                if md.empty:
                    continue
                n_ctx = len([c for c in ctx_labels if c in md["CtxK"].values])
                n_bars = n_ctx * len(budgets) + (n_ctx - 1) * 0.5
                model_centers.append((p + p + n_bars - 1) / 2)
                p += n_bars + 1
            for mc, model in zip(model_centers, [m for m in models if m in best["Model"].values]):
                ax.text(mc, -0.18, model, ha="center", va="top", fontsize=9, fontweight="bold",
                        transform=ax.get_xaxis_transform())

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(len(budgets))]
    fig.legend(handles, budgets, loc="upper center", ncol=len(budgets), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Speedups from Pipelined Sharding (best of ubatch 1024/2048)", fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {args.out}")

if __name__ == "__main__":
    main()
