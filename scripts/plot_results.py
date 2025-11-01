"""
Quick plots for README figures from your CSVs.
Creates results/figures/scaling_species.png and util_before_after.png if columns exist.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TABLES = Path("results/tables")
FIGS = Path("results/figures"); FIGS.mkdir(parents=True, exist_ok=True)

# 1) species scaling (expects: file, mode, wall_s, N, species)
for fname in ["timings_chtc.csv", "timings_local.csv"]:
    f = TABLES / fname
    if not f.exists():
        continue
    df = pd.read_csv(f)
    needed = {"file","mode","wall_s","N","species"}
    if needed.issubset(df.columns):
        df["norm_time"] = df["wall_s"] / (df["N"]**2)
        pivot = df.pivot_table(index="species", columns="mode", values="norm_time", aggfunc="mean")
        ax = pivot.plot(marker="o")
        ax.set_title("Runtime / N^2 vs #Species")
        ax.set_ylabel("seconds / N^2 (mean)")
        ax.set_xlabel("# species")
        ax.grid(True, axis="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS / "scaling_species.png", dpi=160)
        plt.close()
        break

# 2) util_before_after from summary file if present (expects: tag, util)
sumf = TABLES / "summary_local.csv"
if sumf.exists():
    try:
        df = pd.read_csv(sumf)
        if {"tag","util"}.issubset(df.columns):
            ax = df.set_index("tag")["util"].plot(kind="bar")
            ax.set_title("GPU Utilization: before vs after")
            ax.set_ylabel("util (%)")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGS / "util_before_after.png", dpi=160)
            plt.close()
    except Exception as e:
        print("skip util plot:", e)

print("Saved figures under results/figures/.")
