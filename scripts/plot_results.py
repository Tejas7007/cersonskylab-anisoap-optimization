import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TABLES = Path("results/tables")
FIGS = Path("results/figures"); FIGS.mkdir(parents=True, exist_ok=True)

def try_scaling_plot(fname):
    f = TABLES / fname
    if not f.exists():
        return False
    df = pd.read_csv(f)
    need = {"file","mode","wall_s","N","species"}
    if not need.issubset(df.columns):
        print(f"skip {fname}: missing columns {need - set(df.columns)}")
        return False
    df["norm_time"] = df["wall_s"] / (df["N"]**2)
    pivot = df.pivot_table(index="species", columns="mode", values="norm_time", aggfunc="mean").sort_index()
    ax = pivot.plot(marker="o")
    ax.set_title("Runtime / N^2 vs #Species")
    ax.set_ylabel("seconds / N^2 (mean)")
    ax.set_xlabel("# species")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "scaling_species.png", dpi=160)
    plt.close()
    print("wrote results/figures/scaling_species.png")
    return True

for name in ["timings_chtc.csv","timings_local.csv","timings.csv"]:
    if try_scaling_plot(name):
        break
