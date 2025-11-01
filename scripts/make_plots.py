import json, re
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

TABLES = Path("results/tables")
FIGS = Path("results/figures"); FIGS.mkdir(parents=True, exist_ok=True)

def infer_species(name: str):
    s = name.lower()
    if "one_species"   in s: return 1
    if "three_species" in s: return 3
    if "four_species"  in s: return 4
    if "benzenes"      in s: return 2
    m = re.search(r'(\d+)_?species', s)
    return int(m.group(1)) if m else None

rows = []
for p in TABLES.glob("*.metrics.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue

    stem = p.name.replace(".metrics.json","")  # e.g., 'three_species_numpy'
    if   stem.endswith("_numpy"): mode, file_name = "numpy", stem[:-6]
    elif stem.endswith("_torch"): mode, file_name = "torch", stem[:-6]
    else:                         mode, file_name = d.get("mode","unknown"), stem

    wall = d.get("wall_s") or d.get("wall") or d.get("time_s") or d.get("wall_time") or d.get("elapsed_s") or d.get("elapsed")
    N    = d.get("N") or d.get("n_atoms") or d.get("atoms")
    sp   = d.get("species") or d.get("n_species") or infer_species(file_name)

    rows.append({"file": file_name, "mode": mode, "wall_s": wall, "N": N, "species": sp})

df = pd.DataFrame(rows)
print("rows:", len(df))
if df.empty:
    raise SystemExit("No metrics JSONs found under results/tables")

# Save combined CSV for transparency
out_csv = TABLES / "combined_from_metrics.csv"
df.to_csv(out_csv, index=False)
print("wrote", out_csv)

# Plot A: wall time by file × mode (bar)
pv = df.pivot_table(index="file", columns="mode", values="wall_s", aggfunc="mean").sort_index()
ax = pv.plot(kind="bar")
ax.set_title("Wall time by system and backend")
ax.set_ylabel("seconds (mean)")
ax.set_xlabel("system/file")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "wall_time_by_system.png", dpi=160)
plt.close()
print("wrote", FIGS / "wall_time_by_system.png")

# Plot B: wall time vs species (line), if species is known
if df["species"].notna().any():
    sdf = df.dropna(subset=["species","wall_s"]).copy()
    if not sdf.empty:
        pv2 = sdf.pivot_table(index="species", columns="mode", values="wall_s", aggfunc="mean").sort_index()
        ax = pv2.plot(marker="o")
        ax.set_title("Wall time vs #Species")
        ax.set_ylabel("seconds (mean)")
        ax.set_xlabel("# species")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS / "wall_time_vs_species.png", dpi=160)
        plt.close()
        print("wrote", FIGS / "wall_time_vs_species.png")

# Plot C: normalized by N^2 (only if N is present)
if df["N"].notna().any():
    sdf = df.dropna(subset=["N","species","wall_s"]).copy()
    if not sdf.empty:
        sdf["norm_time"] = sdf["wall_s"] / (sdf["N"]**2)
        pv3 = sdf.pivot_table(index="species", columns="mode", values="norm_time", aggfunc="mean").sort_index()
        ax = pv3.plot(marker="o")
        ax.set_title("Runtime / N^2 vs #Species")
        ax.set_ylabel("seconds / N^2 (mean)")
        ax.set_xlabel("# species")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS / "scaling_species.png", dpi=160)
        plt.close()
        print("wrote", FIGS / "scaling_species.png")

