#!/usr/bin/env python3
"""Regenerate comparison figures from the committed result tables.

The script intentionally writes to ``results/generated`` and never overwrites
figures preserved in the DOI-backed v0.0.0 snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES = ROOT / "results" / "tables"
DEFAULT_OUTPUT = ROOT / "results" / "generated"


def metric_frame(tables: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(tables.glob("*.metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        stem = path.name.removesuffix(".metrics.json")
        if stem.endswith("_numpy"):
            workload, backend = stem.removesuffix("_numpy"), "NumPy"
        elif stem.endswith("_torch"):
            workload, backend = stem.removesuffix("_torch"), "PyTorch"
        else:
            continue
        rows.append(
            {
                "workload": workload.replace("_", " ").title(),
                "backend": backend,
                "elapsed_s": float(data["elapsed_s"]),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 10:
        raise ValueError(f"expected 10 metric records, found {len(frame)}")
    return frame


def plot_committed_metrics(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.pivot(index="workload", columns="backend", values="elapsed_s")
    order = [
        "One Species",
        "Benzenes",
        "Three Species",
        "Four Species",
        "Ellipsoids",
    ]
    pivot = pivot.reindex(order)

    positions = np.arange(len(pivot))
    height = 0.36
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.barh(positions - height / 2, pivot["NumPy"], height=height, label="NumPy")
    axis.barh(
        positions + height / 2,
        pivot["PyTorch"],
        height=height,
        label="PyTorch",
    )
    axis.set_yticks(positions, pivot.index)
    axis.invert_yaxis()
    axis.set_xlabel("Elapsed time (seconds)")
    axis.set_title("Committed per-system measurements")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()

    for index, workload in enumerate(pivot.index):
        numpy_time = float(pivot.loc[workload, "NumPy"])
        torch_time = float(pivot.loc[workload, "PyTorch"])
        ratio = numpy_time / torch_time
        axis.text(
            max(numpy_time, torch_time) * 1.02,
            index,
            f"{ratio:.2f}×",
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(output / "committed_metrics.svg", bbox_inches="tight")
    plt.close(figure)


def plot_local_microbenchmarks(tables: Path, output: Path) -> None:
    frame = pd.read_csv(tables / "summary_local.csv")
    selected = frame[frame["mode"].isin(["numpy_only", "torch_full"])].copy()
    selected["backend"] = selected["mode"].map(
        {"numpy_only": "NumPy only", "torch_full": "PyTorch full"}
    )
    selected["workload"] = (
        selected["file"]
        .str.removesuffix(".xyz")
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    pivot = selected.pivot(index="workload", columns="backend", values="wall_mean")
    errors = selected.pivot(index="workload", columns="backend", values="wall_std")
    order = [
        "One Species",
        "Benzenes",
        "Three Species",
        "Four Species",
        "Ellipsoids",
    ]
    pivot = pivot.reindex(order)
    errors = errors.reindex(order)

    positions = np.arange(len(pivot))
    height = 0.36
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.barh(
        positions - height / 2,
        pivot["NumPy only"],
        xerr=errors["NumPy only"],
        height=height,
        label="NumPy only",
    )
    axis.barh(
        positions + height / 2,
        pivot["PyTorch full"],
        xerr=errors["PyTorch full"],
        height=height,
        label="PyTorch full",
    )
    axis.set_yticks(positions, pivot.index)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Mean elapsed time (seconds, log scale)")
    axis.set_title("Local operation microbenchmarks")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()

    for index, workload in enumerate(pivot.index):
        numpy_time = float(pivot.loc[workload, "NumPy only"])
        torch_time = float(pivot.loc[workload, "PyTorch full"])
        ratio = torch_time / numpy_time
        axis.text(
            torch_time * 1.08,
            index,
            f"{ratio:.2f}× slower",
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(output / "local_microbenchmarks.svg", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = metric_frame(args.tables)
    plot_committed_metrics(frame, args.output)
    plot_local_microbenchmarks(args.tables, args.output)
    print(f"wrote {args.output / 'committed_metrics.svg'}")
    print(f"wrote {args.output / 'local_microbenchmarks.svg'}")


if __name__ == "__main__":
    main()
