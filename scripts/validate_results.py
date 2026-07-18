#!/usr/bin/env python3
"""Validate the committed AniSOAP profiling artifacts.

The validator checks internal consistency only. It does not rerun AniSOAP or
reconstruct benchmark conditions that were not preserved in the DOI snapshot.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
REQUIRED_METRIC_FIELDS = {
    "xyz",
    "mode",
    "returncode",
    "elapsed_s",
    "max_rss_kb",
    "inblock",
    "oublock",
}
WORKLOADS = {
    "one_species",
    "benzenes",
    "three_species",
    "four_species",
    "ellipsoids",
}
MODES = {"numpy", "torch"}


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=tolerance)


def load_metric_records() -> dict[tuple[str, str], dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(TABLES.glob("*.metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED_METRIC_FIELDS.difference(data)
        if missing:
            fail(f"{path.name} is missing fields: {sorted(missing)}")
        if data["returncode"] != 0:
            fail(f"{path.name} records a non-zero return code")
        elapsed = float(data["elapsed_s"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            fail(f"{path.name} has an invalid elapsed time")

        stem = path.name.removesuffix(".metrics.json")
        if stem.endswith("_numpy"):
            workload, mode = stem.removesuffix("_numpy"), "numpy"
        elif stem.endswith("_torch"):
            workload, mode = stem.removesuffix("_torch"), "torch"
        else:
            fail(f"cannot infer workload and mode from {path.name}")

        key = (workload, mode)
        if key in records:
            fail(f"duplicate metric record for {key}")
        records[key] = data

    expected = {(workload, mode) for workload in WORKLOADS for mode in MODES}
    if set(records) != expected:
        missing = sorted(expected.difference(records))
        extra = sorted(set(records).difference(expected))
        fail(f"unexpected metric set; missing={missing}, extra={extra}")
    return records


def validate_combined_csv(records: dict[tuple[str, str], dict[str, object]]) -> None:
    path = TABLES / "combined_from_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        fail(f"{path.name} should contain 10 rows, found {len(rows)}")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["file"], row["mode"])
        if key not in records:
            fail(f"{path.name} contains unknown record {key}")
        if key in seen:
            fail(f"{path.name} contains duplicate record {key}")
        seen.add(key)
        if not close(float(row["wall_s"]), float(records[key]["elapsed_s"])):
            fail(f"{path.name} disagrees with metric JSON for {key}")


def validate_legacy_timings(records: dict[tuple[str, str], dict[str, object]]) -> None:
    path = TABLES / "timings.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 10 or any(len(row) != 4 for row in rows):
        fail(f"{path.name} must remain a 10-row, four-column legacy export")

    seen: set[tuple[str, str]] = set()
    for run_id, xyz, mode, elapsed_text in rows:
        workload = run_id.removesuffix(f"_{mode}")
        key = (workload, mode)
        if key not in records:
            fail(f"{path.name} contains unknown record {key}")
        if str(records[key]["xyz"]) != xyz:
            fail(f"{path.name} has an XYZ mismatch for {key}")
        if not close(
            float(elapsed_text),
            float(records[key]["elapsed_s"]),
            tolerance=5e-4,
        ):
            fail(f"{path.name} disagrees with metric JSON for {key}")
        seen.add(key)
    if len(seen) != 10:
        fail(f"{path.name} does not cover all metric records")


def load_local_rows() -> list[dict[str, str]]:
    path = TABLES / "timings_local.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "file",
        "mode",
        "rep",
        "N",
        "species_unique",
        "wall_s",
        "compute_s",
        "conv_in_s",
        "conv_out_s",
        "time_per_N2",
    }
    if not rows or set(rows[0]) != required:
        fail(f"{path.name} has an unexpected schema")
    for row in rows:
        wall = float(row["wall_s"])
        compute = float(row["compute_s"])
        if not all(math.isfinite(value) and value >= 0 for value in (wall, compute)):
            fail(f"{path.name} contains a non-finite or negative timing")
        if compute > wall + 1e-9:
            fail(f"{path.name} has compute time greater than wall time")
    return rows


def validate_local_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["file"], row["mode"])].append(row)

    path = TABLES / "summary_local.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    if len(summaries) != 15:
        fail(f"{path.name} should contain 15 grouped rows")

    summary_lookup = {(row["file"], row["mode"]): row for row in summaries}
    if set(summary_lookup) != set(groups):
        fail(f"{path.name} groups do not match {TABLES / 'timings_local.csv'}")

    for key, members in groups.items():
        walls = [float(row["wall_s"]) for row in members]
        mean = sum(walls) / len(walls)
        stored = float(summary_lookup[key]["wall_mean"])
        if not close(mean, stored):
            fail(f"{path.name} has an incorrect wall mean for {key}")
        if int(summary_lookup[key]["runs"]) != len(members):
            fail(f"{path.name} has an incorrect run count for {key}")

    ratios: dict[str, float] = {}
    files = sorted({file_name for file_name, _ in groups})
    for file_name in files:
        numpy_mean = float(summary_lookup[(file_name, "numpy_only")]["wall_mean"])
        torch_mean = float(summary_lookup[(file_name, "torch_full")]["wall_mean"])
        ratios[file_name] = torch_mean / numpy_mean
    return ratios


def validate_environment_report() -> None:
    path = TABLES / "env_report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"python", "pytorch", "cuda_available", "cuda_version", "cudnn", "devices"}
    if set(data) != required:
        fail(f"{path.name} has an unexpected schema")


def main() -> None:
    records = load_metric_records()
    validate_combined_csv(records)
    validate_legacy_timings(records)
    local_rows = load_local_rows()
    local_ratios = validate_local_summary(local_rows)
    validate_environment_report()

    print("Committed per-system NumPy/PyTorch ratios:")
    for workload in sorted(WORKLOADS):
        numpy_time = float(records[(workload, "numpy")]["elapsed_s"])
        torch_time = float(records[(workload, "torch")]["elapsed_s"])
        print(f"  {workload:14s} {numpy_time / torch_time:6.2f}x")

    print("Local torch_full/NumPy time ratios:")
    for file_name, ratio in sorted(local_ratios.items()):
        print(f"  {file_name:18s} {ratio:6.2f}x slower")

    print("All committed result artifacts are internally consistent.")


if __name__ == "__main__":
    main()
