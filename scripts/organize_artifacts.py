"""
Ingests your bundles and places files into the repo structure.
Usage:
  python scripts/organize_artifacts.py --chtc profiling_artifacts.tgz --local profiling_local.zip
"""
import argparse, shutil, tarfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
LOGS_CHTC = RESULTS / "logs" / "chtc"
LOGS_LOCAL = RESULTS / "logs" / "local"
TRACES = RESULTS / "profiler_traces"

for p in [TABLES, LOGS_CHTC, LOGS_LOCAL, TRACES]:
    p.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--chtc", type=str, required=False, help="path to profiling_artifacts.tgz")
ap.add_argument("--local", type=str, required=False, help="path to profiling_local.zip")
args = ap.parse_args()

if args.chtc:
    tgz = Path(args.chtc)
    if tgz.exists():
        with tarfile.open(tgz, "r:gz") as tf:
            tmp = RESULTS / "_tmp_chtc"
            tmp.mkdir(exist_ok=True)
            tf.extractall(tmp)
            for csv in tmp.rglob("timings*.csv"):
                shutil.move(str(csv), TABLES / csv.name)
            for log in tmp.rglob("*.out"):
                shutil.move(str(log), LOGS_CHTC / log.name)
            for js in tmp.rglob("*.json"):
                dest = TRACES / js.name if "trace" in js.name else TABLES / js.name
                shutil.move(str(js), dest)
            shutil.rmtree(tmp)

if args.local:
    zf = Path(args.local)
    if zf.exists():
        with zipfile.ZipFile(zf) as z:
            tmp = RESULTS / "_tmp_local"
            tmp.mkdir(exist_ok=True)
            z.extractall(tmp)
            for name in ["timings_local.csv", "summary_local.csv"]:
                src = tmp / name
                if src.exists():
                    shutil.move(str(src), TABLES / src.name)
            for log in tmp.rglob("*.log"):
                shutil.move(str(log), LOGS_LOCAL / log.name)
            shutil.rmtree(tmp)

print("Artifacts organized under results/.")
