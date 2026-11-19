#!/usr/bin/env python3
"""Build the data archives (encmonitor_data_*.tar.gz, CHECKSUMS.txt, FILELIST.tsv) from the outputs of this
repository. Read only on all inputs; symbolic links are followed. CPU only.

Inputs (defaults relative to the repository root):
  <outputs>/main/mnist, <outputs>/main/cifar10                    scripts/14, 15 (per seed: manifest.npz,
  <outputs>/camelyon/main/camelyon17                              records.jsonl, conditions.json, RUN_INFO.json)
  <outputs>/camelyon/patient_split/camelyon17                     experiments/camelyon_patient_split, scripts/33
  experiments/camelyon_slide_split/cache/camelyon17               experiments/camelyon_slide_split
  <outputs>/census/records_n250, <outputs>/census/records_n1000   experiments/census/windows.py
  <outputs>/tables/*.csv, <outputs>/tables/data/*.csv             analysis/tables/
  <outputs>/analysis/streams/*.csv                                analysis/streams.py
  --census-pools: output folder of tools/export_census_pools.py
  --weak: output folder of analysis/tables/export_weak_all_windows.py
Only manifest.npz, records.jsonl, conditions.json and RUN_INFO.json are taken from the seed folders (no
representation, preprocessing, random-feature or reference-sketch file).
Output: <dest>/encmonitor_data_{mnist_cifar10,camelyon17,census,tables_streams}.tar.gz, CHECKSUMS.txt, FILELIST.tsv.
The archives are deterministic (sorted names, fixed modification time, numeric owner 0, gzip -n).
Keep <staging> and <dest> outside the repository folder.
Usage (from the repository root):
  python -B tools/build_release.py --census-pools <folder> --weak <folder> --staging <folder> --dest <folder>
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]           # repository root
ap = argparse.ArgumentParser(description="Build the data archives")
ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
ap.add_argument("--slide-cache", type=Path, default=ROOT / "experiments/camelyon_slide_split/cache/camelyon17")
ap.add_argument("--census-pools", type=Path, required=True)
ap.add_argument("--weak", type=Path, required=True)
ap.add_argument("--staging", type=Path, required=True)
ap.add_argument("--dest", type=Path, required=True)
a = ap.parse_args()
RUN, CEN, WEAK = a.outputs.resolve(), a.census_pools.resolve(), a.weak.resolve()
ST, DEST = a.staging.resolve(), a.dest.resolve()
TOP = ST / "encmonitor_data"
if ST.exists():
    shutil.rmtree(ST)
TOP.mkdir(parents=True)
DEST.mkdir(parents=True, exist_ok=True)
KEEP = ("manifest.npz", "records.jsonl", "conditions.json", "RUN_INFO.json")


def seeds(src, dst, keep=KEEP):
    n = 0
    for sd in sorted(Path(src).glob("seed_*")):
        for f in keep:
            p = sd / f
            if p.exists():
                q = TOP / dst / sd.name / f
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, q)
                n += 1
    return n


FAM = {
    "mnist_cifar10": [(RUN / "main/mnist", "mnist"), (RUN / "main/cifar10", "cifar10")],
    "camelyon17": [(RUN / "camelyon/main/camelyon17", "camelyon17/patch_split"),
                   (RUN / "camelyon/patient_split/camelyon17", "camelyon17/patient_split"),
                   (a.slide_cache.resolve(), "camelyon17/slide_split_5seeds")],
    "census": [(RUN / "census/records_n250", "census/n250"), (RUN / "census/records_n1000", "census/n1000")],
}
for fam, items in FAM.items():
    for src, dst in items:
        print(fam, dst, seeds(src, dst), "files")
# weak-shift windows (analysis/tables/export_weak_all_windows.py)
nw = 0
for dst in ("mnist", "cifar10", "camelyon17/patch_split"):
    for p in sorted((WEAK / dst).glob("seed_*/weak_shift_records.jsonl")):
        q = TOP / dst / p.parent.name / p.name
        assert q.parent.is_dir(), q                       # only into existing seed folders
        shutil.copyfile(p, q)
        nw += 1
print("weak-shift files", nw)
# census pools (tools/export_census_pools.py)
shutil.copytree(CEN / "census/pools", TOP / "census/pools")
shutil.copyfile(CEN / "census_pool_verification.json", TOP / "census/pools/verification.json")
# tables and streams
T = RUN / "tables"
(TOP / "tables").mkdir()
for f in ["table1.csv", "table1_per_seed.csv", "table3.csv", "table4.csv", "table5.csv", "matched_fpr.csv",
          "paired.csv", "paired_per_seed.csv", "interval_comparison.csv"]:
    shutil.copyfile(T / f, TOP / "tables" / f)
(TOP / "tables/per_seed").mkdir()
for f in ["per_seed_rates.csv", "record_inventory.csv"]:
    shutil.copyfile(T / "data" / f, TOP / "tables/per_seed" / f)
(TOP / "streams").mkdir()
for f in ["per_stream.csv", "summary.csv", "autocorr.csv"]:
    shutil.copyfile(RUN / "analysis/streams" / f, TOP / "streams" / f)

ARCH = {"encmonitor_data_mnist_cifar10.tar.gz": ["encmonitor_data/mnist", "encmonitor_data/cifar10"],
        "encmonitor_data_camelyon17.tar.gz": ["encmonitor_data/camelyon17"],
        "encmonitor_data_census.tar.gz": ["encmonitor_data/census"],
        "encmonitor_data_tables_streams.tar.gz": ["encmonitor_data/tables", "encmonitor_data/streams"]}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for name in ARCH:
    if (DEST / name).exists():
        (DEST / name).unlink()
lines, flist = [], ["archive\tpath\tbytes"]
for name, dirs in ARCH.items():
    subprocess.run(f"tar --sort=name --mtime='2026-11-17 00:00:00' --owner=0 --group=0 --numeric-owner "
                   f"-cf - {' '.join(dirs)} | gzip -n -9 > '{DEST / name}'", shell=True, check=True, cwd=ST)
    lines.append(f"{sha(DEST / name)}  {name}")
    for d in dirs:
        for p in sorted((ST / d).rglob("*")):
            if p.is_file():
                flist.append(f"{name}\t{p.relative_to(ST).as_posix()}\t{p.stat().st_size}")
inner = [f"{sha(p)}  {p.relative_to(ST).as_posix()}" for p in sorted(TOP.rglob("*")) if p.is_file()]
(DEST / "CHECKSUMS.txt").write_text(
    "# SHA-256 (sha256sum format). Part 1: archives. Part 2: every file inside the archives (paths after unpacking).\n"
    "# Check: sha256sum -c CHECKSUMS.txt (archives in the current folder, unpacked tree in the same folder).\n"
    + "\n".join(lines) + "\n" + "\n".join(inner) + "\n")
(DEST / "FILELIST.tsv").write_text("\n".join(flist) + "\n")
for name in ARCH:
    print(name, round((DEST / name).stat().st_size / 1e6, 1), "MB")
print("files inside:", len(inner))
