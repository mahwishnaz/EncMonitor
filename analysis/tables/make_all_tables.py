#!/usr/bin/env python3
"""Regenerate every table from the records: compute_rates (scoring of all sets and seeds), then one script per
table. Usage (from anywhere): python analysis/tables/make_all_tables.py [--skip-rates]. Writes outputs/tables/."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["compute_rates.py", "table1.py", "paired.py", "matched_fpr.py", "table3.py", "table4.py", "table5.py",
           "supplementary.py"]
for s in SCRIPTS:
    if s == "compute_rates.py" and "--skip-rates" in sys.argv:
        continue
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", str(HERE / s)], cwd=HERE, check=True)
