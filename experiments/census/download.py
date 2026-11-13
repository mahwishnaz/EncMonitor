#!/usr/bin/env python3
"""Download ACS 1-year person files for the five states and years 2014-2018 into data/ (Folktables)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from folktables import ACSDataSource  # noqa: E402

STATES = ["CA", "TX", "FL", "NY", "PA"]
for year in range(2014, 2019):
    ds = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=str(HERE / "data"))
    df = ds.get_data(states=STATES, download=True)
    print(year, df.shape, flush=True)
