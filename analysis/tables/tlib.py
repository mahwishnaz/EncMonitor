"""Helpers for the table scripts (read outputs/tables/data/per_seed_rates.csv; write CSV + Markdown with the commit
in the header to outputs/tables/). Not run directly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shared as c2  # noqa: E402

OUTD = c2.OUT / "tables"
DATA = OUTD / "data"
LABEL = {"pooled": "Pooled", "encmon_joint": "EncMonitor", "encmon_or": "OR rule", "localany": "Local-any",
         "hybrid": "Hybrid", "exact": "Exact-MMD", "encmon_norm": "EncMonitor-norm (i)"}
SETTING = {"mnist": "MNIST", "cifar10": "CIFAR-10", "camelyon17_patch": "Camelyon17, patch split",
           "camelyon17_slide": "Camelyon17, slide split (5 seeds)", "camelyon17_patient": "Camelyon17, patient split",
           "census_n250": "Census, n = 250", "census_n1000": "Census, n = 1,000"}


def rates() -> pd.DataFrame:
    return c2.read_csv(DATA / "per_seed_rates.csv")


def weak(r: pd.DataFrame, geo="all") -> pd.DataFrame:
    """Weak shifts pooled over the three datasets (6 conditions each): per seed mean of the dataset means."""
    w = r[r.set.str.startswith("weak_") & (r["aggregate"] == f"weak_{geo}")]
    return w.groupby(["method", "seed"])[["default", "matched"]].mean().reset_index().assign(
        set="weak_all_site" if geo == "all" else f"weak_{geo}", aggregate="weak", conditions=18)


def pick(r, set_, agg, seeds=None):
    g = r[(r.set == set_) & (r["aggregate"] == agg)]
    if seeds is not None:
        g = g[g.seed.isin(seeds)]
    return g


def fmt(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(s: dict, d=3) -> str:
    return f"{fmt(s['mean'], d)} [{fmt(s['t_lo'], d)}, {fmt(s['t_hi'], d)}]"


def to_md(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else (f"{v:.3f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}")
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save(df: pd.DataFrame, name: str, title: str, note: str = "") -> None:
    c2.write_csv(df, OUTD / f"{name}.csv", title)
    md = [f"# {title}", "", f"Commit {c2.commit()}. {note}".strip(), "", to_md(df)]
    (OUTD / f"{name}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {name}.csv/.md ({len(df)} rows)")
