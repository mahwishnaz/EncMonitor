#!/usr/bin/env python3
"""Site-identity leakage of L1 releases (binary / coarse alert) vs L2, same attack as leakage_dp/run.py (a).

Usage (bundle root): python experiments/l1_leakage/run.py > experiments/l1_leakage/run_log.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_common import RECORD_DIRS, load, score_seed  # noqa: E402
from rx_windows import SeedContext, stats_from_gram  # noqa: E402

DATASETS = ["mnist", "cifar10", "camelyon17"]


def main(rng_seed: int = 20261023) -> int:
    rows = []
    for ds in DATASETS:
        rec = load(ds)
        for seed, g in rec.groupby("partition_seed"):
            _, info = score_seed(g, 0.95)
            ctx = SeedContext(ds, int(seed))
            for fam, sev in (("vector", 1.0), ("label_prior", 0.35)):
                X2, B, Cc, y = [], [], [], []
                for s in range(ctx.S):
                    dirs = [0.0] * ctx.S
                    dirs[s] = 1.0
                    c = {"kind": fam, "family": fam, "severity": sev, "directions": dirs}
                    st = stats_from_gram(ctx.run(ctx.condition_feats(c), 50, int(seed) + 6000 + s)["gram"], ctx.alpha)
                    a, b = st["d_mix"] / info["t_mix"], st["d_site"] / info["t_site"]
                    alert = np.maximum(a, b) > info["c_joint"]
                    X2.append(np.log(np.c_[st["d_mix"], st["d_site"], np.maximum(st["heterogeneity"], 1e-12)]))
                    B.append(alert.astype(int))
                    Cc.append(np.where(~alert, 0, np.where(a > 1.0, 2, 1)))
                    y.append(np.full(50, s))
                X2, B, Cc, y = map(np.concatenate, (X2, B, Cc, y))
                cv = StratifiedKFold(5, shuffle=True, random_state=rng_seed)
                lr = lambda: LogisticRegression(max_iter=5000)
                r = {"dataset": ds, "partition_seed": int(seed), "family": fam, "severity": sev, "chance": 1 / ctx.S,
                     "alert_share": float(B.mean())}
                r["L1_binary"] = float(cross_val_score(make_pipeline(OneHotEncoder(handle_unknown="ignore"), lr()), B[:, None], y, cv=cv).mean())
                r["L1_coarse"] = float(cross_val_score(make_pipeline(OneHotEncoder(handle_unknown="ignore"), lr()), Cc[:, None], y, cv=cv).mean())
                r["L2_scalars"] = float(cross_val_score(make_pipeline(StandardScaler(), lr()), X2, y, cv=cv).mean())
                rows.append(r)
                print(ds, seed, fam, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "l1_leakage_per_seed.csv", index=False)
    summ = df.groupby(["dataset", "family"])[["alert_share", "L1_binary", "L1_coarse", "L2_scalars"]].agg(["mean", "min", "max"])
    summ.to_csv(HERE / "l1_leakage.csv")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    e6 = pd.read_csv(HERE.parent / "leakage_dp" / "a_site_identity_per_seed.csv")
    m = df.merge(e6[["dataset", "partition_seed", "family", "L2_scalars"]], on=["dataset", "partition_seed", "family"], suffixes=("", "_E6"))
    print("max |L2 - L2_E6| =", float((m.L2_scalars - m.L2_scalars_E6).abs().max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
