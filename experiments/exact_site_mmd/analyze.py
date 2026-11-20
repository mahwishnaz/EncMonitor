#!/usr/bin/env python3
"""Exact-kernel analysis (CPU): agreement of RFF site energies with exact MMD^2 and oracle detection.

Usage (from the repository root): python experiments/exact_site_mmd/analyze.py > experiments/exact_site_mmd/analysis_log.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_common import aggregate_masks, load, score_seed  # noqa: E402
from rx_windows import conditions, stats_from_gram  # noqa: E402

DATASETS = ["mnist", "cifar10", "camelyon17"]
MS = [32, 64, 128, 256]
RULES = ["pooled", "site", "encmon_joint", "encmon_or", "localany"]


def frame(names, gram, seed, cmeta):
    """Records-like DataFrame (fields used by rx_common.score_seed) from Gram matrices."""
    S = gram.shape[1]
    st = stats_from_gram(gram, np.ones(S) / S)
    df = pd.DataFrame({"condition": names, "d_mix": st["d_mix"], "d_site": st["d_site"],
                       "heterogeneity": st["heterogeneity"], "exact_mmd": 0.0, "partition_seed": seed})
    df["site_energy"] = list(st["site_energy"])
    df["contributions"] = list(st["contributions"])
    for k in ("family", "geometry", "severity"):
        df[k] = [cmeta.get(n, {}).get(k) for n in names]
    return df


def main() -> int:
    agree, det, nulls, check = [], [], [], {}
    for ds in DATASETS:
        cmeta = {c["condition"]: c for c in conditions(ds)}
        saved = load(ds)
        for f in sorted((HERE / "windows").glob(f"{ds}_seed*.npz")):
            z = np.load(f)
            seed = int(f.stem.split("seed")[1])
            names = z["condition"].astype(str)
            ev = z["exact_v"]
            S = ev.shape[1]
            eu = ev.copy()
            eu[:, np.arange(S), np.arange(S)] = z["exact_u_diag"]
            variants = {"exact_V": ev, "exact_U": eu, **{f"rff{m}": z[f"rff{m}"] for m in MS}}
            # reproduction check: m = 64 against the saved records (same window order per condition)
            sv = saved[saved.partition_seed == seed]
            st64 = stats_from_gram(z["rff64"], np.ones(S) / S)
            diffs = []
            for cn in np.unique(names):
                a = sv[sv.condition == cn].d_mix.to_numpy()
                b = st64["d_mix"][names == cn]
                if a.size == b.size:
                    diffs.append(float(np.abs(a - b).max()))
            check[f"{ds}/seed_{seed}"] = {"max_abs_diff_d_mix": max(diffs), "base_check": float(z["base_check"])}

            Ev = np.einsum("wss->ws", ev)
            Eu = z["exact_u_diag"]
            held = names == "clean_heldout"
            shifted = ~np.isin(names, ["clean_calibration", "clean_heldout"])
            for m in MS:
                Er = np.einsum("wss->ws", z[f"rff{m}"])
                stm = stats_from_gram(z[f"rff{m}"], np.ones(S) / S)
                stv = stats_from_gram(ev, np.ones(S) / S)
                for subset, mask in (("all", np.ones(names.size, bool)), ("clean_heldout", held), ("shifted", shifted)):
                    x, yv, yu = Er[mask].ravel(), Ev[mask].ravel(), Eu[mask].ravel()
                    agree.append({"dataset": ds, "partition_seed": seed, "m": m, "subset": subset,
                                  "pearson_site_energy_V": pearsonr(x, yv)[0], "spearman_site_energy_V": spearmanr(x, yv)[0],
                                  "pearson_site_energy_U": pearsonr(x, yu)[0],
                                  "median_rel_err_site_energy_V": float(np.median(np.abs(x - yv) / yv)),
                                  "pearson_d_mix": pearsonr(stm["d_mix"][mask], stv["d_mix"][mask])[0],
                                  "pearson_d_site": pearsonr(stm["d_site"][mask], stv["d_site"][mask])[0],
                                  "spearman_d_mix": spearmanr(stm["d_mix"][mask], stv["d_mix"][mask])[0],
                                  "spearman_d_site": spearmanr(stm["d_site"][mask], stv["d_site"][mask])[0]})
            for s in range(S):
                nulls.append({"dataset": ds, "partition_seed": seed, "site": s, "ref_size": int(z["ref_sizes"][s]),
                              "clean_mean_rff64": float(np.einsum("wss->ws", z["rff64"])[held, s].mean()),
                              "clean_mean_exact_V": float(Ev[held, s].mean()),
                              "clean_mean_exact_U": float(Eu[held, s].mean())})
            for vname, g in variants.items():
                sc, _ = score_seed(frame(names, g, seed, cmeta), 0.95)
                for cn, gg in sc.groupby("condition", sort=False):
                    r = {"dataset": ds, "partition_seed": seed, "variant": vname, "condition": cn}
                    r.update({k: float(gg[k].mean()) for k in RULES})
                    det.append(r)

    agree = pd.DataFrame(agree)
    det = pd.DataFrame(det)
    nulls = pd.DataFrame(nulls)
    agree.to_csv(HERE / "agreement_per_seed.csv", index=False)
    det.to_csv(HERE / "detection_per_seed_condition.csv", index=False)
    nulls.to_csv(HERE / "null_levels.csv", index=False)
    (HERE / "reproduction_check.json").write_text(json.dumps(check, indent=2))

    a = agree.groupby(["dataset", "subset", "m"]).mean(numeric_only=True).drop(columns="partition_seed").reset_index()
    a.to_csv(HERE / "agreement.csv", index=False)

    # detection: clean FPR and pre-specified aggregates, per seed then mean over seeds
    rows = []
    for ds in DATASETS:
        cm = pd.DataFrame([c for c in conditions(ds)])
        masks = aggregate_masks(cm)
        d = det[det.dataset == ds]
        for v, dv in d.groupby("variant"):
            r = {"dataset": ds, "variant": v}
            fp = dv[dv.condition == "clean_heldout"]
            for k in RULES:
                r[f"fpr_{k}"] = float(fp[k].mean())
            for agg, mk in masks.items():
                conds = set(cm.condition[mk])
                sel = dv[dv.condition.isin(conds)]
                for k in ("pooled", "encmon_joint", "localany"):
                    r[f"{agg}_{k}"] = float(sel.groupby("partition_seed")[k].mean().mean())
            rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(HERE / "detection_summary.csv", index=False)
    # best-case rows (joint calibration selection) under every variant
    best = pd.read_csv(HERE.parent / "equal_fpr" / "best_rows.csv")
    br = det.merge(best[["dataset", "condition"]].drop_duplicates(), on=["dataset", "condition"])
    br = br.groupby(["dataset", "condition", "variant"])[["pooled", "encmon_joint", "localany"]].mean().reset_index()
    br.to_csv(HERE / "best_rows_by_variant.csv", index=False)
    nl = nulls.groupby("dataset")[["clean_mean_rff64", "clean_mean_exact_V", "clean_mean_exact_U"]].describe().T
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    f = lambda x: f"{x:.3f}"
    print("Reproduction check:", json.dumps(check))
    print("\n== Agreement (mean over seeds) ==")
    print(a.to_string(index=False, float_format=f))
    print("\n== Detection summary (mean over seeds) ==")
    print(summ.to_string(index=False, float_format=f))
    print("\n== Best-case rows by variant ==")
    print(br.pivot_table(index=["dataset", "condition"], columns="variant", values="encmon_joint").to_string(float_format=f))
    print(br.pivot_table(index=["dataset", "condition"], columns="variant", values="pooled").to_string(float_format=f))
    print("\n== Null levels (clean held-out mean site energy) ==")
    print(nulls.groupby("dataset")[["ref_size", "clean_mean_rff64", "clean_mean_exact_V", "clean_mean_exact_U"]].agg(["min", "median", "max"]).to_string(float_format=lambda x: f"{x:.5f}"))
    print("Spearman(ref_size, clean exact V) per dataset:",
          {ds: round(float(spearmanr(g.ref_size, g.clean_mean_exact_V)[0]), 3) for ds, g in nulls.groupby("dataset")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
