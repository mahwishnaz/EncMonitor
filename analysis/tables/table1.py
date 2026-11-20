#!/usr/bin/env python3
"""The main detection table (Table 4 of the article; internal name table1): detection (pre-specified aggregates) and held-out clean FPR per method and setting; mean over seeds and
t-interval (t = 2.093 for 20 seeds, 2.776 for 5). Per-seed values: table1_per_seed.csv (Supplementary).
Usage: python analysis/tables/table1.py   (after compute_rates.py)
Writes outputs/tables/table1*.csv and .md.
"""
from tlib import LABEL, SETTING, c2, pd, pick, rates, save, weak

ROWS = [("mnist", "A1_natural_opposing_mixed"), ("cifar10", "A1_natural_opposing_mixed"),
        ("camelyon17_patch", "A1_natural_opposing_mixed"), ("camelyon17_patient", "A1_natural_opposing_mixed"),
        ("camelyon17_slide", "A1_natural_opposing_mixed"), ("census_n250", "A1_natural_opposing_mixed"),
        ("census_n1000", "A1_natural_opposing_mixed"), ("census_n1000", "year_one_or_two")]
METHODS = ["pooled", "encmon_joint", "localany", "hybrid", "exact"]


def main(seeds=None, name="table1", title="Table 1: detection of opposing/mixed shifts and clean FPR"):
    r = rates()
    out, per = [], []
    for set_, agg in ROWS + [("weak_all_site", "weak")]:
        lab = SETTING.get(set_, set_)
        if agg == "year_one_or_two":
            lab = "Real census drift, one/two states (n = 1,000)"
        if set_ == "weak_all_site":
            lab = "Weak shift at all five sites (18 conditions)"
        for k in METHODS:
            g = weak(r) if set_ == "weak_all_site" else pick(r, set_, agg)
            g = g[g.method == k]
            if seeds is not None:
                g = g[g.seed.isin(seeds)]
            if g.empty:
                continue
            f = pick(r, set_, "clean_fpr")
            f = f[f.method == k]
            if seeds is not None:
                f = f[f.seed.isin(seeds)]
            f = f.set_index("seed").reindex(g.seed) if agg == "A1_natural_opposing_mixed" else None
            sd = c2.summ(g["default"].to_numpy())
            row = {"setting": lab, "method": LABEL[k], "seeds": sd["n_seeds"], "conditions": int(g.conditions.iloc[0]),
                   "detection_mean": sd["mean"], "detection_sd": sd["sd"], "detection_t_lo": sd["t_lo"], "detection_t_hi": sd["t_hi"]}
            if f is not None:
                sf = c2.summ(f["default"].to_numpy())
                row.update({"fpr_mean": sf["mean"], "fpr_sd": sf["sd"], "fpr_t_lo": sf["t_lo"], "fpr_t_hi": sf["t_hi"]})
            out.append(row)
            for _, x in g.iterrows():
                per.append({"setting": lab, "method": LABEL[k], "seed": int(x.seed), "detection": x["default"],
                            "fpr": float(f.loc[x.seed, "default"]) if f is not None else None})
    df = pd.DataFrame(out)
    save(df, name, title, "Mean over partition seeds with t-interval mean ± t·SD/√n; Exact-MMD only where the records contain it.")
    save(pd.DataFrame(per), f"{name}_per_seed", title + " (per seed)")
    return df


if __name__ == "__main__":
    main()
