#!/usr/bin/env python3
"""Matched-FPR detection: per setting, method and seed the threshold is the 95th pct ("higher") of the method's
score over the same 200 held-out clean windows on which the FPR is measured (FPR = 9/200 = 0.045 by construction).
Reports default and matched detection and EncMonitor − Local-any (matched) for every setting.
Usage: python analysis/tables/matched_fpr.py   (after compute_rates.py)
Writes outputs/tables/matched_fpr*.csv and .md.
"""
from tlib import LABEL, SETTING, c2, pd, pick, rates, save, weak

SETS = [("mnist", "A1_natural_opposing_mixed"), ("cifar10", "A1_natural_opposing_mixed"),
        ("camelyon17_patch", "A1_natural_opposing_mixed"), ("camelyon17_patient", "A1_natural_opposing_mixed"),
        ("camelyon17_slide", "A1_natural_opposing_mixed"), ("census_n250", "A1_natural_opposing_mixed"),
        ("census_n1000", "A1_natural_opposing_mixed"), ("census_n1000", "year_one_or_two"),
        ("census_n250", "year_all"), ("census_n250", "year_one_or_two"), ("census_n1000", "year_all"), ("weak_all_site", "weak")]
LAB = {("census_n1000", "year_one_or_two"): "Real census drift, one/two states (n = 1,000)",
       ("census_n250", "year_all"): "Real census drift, all five states (n = 250)",
       ("census_n250", "year_one_or_two"): "Real census drift, one/two states (n = 250)",
       ("census_n1000", "year_all"): "Real census drift, all five states (n = 1,000)",
       ("weak_all_site", "weak"): "Weak shift at all five sites (18 conditions)"}


def main(seeds=None, name="matched_fpr"):
    r = rates()
    out = []
    for s, a in SETS:
        g = weak(r) if s == "weak_all_site" else pick(r, s, a)
        if seeds is not None:
            g = g[g.seed.isin(seeds)]
        if g.empty:
            continue
        f = pick(r, "mnist" if s == "weak_all_site" else s, "clean_fpr")
        if seeds is not None:
            f = f[f.seed.isin(seeds)]
        row = {"setting": LAB.get((s, a), SETTING.get(s, s)), "seeds": int(g.seed.nunique())}
        for k in ("pooled", "encmon_joint", "localany", "hybrid", "exact"):
            h = g[g.method == k]
            if h.empty:
                continue
            row[f"{LABEL[k]} default"] = h["default"].mean()
            row[f"{LABEL[k]} matched"] = h["matched"].mean()
        if s != "weak_all_site":
            row["matched FPR (all methods, min-max over seeds)"] = f"{f.matched.min():.3f}-{f.matched.max():.3f}"
        d = (g[g.method == "encmon_joint"].set_index("seed").matched - g[g.method == "localany"].set_index("seed").matched).dropna()
        sm = c2.summ(d.to_numpy())
        row.update({"EncMonitor − Local-any (matched)": sm["mean"], "t_lo": sm["t_lo"], "t_hi": sm["t_hi"]})
        out.append(row)
    save(pd.DataFrame(out), name, "Matched-FPR detection (held-out FPR 0.045 = 9/200)",
         "Thresholds set on the same held-out windows on which the FPR is measured; the FPR is equal by construction. "
         "Weak all-site rows use the held-out windows of their dataset.")


if __name__ == "__main__":
    main()
