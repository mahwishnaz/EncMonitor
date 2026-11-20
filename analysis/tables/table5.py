#!/usr/bin/env python3
"""The census table (Table 7 of the article; internal name table5): census data (n = 250 and n = 1,000): clean FPR, injected opposing/mixed shifts (16 conditions), real
drift at all five states and at one/two states (mean of the four years 2015-2018).
Usage: python analysis/tables/table5.py   (after compute_rates.py)
Writes outputs/tables/table5*.csv and .md.
"""
from tlib import LABEL, SETTING, c2, pd, pick, rates, save

ROWS = [("clean_fpr", "clean FPR"), ("A1_natural_opposing_mixed", "injected opposing/mixed (16 conditions)"),
        ("year_all", "real drift, all five states (mean of 4 years)"),
        ("year_one_or_two", "real drift, one/two states (mean of 4 years)")]


def main(seeds=None, name="table5"):
    r = rates()
    out = []
    for s in ("census_n250", "census_n1000"):
        for agg, lab in ROWS:
            for k in ("pooled", "encmon_joint", "localany", "hybrid"):
                g = pick(r, s, agg)
                g = g[g.method == k]
                if seeds is not None:
                    g = g[g.seed.isin(seeds)]
                if g.empty:
                    continue
                sm = c2.summ(g["default"].to_numpy())
                out.append({"setting": SETTING[s], "row": lab, "method": LABEL[k], "seeds": sm["n_seeds"],
                            "mean": sm["mean"], "sd": sm["sd"], "t_lo": sm["t_lo"], "t_hi": sm["t_hi"],
                            "seed_min": g["default"].min(), "seed_max": g["default"].max()})
    save(pd.DataFrame(out), name, "Table 5: census data")


if __name__ == "__main__":
    main()
