#!/usr/bin/env python3
"""Table 4: Camelyon17 hospital data. Patch split and patient-level split (20 seeds),
slide split (5 seeds, for reference). Clean FPR and detection of the 36 natural opposing/mixed conditions.
Intervals are descriptive (seeds share patches and patients).
Usage: python analysis/tables/table4.py   (after compute_rates.py)
Writes outputs/tables/table4*.csv and .md.
"""
from tlib import LABEL, SETTING, c2, pd, pick, rates, save


def main(seeds=None, name="table4"):
    r = rates()
    out = []
    for s in ("camelyon17_patch", "camelyon17_patient", "camelyon17_slide"):
        for k in ("pooled", "encmon_joint", "localany", "hybrid", "exact"):
            if pick(r, s, "clean_fpr").query("method == @k").empty:
                continue
            row = {"setting": SETTING[s], "method": LABEL[k]}
            for agg, col in (("clean_fpr", "fpr"), ("A1_natural_opposing_mixed", "detection")):
                g = pick(r, s, agg)
                g = g[g.method == k]
                if seeds is not None:
                    g = g[g.seed.isin(seeds)]
                sm = c2.summ(g["default"].to_numpy())
                row.update({"seeds": sm["n_seeds"], f"{col}_mean": sm["mean"], f"{col}_t_lo": sm["t_lo"], f"{col}_t_hi": sm["t_hi"]})
                if col == "detection":
                    row["conditions"] = int(g.conditions.iloc[0])
            out.append(row)
    save(pd.DataFrame(out), name, "Table 4: Camelyon17 hospital data (patch, patient and slide split)",
         "t-intervals over seeds are descriptive: the seeds of one hospital share patches and patients.")


if __name__ == "__main__":
    main()
