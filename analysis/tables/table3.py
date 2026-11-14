#!/usr/bin/env python3
"""Table 3: held-out clean FPR per method (mean over seeds, t-interval, seed range) and ARL0 of the 2-of-3 rule from
the FPR pooled over all held-out windows of all seeds (independent-window Markov model).
Usage: python analysis/tables/table3.py   (after compute_rates.py)
Writes outputs/tables/table3*.csv and .md.
"""
from tlib import LABEL, SETTING, c2, pd, pick, rates, save


def main(seeds=None, name="table3"):
    r = rates()
    out = []
    for s in ("mnist", "cifar10", "camelyon17_patch", "camelyon17_patient"):
        f = pick(r, s, "clean_fpr")
        if seeds is not None:
            f = f[f.seed.isin(seeds)]
        for k in ("pooled", "encmon_joint", "encmon_or", "localany", "hybrid"):
            g = f[f.method == k]
            if g.empty:
                continue
            sm = c2.summ(g["default"].to_numpy())
            n = int(g.heldout_windows.sum())
            alerts = float((g["default"] * g.heldout_windows).sum())
            p = alerts / n
            out.append({"setting": SETTING[s], "method": LABEL[k], "seeds": sm["n_seeds"], "heldout_windows_total": n,
                        "false_alerts_total": int(round(alerts)), "fpr_pooled": p, "fpr_mean": sm["mean"],
                        "fpr_t_lo": sm["t_lo"], "fpr_t_hi": sm["t_hi"], "fpr_seed_min": g["default"].min(),
                        "fpr_seed_max": g["default"].max(), "arl0_2of3_pooled_fpr": c2.arl_2of3(p)})
    save(pd.DataFrame(out), name, "Table 3: clean FPR and ARL0 (2-of-3 rule, FPR pooled over seeds)",
         "ARL0 = expected window index of the first confirmed false alarm for i.i.d. alerts with the pooled FPR.")


if __name__ == "__main__":
    main()
