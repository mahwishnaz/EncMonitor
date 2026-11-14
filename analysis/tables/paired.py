#!/usr/bin/env python3
"""Paired comparison per seed: EncMonitor − Local-any and EncMonitor − Pooled, default and matched thresholds;
mean difference with t-interval and exact two-sided sign test (zeros dropped).
Usage: python analysis/tables/paired.py   (after compute_rates.py)
Writes outputs/tables/paired*.csv and .md.
"""
from tlib import SETTING, c2, pd, pick, rates, save, weak

SETS = [("mnist", "A1_natural_opposing_mixed"), ("cifar10", "A1_natural_opposing_mixed"),
        ("camelyon17_patch", "A1_natural_opposing_mixed"), ("camelyon17_patient", "A1_natural_opposing_mixed"),
        ("camelyon17_slide", "A1_natural_opposing_mixed"), ("census_n250", "A1_natural_opposing_mixed"),
        ("census_n1000", "A1_natural_opposing_mixed"), ("census_n1000", "year_one_or_two"), ("weak_all_site", "weak")]


def label(s, a):
    if a == "year_one_or_two":
        return "Real census drift, one/two states (n = 1,000)"
    return "Weak shift at all five sites (18 conditions)" if s == "weak_all_site" else SETTING[s]


def main(seeds=None, name="paired"):
    r = rates()
    out, per = [], []
    for s, a in SETS:
        g = weak(r) if s == "weak_all_site" else pick(r, s, a)
        if seeds is not None:
            g = g[g.seed.isin(seeds)]
        if g.empty:
            continue
        for other in ("localany", "pooled"):
            for th in ("default", "matched"):
                x = g[g.method == "encmon_joint"].set_index("seed")[th]
                y = g[g.method == other].set_index("seed")[th]
                d = (x - y).dropna()
                sm = c2.summ(d.to_numpy())
                pos, neg, zero, p = c2.sign_test(d.to_numpy())
                comp = f"EncMonitor − {'Local-any' if other == 'localany' else 'Pooled'}"
                out.append({"setting": label(s, a), "comparison": comp, "thresholds": th, "seeds": sm["n_seeds"],
                            "mean_diff": sm["mean"], "sd": sm["sd"], "t_lo": sm["t_lo"], "t_hi": sm["t_hi"],
                            "n_pos": pos, "n_neg": neg, "n_zero": zero, "sign_test_p": p})
                for sd_, v in d.items():
                    per.append({"setting": label(s, a), "comparison": comp, "thresholds": th, "seed": int(sd_), "difference": v})
    save(pd.DataFrame(out), name, "Paired per-seed comparison (EncMonitor vs Local-any and vs Pooled)",
         "Exact two-sided sign test; with 20 seeds the smallest p is 2/2^20 = 1.9e-6, with 5 seeds 0.0625.")
    save(pd.DataFrame(per), f"{name}_per_seed", "Paired differences per seed")


if __name__ == "__main__":
    main()
