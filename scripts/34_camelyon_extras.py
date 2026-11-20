#!/usr/bin/env python3
"""Camelyon17 analyses that the analysis script 16 does not cover.

Reads the records of 33_camelyon_windows.py (same thresholds as 16: per partition seed,
95th percentile "higher" of the 500 clean calibration windows).

  A. False alarms: single statistics, the combined rule, 97.5th-percentile thresholds and
     joint calibration (one threshold c on max(D_mix/tau_mix, D_site/tau_site)).
  B. Global reference on real hospitals: the same windows scored against ONE global reference instead of
     site references -- false-positive rate and dual detection in opposing/mixed conditions.
  C. Accuracy link: per window and site, accuracy drop = clean accuracy of the site's whole
     monitoring pool minus the accuracy in the window; harmful = some site loses >= 5 points.
     Detection of harmful windows by the pooled score and by EncMonitor, rank correlation of
     the statistics with the largest site loss, worst-site vs population loss, harmful share of
     EncMonitor-only alerts, and how often the top-ranked site (largest A_s) is the worst site.

Usage (from the repository root):
  python scripts/34_camelyon_extras.py
Writes outputs/camelyon/analysis/extras.json and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import write_json  # noqa: E402


def qh(v, q):
    return float(np.quantile(np.asarray(v, dtype=float), q, method="higher"))


def load(cache_dir: Path):
    frames, clean_acc = [], {}
    for d in sorted(cache_dir.glob("seed_*")):
        if not (d / "records.jsonl").exists():
            continue
        frames.append(pd.read_json(d / "records.jsonl", lines=True))
        meta = json.loads((d / "conditions.json").read_text())
        clean_acc[int(meta["partition_seed"])] = np.asarray(meta["clean_accuracy_per_site"], dtype=float)
    if not frames:
        raise SystemExit(f"no records under {cache_dir}")
    return pd.concat(frames, ignore_index=True), clean_acc


def score(df: pd.DataFrame, q: float, suffix: str = "") -> pd.DataFrame:
    """Threshold one statistic set (site refs: suffix '', global ref: '_global') per seed."""
    parts = []
    for seed, g in df.groupby("partition_seed"):
        cal = g[g.condition == "clean_calibration"]
        t_mix, t_site, t_h = (qh(cal["d_mix" + suffix], q), qh(cal["d_site" + suffix], q),
                              qh(cal["heterogeneity" + suffix], q))
        s = g[g.condition != "clean_calibration"].copy()
        s["pooled"] = s["d_mix" + suffix] > t_mix
        s["site"] = s["d_site" + suffix] > t_site
        s["H"] = s["heterogeneity" + suffix] > t_h
        s["encmon"] = s.pooled | s.site
        s["dual"] = s.site | s.H
        s["hidden"] = s.dual & ~s.pooled
        score_cal = np.maximum(cal["d_mix" + suffix] / t_mix, cal["d_site" + suffix] / t_site)
        c = qh(score_cal, q)
        s["joint"] = np.maximum(s["d_mix" + suffix] / t_mix, s["d_site" + suffix] / t_site) > c
        s["joint_c"] = c
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Camelyon17 extra analyses (false alarms, H2, accuracy link)")
    ap.add_argument("--cache-dir", default="outputs/camelyon/main/camelyon17")
    ap.add_argument("--out-dir", default="outputs/camelyon/analysis")
    ap.add_argument("--harm", type=float, default=0.05)
    args = ap.parse_args()

    df, clean_acc = load(ROOT / args.cache_dir)
    res: dict = {"seeds": sorted(clean_acc), "clean_accuracy_per_site": {str(k): v.tolist() for k, v in clean_acc.items()}}
    pd.set_option("display.width", 200)

    # ---------------------------------------------------------------- A. false alarms
    s95, s975 = score(df, 0.95), score(df, 0.975)
    held95, held975 = s95[s95.condition == "clean_heldout"], s975[s975.condition == "clean_heldout"]
    fa = {"windows": int(len(held95)),
          "fpr_dmix": float(held95.pooled.mean()), "fpr_dsite": float(held95.site.mean()), "fpr_H": float(held95.H.mean()),
          "fpr_encmon_95": float(held95.encmon.mean()), "fpr_encmon_975": float(held975.encmon.mean()),
          "fpr_joint": float(held95.joint.mean()), "joint_c_per_seed": [float(x) for x in s95.groupby("partition_seed").joint_c.first()]}
    res["false_alarms"] = fa
    print("== A. Clean held-out false-positive rates ==")
    print(json.dumps(fa, indent=2))

    # ---------------------------------------------------------------- B. site vs global reference
    g95 = score(df, 0.95, "_global")
    heldg = g95[g95.condition == "clean_heldout"]
    shifted = s95[(s95.condition != "clean_heldout") & s95.geometry.isin(["opposing", "mixed"])]
    shifted_g = g95[(g95.condition != "clean_heldout") & g95.geometry.isin(["opposing", "mixed"])]
    rows = []
    for (fam, cond), g in shifted.groupby(["family", "condition"]):
        gg = shifted_g[shifted_g.condition == cond]
        rows.append({"family": fam, "condition": cond, "pooled": g.pooled.mean(), "dual_site_ref": g.dual.mean(),
                     "hidden_site_ref": g.hidden.mean(), "encmon_site_ref": g.encmon.mean(),
                     "dual_global_ref": gg.dual.mean(), "encmon_global_ref": gg.encmon.mean()})
    h2 = pd.DataFrame(rows)
    best = h2.sort_values("hidden_site_ref", ascending=False).groupby("family").head(1).sort_values("family")
    res["h2"] = {"fpr_encmon_site_ref": float(held95.encmon.mean()), "fpr_encmon_global_ref": float(heldg.encmon.mean()),
                 "fpr_dual_site_ref": float(held95.dual.mean()), "fpr_dual_global_ref": float(heldg.dual.mean()),
                 "best_condition_per_family": best.to_dict("records")}
    print("\n== B. H2: site references vs one global reference ==")
    print(f"FPR EncMonitor: site refs {held95.encmon.mean():.3f}  global ref {heldg.encmon.mean():.3f}")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---------------------------------------------------------------- C. accuracy link
    sh = s95[s95.condition != "clean_heldout"].copy()
    acc = np.stack(sh.acc.to_numpy())
    ref = np.stack([clean_acc[int(p)] for p in sh.partition_seed])
    drop = ref - acc
    sh["max_drop"] = drop.max(axis=1)
    sh["pop_drop"] = drop.mean(axis=1)          # uniform weights, equal window sizes
    sh["harmful"] = sh.max_drop >= args.harm
    A = np.stack(sh.contributions.to_numpy())
    sh["top_is_worst"] = A.argmax(axis=1) == drop.argmax(axis=1)
    clean_h = s95[s95.condition == "clean_heldout"].copy()
    acc_c = np.stack(clean_h.acc.to_numpy())
    ref_c = np.stack([clean_acc[int(p)] for p in clean_h.partition_seed])
    clean_harm = float(((ref_c - acc_c).max(axis=1) >= args.harm).mean())

    from scipy.stats import spearmanr

    def link(g):
        h = g[g.harmful]
        only = g[g.encmon & ~g.pooled]
        alerted_h = h[h.encmon]
        return {"windows": int(len(g)), "harmful": int(len(h)), "harmful_share": float(g.harmful.mean()),
                "rho_dsite": float(spearmanr(g.d_site, g.max_drop).statistic) if len(g) > 2 else float("nan"),
                "rho_dmix": float(spearmanr(g.d_mix, g.max_drop).statistic) if len(g) > 2 else float("nan"),
                "detect_harmful_encmon": float(h.encmon.mean()) if len(h) else float("nan"),
                "detect_harmful_pooled": float(h.pooled.mean()) if len(h) else float("nan"),
                "worst_site_loss_pts": float(100 * g.max_drop.mean()), "population_loss_pts": float(100 * g.pop_drop.mean()),
                "encmon_only_alerts": int(len(only)), "encmon_only_harmful": float(only.harmful.mean()) if len(only) else float("nan"),
                "top1_is_worst_in_harmful_alerts": float(alerted_h.top_is_worst.mean()) if len(alerted_h) else float("nan")}

    res["accuracy_link"] = {"harm_threshold": args.harm, "clean_windows_harmful": clean_harm,
                            "all_shifted": link(sh),
                            "opposing_mixed": link(sh[sh.geometry.isin(["opposing", "mixed"])]),
                            "per_family": {f: link(g) for f, g in sh.groupby("family")}}
    print("\n== C. Accuracy link (harmful = a site loses >= {:.0f} points) ==".format(100 * args.harm))
    print(f"clean held-out windows that look harmful (sampling only): {clean_harm:.3f}")
    tab = pd.DataFrame({k: link(v) for k, v in [("all", sh), ("opp/mixed", sh[sh.geometry.isin(['opposing', 'mixed'])])]
                        + list(sh.groupby("family"))}).T
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    h2.to_csv(out / "h2_site_vs_global.csv", index=False)
    tab.to_csv(out / "accuracy_link.csv")
    write_json(out / "extras.json", res)
    print(f"\nwritten: {out / 'extras.json'}, h2_site_vs_global.csv, accuracy_link.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
