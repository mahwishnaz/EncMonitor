#!/usr/bin/env python3
"""Screening analysis of the main experiment.

Per dataset and partition seed, thresholds are calibrated on that seed's 500 clean
calibration windows with encmonitor.thresholds.calibrate_thresholds (95th pct, "higher");
detection is evaluated on the independent held-out clean windows and each shifted condition.

Methods compared on the SAME windows:
  pooled   : D_mix > tau_mix  (plaintext RFF pooled-only = secure-aggregation analogue = EncMonitor global-only)
  site     : D_site > tau_site
  H        : H > tau_H
  encmon   : D_mix > tau_mix OR D_site > tau_site        (EncMonitor OR rule)
  dual     : D_site > tau_site OR H > tau_H              (the statistics a pooled monitor lacks)
  hidden   : dual AND NOT pooled                          (hidden cancellation)
  exact    : centralized exact-kernel MMD^2 > its own 95th-pct clean threshold
  localmax : max_s ||delta_s||^2 / tau_s > its own 95th-pct clean threshold (tau_s = per-site 95th pct)
  majority : #{s : ||delta_s||^2 > tau_s} >= ceil((S+1)/2)
Continuous scores for AUROC/AUPRC: d_mix, d_site, heterogeneity, encmon = max(D_mix/tau_mix, D_site/tau_site),
exact_mmd, localmax, votes.

Outputs (per quantile): condition_summary.csv, clean_fpr.csv, family_gate.csv,
analysis_summary.json, and the figures detection rate vs severity (F3_detection_<dataset>.png), hidden-drift rate
vs severity (F4_hidden_drift_<dataset>.png) and site attribution (F5_attribution_<dataset>.png).
Confidence intervals: 95% bootstrap over windows, stratified by partition seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import write_json
from encmonitor.thresholds import calibrate_thresholds

METHODS = ["pooled", "site", "H", "encmon", "dual", "hidden", "exact", "localmax", "majority"]
SCORES = ["d_mix", "d_site", "heterogeneity", "encmon_score", "exact_mmd", "localmax_score", "votes"]


def load(cache_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(cache_dir.glob("seed_*/records.jsonl")):
        frames.append(pd.read_json(f, lines=True))
    if not frames:
        raise RuntimeError(f"No records under {cache_dir}")
    return pd.concat(frames, ignore_index=True)


def q_higher(v, q):
    return float(np.quantile(np.asarray(v, dtype=float), q, method="higher"))


def score_seed(df: pd.DataFrame, q: float, S: int) -> tuple[pd.DataFrame, dict]:
    cal = df[df.condition == "clean_calibration"]
    th = calibrate_thresholds(cal.to_dict("records"), q)
    E_cal = np.stack(cal.site_energy.to_numpy())
    tau_s = np.array([q_higher(E_cal[:, s], q) for s in range(S)])
    lm_cal = (E_cal / tau_s).max(axis=1)
    tau_lm = q_higher(lm_cal, q)
    tau_exact = q_higher(cal.exact_mmd, q) if cal.exact_mmd.notna().all() else np.nan
    need = int(np.ceil((S + 1) / 2))

    out = df[df.condition != "clean_calibration"].copy()
    E = np.stack(out.site_energy.to_numpy())
    out["localmax_score"] = (E / tau_s).max(axis=1)
    out["votes"] = (E > tau_s).sum(axis=1)
    out["encmon_score"] = np.maximum(out.d_mix / th.d_mix, out.d_site / th.d_site)
    out["pooled"] = out.d_mix > th.d_mix
    out["site"] = out.d_site > th.d_site
    out["H"] = out.heterogeneity > th.heterogeneity
    out["encmon"] = out.pooled | out.site
    out["dual"] = out.site | out.H
    out["hidden"] = out.dual & ~out.pooled
    out["exact"] = out.exact_mmd > tau_exact
    out["localmax"] = out.localmax_score > tau_lm
    out["majority"] = out.votes >= need
    thr = {"tau_mix": th.d_mix, "tau_site": th.d_site, "tau_h": th.heterogeneity, "tau_exact_mmd": tau_exact,
           "tau_localmax": tau_lm, "tau_site_energy": tau_s.tolist(), "majority_votes": need}
    return out, thr


def boot_rate(x: np.ndarray, strata: np.ndarray, reps: int, rng) -> tuple[float, float]:
    groups = [x[strata == s] for s in np.unique(strata)]
    stats = [np.mean(np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])) for _ in range(reps)]
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def auc_pair(clean: np.ndarray, shifted: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.r_[np.zeros(clean.size), np.ones(shifted.size)]
    s = np.r_[clean, shifted]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def summarize(scored: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    clean = scored[scored.condition == "clean_heldout"]
    fpr_rows = []
    for ds, g in clean.groupby("dataset"):
        row = {"dataset": ds, "windows": len(g)}
        for mth in METHODS:
            row[f"fpr_{mth}"] = float(g[mth].mean())
        for seed_, gs in g.groupby("partition_seed"):
            row[f"fpr_encmon_seed{seed_}"] = float(gs.encmon.mean())
        fpr_rows.append(row)

    rows = []
    shifted = scored[scored.condition != "clean_heldout"]
    for (ds, cond), g in shifted.groupby(["dataset", "condition"], sort=False):
        c = clean[clean.dataset == ds]
        row = {"dataset": ds, "condition": cond, "family": g.family.iloc[0], "severity": g.severity.iloc[0],
               "geometry": g.geometry.iloc[0], "windows": len(g), "seeds": g.partition_seed.nunique()}
        for k in ("d_mix", "d_site", "heterogeneity", "exact_mmd"):
            row[f"{k}_mean"] = float(g[k].mean())
        row["dsite_over_dmix"] = float(g.d_site.mean() / max(g.d_mix.mean(), 1e-12))
        strata = g.partition_seed.to_numpy()
        for mth in METHODS:
            x = g[mth].to_numpy(dtype=float)
            row[f"tpr_{mth}"] = float(x.mean())
            if mth in ("pooled", "dual", "hidden", "encmon"):
                row[f"tpr_{mth}_ci_low"], row[f"tpr_{mth}_ci_high"] = boot_rate(x, strata, reps, rng)
        for sc in SCORES:
            if g[sc].notna().all() and c[sc].notna().all():
                row[f"auroc_{sc}"], row[f"auprc_{sc}"] = auc_pair(c[sc].to_numpy(float), g[sc].to_numpy(float))
        for k in ("top1", "topk_recall", "ndcg", "spearman"):
            if k in g:
                row[f"{k}_mean"] = float(pd.to_numeric(g[k], errors="coerce").mean())
        x = g.top1.to_numpy(dtype=float)
        row["top1_ci_low"], row["top1_ci_high"] = boot_rate(x, strata, reps, rng)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(fpr_rows)


def gate(summary: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    gc = cfg["gate"]
    rows = []
    for (ds, fam), g in summary.groupby(["dataset", "family"]):
        if fam not in cfg["realistic_families"] and fam != "vector":
            continue
        cand = g[g.geometry.isin(gc["geometries"])]
        hit = cand[(cand.tpr_pooled <= gc["max_pooled_rate"]) & (cand.tpr_dual >= gc["min_dual_rate"])]
        best = cand.sort_values("tpr_hidden", ascending=False).head(1)
        rows.append({
            "dataset": ds, "family": fam, "realistic": fam in cfg["realistic_families"],
            "reproduces_cancellation": bool(len(hit) > 0),
            "qualifying_conditions": ";".join(hit.condition) if len(hit) else "",
            "best_condition": best.condition.iloc[0] if len(best) else "",
            "best_hidden_rate": float(best.tpr_hidden.iloc[0]) if len(best) else np.nan,
            "best_pooled_rate": float(best.tpr_pooled.iloc[0]) if len(best) else np.nan,
            "best_dual_rate": float(best.tpr_dual.iloc[0]) if len(best) else np.nan,
        })
    fg = pd.DataFrame(rows)
    real = fg[fg.realistic]
    per_ds = {ds: sorted(g[g.reproduces_cancellation].family.tolist()) for ds, g in real.groupby("dataset")}
    overall = sorted(set(real[real.reproduces_cancellation].family))
    decision = {
        "rule": f"opposing/mixed with pooled rate <= {gc['max_pooled_rate']} and dual rate >= {gc['min_dual_rate']}; "
                f"The main experiment passes if >= {gc['min_families']} realistic families reproduce",
        "families_reproducing_per_dataset": per_ds,
        "families_reproducing_any_dataset": overall,
        "g2_pass_any_dataset": len(overall) >= gc["min_families"],
        "g2_pass_every_dataset": all(len(v) >= gc["min_families"] for v in per_ds.values()) if per_ds else False,
    }
    return fg, decision


def figures(summary: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ds, gd in summary.groupby("dataset"):
        fams = list(dict.fromkeys(gd.family))
        # figure: detection rate vs severity, opposing geometry (single for blur/noise)
        fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True)
        for ax, fam in zip(np.atleast_1d(axes), fams):
            gf = gd[gd.family == fam]
            geo = "opposing" if "opposing" in set(gf.geometry) else "coherent"
            gg = gf[gf.geometry == geo].sort_values("severity")
            for mth, lab, ls, col in (("pooled", "pooled D_mix", "-", "C0"), ("encmon", "EncMonitor (D_mix or D_site)", "-", "C1"),
                                      ("exact", "exact-kernel MMD", "--", "C2"), ("localmax", "local alarms (max)", ":", "C3")):
                ax.plot(gg.severity, gg[f"tpr_{mth}"], ls, color=col, marker="o", label=lab)
            ax.set_title(f"{fam} ({geo})", fontsize=9)
            ax.set_xlabel("severity")
            ax.set_ylim(-0.02, 1.02)
        np.atleast_1d(axes)[0].set_ylabel("detection rate")
        np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{ds}: detection rate vs severity at calibrated thresholds")
        fig.tight_layout()
        fig.savefig(out / f"F3_detection_{ds}.png", dpi=200)
        plt.close(fig)

        # figure: hidden-drift rate vs severity, opposing and mixed
        fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True)
        for ax, fam in zip(np.atleast_1d(axes), fams):
            gf = gd[gd.family == fam]
            for geo, ls, col in (("opposing", "-", "C0"), ("mixed", "--", "C1"), ("coherent", ":", "C2"), ("single", "-.", "C3")):
                gg = gf[gf.geometry == geo].sort_values("severity")
                if len(gg):
                    ax.plot(gg.severity, gg.tpr_hidden, ls, color=col, marker="o", label=geo)
            ax.set_title(fam, fontsize=9)
            ax.set_xlabel("severity")
            ax.set_ylim(-0.02, 1.02)
        np.atleast_1d(axes)[0].set_ylabel("hidden-drift rate\n(D_site/H alert, D_mix silent)")
        np.atleast_1d(axes)[-1].legend(fontsize=7)
        fig.suptitle(f"{ds}: drift missed by the pooled score but caught by the dual statistic")
        fig.tight_layout()
        fig.savefig(out / f"F4_hidden_drift_{ds}.png", dpi=200)
        plt.close(fig)

        # figure: site attribution (top-1 for single, top-k recall otherwise), mean over severities
        geos = ["single", "coherent", "opposing", "mixed"]
        mat = np.full((len(fams), len(geos)), np.nan)
        for i, fam in enumerate(fams):
            for j, geo in enumerate(geos):
                gg = gd[(gd.family == fam) & (gd.geometry == geo)]
                if len(gg):
                    mat[i, j] = gg["top1_mean" if geo == "single" else "topk_recall_mean"].mean()
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(fams) + 1.8))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(geos)), ["single\n(top-1)", "coherent\n(top-k)", "opposing\n(top-k)", "mixed\n(top-k)"])
        ax.set_yticks(range(len(fams)), fams)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="attribution accuracy")
        ax.set_title(f"{ds}: site attribution, mean over severities")
        fig.tight_layout()
        fig.savefig(out / f"F5_attribution_{ds}.png", dpi=200)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze screening records")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--cache-dirs", nargs="+", required=True, help="e.g. outputs/main/mnist outputs/main/cifar10")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seeds", default="", help="comma-separated partition seeds to include (default: all), for sensitivity checks")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    S = int(cfg["sites"])
    scored, thresholds = [], {}
    for cd in args.cache_dirs:
        cd = Path(cd)
        df = load(cd)
        df["dataset"] = cd.name
        if args.seeds:
            df = df[df.partition_seed.isin([int(x) for x in args.seeds.split(",") if x.strip()])]
        for seed, g in df.groupby("partition_seed"):
            s, thr = score_seed(g, args.quantile, S)
            scored.append(s)
            thresholds[f"{cd.name}/seed_{seed}"] = thr
    scored = pd.concat(scored, ignore_index=True)
    summary, fpr = summarize(scored, args.bootstrap, 20260818)
    fg, decision = gate(summary, cfg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "condition_summary.csv", index=False)
    fpr.to_csv(out / "clean_fpr.csv", index=False)
    fg.to_csv(out / "family_gate.csv", index=False)
    write_json(out / "analysis_summary.json", {"quantile": args.quantile, "seeds": args.seeds or "all", "thresholds": thresholds,
                                               "clean_fpr": fpr.to_dict("records"), "gate": decision})
    figures(summary, out)

    pd.set_option("display.width", 220)
    print(f"== Clean held-out false-positive rates (q={args.quantile}) ==")
    print(fpr[["dataset", "windows"] + [f"fpr_{m}" for m in ("pooled", "site", "H", "encmon", "exact", "localmax", "majority")]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== Cancellation per family (opposing/mixed; best condition by hidden-drift rate) ==")
    print(fg[["dataset", "family", "reproduces_cancellation", "best_condition", "best_pooled_rate", "best_dual_rate",
              "best_hidden_rate"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n== Main-experiment decision ==")
    print(json.dumps(decision, indent=2))
    print(f"\nWrote condition_summary.csv, clean_fpr.csv, family_gate.csv, analysis_summary.json and the figures F3_*.png, F4_*.png, F5_*.png (detection, hidden drift, attribution) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
