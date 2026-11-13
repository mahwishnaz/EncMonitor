#!/usr/bin/env python3
"""Ablation analysis.

Per dataset, setting and partition seed, thresholds are calibrated on the 500 clean calibration
windows with encmonitor.thresholds.calibrate_thresholds; rates are measured on the held-out clean windows
and on the representative conditions (same alert definitions as the main experiment).

Outputs: ablation_summary.csv, axis_gate.csv, rff_fidelity.csv, analysis_summary.json,
figures ablation_axes_<dataset>.png and F6_rff_tradeoff.png.
"""
from __future__ import annotations

import argparse
import importlib.util
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


abl = _load("18_run_ablation")
KEYS = abl.KEYS


def score(df: pd.DataFrame, q: float) -> pd.DataFrame:
    out = []
    for (ds, sid, seed), g in df.groupby(["dataset", "setting", "partition_seed"]):
        th = calibrate_thresholds(g[g.condition == "clean_calibration"].to_dict("records"), q)
        h = g[g.condition != "clean_calibration"].copy()
        h["pooled"] = h.d_mix > th.d_mix
        h["site"] = h.d_site > th.d_site
        h["H"] = h.heterogeneity > th.heterogeneity
        h["encmon"] = h.pooled | h.site
        h["dual"] = h.site | h.H
        h["hidden"] = h.dual & ~h.pooled
        h["encmon_score"] = np.maximum(h.d_mix / th.d_mix, h.d_site / th.d_site)
        out.append(h)
    return pd.concat(out, ignore_index=True)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rows = []
    for (ds, sid), g in scored.groupby(["dataset", "setting"]):
        clean = g[g.condition == "clean_heldout"]
        first = g.iloc[0]
        common = {"dataset": ds, "setting": sid, **{k: first[k] for k in KEYS}}
        fpr = {f"fpr_{m}": float(clean[m].mean()) for m in ("pooled", "site", "H", "encmon")}
        for cond, gc in g[g.condition != "clean_heldout"].groupby("condition"):
            row = {**common, "condition": cond, **fpr, "windows": len(gc)}
            for m in ("pooled", "site", "H", "encmon", "dual", "hidden"):
                row[f"tpr_{m}"] = float(gc[m].mean())
            y = np.r_[np.zeros(len(clean)), np.ones(len(gc))]
            for sc in ("d_mix", "encmon_score"):
                row[f"auroc_{sc}"] = float(roc_auc_score(y, np.r_[clean[sc], gc[sc]]))
            row["top1_mean"] = float(pd.to_numeric(gc.top1, errors="coerce").mean())
            row["topk_recall_mean"] = float(pd.to_numeric(gc.topk_recall, errors="coerce").mean())
            rows.append(row)
    return pd.DataFrame(rows)


def axis_gate(summary: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    base = cfg["base"]
    gc = cfg["gate"]
    rows = []
    for ds, gd in summary.groupby("dataset"):
        nat = cfg["natural_cancellation_conditions"][ds]
        for key in KEYS:
            others = [k for k in KEYS if k != key]
            mask = np.ones(len(gd), dtype=bool)
            for k in others:
                bv = base[k]
                mask &= (gd[k].astype(str) == str(bv)) if isinstance(bv, str) else np.isclose(gd[k].astype(float), float(bv))
            sl = gd[mask]
            for val, gv in sl.groupby(key):
                gn = gv[gv.condition.isin(nat)]
                hit = gn[(gn.tpr_pooled <= gc["max_pooled_rate"]) & (gn.tpr_dual >= gc["min_dual_rate"])]
                rows.append({"dataset": ds, "axis": key, "value": val, "reproduces": bool(len(hit) > 0),
                             "conditions_reproducing": ";".join(hit.condition),
                             "fpr_encmon": float(gv.fpr_encmon.iloc[0]),
                             **{f"{c}__pooled": float(gn[gn.condition == c].tpr_pooled.iloc[0]) for c in nat if (gn.condition == c).any()},
                             **{f"{c}__dual": float(gn[gn.condition == c].tpr_dual.iloc[0]) for c in nat if (gn.condition == c).any()}})
    ag = pd.DataFrame(rows)
    per = {}
    gated = set(gc.get("axes", KEYS))
    for (ds, key), g in ag.groupby(["dataset", "axis"]):
        per.setdefault(ds, {})[key] = {"values_reproducing": [str(v) for v in g[g.reproduces].value],
                                       "values_tested": [str(v) for v in g.value],
                                       "gated": key in gated,
                                       "passes": int(g.reproduces.sum()) >= gc["min_values_per_axis"]}
    decision = {"rule": f"per axis, >= {gc['min_values_per_axis']} values reproduce the main-experiment cancellation rule "
                        f"(pooled <= {gc['max_pooled_rate']}, dual >= {gc['min_dual_rate']}) on a natural condition",
                "per_dataset_axis": per,
                "failing_axes": [f"{ds}:{k}" for ds, d in per.items() for k, v in d.items() if v["gated"] and not v["passes"]],
                "reported_not_gated": sorted({k for d in per.values() for k, v in d.items() if not v["gated"]})}
    decision["g3_pass"] = len(decision["failing_axes"]) == 0
    return ag, decision


def fidelity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    ex = df[df.exact_mmd.notna()]
    for (ds, m), g in ex.groupby(["dataset", "rff_dim"]):
        from scipy.stats import pearsonr, spearmanr
        rows.append({"dataset": ds, "rff_dim": int(m), "windows": len(g),
                     "pearson_dmix_vs_exact": float(pearsonr(g.d_mix, g.exact_mmd).statistic),
                     "spearman_dmix_vs_exact": float(spearmanr(g.d_mix, g.exact_mmd).statistic)})
    return pd.DataFrame(rows)


def figures(summary: pd.DataFrame, fid: pd.DataFrame, cfg: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = cfg["base"]
    for ds, gd in summary.groupby("dataset"):
        nat = cfg["natural_cancellation_conditions"][ds]
        fig, axes = plt.subplots(1, len(KEYS), figsize=(2.7 * len(KEYS), 3.3), sharey=True)
        for ax, key in zip(axes, KEYS):
            mask = np.ones(len(gd), dtype=bool)
            for k in KEYS:
                if k == key:
                    continue
                bv = base[k]
                mask &= (gd[k].astype(str) == str(bv)) if isinstance(bv, str) else np.isclose(gd[k].astype(float), float(bv))
            sl = gd[mask]
            present = list(dict.fromkeys(sl.sort_values(key)[key]))
            order = next((spec[key] for spec in cfg["axes"].values() if key in spec), present)
            vals = [v for v in order if any(str(v) == str(p) or (not isinstance(v, str) and np.isclose(float(v), float(p)))
                                            for p in present)]
            vals = [next(p for p in present if str(v) == str(p) or (not isinstance(v, str) and np.isclose(float(v), float(p))))
                    for v in vals]
            x = np.arange(len(vals))
            for i, c in enumerate(nat):
                gc = sl[sl.condition == c].set_index(key).reindex(vals)
                ax.plot(x, gc.tpr_pooled, "-", color=f"C{2*i}", marker="o", label=f"{c}: pooled")
                ax.plot(x, gc.tpr_dual, "--", color=f"C{2*i+1}", marker="s", label=f"{c}: D_site/H")
            ax.set_xticks(x, [str(v) for v in vals], fontsize=8)
            ax.set_title(key, fontsize=9)
            ax.set_ylim(-0.02, 1.02)
            ax.axhline(0.5, color="grey", lw=0.5, ls=":")
            ax.axhline(0.8, color="grey", lw=0.5, ls=":")
        axes[0].set_ylabel("detection rate")
        axes[-1].legend(fontsize=6, loc="lower right")
        fig.suptitle(f"Ablation ({ds}): pooled vs dual detection across single-factor ablations (others at base)")
        fig.tight_layout()
        fig.savefig(out / f"ablation_axes_{ds}.png", dpi=200)
        plt.close(fig)

    if len(fid):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ds, g in fid.groupby("dataset"):
            axes[0].plot(g.rff_dim, g.pearson_dmix_vs_exact, marker="o", label=ds)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("RFF dimension m")
        axes[0].set_ylabel("Pearson r (D_mix vs exact MMD²)")
        axes[0].set_title("Sketch fidelity")
        axes[0].legend(fontsize=8)
        for ds, gd in summary.groupby("dataset"):
            sl = gd[(gd.sites == base["sites"]) & (gd.window_size == base["window_size"]) & (gd.layer == base["layer"])
                    & np.isclose(gd.bandwidth_multiplier.astype(float), 1.0) & (gd.baseline == "site") & (gd.weights == "uniform")]
            for c in cfg["natural_cancellation_conditions"][ds]:
                g = sl[sl.condition == c].sort_values("rff_dim")
                axes[1].plot(g.rff_dim, g.tpr_hidden, marker="o", label=f"{ds}: {c}")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("RFF dimension m")
        axes[1].set_ylabel("hidden-drift rate")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title("Cancellation exposure")
        axes[1].legend(fontsize=6)
        fig.suptitle("F6: RFF trade-off (latency panel added after the CKKS runs)")
        fig.tight_layout()
        fig.savefig(out / "F6_rff_tradeoff.png", dpi=200)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze ablation runs")
    ap.add_argument("--config", default="configs/ablation.yaml")
    ap.add_argument("--runs-dir", default="outputs/ablation/runs")
    ap.add_argument("--datasets", default="mnist,cifar10")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    frames = []
    for ds in [d for d in args.datasets.split(",") if d.strip()]:
        for f in sorted((Path(args.runs_dir) / ds).glob("*.jsonl")):
            g = pd.read_json(f, lines=True)
            g["dataset"] = ds
            frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    scored = score(df, args.quantile)
    summary = summarize(scored)
    ag, decision = axis_gate(summary, cfg)
    fid = fidelity(df, cfg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    ag.to_csv(out / "axis_gate.csv", index=False)
    fid.to_csv(out / "rff_fidelity.csv", index=False)
    write_json(out / "analysis_summary.json", {"quantile": args.quantile, "gate": decision,
                                               "rff_fidelity": fid.to_dict("records")})
    figures(summary, fid, cfg, out)

    pd.set_option("display.width", 220)
    cols = [c for c in ag.columns if c.endswith("__pooled") or c.endswith("__dual")]
    print(f"== Ablation single-factor axes (q={args.quantile}; others at base) ==")
    print(ag[["dataset", "axis", "value", "reproduces", "fpr_encmon"] + cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if len(fid):
        print("\n== RFF fidelity: D_mix vs exact-kernel MMD^2 (same windows) ==")
        print(fid.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== Ablation decision ==")
    print(json.dumps({k: v for k, v in decision.items() if k != "per_dataset_axis"}, indent=2))
    print(f"\nWrote ablation_summary.csv, axis_gate.csv, rff_fidelity.csv, analysis_summary.json and figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
