#!/usr/bin/env python3
"""Encrypted Local-any and a hybrid vs EncMonitor under the same disclosure policy.

Detection, FPR, ARL0 and leakage (CPU). The CKKS cost is in ckks.py.
Usage (from the repository root): python experiments/private_localany/run.py > experiments/private_localany/run_log.txt
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RX = HERE.parent
sys.path.insert(0, str(RX / "common"))
import rx_common as rc  # noqa: E402
from rx_windows import SeedContext, stats_from_gram  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


arl_2of3 = _load("e7", RX / "arl" / "run.py").arl_2of3
E8 = _load("e8", RX / "normalized_weights" / "run.py")

SETS = {"mnist": ("mnist", rc.RECORD_DIRS["mnist"]), "cifar10": ("cifar10", rc.RECORD_DIRS["cifar10"]),
        "camelyon17_patch": ("camelyon17", rc.RECORD_DIRS["camelyon17"]),
        "camelyon17_slide": ("camelyon17", RX / "camelyon_slide_split" / "cache" / "camelyon17"),
        "census_n250": ("folktables", RX / "census" / "records"),
        "census_n1000": ("folktables", RX / "census" / "records_n1000")}
RULES = ["pooled", "encmon_joint", "encmon_norm", "localany", "plocal_joint", "plocal_sidak", "hybrid"]
CLUSTER_REPS = 2000


def score(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """rx_common scoring + EncMonitor-norm (normalized_weights) + encrypted Local-any (i)/(ii) + hybrid, one partition seed."""
    sc, info = E8.add_norm(g)
    cal = g[g.condition == "clean_calibration"]
    E_cal = np.stack(cal.site_energy.to_numpy())
    S = E_cal.shape[1]
    tau_s, tau_lm = info["tau_s"], info["tau_lm"]
    a_sidak = 1 - (1 - 0.05) ** (1 / S)
    t_sidak = np.array([rc.qh(E_cal[:, s], 1 - a_sidak) for s in range(S)])
    hyb_cal = np.maximum(cal.d_mix.to_numpy() / info["t_mix"], (E_cal / tau_s).max(axis=1))
    c_h = rc.qh(hyb_cal, 0.95)
    E = np.stack(sc.site_energy.to_numpy())
    bits_joint = E > tau_lm * tau_s                      # site bits, threshold (i)
    bits_sidak = E > t_sidak                             # site bits, threshold (ii)
    bits_h = E > c_h * tau_s                             # hybrid site bits
    sc["K_joint"], sc["K_sidak"], sc["K_hybrid"] = bits_joint.sum(1), bits_sidak.sum(1), bits_h.sum(1)
    sc["plocal_joint"] = sc.K_joint >= 1
    sc["plocal_sidak"] = sc.K_sidak >= 1
    sc["hybrid"] = (sc.K_hybrid >= 1) | (sc.d_mix.to_numpy() > c_h * info["t_mix"])
    sc["hybrid_score"] = np.maximum(sc.d_mix.to_numpy() / info["t_mix"], (E / tau_s).max(axis=1))
    info.update({"t_sidak": t_sidak, "alpha_sidak": a_sidak, "c_h": c_h})
    return sc, info


def masks_for(cond: pd.DataFrame, key: str) -> dict:
    if key == "folktables":
        inj = cond.family.isin(["label_prior", "age_mix"])
        return {"A1_natural_opposing_mixed": inj & cond.geometry.isin(rc.OPP_MIX),
                "A4_single": inj & (cond.geometry == "single"), "A5_coherent": inj & (cond.geometry == "coherent"),
                "year_all": (cond.family == "year") & (cond.geometry == "all"),
                "year_one_or_two": (cond.family == "year") & cond.geometry.isin(["single", "coherent"])}
    m = rc.aggregate_masks(cond)
    return {k: m[k] for k in ("A1_natural_opposing_mixed", "A4_single", "A5_coherent")}


def records_part(rng, checks):
    sum_rows, cond_rows, fpr_rows = [], [], []
    e1 = pd.read_csv(RX / "equal_fpr" / "condition_full.csv")
    e8 = pd.read_csv(RX / "normalized_weights" / "records_summary.csv").set_index("set")
    for name, (key, d) in SETS.items():
        rc.RECORD_DIRS[key] = d
        df = rc.load(key)
        sc = pd.concat([score(g)[0] for _, g in df.groupby("partition_seed")], ignore_index=True)
        seeds = sorted(sc.partition_seed.unique())
        checks[f"{name}: plocal_joint == localany (all windows)"] = bool((sc.plocal_joint == sc.localany).all())
        held = sc[sc.condition == "clean_heldout"]
        for k in RULES:
            per = held.groupby("partition_seed")[k].mean()
            p = float(held[k].mean())
            fpr_rows.append({"set": name, "method": k, "fpr": p, "fpr_seed_min": float(per.min()), "fpr_seed_max": float(per.max()),
                             "arl0_2of3": arl_2of3(p), "arl0_2of3_seed_min": min(arl_2of3(x) for x in per),
                             "arl0_2of3_seed_max": max(arl_2of3(x) for x in per)})
        sh = sc[~sc.condition.str.startswith("clean")]
        per_seed = sh.groupby(["partition_seed", "condition", "family", "geometry", "severity"])[RULES].mean().reset_index()
        cond = per_seed.groupby(["condition", "family", "geometry", "severity"])[RULES].mean().reset_index()
        cond.insert(0, "set", name)
        cond_rows.append(cond)
        r = {"set": name}
        for agg, mk in masks_for(cond, key).items():
            if not mk.any():
                continue
            cs = set(cond[mk].condition)
            ps = per_seed[per_seed.condition.isin(cs)]
            mat = {k: ps.pivot(index="partition_seed", columns="condition", values=k).loc[seeds].to_numpy() for k in RULES}
            r[f"{agg}_conditions"] = int(mk.sum())
            for k in RULES:
                r[f"{agg}_{k}"] = float(cond[mk][k].mean())
            if agg == "A1_natural_opposing_mixed":
                for k in RULES:
                    r[f"{agg}_{k}_lo"], r[f"{agg}_{k}_hi"] = rc.cluster_boot(mat[k], CLUSTER_REPS, rng)
                for k in ("plocal_joint", "plocal_sidak", "hybrid"):
                    diff = mat[k] - mat["encmon_joint"]
                    r[f"{agg}_{k}_minus_encmon"] = float(diff.mean())
                    r[f"{agg}_{k}_minus_encmon_lo"], r[f"{agg}_{k}_minus_encmon_hi"] = rc.cluster_boot(diff, CLUSTER_REPS, rng)
        sum_rows.append(r)
        # reproduction against normalized_weights/records_summary.csv where the set exists there
        e8name = {"camelyon17_patch": "camelyon17_patch", "camelyon17_slide": "camelyon17_slide", "census_n250": "folktables"}.get(name, name)
        if e8name in e8.index:
            for k in ("pooled", "encmon_joint", "encmon_norm", "localany"):
                checks[f"{name}: A1 {k} vs E8"] = abs(r["A1_natural_opposing_mixed_" + k] - e8.loc[e8name, "A1_natural_opposing_mixed_" + k])
                checks[f"{name}: FPR {k} vs E8"] = abs(float(held[k].mean()) - e8.loc[e8name, "fpr_" + k])
        if name in ("mnist", "cifar10"):
            sub = e1[e1.dataset == name].set_index("condition")
            m = cond.set_index("condition")
            for k, col in (("pooled", "tpr_pooled"), ("encmon_joint", "tpr_encmon_joint"), ("localany", "tpr_localany")):
                checks[f"{name}: per-condition {k} vs E1 (max abs)"] = float((m[k] - sub.loc[m.index, col]).abs().max())
        print(f"[records {name}] done", flush=True)
    return pd.DataFrame(sum_rows), pd.concat(cond_rows, ignore_index=True), pd.DataFrame(fpr_rows)


def e3_part(checks):
    rows = []
    rc.RECORD_DIRS["camelyon17"] = SETS["camelyon17_patch"][1]
    for ds in ("mnist", "cifar10", "camelyon17"):
        rec = rc.load(ds)
        for seed, saved in rec.groupby("partition_seed"):
            ctx = SeedContext(ds, seed)
            new = []
            for fam, sevs in E8.E3_FAMILIES.items():
                for sev in sevs:
                    for geo, dirs in E8.E3_GEOMETRIES.items():
                        c = {"condition": f"{fam}{sev:g}__{geo}", "family": fam, "kind": fam, "severity": float(sev),
                             "geometry": geo, "directions": dirs}
                        st = stats_from_gram(ctx.run(ctx.condition_feats(c), 50, seed + 2000)["gram"], ctx.alpha)
                        for w in range(50):
                            new.append({"condition": c["condition"], "family": fam, "severity": float(sev), "geometry": geo,
                                        "window": w, "d_mix": st["d_mix"][w], "d_site": st["d_site"][w],
                                        "heterogeneity": st["heterogeneity"][w], "site_energy": st["site_energy"][w].tolist(),
                                        "contributions": st["contributions"][w].tolist(), "exact_mmd": np.nan, "partition_seed": seed})
            base = saved[saved.condition.isin(["clean_calibration", "clean_heldout"])]
            sc, _ = score(pd.concat([base, pd.DataFrame(new)], ignore_index=True))
            for cn, g in sc[~sc.condition.str.startswith("clean")].groupby("condition", sort=False):
                r = {"dataset": ds, "partition_seed": seed, "condition": cn, "family": g.family.iloc[0],
                     "severity": g.severity.iloc[0], "geometry": g.geometry.iloc[0]}
                r.update({k: float(g[k].mean()) for k in RULES})
                rows.append(r)
            print(f"[E3 {ds} {seed}] done", flush=True)
    per = pd.DataFrame(rows)
    out = per.groupby(["dataset", "family", "severity", "geometry"], sort=False)[RULES].mean().reset_index()
    ref = pd.read_csv(RX / "normalized_weights" / "e3_windows.csv")
    mm = out.merge(ref, on=["dataset", "family", "severity", "geometry"], suffixes=("", "_e8"))
    for k in ("pooled", "encmon_joint", "encmon_norm", "localany"):
        checks[f"E3 {k} vs E8 (max abs)"] = float((mm[k] - mm[k + "_e8"]).abs().max())
    return out


def leakage_part(checks):
    rows = []
    e9 = pd.read_csv(RX / "l1_leakage" / "l1_leakage_per_seed.csv")
    rc.RECORD_DIRS["camelyon17"] = SETS["camelyon17_patch"][1]
    for ds in ("mnist", "cifar10", "camelyon17"):
        rec = rc.load(ds)
        for seed, g in rec.groupby("partition_seed"):
            _, info = score(g)
            ctx = SeedContext(ds, int(seed))
            tau_s, tau_lm, t_sid, c_h = info["tau_s"], info["tau_lm"], info["t_sidak"], info["c_h"]
            for fam, sev in (("vector", 1.0), ("label_prior", 0.35)):
                F = {k: [] for k in ("enc_L1", "enc_L2", "pl_L1", "pls_L1", "K", "hyb_L1", "hyb_auth")}
                y, top_enc, top_loc, al_enc, al_loc = [], [], [], [], []
                for s in range(ctx.S):
                    dirs = [0.0] * ctx.S
                    dirs[s] = 1.0
                    c = {"kind": fam, "family": fam, "severity": sev, "directions": dirs}
                    st = stats_from_gram(ctx.run(ctx.condition_feats(c), 50, int(seed) + 6000 + s)["gram"], ctx.alpha)
                    E = st["site_energy"]
                    enc = np.maximum(st["d_mix"] / info["t_mix"], st["d_site"] / info["t_site"]) > info["c_joint"]
                    K = (E > tau_lm * tau_s).sum(1)
                    Kh = (E > c_h * tau_s).sum(1)
                    hyb = (Kh >= 1) | (st["d_mix"] > c_h * info["t_mix"])
                    F["enc_L1"].append(enc.astype(int)[:, None])
                    F["enc_L2"].append(np.log(np.c_[st["d_mix"], st["d_site"], np.maximum(st["heterogeneity"], 1e-12)]))
                    F["pl_L1"].append((K >= 1).astype(int)[:, None])
                    F["pls_L1"].append(((E > t_sid).sum(1) >= 1).astype(int)[:, None])
                    F["K"].append(K[:, None])
                    F["hyb_L1"].append(hyb.astype(int)[:, None])
                    F["hyb_auth"].append(np.c_[Kh, np.log(st["d_mix"])])
                    y.append(np.full(50, s))
                    top_enc.append(np.argmax(st["contributions"], 1) == s)
                    top_loc.append(np.argmax(E / tau_s, 1) == s)
                    al_enc.append(enc)
                    al_loc.append(K >= 1)
                F = {k: np.concatenate(v) for k, v in F.items()}
                y = np.concatenate(y)
                cv = StratifiedKFold(5, shuffle=True, random_state=20261023)
                lr = lambda: LogisticRegression(max_iter=5000)
                cat = lambda: make_pipeline(OneHotEncoder(handle_unknown="ignore"), lr())
                num = lambda: make_pipeline(StandardScaler(), lr())
                r = {"dataset": ds, "partition_seed": int(seed), "family": fam, "severity": sev, "chance": 1 / ctx.S}
                for k in ("enc_L1", "pl_L1", "pls_L1", "K", "hyb_L1"):
                    r[k] = float(cross_val_score(cat(), F[k], y, cv=cv).mean())
                for k in ("enc_L2", "hyb_auth"):
                    r[k] = float(cross_val_score(num(), F[k], y, cv=cv).mean())
                te, tl, ae, alc = map(np.concatenate, (top_enc, top_loc, al_enc, al_loc))
                r.update({"alert_share_enc": float(ae.mean()), "alert_share_plocal": float(alc.mean()),
                          "post_alert_top1_enc": float(te[ae].mean()) if ae.any() else np.nan,
                          "post_alert_top1_plocal": float(tl[alc].mean()) if alc.any() else np.nan})
                rows.append(r)
            print(f"[leakage {ds} {seed}] done", flush=True)
    df = pd.DataFrame(rows)
    m = df.merge(e9[["dataset", "partition_seed", "family", "L1_binary", "L2_scalars"]], on=["dataset", "partition_seed", "family"])
    checks["leakage enc_L1 vs E9 (max abs)"] = float((m.enc_L1 - m.L1_binary).abs().max())
    checks["leakage enc_L2 vs E9 (max abs)"] = float((m.enc_L2 - m.L2_scalars).abs().max())
    return df


def main() -> int:
    rng = np.random.default_rng(20261022)
    checks: dict = {}
    summ, cond, fpr = records_part(rng, checks)
    summ.to_csv(HERE / "records_summary.csv", index=False)
    cond.to_csv(HERE / "condition_rates.csv", index=False)
    fpr.to_csv(HERE / "fpr_arl.csv", index=False)
    e3 = e3_part(checks)
    e3.to_csv(HERE / "e3_rates.csv", index=False)
    leak = leakage_part(checks)
    leak.to_csv(HERE / "leakage_per_seed.csv", index=False)
    cols = ["enc_L1", "enc_L2", "pl_L1", "pls_L1", "K", "hyb_L1", "hyb_auth", "alert_share_enc", "alert_share_plocal",
            "post_alert_top1_enc", "post_alert_top1_plocal"]
    ls = leak.groupby(["dataset", "family"])[cols].agg(["mean", "min", "max"])
    ls.to_csv(HERE / "leakage.csv")
    (HERE / "checks.json").write_text(json.dumps(checks, indent=2))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 500)
    f = lambda x: f"{x:.3f}"
    print("\n== Checks ==\n" + json.dumps(checks, indent=1))
    print("\n== Clean FPR and ARL0 (2-of-3) ==")
    print(fpr.to_string(index=False, float_format=f))
    print("\n== Aggregates ==")
    print(summ.set_index("set").T.to_string(float_format=f))
    print("\n== E3 (all-site, one-site, two-site) ==")
    print(e3.to_string(index=False, float_format=f))
    print("\n== Leakage (accuracy; chance 0.2) ==")
    print(ls.xs("mean", axis=1, level=1).to_string(float_format=f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
