#!/usr/bin/env python3
"""Compare a re-run of partition seed 2026 (scripts/reproduce_seed2026.sh) with the saved records of seed 2026.

Acceptance rule (fixed in advance):
  identical window indices, every statistic within a relative difference of 1e-6, identical alert decisions.

Window indices: the window runners draw idx = default_rng(window_seed).choice(N_s, n) per site and window, so the
indices are identical iff the window seeds, the order of draws and the pool sizes N_s are identical. Checked:
manifests (pool membership and order), pool sizes of every condition (pieces, clean pools), and the
(condition, window, window_seed) sequence of the records. In addition the index arrays of the clean windows are
regenerated from both caches and compared.
Relative difference of a value pair (a, b): |a - b| / max(|a|, |b|), 0 if both are exactly 0.
Decisions: every alert column of private_localany/run.py `score` (+ matched-FPR alerts), both runs scored with the
same code.

Saved records: outputs/main/{mnist,cifar10}, outputs/camelyon/main/camelyon17,
experiments/camelyon_slide_split/cache/camelyon17, experiments/census/records and records_n1000.
Usage: python scripts/compare_seed2026.py   (after scripts/reproduce_seed2026.sh)
Writes outputs/seed2026_check/{seed2026_check.json, seed2026_check_decisions.csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLONE = Path(__file__).resolve().parents[1]   # repository root
NEW = CLONE / "outputs/seed2026_check"
sys.path.insert(0, str(CLONE / "experiments/common"))
sys.path.insert(0, str(CLONE))
import importlib.util  # noqa: E402

import rx_common as rc  # noqa: E402

spec = importlib.util.spec_from_file_location("p1", CLONE / "experiments/private_localany/run.py")
P1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P1)

SETS = {  # name: (key, saved dir, new dir)
    "mnist": ("mnist", CLONE / "outputs/main/mnist", NEW / "main/mnist"),
    "cifar10": ("cifar10", CLONE / "outputs/main/cifar10", NEW / "main/cifar10"),
    "camelyon17_patch": ("camelyon17", CLONE / "outputs/camelyon/main/camelyon17", NEW / "camelyon_patch"),
    "camelyon17_slide": ("camelyon17", CLONE / "experiments/camelyon_slide_split/cache/camelyon17", NEW / "camelyon_slide"),
    "census_n250": ("folktables", CLONE / "experiments/census/records", NEW / "census_n250"),
    "census_n1000": ("folktables", CLONE / "experiments/census/records_n1000", NEW / "census_n1000"),
}
STAT_FIELDS = ["d_mix", "d_site", "heterogeneity", "contributions", "directional", "site_energy", "exact_mmd",
               "d_mix_global", "d_site_global", "heterogeneity_global", "acc", "top1", "topk_recall", "ndcg", "spearman"]
DECISIONS = ["pooled", "site", "H", "encmon_or", "encmon_joint", "localany", "exact", "majority", "hidden",
             "encmon_norm", "plocal_joint", "plocal_sidak", "hybrid"]
MATCHED = {"pooled": "d_mix", "encmon_joint": "encmon_score", "localany": "localmax_score", "hybrid": "hybrid_score"}
LABEL = {"pooled": "Pooled", "encmon_joint": "EncMonitor", "localany": "Local-any", "hybrid": "Hybrid", "exact": "Exact-MMD"}
TOL = 1e-6


def reldiff(a, b) -> tuple[float, float]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    both_nan = np.isnan(a) & np.isnan(b)
    a, b = a[~both_nan], b[~both_nan]
    if a.size == 0:
        return 0.0, 0.0
    d = np.abs(a - b)
    den = np.maximum(np.abs(a), np.abs(b))
    r = np.where(den > 0, d / np.where(den > 0, den, 1), 0.0)
    r[np.isnan(d)] = np.inf
    return float(np.max(d)), float(np.max(r))


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in open(path)]


def npz_equal(p: Path, q: Path) -> tuple[bool, float]:
    a, b = np.load(p), np.load(q)
    if set(a.files) != set(b.files):
        return False, np.inf
    same, mx = True, 0.0
    for k in a.files:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            return False, np.inf
        if x.dtype.kind in "fc":
            mx = max(mx, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0)
            same &= bool(np.array_equal(x, y))
        else:
            same &= bool(np.array_equal(x, y))
    return same, mx


def score(rows: list[dict]) -> pd.DataFrame:
    g = pd.DataFrame(rows)
    sc, _ = P1.score(g)
    held = sc[sc.condition == "clean_heldout"]
    for k, col in MATCHED.items():
        sc[f"m_{k}"] = sc[col] > rc.qh(held[col], 0.95)
    return sc


def main() -> int:
    out, det = {}, []
    for name, (key, old_root, new_root) in SETS.items():
        r = {"set": name}
        od, nd = old_root / "seed_2026", new_root / "seed_2026"
        # pools
        if (od / "manifest.npz").exists():
            r["manifest_identical"] = npz_equal(od / "manifest.npz", nd / "manifest.npz")[0]
            for part in ("baseline_clean", "monitor_clean"):
                same, mx = npz_equal(od / f"{part}.npz", nd / f"{part}.npz")
                r[f"{part}_bit_identical"], r[f"{part}_max_abs_diff"] = same, mx
            po, pn = np.load(od / "pieces.npz"), np.load(nd / "pieces.npz")
            r["pieces_same_keys_and_shapes"] = set(po.files) == set(pn.files) and all(po[k].shape == pn[k].shape for k in po.files)
            r["pieces_bit_identical"] = r["pieces_same_keys_and_shapes"] and all(np.array_equal(po[k], pn[k]) for k in po.files)
            r["pieces_max_abs_diff"] = max(float(np.max(np.abs(po[k].astype(np.float64) - pn[k].astype(np.float64)))) for k in po.files)
            # regenerate the clean-window index arrays from both caches' pool sizes
            mo, mn = np.load(od / "monitor_clean.npz"), np.load(nd / "monitor_clean.npz")
            S = sum(1 for k in mo.files if k.startswith("site_"))
            Ns_o = [mo[f"site_{s:03d}"].shape[0] for s in range(S)]
            Ns_n = [mn[f"site_{s:03d}"].shape[0] for s in range(S)]
            same_idx = Ns_o == Ns_n
            for wseed, W in ((2026, 500), (3026, 200)):
                ro, rn = np.random.default_rng(wseed), np.random.default_rng(wseed)
                for _ in range(W):
                    for s in range(S):
                        a = ro.choice(Ns_o[s], 250, replace=Ns_o[s] < 250)
                        b = rn.choice(Ns_n[s], 250, replace=Ns_n[s] < 250)
                        same_idx &= bool(np.array_equal(a, b))
            r["clean_window_indices_identical"] = bool(same_idx)
        else:   # census: pools are not stored; compare the pool sizes and positive rates of conditions.json
            co, cn = json.load(open(od / "conditions.json")), json.load(open(nd / "conditions.json"))
            r["pool_sizes_identical"] = co["pool_sizes"] == cn["pool_sizes"]
            r["positive_rates_identical"] = co["positive_rate"] == cn["positive_rate"]
            r["bandwidth_identical"] = co["bandwidth"] == cn["bandwidth"]
        ro_, rn_ = load(od / "records.jsonl"), load(nd / "records.jsonl")
        r["windows_saved"], r["windows_new"] = len(ro_), len(rn_)
        seq_o = [(x["condition"], x["window"], x["window_seed"]) for x in ro_]
        seq_n = [(x["condition"], x["window"], x["window_seed"]) for x in rn_]
        r["window_sequence_identical"] = seq_o == seq_n
        stat = {}
        for f in STAT_FIELDS:
            pairs = [(a[f], b[f]) for a, b in zip(ro_, rn_) if f in a and a[f] is not None]
            if not pairs:
                continue
            missing = sum(1 for a, b in zip(ro_, rn_) if (f in a) != (f in b))
            mabs, mrel = 0.0, 0.0
            for a, b in pairs:
                x, y = reldiff(a, b)
                mabs, mrel = max(mabs, x), max(mrel, y)
            stat[f] = {"max_abs": mabs, "max_rel": mrel, "n": len(pairs), "missing_mismatch": missing}
        r["stats"] = stat
        r["max_rel_all_stats"] = max(v["max_rel"] for v in stat.values())
        # decisions
        so, sn = score(ro_), score(rn_)
        cols = [c for c in DECISIONS if c in so.columns] + [f"m_{k}" for k in MATCHED]
        if key == "folktables":
            cols = [c for c in cols if c != "exact"]
        for c in cols:
            flips = int((so[c].to_numpy() != sn[c].to_numpy()).sum())
            det.append({"set": name, "decision": c, "windows": len(so), "flips": flips,
                        "alert_rate_saved": float(so[c].mean()), "alert_rate_new": float(sn[c].mean())})
        r["decision_flips_total"] = int(sum(d["flips"] for d in det if d["set"] == name))
        r["pass"] = bool(r["window_sequence_identical"] and r.get("manifest_identical", True)
                         and r.get("clean_window_indices_identical", True) and r.get("pool_sizes_identical", True)
                         and r["max_rel_all_stats"] <= TOL and r["decision_flips_total"] == 0)
        out[name] = r
        print(f"[{name}] pass={r['pass']} max_rel={r['max_rel_all_stats']:.2e} flips={r['decision_flips_total']}", flush=True)
    (NEW / "seed2026_check.json").write_text(json.dumps(out, indent=1, default=str))
    pd.DataFrame(det).to_csv(NEW / "seed2026_check_decisions.csv", index=False)
    print("ALL PASS" if all(v["pass"] for v in out.values()) else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
