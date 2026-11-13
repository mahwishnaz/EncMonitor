#!/usr/bin/env python3
"""Census experiment, step 2 (CPU): pools, RFF sketch, windows and records in the format of scripts 15/33.

Usage (bundle root): python experiments/census/windows.py > experiments/census/windows_log.txt
  Larger windows: add --window 500 --out records_n500 --only-real (and 1000 / records_n1000)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "common"))
from encmonitor.embeddings import EmbeddingStandardizer  # noqa: E402
from encmonitor.metrics import top1_accuracy, topk_recall  # noqa: E402
from encmonitor.rff import RFFMap, median_bandwidth  # noqa: E402
from encmonitor.statistics import compute_dual_statistics  # noqa: E402
from rx_windows import label_prior_idx  # noqa: E402

FIPS = [6, 48, 12, 36, 42]                      # CA, TX, FL, NY, PA = sites 0..4
YEARS = [2015, 2016, 2017, 2018]
SEEDS = [int(s) for s in yaml.safe_load(open(ROOT / "configs" / "census.yaml"))["partition_seeds"]]
S, N_POOL, n, M = 5, 2500, 250, 64
CAL, HELD, PER = 500, 200, 50
GEO = {"single": [1, 0, 0, 0, 0], "coherent": [1, 1, 0, 0, 0], "opposing": [1, -1, 0, 0, 0], "mixed": [1, -0.5, 0.75, 0, 0],
       "all": [1, 1, 1, 1, 1]}
SEV = [0.1, 0.2, 0.35, 0.5]
VSEV = [0.25, 0.5, 1.0, 2.0]


def household_sample(serial: np.ndarray, rows: np.ndarray, size: int, rng) -> np.ndarray:
    """Rows of whole households in random order until `size` rows (last household truncated)."""
    hh = np.unique(serial[rows])
    rng.shuffle(hh)
    order = np.argsort(serial[rows], kind="stable")
    srt = serial[rows][order]
    out, total = [], 0
    for h in hh:
        lo, hi = np.searchsorted(srt, h, "left"), np.searchsorted(srt, h, "right")
        take = rows[order[lo:hi]]
        out.append(take)
        total += take.size
        if total >= size:
            break
    return np.concatenate(out)[:size]


def main() -> int:
    import argparse
    global n
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--out", default="records")
    ap.add_argument("--only-real", action="store_true", help="clean, year, label_prior/age_mix opposing+mixed only")
    ap.add_argument("--seeds", default="", help="comma-separated subset of the partition seeds (default: all in configs/census.yaml)")
    args = ap.parse_args()
    n = args.window
    data = {y: dict(np.load(HERE / "cache" / f"acs_{y}.npz", allow_pickle=True)) for y in [2014] + YEARS}
    W = __import__("torch").load(HERE / "cache" / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    fcW, fcb = W["fc.weight"].double().numpy(), W["fc.bias"].double().numpy()
    conds = []
    for y in YEARS:
        for g in ("single", "coherent", "all"):
            conds.append({"condition": f"year{y}__{g}", "family": "year", "kind": "year", "severity": float(y - 2014), "year": y, "geometry": g})
    for fam in ("label_prior", "age_mix"):
        for q in SEV:
            for g in ("single", "coherent", "opposing", "mixed"):
                conds.append({"condition": f"{fam}{q:g}__{g}", "family": fam, "kind": fam, "severity": q, "geometry": g})
    for e in VSEV:
        for g in ("single", "coherent", "opposing", "mixed"):
            conds.append({"condition": f"vector{e:g}__{g}", "family": "vector", "kind": "vector", "severity": e, "geometry": g})
    for c in conds:
        c["directions"] = [float(v) for v in GEO[c["geometry"]]]
    if args.only_real:
        conds = [c for c in conds if c["family"] == "year" or (c["family"] in ("label_prior", "age_mix") and c["geometry"] in ("opposing", "mixed"))]

    out_root = HERE / args.out
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or SEEDS
    for seed in seeds:
        d14 = data[2014]
        pools = {}
        for s, st in enumerate(FIPS):
            rows = np.flatnonzero((d14["st"] == st) & ~d14["train"])
            rng = np.random.default_rng([seed, s, 41])
            both = household_sample(d14["serial"], rows, 2 * N_POOL, rng)
            # reference = first households, monitoring = the rest (households never split: truncation only at the end)
            cut_h = d14["serial"][both[N_POOL - 1]]
            ref = both[:N_POOL]
            mon = both[N_POOL:]
            mon = mon[d14["serial"][mon] != cut_h]          # drop the household that straddles the cut
            pools[(s, "ref")], pools[(s, 2014)] = ref, mon
            for y in YEARS:
                dy = data[y]
                rows_y = np.flatnonzero(dy["st"] == st)
                pools[(s, y)] = household_sample(dy["serial"], rows_y, N_POOL, np.random.default_rng([seed, s, y]))
        ref_raw = [d14["rep"][pools[(s, "ref")]].astype(np.float64) for s in range(S)]
        x = np.concatenate(ref_raw)
        prep = EmbeddingStandardizer.fit(x, clip_radius=8.0)
        sigma = median_bandwidth(prep.transform(x), max_samples=2000, seed=1729)
        rff = RFFMap.fit(x.shape[1], M, sigma, 1729)
        base = np.stack([rff.mean_sketch(prep.transform(r)) for r in ref_raw])
        mon_raw = [d14["rep"][pools[(s, 2014)]].astype(np.float64) for s in range(S)]
        mon_lab = [d14["label"][pools[(s, 2014)]] for s in range(S)]
        mon_age = [(d14["agep"][pools[(s, 2014)]] >= 45).astype(int) for s in range(S)]
        clean_acc = [float(((r @ fcW.T + fcb).argmax(1) == l).mean()) for r, l in zip(mon_raw, mon_lab)]
        v = np.random.default_rng([seed, 7]).standard_normal(x.shape[1]); v /= np.linalg.norm(v)
        weights = np.ones(S) / S

        def pool(raw, lab):
            return rff.transform(prep.transform(raw)), ((raw @ fcW.T + fcb).argmax(1) == lab)

        clean = [pool(mon_raw[s], mon_lab[s]) for s in range(S)]
        rows_out = []

        def run(pls, W_, wseed, meta):
            rng = np.random.default_rng(wseed)
            dirs = np.asarray(meta.get("directions", [0.0] * S)); mask = dirs != 0
            for w in range(W_):
                means, acc = [], []
                for s in range(S):
                    f, corr = pls[s]
                    idx = rng.choice(f.shape[0], n, replace=f.shape[0] < n)
                    means.append(f[idx].mean(0)); acc.append(float(corr[idx].mean()))
                res = np.stack(means) - base
                stt = compute_dual_statistics(res, weights)
                r = {k: meta[k] for k in ("condition", "family", "severity", "geometry")}
                r.update({"window": w, "window_seed": wseed, "d_mix": stt.d_mix, "d_site": stt.d_site,
                          "heterogeneity": stt.heterogeneity, "contributions": stt.contributions.tolist(),
                          "site_energy": np.einsum("sm,sm->s", res, res).tolist(), "shifted_mask": mask.tolist(),
                          "exact_mmd": None, "acc": acc, "partition_seed": seed})
                if mask.any():
                    r["top1"] = top1_accuracy(stt.contributions, mask)
                    r["topk_recall"] = topk_recall(stt.contributions, mask)
                rows_out.append(r)

        run(clean, CAL, seed, {"condition": "clean_calibration", "family": "clean", "severity": 0.0, "geometry": "clean"})
        run(clean, HELD, seed + 1000, {"condition": "clean_heldout", "family": "clean", "severity": 0.0, "geometry": "clean"})
        for c in conds:
            pls = []
            for s in range(S):
                f = c["directions"][s]
                if f == 0:
                    pls.append(clean[s]); continue
                if c["kind"] == "year":
                    dy = data[c["year"]]
                    pls.append(pool(dy["rep"][pools[(s, c["year"])]].astype(np.float64), dy["label"][pools[(s, c["year"])]]))
                elif c["kind"] in ("label_prior", "age_mix"):
                    q = c["severity"] * abs(f)
                    groups = mon_lab[s] if c["kind"] == "label_prior" else mon_age[s]
                    tag = 3 if c["kind"] == "label_prior" else 8
                    idx = label_prior_idx(groups, 1, q, np.sign(f), np.random.default_rng([seed, tag, s, int(round(q * 1000)), int(f > 0)]))
                    pls.append(pool(mon_raw[s][idx], mon_lab[s][idx]))
                else:
                    pls.append(pool(mon_raw[s] + f * c["severity"] * v * prep.scale, mon_lab[s]))
            run(pls, PER, seed + 2000, c)
        d = out_root / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "records.jsonl", "w") as fh:
            for r in rows_out:
                fh.write(json.dumps(r) + "\n")
        info = {"partition_seed": seed, "clean_accuracy_per_site": clean_acc, "bandwidth": sigma,
                "pool_sizes": {f"{s}_{k}": int(v_.size) for (s, k), v_ in pools.items()},
                "ref_positive_rate": [float(d14["label"][pools[(s, 'ref')]].mean()) for s in range(S)]}
        info["positive_rate"] = {str(y): [float((data[y]["label"] if y != 2014 else d14["label"])[pools[(s, y)]].mean()) for s in range(S)] for y in [2014] + YEARS}
        (d / "conditions.json").write_text(json.dumps({**info, "conditions": conds}, indent=2, default=str))
        print(f"[seed {seed}] {len(rows_out)} windows, clean acc {np.round(clean_acc, 3).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
