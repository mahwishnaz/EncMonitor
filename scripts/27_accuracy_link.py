#!/usr/bin/env python3
"""Do EncMonitor alerts track accuracy loss?

No new data, model training or encryption. For every window of the main experiment this script rebuilds
exactly the same per-site sample indices as 15_run_screening.py (same seeds, same random calls), and adds the
frozen model's accuracy on those samples. Predictions come from the model's own final linear layer
applied to the cached penultimate embeddings (LeNet: `classifier`, ResNet-18: `backbone.fc`), so
no image is processed again. Labels are the cached test labels of each site's monitoring pool.

Check: the recomputed D_mix and D_site must equal the saved records (printed as max difference).

Per window and site: accuracy drop = clean accuracy of that site's whole monitoring pool minus
the accuracy in the window. A window is "harmful" when at least one site loses >= --harm
(default 0.05, i.e. 5 percentage points).

Usage (from the repository root):
  python scripts/27_accuracy_link.py --dataset mnist   --cache-dir outputs/main/mnist
  python scripts/27_accuracy_link.py --dataset cifar10 --cache-dir outputs/main/cifar10
  (add --checkpoint PATH if the path stored in cache_meta.json has moved)
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

from encmonitor.embeddings import EmbeddingStandardizer  # noqa: E402
from encmonitor.rff import RFFMap  # noqa: E402
from encmonitor.statistics import compute_dual_statistics  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label_prior_idx(labels, target, q, sign, rng):
    """Same random calls as label_prior_pool() in 15_run_screening.py, but returns the indices."""
    classes = np.unique(labels)
    p = np.array([np.mean(labels == c) for c in classes])
    onehot = (classes == target).astype(float)
    if sign > 0:
        p_new = (1 - q) * p + q * onehot
    else:
        rest = p * (1 - onehot)
        p_new = (1 - q) * p + q * rest / rest.sum()
    counts = rng.multinomial(labels.size, p_new)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), k, replace=True)
                          for c, k in zip(classes, counts) if k > 0])
    rng.shuffle(idx)
    return idx


def head_weights(checkpoint: str):
    import torch
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    for k in ("classifier", "backbone.fc", "fc"):
        if f"{k}.weight" in sd:
            return sd[f"{k}.weight"].double().numpy(), sd[f"{k}.bias"].double().numpy()
    raise KeyError("final linear layer not found in checkpoint")


def qh(v, q):
    return float(np.quantile(np.asarray(v, dtype=float), q, method="higher"))


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(spearmanr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Accuracy loss of the frozen model on the exact main-experiment windows")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--harm", type=float, default=0.05, help="accuracy drop at one site that counts as harmful")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    scr = _load("15_run_screening")
    cfg = yaml.safe_load(open(args.config))
    n, S = int(cfg["window_size"]), int(cfg["sites"])
    weights = np.ones(S) / S
    conds = scr.conditions(cfg)
    cache = Path(args.cache_dir)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [int(s) for s in cfg["partition_seeds"]]
    ckpt = args.checkpoint or json.load(open(cache / f"seed_{seeds[0]}" / "cache_meta.json"))["checkpoint"]
    W, b = head_weights(ckpt)
    predict = lambda e: np.argmax(e @ W.T + b, axis=1)

    rows, check = [], []
    for seed in seeds:
        d = cache / f"seed_{seed}"
        prep = EmbeddingStandardizer.load(d / "preprocessor.npz")
        rff = RFFMap.load(str(d / "rff_map.npz"))
        base_z = np.load(d / "baselines.npz")
        base = [np.asarray(base_z[f"site_{s:03d}"], dtype=np.float64) for s in range(S)]
        mc = np.load(d / "monitor_clean.npz")
        pieces = np.load(d / "pieces.npz")
        clean_raw = [mc[f"site_{s:03d}"].astype(np.float64) for s in range(S)]
        labels = [mc[f"labels_{s:03d}"] for s in range(S)]
        clean_pred = [predict(x) for x in clean_raw]
        clean_acc = np.array([np.mean(clean_pred[s] == labels[s]) for s in range(S)])
        feat = lambda raw: rff.transform(prep.transform(raw))
        clean_f = [feat(x) for x in clean_raw]
        classes = np.unique(np.concatenate(labels))
        target = int(classes[np.argmax([min(np.sum(labels[s] == c) for s in range(3)) for c in classes])])
        v = np.random.default_rng([seed, 7]).standard_normal(clean_raw[0].shape[1])
        v /= np.linalg.norm(v)
        rec = pd.read_json(d / "records.jsonl", lines=True).set_index(["condition", "window"])

        def run(feats, correct, wseed, nwin, meta):
            rng = np.random.default_rng(wseed)
            for w in range(nwin):
                res, acc = [], np.zeros(S)
                for s in range(S):
                    N = feats[s].shape[0]
                    idx = rng.choice(N, n, replace=N < n)
                    res.append(feats[s][idx].mean(axis=0) - base[s])
                    acc[s] = correct[s][idx].mean()
                st = compute_dual_statistics(np.stack(res), weights)
                r0 = rec.loc[(meta["condition"], w)]
                check.append(max(abs(st.d_mix - r0.d_mix), abs(st.d_site - r0.d_site)))
                drop = clean_acc - acc
                rows.append({"partition_seed": seed, "condition": meta["condition"], "family": meta["family"],
                             "severity": meta["severity"], "geometry": meta["geometry"], "window": w,
                             "d_mix": st.d_mix, "d_site": st.d_site, "heterogeneity": st.heterogeneity,
                             "contributions": st.contributions.tolist(), "acc": acc.tolist(), "site_drop": drop.tolist(),
                             "shifted_mask": list(np.asarray(meta.get("directions", [0] * S)) != 0)})

        correct_clean = [clean_pred[s] == labels[s] for s in range(S)]
        clean_meta = {"family": "clean", "severity": 0.0, "geometry": "clean"}
        run(clean_f, correct_clean, seed, int(cfg["calibration_windows"]), {"condition": "clean_calibration", **clean_meta})
        run(clean_f, correct_clean, seed + 1000, int(cfg["heldout_windows"]), {"condition": "clean_heldout", **clean_meta})
        for c in conds:
            feats, correct = [], []
            for s in range(S):
                f = c["directions"][s]
                if f == 0:
                    feats.append(clean_f[s]); correct.append(correct_clean[s]); continue
                if c["kind"] == "image":
                    raw = pieces[f"{c['family']}|{c['severity']:g}|{f:+g}|site_{s:03d}"].astype(np.float64)
                    lab = labels[s]
                elif c["kind"] == "label_prior":
                    q = c["severity"] * abs(f)
                    rng = np.random.default_rng([seed, 3, s, int(round(q * 1000)), int(f > 0)])
                    idx = label_prior_idx(labels[s], target, q, np.sign(f), rng)
                    raw, lab = clean_raw[s][idx], labels[s][idx]
                elif c["kind"] == "vector":
                    raw, lab = clean_raw[s] + f * c["severity"] * v * prep.scale, labels[s]
                else:
                    raise ValueError(c["kind"])
                feats.append(feat(raw)); correct.append(predict(raw) == lab)
            run(feats, correct, seed + 2000, int(cfg["windows_per_condition"]), c)
        print(f"[{args.dataset} seed {seed}] clean pool accuracy per site: {np.round(clean_acc, 4).tolist()}")

    df = pd.DataFrame(rows)
    print(f"check vs saved records: max |difference| in D_mix/D_site = {max(check):.3e} (must be ~0)")

    # thresholds per seed, as in 16_analyze_screening.py
    parts = []
    for seed, g in df.groupby("partition_seed"):
        cal = g[g.condition == "clean_calibration"]
        tm, ts = qh(cal.d_mix, args.quantile), qh(cal.d_site, args.quantile)
        h = g[g.condition != "clean_calibration"].copy()
        h["pooled"], h["encmon"] = h.d_mix > tm, (h.d_mix > tm) | (h.d_site > ts)
        parts.append(h)
    sc = pd.concat(parts, ignore_index=True)
    D = np.stack(sc["site_drop"].to_numpy()); A = np.stack(sc.contributions.to_numpy())
    sc["max_drop"] = D.max(axis=1)
    sc["mean_drop"] = D.mean(axis=1)                      # = drop of the fixed-weight pooled population
    sc["harmful"] = sc.max_drop >= args.harm
    sc["top1_A_is_worst_site"] = A.argmax(axis=1) == D.argmax(axis=1)
    sh = sc[sc.condition != "clean_heldout"]
    cl = sc[sc.condition == "clean_heldout"]

    summ = {
        "dataset": args.dataset, "harm_threshold": args.harm, "shifted_windows": int(len(sh)),
        "clean_heldout_harmful_rate": float(cl.harmful.mean()),
        "spearman_Dsite_vs_maxdrop": spearman(sh.d_site, sh.max_drop),
        "spearman_Dmix_vs_maxdrop": spearman(sh.d_mix, sh.max_drop),
        "spearman_Dmix_vs_pooled_drop": spearman(sh.d_mix, sh.mean_drop),
        "spearman_As_vs_site_drop": spearman(A[sh.index].ravel(), D[sh.index].ravel()),
        "harmful_windows": int(sh.harmful.sum()),
        "harmful_detected_pooled": float(sh[sh.harmful].pooled.mean()),
        "harmful_detected_encmon": float(sh[sh.harmful].encmon.mean()),
        "encmon_alerts_that_are_harmful": float(sh[sh.encmon].harmful.mean()),
        "hidden_windows_harmful": float(sh[sh.encmon & ~sh.pooled].harmful.mean()),
        "top1_A_is_worst_site_in_harmful_alerts": float(sh[sh.harmful & sh.encmon].top1_A_is_worst_site.mean()),
    }
    om = sh[sh.geometry.isin(["opposing", "mixed"])]
    summ.update({
        "opp_mixed_harmful_windows": int(om.harmful.sum()),
        "opp_mixed_harmful_detected_pooled": float(om[om.harmful].pooled.mean()),
        "opp_mixed_harmful_detected_encmon": float(om[om.harmful].encmon.mean()),
        "opp_mixed_mean_pooled_drop": float(om.mean_drop.mean()),
        "opp_mixed_mean_max_site_drop": float(om.max_drop.mean()),
    })
    by_fam = (sh.groupby(["family", "geometry"])
                .agg(windows=("harmful", "size"), mean_max_site_drop=("max_drop", "mean"),
                     mean_pooled_drop=("mean_drop", "mean"), harmful_rate=("harmful", "mean"),
                     detected_pooled=("pooled", "mean"), detected_encmon=("encmon", "mean"))
                .reset_index())

    out = Path(args.out_dir or f"outputs/main/analysis/accuracy_link/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    sc.drop(columns=["contributions", "acc", "site_drop", "shifted_mask"]).to_csv(out / "windows_accuracy.csv", index=False)
    by_fam.to_csv(out / "accuracy_by_family_geometry.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 200)
    print("\n== Summary ==")
    for k, v in summ.items():
        print(f"  {k:42s} {v:.3f}" if isinstance(v, float) else f"  {k:42s} {v}")
    print("\n== By family and geometry ==")
    print(by_fam.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote windows_accuracy.csv, accuracy_by_family_geometry.csv, summary.json to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
