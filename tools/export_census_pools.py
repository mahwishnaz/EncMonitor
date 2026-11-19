#!/usr/bin/env python3
"""Export the census pool indices and verify them against the saved census records.

The census experiment (experiments/census/windows.py) draws its reference and monitoring pools from the partition
seed at run time and does not save them. This script draws them again with the same code and seeds:
household_sample() is imported from experiments/census/windows.py, and the pool block below is checked to be a
verbatim copy of that file. CPU only. Verification per seed:
  (1) pool sizes and positive rates equal those stored in conditions.json (window sizes n = 250 and n = 1,000);
  (2) the clean calibration and held-out windows (500 + 200) recomputed from the exported pools reproduce d_mix,
      d_site and heterogeneity of records.jsonl (n = 250 and n = 1,000).
Inputs (read only): experiments/census/cache/acs_<year>.npz (written by experiments/census/model.py) and
<records>/records_n250, <records>/records_n1000 (written by experiments/census/windows.py, see scripts/run_seeds.sh).
Output: <out>/census/pools/seed_<seed>.npz with int32 row indices into experiments/census/cache/acs_<year>.npz:
  ref_site_<s> (reference pool, 2014), mon_site_<s>_<year> (monitoring pool; 2014 = clean monitoring pool), and
  <out>/census_pool_verification.json.
Usage (from the repository root):
  python -B tools/export_census_pools.py --out <folder> [--records outputs/census] [--cache experiments/census/cache]"""
import argparse, sys, json, importlib.util
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, yaml

REL = Path(__file__).resolve().parents[1]            # repository root
ap = argparse.ArgumentParser(description="Export and verify the census pool indices")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--records", type=Path, default=REL / "outputs/census")
ap.add_argument("--cache", type=Path, default=REL / "experiments/census/cache")
args = ap.parse_args()
CACHE = args.cache.resolve()
REC = {250: args.records.resolve() / "records_n250", 1000: args.records.resolve() / "records_n1000"}
OUT = args.out.resolve() / "census" / "pools"; OUT.mkdir(parents=True, exist_ok=True)

src = (REL / "experiments/census/windows.py").read_text()
BLOCK = '''        for s, st in enumerate(FIPS):
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
                pools[(s, y)] = household_sample(dy["serial"], rows_y, N_POOL, np.random.default_rng([seed, s, y]))'''
assert BLOCK in src, "pool block differs from the release file"
sys.path.insert(0, str(REL)); sys.path.insert(0, str(REL / "experiments/common"))
spec = importlib.util.spec_from_file_location("cw", REL / "experiments/census/windows.py")
cw = importlib.util.module_from_spec(spec); spec.loader.exec_module(cw)
from encmonitor.embeddings import EmbeddingStandardizer
from encmonitor.rff import RFFMap, median_bandwidth
from encmonitor.statistics import compute_dual_statistics
household_sample, FIPS, YEARS, N_POOL, S, M = cw.household_sample, cw.FIPS, cw.YEARS, cw.N_POOL, cw.S, cw.M
SEEDS = [int(s) for s in yaml.safe_load(open(REL / "configs/census.yaml"))["partition_seeds"]]
data = {y: dict(np.load(CACHE / f"acs_{y}.npz", allow_pickle=True)) for y in [2014] + YEARS}

report = []
for seed in SEEDS:
    d14 = data[2014]; pools = {}
    exec(compile("if True:\n" + BLOCK, "pool_block", "exec"))
    # (1) sizes and positive rates against conditions.json
    ok1 = True
    for n, root in REC.items():
        info = json.loads((root / f"seed_{seed}" / "conditions.json").read_text())
        ok1 &= info["pool_sizes"] == {f"{s}_{k}": int(v.size) for (s, k), v in pools.items()}
        ok1 &= info["ref_positive_rate"] == [float(d14["label"][pools[(s, 'ref')]].mean()) for s in range(S)]
        ok1 &= info["positive_rate"] == {str(y): [float(data[y]["label"][pools[(s, y)]].mean()) for s in range(S)] for y in [2014] + YEARS}
    # (2) clean windows recomputed from the pools (same steps as windows.py main())
    ref_raw = [d14["rep"][pools[(s, "ref")]].astype(np.float64) for s in range(S)]
    x = np.concatenate(ref_raw)
    prep = EmbeddingStandardizer.fit(x, clip_radius=8.0)
    sigma = median_bandwidth(prep.transform(x), max_samples=2000, seed=1729)
    rff = RFFMap.fit(x.shape[1], M, sigma, 1729)
    base = np.stack([rff.mean_sketch(prep.transform(r)) for r in ref_raw])
    feats = [rff.transform(prep.transform(d14["rep"][pools[(s, 2014)]].astype(np.float64))) for s in range(S)]
    w = np.ones(S) / S
    maxdiff = {}
    for n, root in REC.items():
        saved = [json.loads(l) for l in open(root / f"seed_{seed}" / "records.jsonl")]
        saved = [r for r in saved if r["condition"] in ("clean_calibration", "clean_heldout")]
        out = []
        for W_, ws in ((500, seed), (200, seed + 1000)):
            rng = np.random.default_rng(ws)
            for _ in range(W_):
                means = []
                for s in range(S):
                    idx = rng.choice(feats[s].shape[0], n, replace=feats[s].shape[0] < n)
                    means.append(feats[s][idx].mean(0))
                st = compute_dual_statistics(np.stack(means) - base, w)
                out.append((st.d_mix, st.d_site, st.heterogeneity))
        a = np.array(out); b = np.array([(r["d_mix"], r["d_site"], r["heterogeneity"]) for r in saved])
        assert a.shape == b.shape, (seed, n, a.shape, b.shape)
        maxdiff[n] = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    np.savez_compressed(OUT / f"seed_{seed}.npz",
                        **{f"ref_site_{s}": pools[(s, 'ref')].astype(np.int32) for s in range(S)},
                        **{f"mon_site_{s}_{y}": pools[(s, y)].astype(np.int32) for s in range(S) for y in [2014] + YEARS})
    report.append({"seed": seed, "sizes_and_positive_rates_equal": bool(ok1),
                   "clean_windows": 700, "max_rel_diff_n250": maxdiff[250], "max_rel_diff_n1000": maxdiff[1000]})
    print(report[-1], flush=True)
json.dump(report, open(args.out.resolve() / "census_pool_verification.json", "w"), indent=1)
