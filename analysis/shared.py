"""Shared helpers of the analysis scripts (analysis/ and analysis/tables/). Imports the code of this repository.

- SETS: every record set with its folder (20 seeds where they exist; 2026-2030 are the original five seeds).
- score(): the scoring of experiments/private_localany/run.py (Table 1 code), unchanged, plus matched-FPR alerts.
- Ctx: experiments/common/rx_windows.SeedContext for any cache folder (patch, slide, patient split).
- CensusCtx: the pools and window sampling of experiments/census/windows.py, returning residual Gram matrices.
- commit(): the commit hash of this repository (written into every result file header).
Records are read from outputs/ (OUT); analysis results are written below outputs/analysis/ (RESULTS).
Not run directly; imported by the other scripts.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CLONE = Path(__file__).resolve().parents[1]   # repository root
OUT = CLONE / "outputs"
RESULTS = OUT / "analysis"
for p in (CLONE / "experiments/common", CLONE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rx_common as rc  # noqa: E402
import rx_windows as rw  # noqa: E402
from encmonitor.embeddings import EmbeddingStandardizer  # noqa: E402
from encmonitor.rff import RFFMap, median_bandwidth  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P1 = _load("p1_release", CLONE / "experiments/private_localany/run.py")
arl_2of3 = P1.arl_2of3

OLD = [2026, 2027, 2028, 2029, 2030]
NEW = list(range(2031, 2046))
ALL = OLD + NEW

# name: (record key for masks, cache/record folder, dataset of the window generator or None)
SETS = {
    "mnist": ("mnist", OUT / "main/mnist", "mnist"),
    "cifar10": ("cifar10", OUT / "main/cifar10", "cifar10"),
    "camelyon17_patch": ("camelyon17", OUT / "camelyon/main/camelyon17", "camelyon17"),
    "camelyon17_slide": ("camelyon17", CLONE / "experiments/camelyon_slide_split/cache/camelyon17", "camelyon17"),
    "camelyon17_patient": ("camelyon17", OUT / "camelyon/patient_split/camelyon17", "camelyon17"),
    "census_n250": ("folktables", OUT / "census/records_n250", None),
    "census_n1000": ("folktables", OUT / "census/records_n1000", None),
}
METHODS = {"pooled": "d_mix", "encmon_joint": "encmon_score", "localany": "localmax_score", "hybrid": "hybrid_score"}
LABEL = {"pooled": "Pooled", "encmon_joint": "EncMonitor", "localany": "Local-any", "hybrid": "Hybrid",
         "encmon_or": "OR", "exact": "Exact-MMD", "encmon_norm": "EncMonitor-norm (i)"}
T = {5: 2.776, 20: 2.093}


def commit() -> str:
    return subprocess.check_output(["git", "-C", str(CLONE), "rev-parse", "HEAD"], text=True).strip()


def write_csv(df: pd.DataFrame, path: Path, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# commit {commit()}; {note}\n")
        df.to_csv(fh, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def seeds_of(name: str) -> list[int]:
    d = SETS[name][1]
    return sorted(int(p.name.split("_")[1]) for p in d.glob("seed_*") if (p / "records.jsonl").exists())


def load_records(name: str, seeds=None) -> pd.DataFrame:
    d = SETS[name][1]
    seeds = seeds or seeds_of(name)
    frames = [pd.read_json(d / f"seed_{s}" / "records.jsonl", lines=True) for s in seeds]
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = SETS[name][0]
    return df


def score(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Table 1 scoring (private_localany/run.py `score`) for one seed + matched-FPR alerts (held-out 95th pct)."""
    sc, info = P1.score(g)
    held = sc[sc.condition == "clean_heldout"]
    for k, col in METHODS.items():
        sc[f"m_{k}"] = sc[col] > rc.qh(held[col], 0.95)
    if "exact_mmd" in sc and sc.exact_mmd.notna().all():
        sc["m_exact"] = sc.exact_mmd > rc.qh(held.exact_mmd, 0.95)
    return sc, info


def scored(name: str, seeds=None) -> pd.DataFrame:
    df = load_records(name, seeds)
    return pd.concat([score(g)[0] for _, g in df.groupby("partition_seed")], ignore_index=True)


def summ(v, t=None) -> dict:
    v = np.asarray(v, dtype=float)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")
    t = t or T.get(v.size)
    h = t * sd / sqrt(v.size) if t else float("nan")
    return {"n_seeds": int(v.size), "mean": m, "sd": sd, "t": t, "t_lo": m - h, "t_hi": m + h}


def sign_test(d) -> tuple[int, int, int, float]:
    d = np.asarray(d, dtype=float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    k, n = min(pos, neg), pos + neg
    if n == 0:
        return pos, neg, int(d.size), 1.0
    return pos, neg, int(d.size - n), min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---------------------------------------------------------------- window generators
class Ctx(rw.SeedContext):
    """SeedContext for an explicit cache folder (and optionally another clip radius)."""

    def __init__(self, ds: str, seed: int, cache_dir: Path, clip: float | None = None):
        saved = rw.CACHES[ds]
        rw.CACHES[ds] = str(cache_dir)          # absolute path: ROOT / abs = abs
        try:
            super().__init__(ds, seed)
        finally:
            rw.CACHES[ds] = saved
        self.cache_dir = Path(cache_dir) / f"seed_{seed}"
        if clip is not None and clip != self.prep.clip_radius:
            self._refit(clip)

    def _refit(self, clip: float):
        """Same fit as scripts 14/32 with another clip radius: standardizer, median bandwidth, RFF map, baselines."""
        bc = np.load(self.cache_dir / "baseline_clean.npz")
        ref = [bc[f"site_{s:03d}"].astype(np.float64) for s in range(self.S)]
        x = np.concatenate(ref)
        self.prep = EmbeddingStandardizer.fit(x, clip_radius=float(clip))
        sigma0 = median_bandwidth(self.prep.transform(x), max_samples=2000, seed=int(self.cfg["rff_seed"]))
        self.rff = RFFMap.fit(x.shape[1], int(self.cfg["rff_dim"]), sigma0 * float(self.cfg["bandwidth_multiplier"]),
                              int(self.cfg["rff_seed"]))
        self.base = np.stack([self.rff.mean_sketch(self.prep.transform(r)) for r in ref])
        self.clean_feats = [self.feats(r) for r in self.clean_raw]
        self.sigma0 = sigma0

    def all_grams(self, conds: list[dict]) -> dict:
        cfg = self.cfg
        out = {"clean_calibration": self.run(self.clean_feats, int(cfg["calibration_windows"]), self.seed)["gram"],
               "clean_heldout": self.run(self.clean_feats, int(cfg["heldout_windows"]), self.seed + 1000)["gram"]}
        for c in conds:
            out[c["condition"]] = self.run(self.condition_feats(c), int(cfg["windows_per_condition"]), self.seed + 2000)["gram"]
        return out


class CensusCtx:
    """Pools, standardizer, RFF map and window sampling of experiments/census/windows.py (same seeds and streams);
    returns residual Gram matrices instead of statistics. Verified against the saved records by the callers."""

    def __init__(self, seed: int, n: int):
        W = _load("census_windows", CLONE / "experiments/census/windows.py")
        self.W, self.seed, self.n, self.S = W, int(seed), int(n), W.S
        here = CLONE / "experiments/census"
        if not hasattr(CensusCtx, "_data"):
            CensusCtx._data = {y: dict(np.load(here / "cache" / f"acs_{y}.npz", allow_pickle=True)) for y in [2014] + W.YEARS}
        data = CensusCtx._data
        d14 = data[2014]
        pools = {}
        for s, st in enumerate(W.FIPS):
            rows = np.flatnonzero((d14["st"] == st) & ~d14["train"])
            both = W.household_sample(d14["serial"], rows, 2 * W.N_POOL, np.random.default_rng([seed, s, 41]))
            cut_h = d14["serial"][both[W.N_POOL - 1]]
            ref, mon = both[:W.N_POOL], both[W.N_POOL:]
            mon = mon[d14["serial"][mon] != cut_h]
            pools[(s, "ref")], pools[(s, 2014)] = ref, mon
            for y in W.YEARS:
                dy = data[y]
                pools[(s, y)] = W.household_sample(dy["serial"], np.flatnonzero(dy["st"] == st), W.N_POOL,
                                                   np.random.default_rng([seed, s, y]))
        ref_raw = [d14["rep"][pools[(s, "ref")]].astype(np.float64) for s in range(self.S)]
        x = np.concatenate(ref_raw)
        self.prep = EmbeddingStandardizer.fit(x, clip_radius=8.0)
        sigma = median_bandwidth(self.prep.transform(x), max_samples=2000, seed=1729)
        self.rff = RFFMap.fit(x.shape[1], W.M, sigma, 1729)
        self.base = np.stack([self.rff.mean_sketch(self.prep.transform(r)) for r in ref_raw])
        self.mon_raw = [d14["rep"][pools[(s, 2014)]].astype(np.float64) for s in range(self.S)]
        self.mon_lab = [d14["label"][pools[(s, 2014)]] for s in range(self.S)]
        self.mon_age = [(d14["agep"][pools[(s, 2014)]] >= 45).astype(int) for s in range(self.S)]
        v = np.random.default_rng([seed, 7]).standard_normal(x.shape[1])
        self.v = v / np.linalg.norm(v)
        self.pools, self.data = pools, data
        self.clean = [self.feats(r) for r in self.mon_raw]

    def feats(self, raw):
        return self.rff.transform(self.prep.transform(raw))

    def conditions(self, only_real: bool) -> list[dict]:
        W, conds = self.W, []
        for y in W.YEARS:
            for g in ("single", "coherent", "all"):
                conds.append({"condition": f"year{y}__{g}", "family": "year", "kind": "year", "severity": float(y - 2014), "year": y, "geometry": g})
        for fam in ("label_prior", "age_mix"):
            for q in W.SEV:
                for g in ("single", "coherent", "opposing", "mixed"):
                    conds.append({"condition": f"{fam}{q:g}__{g}", "family": fam, "kind": fam, "severity": q, "geometry": g})
        for e in W.VSEV:
            for g in ("single", "coherent", "opposing", "mixed"):
                conds.append({"condition": f"vector{e:g}__{g}", "family": "vector", "kind": "vector", "severity": e, "geometry": g})
        for c in conds:
            c["directions"] = [float(v) for v in W.GEO[c["geometry"]]]
        if only_real:
            conds = [c for c in conds if c["family"] == "year" or (c["family"] in ("label_prior", "age_mix") and c["geometry"] in ("opposing", "mixed"))]
        return conds

    def condition_feats(self, c):
        out = []
        for s in range(self.S):
            f = c["directions"][s]
            if f == 0:
                out.append(self.clean[s])
            elif c["kind"] == "year":
                dy = self.data[c["year"]]
                out.append(self.feats(dy["rep"][self.pools[(s, c["year"])]].astype(np.float64)))
            elif c["kind"] in ("label_prior", "age_mix"):
                q = c["severity"] * abs(f)
                groups = self.mon_lab[s] if c["kind"] == "label_prior" else self.mon_age[s]
                tag = 3 if c["kind"] == "label_prior" else 8
                idx = rw.label_prior_idx(groups, 1, q, np.sign(f), np.random.default_rng([self.seed, tag, s, int(round(q * 1000)), int(f > 0)]))
                out.append(self.feats(self.mon_raw[s][idx]))
            else:
                out.append(self.feats(self.mon_raw[s] + f * c["severity"] * self.v * self.prep.scale))
        return out

    def run(self, feats, windows, wseed):
        rng = np.random.default_rng(wseed)
        gram = np.empty((windows, self.S, self.S))
        for w in range(windows):
            means = np.stack([feats[s][rng.choice(feats[s].shape[0], self.n, replace=feats[s].shape[0] < self.n)].mean(0)
                              for s in range(self.S)])
            res = means - self.base
            gram[w] = res @ res.T
        return gram

    def all_grams(self, conds):
        W = self.W
        out = {"clean_calibration": self.run(self.clean, W.CAL, self.seed),
               "clean_heldout": self.run(self.clean, W.HELD, self.seed + 1000)}
        for c in conds:
            out[c["condition"]] = self.run(self.condition_feats(c), W.PER, self.seed + 2000)
        return out


def grams_for(name: str, seed: int, clip: float | None = None) -> tuple[dict, list[dict]]:
    """All windows of one record set and seed as residual Gram matrices, in the record order."""
    key, d, ds = SETS[name]
    if ds is None:
        n = 1000 if name.endswith("n1000") else 250
        ctx = CensusCtx(seed, n)
        conds = ctx.conditions(only_real=(n == 1000))
    else:
        ctx = Ctx(ds, seed, d, clip)
        conds = rw.conditions(ds)
    return ctx.all_grams(conds), conds


def w_inverse_energy(cal_gram: np.ndarray) -> np.ndarray:
    e0 = np.einsum("wss->ws", cal_gram).mean(axis=0)
    w = 1.0 / e0
    return w / w.sum()


def json_dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=1, default=float))


def cfg(name: str) -> dict:
    return yaml.safe_load(open(CLONE / ("configs/camelyon.yaml" if "camelyon" in name else "configs/main.yaml")))
